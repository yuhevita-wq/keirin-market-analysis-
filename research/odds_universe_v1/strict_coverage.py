"""Strict per-race historical odds availability audit across CSV and grade ZIPs.

Checks numeric completeness by actual combinations (not file existence).
No user-facing positive coverage claim is made for a race whose offered book
contains missing, duplicate, nonnumeric, or unavailable combinations.
Uses result records only to mark the presence of grading labels.
"""
from __future__ import annotations
import csv, io, itertools, json, math, zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/audits/strict_market_coverage.json"

def iter_csv(path):
    with open(path,encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)

def iter_zip(z,name):
    with z.open(name) as raw:
        t=io.TextIOWrapper(raw,encoding="utf-8-sig",newline="")
        yield from csv.DictReader(t)

def encoded_combo(cars,is_trifecta):
    if is_trifecta:
        return (cars[0]-1)*81+(cars[1]-1)*9+(cars[2]-1)
    cars=sorted(cars)
    return (cars[0]-1)*81+(cars[1]-1)*9+(cars[2]-1)

def expected_masks(cars):
    tri=0;trio=0
    for c in itertools.permutations(cars,3):
        tri|=1<<encoded_combo(c,True)
    for c in itertools.combinations(cars,3):
        trio|=1<<encoded_combo(c,False)
    return tri,trio

def status(row):
    if row.get("odds_status")!="available":return False
    if row.get("odds_phase") not in ("final",):
        return False
    try:
        v=float(row["odds"])
        return math.isfinite(v) and v>=1
    except (KeyError,ValueError):
        return False

def outcome_ids(rows):
    d=defaultdict(set)
    for row in rows:
        try:
            p=int(row.get("finish_position",0))
            if p in (1,2,3):d[row["race_id"]].add(p)
        except (KeyError,ValueError):pass
    return {r for r,p in d.items() if p=={1,2,3}}

def evaluate(source_name,files):
    """files provides streaming iterators for races/entries/results/trifecta/trio."""
    races={}
    for r in files["races"]:
        rid=r.get("race_id","")
        if rid:races[rid]=r
    cars=defaultdict(set)
    for e in files["entries"]:
        try:
            rid=e["race_id"]
            c=int(e["car_no"])
            if rid in races and 1<=c<=9: cars[rid].add(c)
        except (KeyError,ValueError):pass
    labelled=outcome_ids(files["results"]) if "results" in files else set()
    market={}
    for kind in ("trifecta","trio"):
        d=defaultdict(lambda:{"seen":0,"available":0,"bad":0,"raw":0,"duplicates":0})
        for r in files.get(kind,()):
            rid=r.get("race_id","")
            if rid not in races:continue
            v=d[rid];v["raw"]+=1
            try:
                key=tuple(int(x) for x in r["combination"].split("-" if kind=="trifecta" else "="))
                if len(key)!=3 or len(set(key))!=3 or any(c not in cars[rid] for c in key):
                    v["bad"]+=1;continue
                ix=encoded_combo(key,kind=="trifecta")
                bit=1<<ix
                if v["seen"] & bit:v["duplicates"]+=1
                v["seen"]|=bit
                if status(r):v["available"]|=bit
                else:v["bad"]+=1
            except (KeyError,ValueError):
                v["bad"]+=1
        market[kind]=d
    counts=Counter()
    by_year=defaultdict(Counter)
    example=defaultdict(list)
    for rid,race in races.items():
        year=str(race.get("race_date",""))[:4]
        n=len(cars[rid])
        counts["races"]+=1
        by_year[year]["races"]+=1
        if rid in labelled:
            counts["result_labels"]+=1
            by_year[year]["result_labels"]+=1
        if n<3:
            tag="incomplete_entries"
        else:
            exp3,expt=expected_masks(sorted(cars[rid]))
            v3=market["trifecta"][rid]
            vt=market["trio"][rid]
            if v3["available"]==exp3 and vt["available"]==expt and not (v3["bad"] or vt["bad"] or v3["duplicates"] or vt["duplicates"]):
                tag="full_numeric_odds"
                counts["labelled_and_full_odds"]+=int(rid in labelled)
                by_year[year]["labelled_and_full_odds"]+=int(rid in labelled)
            elif v3["raw"]==0 and vt["raw"]==0:
                tag="no_odds"
            else:
                tag="partial_or_invalid_odds"
        counts[tag]+=1
        by_year[year][tag]+=1
        if tag!="full_numeric_odds" and len(example[tag])<10:
            example[tag].append({"race_id":rid,"race_date":race.get("race_date",""),
                                 "n":n,"tri3_count":market["trifecta"][rid]["raw"],
                                 "trio_count":market["trio"][rid]["raw"]})
    return {"source":source_name,"races":len(races),"counts":dict(counts),
            "by_year":{y:dict(c) for y,c in sorted(by_year.items())},
            "missing_examples":dict(example)}

def csv_sources():
    from inventory import ROOT as _root
    for p in sorted((ROOT/"data").glob("a_class_12/*/races.csv")):
        yield p.parent
    for p in sorted((ROOT/"data").glob("a3_challenge/*/races.csv")):
        yield p.parent
    for year in ("2023","2024","2025","2026_h1","2026_future_block1"):
        for p in sorted((ROOT/"data"/year).rglob("races.csv")):
            yield p.parent

def main():
    reports=[]
    for directory in csv_sources():
        streams={"races":iter_csv(directory/"races.csv"),
                 "entries":iter_csv(directory/"entries.csv") if (directory/"entries.csv").exists() else (),
                 "results":iter_csv(directory/"results.csv") if (directory/"results.csv").exists() else ()}
        p=directory/"trifecta_final_odds.csv"
        if p.exists():streams["trifecta"]=iter_csv(p)
        else:
            parts=sorted(directory.glob("trifecta_final_odds_part*.csv"))
            streams["trifecta"]=itertools.chain.from_iterable(iter_csv(x) for x in parts)
        p=directory/"trio_final_odds.csv"
        streams["trio"]=iter_csv(p) if p.exists() else ()
        x=evaluate(str(directory.relative_to(ROOT)),streams)
        reports.append(x)
        print("CSV",x["source"],x["counts"],flush=True)
    for path in sorted((ROOT/"data/grade_races").rglob("*.zip")):
        with zipfile.ZipFile(path) as z:
            entries=set(z.namelist())
            def get(name):
                matches=[m for m in entries if m.rsplit("/",1)[-1]==name]
                return iter_zip(z,matches[0]) if len(matches)==1 else ()
            if not any(m.endswith("races.csv") for m in entries):continue
            streams={"races":get("races.csv"),"entries":get("entries.csv"),
                     "results":get("results.csv"),"trifecta":get("trifecta_final_odds.csv"),
                     "trio":get("trio_final_odds.csv")}
            x=evaluate(str(path.relative_to(ROOT)),streams)
            reports.append(x)
            print("ZIP",x["source"],x["counts"],flush=True)
    # Overlaps are not claimed unique: one source row might reappear in another.
    family=defaultdict(Counter)
    for r in reports:
        path=r["source"]
        group="A1/A2" if "/a_class_12/" in path else ("A3" if "/a3_challenge/" in path
             else "G1G2" if "/grade_races/g1g2/" in path else
                  "G3" if "/grade_races/g3/" in path else "S_and_other_csv")
        family[group].update(r["counts"])
    report={"status":"STRICT_COMPLETE_NUMERIC_MARKET_COVERAGE",
            "asof_utc":datetime.now(timezone.utc).isoformat(),
            "definition":"full numeric 3連単 and 3連複 exact combination coverage, using entrant numbers and recorded final odds",
            "models_can_use":"only race IDs with full_numeric_odds AND result_labels; do not treat dataset rows as unique across sources",
            "known_limits":"CSV/ZIP archive sources only; nationwide racing calendar, historical gaps, and girls not yet independently reconciled",
            "family_source_row_counts":{k:dict(v) for k,v in sorted(family.items())},
            "sources":reports}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY",json.dumps(report["family_source_row_counts"],ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

"""Resumable, source-agnostic historical market odds collection.

Existing race-card CSVs are used ONLY to locate the odds URL and valid entrants.
Research features remain odds only; results are NEVER used in collection.
For each attempt, retain explicit failures and complete numeric market verification.
Writes one immutable compressed batch, so interrupted historical backfills can resume.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import itertools
import json
import math
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit

from keirin_market_analysis.historical_final_odds import scrape_race

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/odds_universe_v1"
FIELDS=[
    "race_id","race_date","track","race_no","race_type","ticket_type","combination",
    "odds","market_rank","odds_status","total_votes","odds_as_of","odds_phase",
    "odds_source","odds_source_url","captured_at_utc",
]
FAIL_FIELDS=["race_id","source","error","attempted_at_utc"]


def csv_read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)


def source_dirs():
    for parent in ("data/a_class_12","data/a3_challenge"):
        for r in sorted((ROOT/parent).glob("*/races.csv")):
            if (r.parent/"entries.csv").exists():
                yield r.parent


def used_rids(source):
    out=set()
    rel=source.relative_to(ROOT)
    ledger=BASE/rel
    if ledger.exists():
        for p in ledger.glob("batch_*/summary.json"):
            try:
                j=json.loads(p.read_text(encoding="utf-8"))
                out.update(j["attempted_race_ids"])
            except (KeyError,ValueError):pass
    return out


def choose_auto(limit):
    """Balanced A1/A2 and A3 coverage. Resumable from immutable manifests."""
    catalogs={"a_class_12":[],"a3_challenge":[]}
    ledger={}
    for src in source_dirs():
        family=src.relative_to(ROOT).parts[1]
        ledger[src]=used_rids(src)
        for r in csv_read(src/"races.csv"):
            if r.get("race_id") and r["race_id"] not in ledger[src]:
                catalogs[family].append((r.get("race_date",""),src,r))
    for family in catalogs:
        catalogs[family].sort(key=lambda z:(z[0],str(z[1]),z[2]["race_id"]))
    quotas={"a_class_12":limit//2+limit%2,"a3_challenge":limit//2}
    chosen=[]
    for family in ("a_class_12","a3_challenge"):
        chosen += [(src,r) for _,src,r in catalogs[family][:quotas[family]]]
    if len(chosen)<limit:
        already={(str(src),r["race_id"]) for src,r in chosen}
        extra=[(src,r) for family in ("a_class_12","a3_challenge")
               for _,src,r in catalogs[family] if (str(src),r["race_id"]) not in already]
        chosen += extra[:limit-len(chosen)]
    return chosen[:limit]

def validate(race,entrants,three,trio):
    n=len(entrants)
    if n<3 or n>9:raise ValueError(f"unsupported entrant count: {n}")
    ec3=n*(n-1)*(n-2)
    ect=math.comb(n,3)
    if len(three)!=ec3 or len(trio)!=ect:
        raise ValueError(f"incomplete price grid {len(three)}/{ec3} 3連単, {len(trio)}/{ect} 3連複")
    ids={str(r["race_id"]) for r in three+trio}
    if ids!={race["race_id"]}:raise ValueError("race id mismatch")
    for rows,kind in ((three,"3連単"),(trio,"3連複")):
        if any(r["odds_phase"]!="final" or r["ticket_type"]!=kind for r in rows):
            raise ValueError("not verified final-market rows")
        if not all(r.get("odds_as_of") and r.get("total_votes") for r in rows):
            raise ValueError("missing market vote count or time")
        if len({str(r["combination"]) for r in rows})!=len(rows):
            raise ValueError("duplicate market combination")
        for r in rows:
            if r.get("odds_status")!="available":
                raise ValueError(f"incomplete numeric market {kind}: {r['combination']} {r.get('odds_status')}")
            odds=float(r["odds"])
            if not math.isfinite(odds) or odds<1:raise ValueError("nonfinite market odds")
    keys3={tuple(map(int,r["combination"].split("-"))) for r in three}
    keyt={tuple(sorted(map(int,r["combination"].split("=")))) for r in trio}
    if keys3!=set(itertools.permutations(entrants,3)):
        raise ValueError("3連単 exact combination universe mismatch")
    if keyt!=set(itertools.combinations(entrants,3)):
        raise ValueError("3連複 exact combination universe mismatch")
    for typ,rows in (("3連単",three),("3連複",trio)):
        if any(r.get("odds_source_url","").split("/")[2]!="keirin.kdreams.jp" for r in rows):
            raise ValueError("source host mismatch")
    return True


def process(src,race,entries,delay):
    rid=race["race_id"]
    cars=entries.get(rid,[])
    if not re.fullmatch(r"\d{16}",rid):
        raise ValueError("invalid 16-digit KDreams race ID")
    host=urlsplit(race.get("source_url","")).netloc.lower()
    if host!="keirin.kdreams.jp":
        raise ValueError(f"unsupported odds provider {host}")
    if not (3<=len(cars)<=9):
        raise ValueError(f"missing or unsupported entrants ({len(cars)})")
    a,b=scrape_race(race,cars,delay)
    validate(race,cars,a,b)
    return a,b


def collect(tasks,delay,workers=2,tag="manual"):
    if not tasks:
        print("NO_UNATTEMPTED_RACES",flush=True)
        return {"attempted":0,"complete":0,"failed":0,"outputs":[]}
    groups=defaultdict(list)
    for src,r in tasks:groups[src].append(r)
    outputs=[]
    now=datetime.now(timezone.utc).isoformat()
    for src,rs in sorted(groups.items(),key=lambda x:str(x[0])):
        em=defaultdict(set)
        for e in csv_read(src/"entries.csv"):
            if e.get("race_id") and str(e.get("car_no","")).isdigit():
                em[e["race_id"]].add(int(e["car_no"]))
        em={rid:sorted(cars) for rid,cars in em.items()}
        successes=[];failed=[]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fs={executor.submit(process,src,r,em,delay):r for r in rs}
            for f in as_completed(fs):
                r=fs[f]
                try:
                    a,b=f.result()
                    successes.append((r["race_id"],a,b))
                except Exception as e:
                    failed.append({"race_id":r["race_id"],"source":str(src.relative_to(ROOT)),
                                   "error":f"{type(e).__name__}: {str(e)[:700]}",
                                   "attempted_at_utc":datetime.now(timezone.utc).isoformat()})
        successes.sort(key=lambda x:x[0])
        failed.sort(key=lambda x:x["race_id"])
        ids=sorted(r["race_id"] for r in rs)
        # Stable content-addressed path prevents accidental overwrites.
        import hashlib
        digest=hashlib.sha256("\n".join(ids).encode()).hexdigest()[:14]
        out=BASE/src.relative_to(ROOT)/("batch_"+digest)
        if out.exists():
            raise RuntimeError("immutable batch already exists: "+str(out))
        out.mkdir(parents=True,exist_ok=False)
        for fn,ix in (("trifecta_final_odds.csv.gz",1),("trio_final_odds.csv.gz",2)):
            with gzip.open(out/fn,"wt",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore")
                w.writeheader()
                for _,a,b in successes:
                    w.writerows(a if ix==1 else b)
        with (out/"failures.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FAIL_FIELDS)
            w.writeheader();w.writerows(failed)
        summary={
            "status":"BATCH_FINISHED_WITH_ERRORS" if failed else "COMPLETE",
            "batch_source":str(src.relative_to(ROOT)),
            "tag":tag,"created_at_utc":now,
            "attempted":len(rs),"complete":len(successes),"failed":len(failed),
            "attempted_race_ids":ids,
            "complete_race_ids":[rid for rid,_,_ in successes],
            "error_race_ids":[r["race_id"] for r in failed],
            "full_numeric_market_required":True,
            "features_used":"odds only; existing race card used solely as source URL and entrant keys",
            "warning":"Historical final odds, not T-15 market. Missing and unavailable combinations are rejected."
        }
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        outputs.append(str(out.relative_to(ROOT)))
        print("DONE",json.dumps({"out":outputs[-1],"complete":summary["complete"],
                                "failed":summary["failed"],
                                "failure_examples":failed[:3]},ensure_ascii=False),flush=True)
    return {"attempted":len(tasks),"complete":sum(json.loads((ROOT/o/"summary.json").read_text())["complete"] for o in outputs),
            "failed":sum(json.loads((ROOT/o/"summary.json").read_text())["failed"] for o in outputs),"outputs":outputs}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",help="Existing race directory, e.g. data/a_class_12/2025_01_02")
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--limit",type=int,default=16)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--sleep",type=float,default=.5)
    ap.add_argument("--auto",action="store_true")
    ap.add_argument("--tag",default="manual")
    arg=ap.parse_args()
    if not 1<=arg.limit<=200:ap.error("limit must be in [1,200]")
    if not 1<=arg.workers<=3:ap.error("workers must be in [1,3]")
    if not .25<=arg.sleep<=10:ap.error("sleep must be between 0.25 and 10 seconds")
    if arg.auto:
        tasks=choose_auto(arg.limit)
    else:
        if not arg.source:ap.error("--source or --auto required")
        folder=(ROOT/arg.source).resolve()
        if ROOT not in folder.parents:ap.error("--source must resolve within repository")
        if not (folder/"races.csv").exists() or not (folder/"entries.csv").exists():
            ap.error("missing races.csv or entries.csv")
        if not any(folder==src for src in source_dirs()):
            ap.error("source not part of audited A1/A2 or A3 historical directories")
        if arg.offset<0:ap.error("offset cannot be negative")
        races=list(csv_read(folder/"races.csv"))
        tasks=[(folder,r) for r in races[arg.offset:arg.offset+arg.limit]]
    output=collect(tasks,arg.sleep,arg.workers,arg.tag)
    print("FINISHED",json.dumps(output,ensure_ascii=False),flush=True)
    if output["attempted"] and output["complete"]==0:
        raise SystemExit("all attempted races failed; no coverage claimed")


if __name__=="__main__":
    main()

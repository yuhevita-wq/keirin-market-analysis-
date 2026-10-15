"""Recover exact missing whole-book final odds for rescued G1/G2/G3 ZIP archives.

This repairs *gaps* rather than rescraping the 11,000+ already archived races.
Every attempted race is preserved with either complete numeric market and
source provenance, or a recorded failure reason.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,itertools,json,math,zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path

from keirin_market_analysis.historical_final_odds import scrape_race
from backfill import ROOT,BASE,FIELDS,FAIL_FIELDS,validate
from scope import in_scope, girls_race


def read_zcsv(z,name):
    names=[x for x in z.namelist() if x.rsplit("/",1)[-1]==name]
    if len(names)!=1:return
    with z.open(names[0]) as raw:
        text=io.TextIOWrapper(raw,encoding="utf-8-sig",newline="")
        yield from csv.DictReader(text)


def archive_candidates(path):
    with zipfile.ZipFile(path) as z:
        races={r.get("race_id"):r for r in read_zcsv(z,"races.csv") if r.get("race_id")}
        em=defaultdict(set)
        for r in read_zcsv(z,"entries.csv"):
            try:
                rid=r["race_id"];c=int(r["car_no"])
                if rid in races and 1<=c<=9:em[rid].add(c)
            except (KeyError,ValueError):pass
        known={}
        for kind,file in (("trifecta","trifecta_final_odds.csv"),("trio","trio_final_odds.csv")):
            d=defaultdict(lambda:{"good":0,"bad":0})
            for row in read_zcsv(z,file):
                rid=row.get("race_id")
                if rid not in races:continue
                try:
                    odd=float(row["odds"])
                    good=row.get("odds_status")=="available" and math.isfinite(odd) and odd>=1
                except (KeyError,ValueError):good=False
                d[rid]["good" if good else "bad"]+=1
            known[kind]=d
        done=set()
        for rid,cars in em.items():
            n=len(cars)
            tri=known["trifecta"][rid]
            trio=known["trio"][rid]
            if tri["good"]==n*(n-1)*(n-2) and tri["bad"]==0 and trio["good"]==math.comb(n,3) and trio["bad"]==0:
                done.add(rid)
        # Class/type serves only as a girls exclusion; not a predictive feature.
        candidates=[(path,r,sorted(em.get(rid,set()))) for rid,r in sorted(races.items(),key=lambda x:(x[1].get("race_date",""),x[0]))
                    if rid not in done and in_scope(r.get("race_date")) and not girls_race(r.get("race_type"))]
        return candidates


def previous_attempts(archive):
    dest=BASE/"grade_archive_repairs"/archive.stem
    attempted=set()
    for file in dest.glob("batch_*/summary.json"):
        try:
            doc=json.loads(file.read_text(encoding="utf-8"))
            attempted.update(doc["attempted_race_ids"])
        except (ValueError,KeyError):pass
    return attempted


def choose(limit,source=None):
    paths=sorted((ROOT/"data/grade_races").rglob("*.zip"))
    if source:
        selected=(ROOT/source).resolve()
        if selected not in paths:raise ValueError("--source must refer to a graded archive in repository")
        paths=[selected]
    found=[]
    for path in paths:
        existing=previous_attempts(path)
        for orig,r,cars in archive_candidates(path):
            if r["race_id"] not in existing:
                found.append((orig,r,cars))
                if len(found)>=limit:return found
    return found


def run(tasks,delay=0.6,workers=2):
    if not tasks:
        print("NO_MISSING_GRADED_ODDS_REMAIN_TO_ATTEMPT",flush=True)
        return
    grouped=defaultdict(list)
    for arch,r,c in tasks:grouped[arch].append((r,c))
    for arch,rows in sorted(grouped.items(),key=lambda x:str(x[0])):
        failures=[];oks=[]
        def task(r,c):
            a,b=scrape_race(r,c,delay)
            validate(r,c,a,b)
            return a,b
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(task,r,c):r for r,c in rows}
            for future in as_completed(fs):
                r=fs[future]
                try:
                    a,b=future.result()
                    oks.append((r["race_id"],a,b))
                except Exception as exc:
                    failures.append({"race_id":r["race_id"],"source":str(arch.relative_to(ROOT)),
                                     "error":f"{type(exc).__name__}: {exc}"[:700],
                                     "attempted_at_utc":datetime.now(timezone.utc).isoformat()})
        oks.sort(key=lambda x:x[0])
        ids=sorted(r["race_id"] for r,c in rows)
        key=hashlib.sha256("\n".join(ids).encode()).hexdigest()[:14]
        dest=BASE/"grade_archive_repairs"/arch.stem/("batch_"+key)
        if dest.exists():raise ValueError("refusing to overwrite immutable grade odds batch")
        dest.mkdir(parents=True,exist_ok=False)
        for name,idx in (("trifecta_final_odds.csv.gz",1),("trio_final_odds.csv.gz",2)):
            with gzip.open(dest/name,"wt",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore")
                w.writeheader()
                for _,a,b in oks:w.writerows(a if idx==1 else b)
        with (dest/"failures.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FAIL_FIELDS)
            w.writeheader()
            w.writerows(failures)
        summary={
            "source_archive":str(arch.relative_to(ROOT)),
            "attempted":len(rows),"complete":len(oks),"failed":len(failures),
            "attempted_race_ids":ids,
            "complete_race_ids":[rid for rid,_,_ in oks],
            "failed_race_ids":[x["race_id"] for x in failures],
            "status":"COMPLETE" if not failures else "BATCH_FINISHED_WITH_ERRORS",
            "odds_phase":"final",
            "source":"楽天Kドリームス historical confirmed odds",
            "reason":"repair missing G1 G2 and G3 archived full odds",
        }
        (dest/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("GRADE_RECOVERY",json.dumps({"path":str(dest.relative_to(ROOT)),"attempted":summary["attempted"],
                                           "complete":summary["complete"],"failed":summary["failed"],
                                           "examples":failures[:3]},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default=None)
    p.add_argument("--limit",type=int,default=32)
    p.add_argument("--sleep",type=float,default=.6)
    a=p.parse_args()
    if not 1<=a.limit<=250:p.error("limit must be [1,250]")
    if not .25<=a.sleep<=10:p.error("sleep must be [0.25,10]")
    run(choose(a.limit,a.source),a.sleep)


if __name__=="__main__":
    main()

"""Collect races absent from ALL archived sources as discovered by the public calendar.

Input race IDs/URLs come from an independent date census. Race-card fields only
establish runner identity and source provenance; predictor input remains odds.
Full numeric 3連単 + 3連複 is mandatory. Outcome rows are saved as LABELS only.
No undocumented extrapolation of omitted final odds.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,re,math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path

from bs4 import BeautifulSoup
from keirin_market_analysis import s_yosen_2025 as cards
from keirin_market_analysis.historical_final_odds import scrape_race
from keirin_market_analysis.results_payouts_2025 import result_url,parse_result_page
from backfill import ROOT,FIELDS,validate
from compare_calendar import saved_races
from scope import in_scope, girls_race

BASE=ROOT/"data/unarchived_races_v1"
RACE_COLS=["race_id","race_date","track","race_no","race_type","start_time","deadline","entry_count","source_url","captured_at_utc"]
ENTRY_COLS=["race_id","car_no","class"]
RESULT_COLS=["race_id","car_no","finish_position","result_source_url","captured_at_utc"]
ERROR_COLS=["race_id","race_date","stage","error","odds_source_url"]


def read_candidates():
    file=ROOT/"data/audits/nationwide_census_vs_saved.json"
    if not file.exists():return []
    report=json.loads(file.read_text(encoding="utf-8"))
    known=saved_races()
    done=set()
    for s in BASE.glob("batch_*/summary.json"):
        try:done.update(json.loads(s.read_text(encoding="utf-8"))["attempted_ids"])
        except (ValueError,KeyError):pass
    found={}
    for src in report.get("sources",[]):
        for r in src.get("unarchived_races",[]):
            rid=r["race_id"]
            if rid not in known and rid not in done and in_scope(r.get("race_date")):found[rid]=r
    return sorted(found.values(),key=lambda r:(r["race_date"],r["race_id"]))


def fetch_one(item):
    rid=item["race_id"]
    url=item["odds_page_url"]
    if not re.fullmatch(r"\d{16}",rid):raise ValueError("invalid race ID")
    if not in_scope(item.get("race_date")):
        raise ValueError("2026 second-half and out-of-window collection is forbidden")
    session=cards.make_session()
    html=cards.fetch_html(session,url)
    ref=cards.RaceRef(discovered_on=item["race_date"],race_no=int(rid[-4:]),url=url)
    prev=cards.TARGET_RACE_TYPE
    try:
        # The published racecard is NOT part of market-odds prediction features.
        cards.TARGET_RACE_TYPE="UNIVERSE_REPAIR"
        meta,entries=cards.extract_entries(html,ref,require_target_label=False)
    finally:
        cards.TARGET_RACE_TYPE=prev
    if meta["race_id"]!=rid or meta["race_date"]!=item["race_date"]:
        raise ValueError("race-id or date mismatch between census and racecard")
    classes=[str(e.get("class","")) for e in entries]
    if girls_race(meta.get("race_type"), classes):
        raise ValueError("girls race excluded; no odds retrieved")
    # Prefer the source daily-index race type if supplied by the census.
    race_type=str(item.get("race_type","")).strip()
    if race_type and not girls_race(race_type, classes):
        meta["race_type"]=race_type
    cars=sorted({int(e["car_no"]) for e in entries})
    if len(cars)!=len(entries):
        raise ValueError("duplicate racer numbers")
    meta["entry_count"]=len(cars)
    def core(r):
        return {"race_id":rid,"car_no":int(r["car_no"]),"class":str(r.get("class",""))}
    entry_rows=[core(e) for e in entries]
    errors=[]
    try:
        tri,trio=scrape_race(meta,cars,.55)
        validate(meta,cars,tri,trio)
    except Exception as exc:
        errors.append({"race_id":rid,"race_date":item["race_date"],"stage":"odds",
                       "error":f"{type(exc).__name__}: {exc}"[:650],
                       "odds_source_url":url})
        tri,trio=[],[]
    labels=[]
    try:
        res_page=cards.fetch_html(session,result_url(url))
        res,payout=parse_result_page(res_page)
        observed={int(r["car_no"]) for r in res}
        pos=[(int(r["finish_position"]),int(r["car_no"]))
             for r in res if str(r.get("finish_position","")).isdigit()]
        if observed!=set(cars) or len({p for p,_ in pos if p in (1,2,3)})!=3:
            raise ValueError("results not complete for original racer set")
        when=datetime.now(timezone.utc).isoformat()
        labels=[{"race_id":rid,"car_no":r["car_no"],
                 "finish_position":r.get("finish_position",""),
                 "result_source_url":result_url(url),"captured_at_utc":when} for r in res]
    except Exception as exc:
        errors.append({"race_id":rid,"race_date":item["race_date"],"stage":"result",
                       "error":f"{type(exc).__name__}: {exc}"[:650],
                       "odds_source_url":url})
    return meta,entry_rows,labels,tri,trio,errors


def write_csv(path,cols,rows,gz=False):
    opener=gzip.open if gz else open
    with opener(path,"wt" if gz else "w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=16)
    a=ap.parse_args()
    if not 1<=a.limit<=200:ap.error("limit outside [1,200]")
    batch=read_candidates()[:a.limit]
    if not batch:
        print("NO_INDEXED_UNARCHIVED_RACES_TO_FETCH",flush=True)
        return
    rows=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        fs={executor.submit(fetch_one,r):r for r in batch}
        for future in as_completed(fs):
            r=fs[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"race_id":r["race_id"],"race_date":r["race_date"],
                               "stage":"racecard","error":f"{type(exc).__name__}: {exc}"[:650],
                               "odds_source_url":r["odds_page_url"]})
    rows.sort(key=lambda a:a[0]["race_id"])
    rid=sorted(r["race_id"] for r in batch)
    digest=hashlib.sha256("\n".join(rid).encode()).hexdigest()[:14]
    folder=BASE/("batch_"+digest)
    if folder.exists():raise RuntimeError("refusing overwrite of existing missing-race batch")
    folder.mkdir(parents=True,exist_ok=False)
    for fn,cols,idx in (
        ("races.csv.gz",RACE_COLS,0),
        ("entries.csv.gz",ENTRY_COLS,1),
        ("results.csv.gz",RESULT_COLS,2),
        ("trifecta_final_odds.csv.gz",FIELDS,3),
        ("trio_final_odds.csv.gz",FIELDS,4)):
        data=[r for item in rows for r in (item[idx] if idx else [item[idx]])]
        write_csv(folder/fn,cols,data,True)
    allerrors=errors+[err for row in rows for err in row[5]]
    write_csv(folder/"failures.csv",ERROR_COLS,allerrors)
    full_odds=sum(bool(r[3] and r[4]) for r in rows)
    with_label=sum(bool(r[2]) for r in rows)
    report={
        "status":"CENSUS_UNARCHIVED_RACE_RECOVERY",
        "attempted":len(batch),
        "race_cards_recovered":len(rows),
        "complete_odds_books":full_odds,
        "validated_result_labels":with_label,
        "both_market_and_labels":sum(bool(r[2] and r[3] and r[4]) for r in rows),
        "attempted_ids":rid,
        "error_count":len(allerrors),
        "source":"KDreams historical daily census + historical racedetail odds",
        "predictor_features":"odds only; results are labels, racecard used to fetch keys",
        "warning":"Non-discovered races remain unknown until the daily census is reconciled"
    }
    (folder/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RECOVER_NEW_RACES",json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

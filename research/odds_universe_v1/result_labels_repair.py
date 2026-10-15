"""Backfill missing historical 1st/2nd/3rd-place labels for odds research.

Results are TARGET LABELS ONLY and never predictor features.
A source race is eligible for fitting only after validated numeric odds and
top-three result labels exist. Result refunds/scratches remain explicit.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,re,math
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from keirin_market_analysis.s_yosen_2025 import fetch_html,make_session
from keirin_market_analysis.results_payouts_2025 import result_url,parse_result_rows,parse_payout_rows

from backfill import ROOT,source_dirs,csv_read
from scope import eligible

BASE=ROOT/"data/recovered_result_labels_v1"
OUTCOME_FIELDS=[
 "race_id","race_date","track","race_no","race_type","car_no",
 "finish_position","finish_text","result_source_url","captured_at_utc",
]
PAY_FIELDS=[
 "race_id","ticket_type","combination","payout_yen","status","result_source_url"
]
FAIL_FIELDS=["race_id","source","stage","error","attempted_at_utc"]


def result_exists(folder):
    f=folder/"results.csv"
    if not f.exists():return set()
    ranks=defaultdict(set)
    cars=defaultdict(set)
    for r in csv_read(f):
        try:
            p=int(r["finish_position"]);c=int(r["car_no"])
        except (KeyError,ValueError):continue
        if p in (1,2,3):
            ranks[r["race_id"]].add(p);cars[r["race_id"]].add(c)
    return {rid for rid in ranks if ranks[rid]=={1,2,3} and len(cars[rid])==3}


def attempted_ids(folder):
    out=set()
    for p in (BASE/folder.relative_to(ROOT)).glob("batch_*/summary.json"):
        try:out.update(json.loads(p.read_text(encoding="utf-8"))["attempted_race_ids"])
        except (KeyError,ValueError):pass
    return out


def choose(limit,source=None,offset=0):
    folders=sorted(source_dirs())
    if source:
        target=(ROOT/source).resolve()
        if target not in folders:raise ValueError("source must be an existing audited historical CSV directory")
        folders=[target]
    candidates={"A12":[],"A3":[],"S":[]}
    for folder in folders:
        good=result_exists(folder)
        attempted=attempted_ids(folder)
        key="A12" if "/a_class_12/" in str(folder) else "A3" if "/a3_challenge/" in str(folder) else "S"
        for race in csv_read(folder/"races.csv"):
            rid=race.get("race_id")
            if rid and rid not in good and rid not in attempted and eligible(race):
                candidates[key].append((folder,race))
    for k in candidates:
        candidates[k].sort(key=lambda z:(z[1].get("race_date",""),str(z[0]),z[1]["race_id"]))
    if source:
        return [x for group in candidates.values() for x in group][offset:offset+limit]
    selection=[]
    counts={"A12":0,"A3":0,"S":0}
    while len(selection)<limit:
        progressed=False
        for key in ("A12","A3","S"):
            if counts[key]<len(candidates[key]) and len(selection)<limit:
                selection.append(candidates[key][counts[key]])
                counts[key]+=1
                progressed=True
        if not progressed:break
    return selection


def fetch_one(race,expected_cars):
    if not eligible(race):
        raise ValueError("result labels out of scope or girls race")
    url=result_url(race.get("source_url",""))
    if urlsplit(url).netloc!="keirin.kdreams.jp" or not re.fullmatch(r"\d{16}",race.get("race_id","")):
        raise ValueError("untrusted result endpoint")
    ses=make_session()
    html=fetch_html(ses,url)
    soup=BeautifulSoup(html,"lxml")
    rr=parse_result_rows(soup)
    pairs=[]
    for row in rr:
        try:
            pos=int(row["finish_position"]);car=int(row["car_no"])
            if pos in (1,2,3):pairs.append((pos,car))
        except (ValueError,KeyError):continue
    if len(pairs)!=3 or {p for p,_ in pairs}!={1,2,3} or len({c for _,c in pairs})!=3:
        raise ValueError("missing or ambiguous podium")
    if not {c for _,c in pairs}.issubset(expected_cars):
        raise ValueError("podium references a rider absent from the source race")
    now=datetime.now(timezone.utc).isoformat()
    a=[]
    for row in rr:
        a.append({"race_id":race["race_id"],"race_date":race.get("race_date",""),
                  "track":race.get("track",""),"race_no":race.get("race_no",""),
                  "race_type":race.get("race_type",""),"car_no":row["car_no"],
                  "finish_position":row.get("finish_position",""),
                  "finish_text":row.get("finish_text",""),
                  "result_source_url":url,"captured_at_utc":now})
    try:
        payout=parse_payout_rows(soup)
        b=[{"race_id":race["race_id"],"ticket_type":r["ticket_type"],
            "combination":r.get("combination",""),"payout_yen":r.get("payout_yen",""),
            "status":r["status"],"result_source_url":url} for r in payout]
        payout_error=None
    except Exception as exc:
        b=[]
        payout_error=f"{type(exc).__name__}: {exc}"
    return a,b,payout_error


def write_gz(path,fields,rows):
    with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader();w.writerows(rows)


def run(tasks,workers=2):
    if not tasks:
        print("NO_MISSING_RESULT_LABELS_TO_ATTEMPT",flush=True);return
    by_folder=defaultdict(list)
    for folder,r in tasks:by_folder[folder].append(r)
    for folder,rr in sorted(by_folder.items(),key=lambda t:str(t[0])):
        expected=defaultdict(set)
        for r in csv_read(folder/"entries.csv"):
            try:expected[r["race_id"]].add(int(r["car_no"]))
            except (ValueError,KeyError):pass
        resultrows=[];payoutrows=[];errors=[];valid=set()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fs={pool.submit(fetch_one,r,expected[r["race_id"]]):r for r in rr}
            for f in as_completed(fs):
                r=fs[f]
                try:
                    a,b,err=f.result()
                    resultrows+=a;payoutrows+=b;valid.add(r["race_id"])
                    if err:
                        errors.append({"race_id":r["race_id"],"source":str(folder.relative_to(ROOT)),
                                       "stage":"payout","error":err,
                                       "attempted_at_utc":datetime.now(timezone.utc).isoformat()})
                except Exception as exc:
                    errors.append({"race_id":r["race_id"],"source":str(folder.relative_to(ROOT)),
                                   "stage":"result","error":f"{type(exc).__name__}: {exc}"[:650],
                                   "attempted_at_utc":datetime.now(timezone.utc).isoformat()})
        ids=sorted(x["race_id"] for x in rr)
        name="batch_"+hashlib.sha256("\n".join(ids).encode()).hexdigest()[:14]
        out=BASE/folder.relative_to(ROOT)/name
        if out.exists():raise RuntimeError("immutable result-label batch already exists")
        out.mkdir(parents=True,exist_ok=False)
        write_gz(out/"results.csv.gz",OUTCOME_FIELDS,sorted(resultrows,key=lambda r:(r["race_id"],int(r["car_no"]))))
        write_gz(out/"payouts.csv.gz",PAY_FIELDS,payoutrows)
        with (out/"failures.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FAIL_FIELDS);w.writeheader();w.writerows(errors)
        summary={"source":str(folder.relative_to(ROOT)),"attempted":len(rr),
                 "complete_podium":len(valid),
                 "failed_podium":len(rr)-len(valid),
                 "payout_parse_warnings":sum(e["stage"]=="payout" for e in errors),
                 "attempted_race_ids":ids,"complete_podium_ids":sorted(valid),
                 "actual_results_label_only":True,
                 "outcome_not_market_feature":True}
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("RESULT_RECOVERY",json.dumps({"summary":summary,"errors":errors[:3]},ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--auto",action="store_true")
    ap.add_argument("--source")
    ap.add_argument("--offset",type=int,default=0)
    ap.add_argument("--limit",type=int,default=48)
    a=ap.parse_args()
    if not 1<=a.limit<=250:ap.error("limit must be [1,250]")
    run(choose(a.limit,a.source,a.offset),workers=2)


if __name__=="__main__":
    main()

"""Discover missing girls races and collect complete historical odds and labels.

Race class/labels and race URLs are for UNIVERSE DISCOVERY only. Model features
must remain 3連単 and 3連複 odds, with actual results only grading targets.

No fabricated odds and no races silently excluded. Daily discovery misses and
market scraping failures are explicit in the dated source summary.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import re
import unicodedata
import time
from collections import Counter
from datetime import date,datetime,timedelta,timezone
from pathlib import Path

from bs4 import BeautifulSoup

from keirin_market_analysis import s_yosen_2025 as base
from keirin_market_analysis.a_class_12 import racecard_detail_url
from keirin_market_analysis.historical_final_odds import scrape_race
from keirin_market_analysis.results_payouts_2025 import result_url,parse_result_page

from backfill import validate,FIELDS

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/girls_universe"
RACE_FIELDS=[
    "race_id","race_date","track","race_no","race_type","start_time","deadline",
    "entry_count","source_url","captured_at_utc","discovered_on",
    "model_feature_rule",
]
ENTRY_FIELDS=["race_id","car_no","class"]
RESULT_FIELDS=["race_id","race_date","track","race_no","race_type","car_no",
               "finish_position","finish_text","result_source_url","captured_at_utc"]
FAILURE_FIELDS=["stage","date","race_id","url","error"]


def write_csv(path,cols,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        w.writeheader();w.writerows(rows)


def is_girls_label(text):
    label=unicodedata.normalize("NFKC",base.normalize_text(text)).replace(" ","")
    return bool(re.search(r"ガールズ|女子|^L級|^ガ予|^ガ決|^ガ選|^ガ一般",label))


def discover_daily(html, daily_url,day):
    soup=BeautifulSoup(html,"lxml")
    found={}
    for table in soup.find_all("table"):
        rows=table.find_all("tr")
        for ix,row in enumerate(rows):
            cells=base.expand_row_cells(row)
            labels=[base.normalize_text(cell.get_text(" ",strip=True)) for cell in cells]
            for col,label in enumerate(labels):
                if not is_girls_label(label):continue
                href=""
                for other in rows[ix+1:]:
                    cols=base.expand_row_cells(other)
                    if len(cols)<=col:continue
                    for a in cols[col].find_all("a",href=True):
                        if "/racedetail/" in str(a["href"]):
                            href=str(a["href"])
                            break
                    if href:break
                if not href:continue
                url=racecard_detail_url(daily_url,href)
                match=re.search(r"/racedetail/(\d{16})/",url)
                if not match:continue
                rid=match.group(1)
                ref=base.RaceRef(discovered_on=day,race_no=int(rid[-4:]),url=url)
                found[rid]=(ref,label)
    return found


def date_range(start,end):
    now=start
    while now<=end:
        yield now
        now+=timedelta(days=1)


def scrape(start,end,out,delay=0.35):
    if end>=datetime.now(timezone(timedelta(hours=9))).date():
        raise ValueError("research collector requires completed historical dates")
    if out.exists():
        raise ValueError("immutable girls batch already exists: "+str(out))
    s=base.make_session()
    failures=[]
    refs={}
    for d in date_range(start,end):
        day=d.isoformat()
        url=base.KDREAMS_DAILY.format(year=d.year,month=d.month,day=d.day)
        try:
            page=base.fetch_html(s,url)
            for rid,(ref,lab) in discover_daily(page,url,day).items():
                refs.setdefault(rid,(ref,lab))
        except Exception as exc:
            failures.append({"stage":"daily","date":day,"race_id":"","url":url,
                             "error":f"{type(exc).__name__}: {exc}"[:700]})
        if delay:time.sleep(delay)
    races=[];entries=[];results=[];tri=[];trio=[]
    rejected=Counter()
    for no,(rid,(ref,label)) in enumerate(sorted(refs.items()),1):
        try:
            html=base.fetch_html(s,ref.url)
            # TARGET_RACE_TYPE is not a feature. It preserves the source's label.
            previous=base.TARGET_RACE_TYPE
            try:
                base.TARGET_RACE_TYPE=label
                meta,es=base.extract_entries(html,ref,require_target_label=False)
            finally:
                base.TARGET_RACE_TYPE=previous
            actual_date=date.fromisoformat(str(meta["race_date"]))
            if not start<=actual_date<=end:
                rejected["outside_date_range"]+=1
                continue
            classes=[unicodedata.normalize("NFKC",str(e.get("class",""))).replace(" ","").upper() for e in es]
            if not classes or any(not k.startswith("L") for k in classes):
                rejected["non_girls_classes"]+=1
                continue
            # Only a race with its actual field is marked discovered.
            keys=sorted({int(e["car_no"]) for e in es})
            row={**meta,"race_type":label,"discovered_on":ref.discovered_on,
                 "model_feature_rule":"odds_only"}
            row["entry_count"]=len(keys)
            races.append(row)
            for e in es:entries.append({"race_id":rid,"car_no":int(e["car_no"]),"class":str(e.get("class",""))})
            try:
                a,b=scrape_race(row,keys,delay)
                validate(row,keys,a,b)
                tri.extend(a);trio.extend(b)
            except Exception as exc:
                failures.append({"stage":"odds","date":meta["race_date"],"race_id":rid,
                                 "url":ref.url,"error":f"{type(exc).__name__}: {exc}"[:700]})
            try:
                html_res=base.fetch_html(s,result_url(ref.url))
                rr,pp=parse_result_page(html_res)
                got={int(r["car_no"]) for r in rr}
                finishing={int(r["finish_position"]) for r in rr if str(r.get("finish_position","")).isdigit()}
                if got!=set(keys) or not {1,2,3}.issubset(finishing):
                    raise ValueError("incomplete result runner set or 1-3 positions")
                captured=datetime.now(timezone.utc).isoformat()
                for r in rr:
                    results.append({"race_id":rid,"race_date":meta["race_date"],
                                    "track":meta["track"],"race_no":meta["race_no"],
                                    "race_type":label,"car_no":r["car_no"],
                                    "finish_position":r.get("finish_position",""),
                                    "finish_text":r.get("finish_text",""),
                                    "result_source_url":result_url(ref.url),
                                    "captured_at_utc":captured})
            except Exception as exc:
                failures.append({"stage":"results","date":meta["race_date"],"race_id":rid,
                                 "url":result_url(ref.url),
                                 "error":f"{type(exc).__name__}: {exc}"[:700]})
        except Exception as exc:
            failures.append({"stage":"racecard","date":ref.discovered_on,"race_id":rid,"url":ref.url,
                             "error":f"{type(exc).__name__}: {exc}"[:700]})
        if delay:time.sleep(delay)
        if no%25==0:print("PROGRESS",no,len(refs),flush=True)
    out.mkdir(parents=True)
    write_csv(out/"races.csv",RACE_FIELDS,races)
    write_csv(out/"entries.csv",ENTRY_FIELDS,entries)
    write_csv(out/"results.csv",RESULT_FIELDS,results)
    write_csv(out/"failures.csv",FAILURE_FIELDS,failures)
    for fn,rows in (("trifecta_final_odds.csv.gz",tri),("trio_final_odds.csv.gz",trio)):
        with gzip.open(out/fn,"wt",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="ignore")
            w.writeheader();w.writerows(rows)
    resultids={r["race_id"] for r in results}
    oids={r["race_id"] for r in tri}
    counts=Counter(r["stage"] for r in failures)
    summary={
        "status":"HISTORICAL_GIRLS_SOURCE_GAPS_EXPLICIT",
        "date_range":[start.isoformat(),end.isoformat()],
        "source":"楽天Kドリームス published daily calendar and historical final odds",
        "days_attempted":(end-start).days+1,
        "daily_discovery_failures":counts["daily"],
        "discovered_girls_candidates":len(refs),
        "verified_girls_races":len(races),
        "full_odds_races":len(oids),
        "fully_labelled_races":len(resultids),
        "both_market_and_result_races":len(resultids & oids),
        "rejected":dict(rejected),
        "errors_by_stage":dict(counts),
        "model_features":"odds only; class, source race label and field size are solely for discovery and integrity checks",
        "warning":"Discovery depends on KDreams daily-page labels; gaps are not proof no girls race existed."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GIRLS_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def auto_week(today):
    done=set()
    if BASE.exists():
        for p in BASE.glob("*/summary.json"):
            try:
                a=json.loads(p.read_text(encoding="utf-8"))["date_range"]
                done.add((date.fromisoformat(a[0]),date.fromisoformat(a[1])))
            except (ValueError,KeyError):pass
    start=date(2021,1,1)
    yesterday=today-timedelta(days=1)
    while start+timedelta(days=6)<=yesterday:
        end=start+timedelta(days=6)
        if (start,end) not in done:return start,end
        start=end+timedelta(days=1)
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--auto",action="store_true")
    p.add_argument("--sleep",type=float,default=.35)
    a=p.parse_args()
    if not .25<=a.sleep<=10:p.error("sleep outside [0.25,10]")
    if a.auto:
        match=auto_week(datetime.now(timezone(timedelta(hours=9))).date())
        if match is None:
            print("ALL_HISTORICAL_WEEKS_DISCOVERED");return
        start,end=match
    else:
        if not a.start or not a.end:p.error("--start and --end needed")
        start,end=date.fromisoformat(a.start),date.fromisoformat(a.end)
    if start>end or (end-start).days>13:p.error("request a positive window of no more than 14 days")
    output=BASE/f"{start.isoformat()}__{end.isoformat()}"
    scrape(start,end,output,a.sleep)

if __name__=="__main__":
    main()

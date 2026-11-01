#!/usr/bin/env python3
"""2026 July-October: discover ALL men's race categories and recover only missing full entry/results.

Read canonical existing datasets first; never overwrite original input files or models.
Failures and uncategorized index links are recorded rather than imputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup
from keirin_market_analysis import a_class_12, a3_challenge, s_class_f1
from keirin_market_analysis import s_yosen_2025 as base
from keirin_market_analysis import results_payouts_2025 as result_parser

ROOT = Path(__file__).resolve().parents[2]
DAY_LOW, DAY_HIGH = date(2026, 7, 1), date(2026, 10, 31)
DEST = ROOT / "data/rider_condition_gap_2026"
RID = re.compile(r"/racedetail/(\d{16})/")
RACE_FIELDS = ["race_id","race_date","track","meeting_grade","race_no","race_type",
               "start_time","deadline","entry_count","source_url","captured_at_utc"]
ENTRY_FIELDS = ["race_id","race_date","track","meeting_grade","race_no","race_type",
                "start_time","deadline","source_url","captured_at_utc","car_no",
                "player_name","player_profile","prefecture","age","term","class",
                "style","gear","score","s_count","b_count","nige_count","makuri_count",
                "sashi_count","mark_count","first_count","second_count","third_count",
                "outside_count","win_rate","top2_rate","top3_rate","prediction_mark",
                "evaluation","raw_row_json"]
RESULT_FIELDS = ["race_id","race_date","track","race_no","race_type","car_no",
                 "player_name","finish_position","finish_text","margin","last200",
                 "winning_method","sb","result_comment","result_source_url",
                 "result_source","captured_at_utc","raw_row_json"]
FAIL_FIELDS = ["race_id","race_date","stage","url","reason"]

def norm(v):
    return unicodedata.normalize("NFKC",str(v or "")).strip()

def rid_day(rid):
    if not re.fullmatch(r"\d{16}",rid):
        return None
    try:
        number=int(rid[10:12])
        if not (1<=number<=7 and 1<=int(rid[-2:])<=12):
            return None
        return date(int(rid[2:6]),int(rid[6:8]),int(rid[8:10]))+timedelta(days=number-1)
    except ValueError:
        return None

def list_existing():
    # Exact data folders only. Do not scan all_classes/* aliases, which duplicate the originals.
    root=ROOT/"data"
    folders=[root/"2026_future_block1/s_class_f1_20260701_20260830"]
    for group in ("a_class_12","a3_challenge"):
        folders += sorted((root/group).glob("2026_*"))
    complete=set()
    counts={}
    for p in folders:
        rf,ef,tf=[p/(name+".csv") for name in ("races","entries","results")]
        if not all(f.exists() for f in (rf,ef,tf)):
            continue
        with rf.open(encoding="utf-8-sig",newline="") as f:
            ids={r["race_id"] for r in csv.DictReader(f)
                 if DAY_LOW<= (rid_day(r["race_id"]) or date.min)<=DAY_HIGH}
        entries=defaultdict(set)
        with ef.open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                if r["race_id"] in ids:entries[r["race_id"]].add(r["car_no"])
        results=defaultdict(set)
        with tf.open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                if r["race_id"] in ids:results[r["race_id"]].add(r["car_no"])
        good={rid for rid in ids if entries[rid] and entries[rid]==results[rid]}
        complete.update(good)
        counts[str(p.relative_to(ROOT))]={"in_range_race_ids":len(ids),"full_result_ids":len(good)}
    return complete,counts

def candidate_sources(html,url,day):
    sources=(
        ("S",s_class_f1.discover_s_class_from_daily_html(html,url,day.isoformat())),
        ("A12",a_class_12.discover_a_class_from_daily_html(html,url,day.isoformat())),
        ("A3",a3_challenge.discover_challenge_from_daily_html(html,url,day.isoformat())),
    )
    found={}
    conflicts=[]
    for kind,refs in sources:
        for ref in refs:
            m=RID.search(ref.url)
            if not m:continue
            rid=m.group(1)
            day_actual=rid_day(rid)
            if day_actual is None:
                conflicts.append([rid,kind,"invalid encoded race day"])
                continue
            if not DAY_LOW<=day_actual<=DAY_HIGH: continue
            typ=norm(ref.race_type)
            if "ガールズ" in typ or "女子" in typ or typ.startswith("L級"):continue
            existing=found.get(rid)
            if existing:
                if norm(existing[1].race_type)!=typ:
                    conflicts.append([rid,kind,"conflicting daily type labels"])
            else:
                found[rid]=(kind,ref)
    # Daily index may contain kinds not covered by any of the three parsers.
    # Keep them as an AUDIT warning rather than assuming completeness.
    all_ids={i for i in RID.findall(html) if DAY_LOW <= (rid_day(i) or date.min)<=DAY_HIGH}
    return found,conflicts,all_ids

def odds_urls(source):
    parts=urlsplit(source)
    prefix=(parts.scheme,parts.netloc,parts.path)
    return [urlunsplit((*prefix,query,"")) for query in
            ("pageType=odds","pageType=yoso","","pageType=result")]

def collect_one(session, rid, kind, ref, grade, pause):
    errs=[]
    for candidate in odds_urls(ref.url):
        frozen=base.RaceRef(discovered_on=ref.discovered_on,race_no=ref.race_no,url=candidate)
        try:
            html=base.fetch_html(session,candidate)
            previous=base.TARGET_RACE_TYPE
            base.TARGET_RACE_TYPE=ref.race_type
            try:
                race,entries=base.extract_entries(html,frozen,require_target_label=False)
            finally:
                base.TARGET_RACE_TYPE=previous
            if race["race_id"]!=rid or rid_day(rid).isoformat()!=str(race["race_date"]):
                raise ValueError("race identity/date mismatch")
            cars={str(e["car_no"]) for e in entries}
            if not (5<=len(cars)<=9 and len(cars)==len(entries)):
                raise ValueError("invalid or duplicated entrant car numbers")
            cls={norm(e.get("class","")).upper() for e in entries}
            if any(c.startswith("L") for c in cls) or "ガールズ" in norm(ref.race_type):
                return None,None,None,"girls-excluded"
            if not cls or any(not c or c=="-" for c in cls):
                raise ValueError("missing class labels")
            race["race_type"]=ref.race_type
            race["source_url"]=candidate
            race["meeting_grade"]=grade or ""
            for e in entries:
                e["meeting_grade"]=grade or ""
                e["race_type"]=ref.race_type
            result_url=result_parser.result_url(candidate)
            text=base.fetch_html(session,result_url)
            result_rows=result_parser.parse_result_rows(BeautifulSoup(text,"lxml"))
            result_cars=[str(r["car_no"]) for r in result_rows]
            if len(set(result_cars))!=len(result_cars) or set(result_cars)!=cars:
                raise ValueError("result car numbers differ from entrants")
            # Preserve non-numeric finish order exactly: do not treat DNS/DSQ as losses.
            for r in result_rows:
                r.update({"race_id":rid,"race_date":race["race_date"],
                          "track":race["track"],"race_no":race["race_no"],
                          "race_type":race["race_type"],"result_source_url":result_url,
                          "result_source":"楽天Kドリームス 結果・払戻金",
                          "captured_at_utc":datetime.now(timezone.utc).isoformat()})
            return race,entries,result_rows,""
        except Exception as exc:
            errs.append(f"{type(exc).__name__}:{str(exc)[:140]}")
        if pause:time.sleep(pause)
    return None,None,None," | ".join(errs)[:750]

def emit(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",required=True)
    ap.add_argument("--end",required=True)
    ap.add_argument("--pause",type=float,default=.30)
    args=ap.parse_args()
    start,end=date.fromisoformat(args.start),date.fromisoformat(args.end)
    if start>end or start<DAY_LOW or end>DAY_HIGH or (end-start).days>15:
        ap.error("one batch must be inside July-October 2026 and at most 16 days")
    if not 0<=args.pause<=3:ap.error("pause must be between 0 and 3 seconds")
    out=DEST/f"batch_{start:%Y%m%d}_{end:%Y%m%d}"
    if out.exists() and (out/"summary.json").exists():
        raise RuntimeError("immutable completed batch already exists")
    known,known_counts=list_existing()
    print("KNOWN_FULL_RESULT_RACES",len(known),flush=True)
    sess=base.make_session()
    discovered={}
    discovery_failed=[]
    unknown_ids=set()
    daily_discovery={}
    grade_by_day={}
    for n in range((end-start).days+1):
        day=start+timedelta(days=n)
        url=base.KDREAMS_DAILY.format(year=day.year,month=day.month,day=day.day)
        try:
            html=base.fetch_html(sess,url)
            grade_by_day[day.isoformat()]=a3_challenge.discover_meeting_grades_from_daily_html(html)
            refs,conflicts,all_ids=candidate_sources(html,url,day)
            daily_discovery[day.isoformat()]={"found":len(refs),"indexed_link_count":len(all_ids),"conflicts":conflicts[:30]}
            unknown_ids.update(all_ids-set(refs))
            for rid,ref in refs.items():
                if start<=rid_day(rid)<=end:discovered.setdefault(rid,ref)
        except Exception as exc:
            discovery_failed.append({"date":day.isoformat(),"url":url,
                                     "error":f"{type(exc).__name__}: {exc}"[:500]})
        if args.pause:time.sleep(args.pause)
    races,entries,results,failed=[],[],[],[]
    skipped=0
    for i,(rid,(kind,ref)) in enumerate(sorted(discovered.items())):
        if rid in known:
            skipped+=1
            continue
        candidate_grade=grade_by_day.get(ref.discovered_on,{})
        # Grade is annotation only; no guessed F1/F2 from race type.
        actual,entrants,finish,reason=collect_one(
            sess,rid,kind,ref,candidate_grade.get("", ""),args.pause)
        if reason=="girls-excluded":
            continue
        if reason:
            failed.append({"race_id":rid,"race_date":rid_day(rid).isoformat(),
                           "stage":"full-race","url":ref.url,"reason":reason})
        else:
            races.append(actual);entries.extend(entrants);results.extend(finish)
        if args.pause:time.sleep(args.pause)
        if i%30==0:print("PROGRESS",i+1,len(discovered),"recovered",len(races),"errors",len(failed),flush=True)
    emit(out/"races.csv",RACE_FIELDS,races)
    emit(out/"entries.csv",ENTRY_FIELDS,entries)
    emit(out/"results.csv",RESULT_FIELDS,results)
    emit(out/"failures.csv",FAIL_FIELDS,failed)
    status={"period":[start.isoformat(),end.isoformat()],
            "known_existing_full_result_races_across_2026":len(known),
            "known_sources":known_counts,
            "discovered_unique_men_race_ids_in_batch":len(discovered),
            "already_present_skipped":skipped,"new_full_result_races":len(races),
            "new_entrant_rows":len(entries),"new_result_rows":len(results),
            "recovery_failures":len(failed),"daily_discovery_failures":discovery_failed,
            "unmapped_daily_link_ids_sample":sorted(unknown_ids)[:100],
            "unmapped_daily_link_id_count":len(unknown_ids),
            "daily_discovery":daily_discovery,
            "not_nationally_certified":True,
            "source_originals_unchanged":True,
            "exclusions":"girls excluded; any non-numeric final place preserved as unknown",
            "warning":"A daily race link is not proof of a valid men's race; coverage requires class verified full race-card and results."}
    (out/"summary.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("BATCH_REPORT",json.dumps({k:status[k] for k in
         ("period","discovered_unique_men_race_ids_in_batch","already_present_skipped","new_full_result_races",
          "recovery_failures","unmapped_daily_link_id_count")},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

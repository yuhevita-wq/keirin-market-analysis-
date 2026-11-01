#!/usr/bin/env python3
"""Collect ONLY missing 2026 rider entries and complete results from published day indexes.

No original datasets, odds, or predictive model files are changed. Calendar links
are an observed source, not an independently certified nationwide census.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import time
import unicodedata
import zipfile
from collections import Counter,defaultdict
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
from urllib.parse import urljoin,urlsplit,urlunsplit
from bs4 import BeautifulSoup
from keirin_market_analysis import s_yosen_2025 as base
from keirin_market_analysis import results_payouts_2025 as result_parser

ROOT=Path(__file__).resolve().parents[2]
OUTBASE=ROOT/"data/rider_condition_2026_backfill"
RX=re.compile(r"/racedetail/(\d{16})/")
RACE_FIELDS=["race_id","race_date","track","meeting_grade","race_no","race_type","start_time",
             "deadline","entry_count","source_url","captured_at_utc"]
ENTRY_FIELDS=RACE_FIELDS[:-1]+["captured_at_utc","car_no","player_name","player_profile",
 "prefecture","age","term","class","style","gear","score","s_count","b_count",
 "nige_count","makuri_count","sashi_count","mark_count","first_count","second_count",
 "third_count","outside_count","win_rate","top2_rate","top3_rate","prediction_mark",
 "evaluation","raw_row_json","player_id","player_id_status"]
RESULT_FIELDS=["race_id","race_date","track","race_no","race_type","car_no","player_name",
 "finish_position","finish_text","margin","last200","winning_method","sb","result_comment",
 "result_source_url","result_source","captured_at_utc","raw_row_json"]
FAIL_FIELDS=["race_id","race_date","stage","source_url","error"]
RACE_TYPE_RX=re.compile(r"(?:[ＳSＡAＬL]級|チャレンジ|予選|選抜|準決勝|決勝|一般|特選|特別選抜|一次予選|二次予選|敗者戦)")

def norm(s):
    return unicodedata.normalize("NFKC",str(s or "")).strip()

def day_from_id(rid):
    if not re.fullmatch(r"\d{16}",rid):return ""
    try:
        d=date(int(rid[2:6]),int(rid[6:8]),int(rid[8:10]))
        n=int(rid[10:12])
        if not 1<=n<=7 or not 1<=int(rid[-2:])<=12:return ""
        return (d+timedelta(days=n-1)).isoformat()
    except ValueError:return ""

def is_girls(text):
    t=norm(text).upper()
    return any(s in t for s in ("ガールズ","女子","GIRLS","L級","ガ予","ガ決","ガ一般"))

def get_existing():
    existing=set()
    # Use prior verified consolidation, recent category collections and old G3 archives.
    paths=[ROOT/"data/rider_condition_2025_2026h1/races_2026.csv.gz"]
    import gzip
    for p in paths:
        if p.exists():
            with gzip.open(p,"rt",encoding="utf-8-sig") as f:
                existing.update(r["race_id"] for r in csv.DictReader(f) if r["race_id"])
    roots=["data/a_class_12","data/a3_challenge","data/2026_future_block1",
           "data/2026_h1","data/rider_condition_2026_backfill"]
    for root in roots:
        for p in sorted((ROOT/root).rglob("races.csv")):
            # We retain previous backfill partitions for idempotence.
            if "2025" in p.parts:continue
            with p.open(encoding="utf-8-sig",newline="") as f:
                existing.update(r.get("race_id","") for r in csv.DictReader(f))
    for p in (ROOT/"data/grade_races/g3/2026_h1").glob("*.zip"):
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.rsplit("/",1)[-1]=="races.csv":
                    import io
                    with z.open(name) as b,io.TextIOWrapper(b,encoding="utf-8-sig") as f:
                        existing.update(r.get("race_id","") for r in csv.DictReader(f))
    return {rid for rid in existing if day_from_id(rid)}

def guess_label(table_rows,ri,ci):
    for j in range(ri,-1,-1):
        expanded=base.expand_row_cells(table_rows[j])
        if ci>=len(expanded):continue
        c=expanded[ci]
        if c.name not in ("td","th"):continue
        s=base.normalize_text(c.get_text(" ",strip=True))
        if RACE_TYPE_RX.search(s) and len(s)<34:
            return s
    return ""

def discover(html,daily_url,day):
    soup=BeautifulSoup(html,"lxml")
    found={}
    for table in soup.find_all("table"):
        rows=table.find_all("tr")
        for ri,row in enumerate(rows):
            cells=base.expand_row_cells(row)
            for ci,cell in enumerate(cells):
                if not cell:return
                for a in cell.find_all("a",href=True):
                    href=str(a["href"])
                    m=RX.search(href)
                    if not m:continue
                    rid=m.group(1)
                    if day_from_id(rid)!=day:continue
                    url=base.canonical_race_url(daily_url,href)
                    label=guess_label(rows,ri,ci)
                    # Prefer an explicit label. Never replace it with inferred
                    # category from a player's rating or finish position.
                    old=found.get(rid)
                    if old is None or (label and not old[1]):
                        found[rid]=(url,label)
    return found

def parsed_title_type(html):
    soup=BeautifulSoup(html,"lxml")
    title=base.normalize_text(soup.title.get_text(" ",strip=True) if soup.title else "")
    m=re.search(r"レース詳細\s*\|\s*.+?\s+\d+R\s+(.+?)\s*\|",title)
    if m:return m.group(1).strip()
    return ""

def variants(url):
    p=urlsplit(url)
    return [urlunsplit((p.scheme,p.netloc,p.path,q,"")) for q in
            ("pageType=odds","pageType=yoso","","pageType=result")]

def extract_card(session,rid,url,label):
    errors=[]
    for target in variants(url):
        try:
            html=base.fetch_html(session,target)
            typ=parsed_title_type(html) or label
            if not typ or not RACE_TYPE_RX.search(typ):
                raise ValueError("race type unverified from page title and daily index")
            if is_girls(typ):
                return None,None,"girls_excluded"
            ref=base.RaceRef(discovered_on=day_from_id(rid),race_no=int(rid[-2:]),url=target)
            old=base.TARGET_RACE_TYPE
            try:
                base.TARGET_RACE_TYPE=typ
                race,entries=base.extract_entries(html,ref,require_target_label=False)
            finally:
                base.TARGET_RACE_TYPE=old
            if race["race_id"]!=rid or race["race_date"]!=day_from_id(rid):
                raise ValueError("race ID / date mismatch")
            # Girl class is an explicit exclusion even if the race label is ambiguous.
            if any(norm(e.get("class")).upper().startswith("L") for e in entries):
                return None,None,"girls_excluded"
            race["meeting_grade"]=""
            for e in entries:
                e["meeting_grade"]=""
                e["player_id"]=""
                e["player_id_status"]="not_verified"
            return race,entries,None
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return None,None," | ".join(errors)[-1200:]

def extract_result(session,race,entries):
    url=result_parser.result_url(race["source_url"])
    html=base.fetch_html(session,url)
    soup=BeautifulSoup(html,"lxml")
    parsed=result_parser.parse_result_rows(soup)
    cars={int(e["car_no"]) for e in entries}
    rows={int(r["car_no"]):r for r in parsed}
    if cars != set(rows):
        raise ValueError("result car set differs from entrants")
    # Nonstarter/abnormal status stays explicit and is never interpreted as losing.
    now=datetime.now(timezone.utc).isoformat()
    out=[]
    for car in sorted(rows):
        r=dict(rows[car])
        out.append({"race_id":race["race_id"],"race_date":race["race_date"],
                    "track":race["track"],"race_no":race["race_no"],
                    "race_type":race["race_type"],**r,
                    "result_source_url":url,"result_source":"楽天Kドリームス 公開結果",
                    "captured_at_utc":now})
    return out

def save_csv(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=header,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def execute(start,end,pause,out):
    if not ("2026-01-01"<=start<=end<="2026-10-31"):
        raise ValueError("Only 2026-01-01 to 2026-10-31 is allowed")
    if not 0.20<=pause<=3:raise ValueError("pause must be >= 0.20")
    known=get_existing()
    session=base.make_session()
    found={}
    days={}
    failures=[]
    d=date.fromisoformat(start)
    final=date.fromisoformat(end)
    while d<=final:
        ds=d.isoformat()
        daily_url=base.KDREAMS_DAILY.format(year=d.year,month=d.month,day=d.day)
        try:
            html=base.fetch_html(session,daily_url)
            index=discover(html,daily_url,ds)
            days[ds]={"index_links":len(index),"already_saved":sum(x in known for x in index),
                      "new_candidates":sum(x not in known for x in index)}
            for rid, info in index.items():
                if rid not in known:found.setdefault(rid,info)
        except Exception as exc:
            days[ds]={"error":f"{type(exc).__name__}: {exc}"[:300]}
            failures.append({"race_id":"","race_date":ds,"stage":"daily_index",
                "source_url":daily_url,"error":str(exc)[:800]})
        time.sleep(pause)
        d+=timedelta(days=1)
    races,entries,results=[],[],[]
    excluded=[]
    for i,(rid,(url,label)) in enumerate(sorted(found.items(),key=lambda x:(day_from_id(x[0]),x[0])),1):
        race,rows,err=extract_card(session,rid,url,label)
        if err=="girls_excluded":
            excluded.append(rid)
        elif err:
            failures.append({"race_id":rid,"race_date":day_from_id(rid),
                             "stage":"race_entry","source_url":url,"error":err})
        else:
            races.append(race)
            entries.extend(rows)
            try:
                results.extend(extract_result(session,race,rows))
            except Exception as exc:
                failures.append({"race_id":rid,"race_date":day_from_id(rid),
                                 "stage":"full_result","source_url":url,
                                 "error":f"{type(exc).__name__}: {exc}"[:1000]})
        if i%30==0:
            print("BACKFILL_PROGRESS",i,"/",len(found),"races",len(races),"results",len({r["race_id"] for r in results}),flush=True)
        time.sleep(pause)
    races.sort(key=lambda r:(r["race_date"],r["race_id"]))
    entries.sort(key=lambda r:(r["race_date"],r["race_id"],int(r["car_no"])))
    results.sort(key=lambda r:(r["race_date"],r["race_id"],int(r["car_no"])))
    save_csv(out/"races.csv",RACE_FIELDS,races)
    save_csv(out/"entries.csv",ENTRY_FIELDS,entries)
    save_csv(out/"results.csv",RESULT_FIELDS,results)
    save_csv(out/"failures.csv",FAIL_FIELDS,failures)
    report={
        "window":[start,end],"source":"楽天Kドリームス 日別開催一覧と結果",
        "status":"PUBLISHED_DAILY_INDEX_CENSUS_NOT_NATIONWIDE_CERTIFIED",
        "known_existing_ids_at_start":len(known),
        "daily_indexes_succeeded":sum("error" not in x for x in days.values()),
        "daily_indexes_failed":sum("error" in x for x in days.values()),
        "daily_index_race_links":sum(x.get("index_links",0) for x in days.values()),
        "known_from_other_sources":sum(x.get("already_saved",0) for x in days.values()),
        "new_candidate_ids":len(found),
        "girls_excluded":len(excluded),
        "new_races_with_entries":len(races),
        "new_races_with_full_results":len({r["race_id"] for r in results}),
        "new_entrant_rows":len(entries),
        "result_rows":len(results),
        "unresolved_errors":len(failures),
        "errors_by_stage":dict(Counter(f["stage"] for f in failures)),
        "daily":days,
        "excluded_girls_ids":excluded,
        "note":"No original paths edited. Race identity matched exactly. Unknown race type or results retained as failure; no invented facts.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("BACKFILL_SUMMARY",json.dumps({k:v for k,v in report.items() if k not in ("daily","excluded_girls_ids")},ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--start",required=True)
    p.add_argument("--end",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--sleep",type=float,default=0.25)
    args=p.parse_args()
    execute(args.start,args.end,args.sleep,ROOT/args.out)
if __name__=="__main__":
    main()

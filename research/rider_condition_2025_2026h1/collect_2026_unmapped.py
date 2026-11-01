#!/usr/bin/env python3
"""Second pass: recover official race links not classified by the S/A/Challenge daily-table parsers."""
import argparse
import csv
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
from urllib.parse import urljoin,urlsplit,urlunsplit
from bs4 import BeautifulSoup

import collect_2026_gap as gap
from keirin_market_analysis import s_yosen_2025 as base
from keirin_market_analysis import a3_challenge

@dataclass
class Ref:
    discovered_on:str
    race_no:int
    race_type:str
    url:str

def type_from_official_card(html):
    soup=BeautifulSoup(html,"lxml")
    # The page states its exact class and race phase immediately before 発走予定.
    text=base.normalize_text(soup.get_text(" ",strip=True))
    idx=text.find("発走予定")
    prefix=text[max(0,idx-180):idx] if idx>=0 else ""
    hits=re.findall(r"((?:Ｓ級|S級|Ａ級|A級|チャレンジ)[^\s]{1,18})",prefix)
    if not hits:
        return "未分類(要監査)"
    value=hits[-1]
    value=re.split(r"発走|投票|賞金|予想|誘導",value)[0].strip()
    return value or "未分類(要監査)"

def run(start,end,pause):
    known,sources=gap.list_existing()
    # Include all immutable saved first-pass batches, to avoid any duplicated retrieval.
    for p in sorted(gap.DEST.glob("batch_*/races.csv")):
        with p.open(encoding="utf-8-sig",newline="") as f:
            known.update(r["race_id"] for r in csv.DictReader(f))
    out=gap.DEST/f"supplement_{start:%Y%m%d}_{end:%Y%m%d}"
    if (out/"summary.json").exists():
        raise RuntimeError("refusing to overwrite saved supplement")
    session=base.make_session()
    missing={}
    errors=[]
    day_audit={}
    for n in range((end-start).days+1):
        d=start+timedelta(days=n)
        u=base.KDREAMS_DAILY.format(year=d.year,month=d.month,day=d.day)
        try:
            html=base.fetch_html(session,u)
            mapped,_,allids=gap.candidate_sources(html,u,d)
            candidates=allids-set(mapped)
            soup=BeautifulSoup(html,"lxml")
            urls={}
            for a in soup.find_all("a",href=True):
                m=gap.RID.search(a["href"])
                if not m:continue
                rid=m.group(1)
                if rid in candidates and start<=gap.rid_day(rid)<=end:
                    url=urljoin(u,a["href"])
                    parsed=urlsplit(url)
                    if parsed.hostname!="keirin.kdreams.jp":continue
                    urls.setdefault(rid,urlunsplit((parsed.scheme,parsed.netloc,parsed.path,"pageType=odds","")))
            for rid,url in urls.items():
                if rid not in known:
                    missing.setdefault(rid,(d.isoformat(),url))
            day_audit[d.isoformat()]={"indexed_race_ids":len(allids),
                                    "unclassified_race_ids":len(candidates),
                                    "unclassified_url_recovered":len(urls)}
        except Exception as e:
            errors.append({"day":d.isoformat(),"stage":"discovery","error":f"{type(e).__name__}:{e}"[:400]})
        if pause:time.sleep(pause)
    races,entries,results,failures=[],[],[],[]
    ignored_girls=0
    unknown_types=0
    for i,(rid,(discovered_on,url)) in enumerate(sorted(missing.items())):
        try:
            page=base.fetch_html(session,url)
            typ=type_from_official_card(page)
            ref=Ref(discovered_on,int(rid[-2:]),typ,url)
            race,entrant,finish,reason=gap.collect_one(session,rid,"other",ref,"",pause)
            if reason=="girls-excluded":
                ignored_girls+=1
            elif reason:
                failures.append({"race_id":rid,"race_date":gap.rid_day(rid).isoformat(),
                                "stage":"full-race","url":url,"reason":reason})
            else:
                if typ.startswith("未分類"):unknown_types+=1
                races.append(race);entries.extend(entrant);results.extend(finish)
        except Exception as e:
            failures.append({"race_id":rid,"race_date":gap.rid_day(rid).isoformat(),
                            "stage":"type-detection","url":url,
                            "reason":f"{type(e).__name__}:{e}"[:600]})
        if pause:time.sleep(pause)
        if i%30==0:print("SECOND_PASS",i+1,len(missing),"recovered",len(races),flush=True)
    gap.emit(out/"races.csv",gap.RACE_FIELDS,races)
    gap.emit(out/"entries.csv",gap.ENTRY_FIELDS,entries)
    gap.emit(out/"results.csv",gap.RESULT_FIELDS,results)
    gap.emit(out/"failures.csv",gap.FAIL_FIELDS,failures)
    summary={
        "period":[start.isoformat(),end.isoformat()],
        "eligible_unmapped_race_links":len(missing),
        "new_full_result_races":len(races),
        "new_entrant_rows":len(entries),
        "new_result_rows":len(results),
        "recovery_failures":len(failures),
        "excluded_l_class_girls":ignored_girls,
        "unverified_race_type_rows":unknown_types,
        "discovery_errors":errors,
        "day_audit":day_audit,
        "source_originals_unchanged":True,
        "warning":"Race phase labels are copied from the race page where present. 未分類(要監査) must never be interpreted as an observed race phase.",
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUPPLEMENT_RESULT",json.dumps({
        k:summary[k] for k in ("period","eligible_unmapped_race_links","new_full_result_races",
                             "recovery_failures","excluded_l_class_girls","unverified_race_type_rows")
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",required=True);ap.add_argument("--end",required=True)
    ap.add_argument("--pause",type=float,default=.3)
    a=ap.parse_args()
    start,end=date.fromisoformat(a.start),date.fromisoformat(a.end)
    if start>end or start<gap.DAY_LOW or end>gap.DAY_HIGH or (end-start).days>15:
        ap.error("invalid date batch")
    run(start,end,a.pause)

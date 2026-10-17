#!/usr/bin/env python3
"""Collect REAL published podiums matching the seven-bet odds archival DATE SCOPE.

One daily result index, one meeting-result summary per venue. NEVER synthesize
outcomes or use results as market features. No private odds key is required.
No claim of exact coverage of encrypted odds can be made without exact-ID join.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
BEGIN = date(2021, 1, 1)
END = date(2026, 6, 30)
BASE = "https://keirin.kdreams.jp"
INDEX = BASE + "/raceresult/{year:04d}/{month:02d}/{day:02d}/"
VENUE_RX = re.compile(r"^/([a-z0-9_-]+)/raceresult/(\d{14})/?$")
RACE_RX = re.compile(r"/racedetail/(\d{16})/")
DAY_DIR = ROOT / "data/result_universe_v2/daily"
LEDGER_DIR = ROOT / "data/result_universe_v2/ledger"
CURSOR_DIR = ROOT / "data/audits/results_full_shards"
HEADERS = {"User-Agent":"Mozilla/5.0 (compatible; keirin-market-analysis/0.1; +https://github.com/yuhevita-wq/keirin-market-analysis-)",
           "Accept-Language":"ja,en;q=0.8"}


def race_date_from_id(rid):
    if not re.fullmatch(r"\d{16}",rid):
        return None
    try:
        n=int(rid[10:12])
        if not 1<=n<=7 or not 1<=int(rid[12:])<=12:
            return None
        return date(int(rid[2:6]),int(rid[6:8]),int(rid[8:10]))+timedelta(days=n-1)
    except ValueError:
        return None


def meeting_date_from_id(mid):
    if not re.fullmatch(r"\d{14}",mid) or mid[-2:]!="00":
        return None
    return race_date_from_id(mid[:12]+"0001")


def is_girls(label):
    n=unicodedata.normalize("NFKC",label).upper()
    return any(s in n for s in ("ガールズ","L級","GIRLS","女子"))


def daily_meetings(html, day):
    soup=BeautifulSoup(html,"lxml")
    found={}
    for a in soup.find_all("a",href=True):
        href=urljoin(INDEX.format(year=day.year,month=day.month,day=day.day),a["href"])
        parsed=urlsplit(href)
        m=VENUE_RX.fullmatch(parsed.path)
        if parsed.hostname!="keirin.kdreams.jp" or not m or a.get_text(" ",strip=True)!="一覧":
            continue
        mid=m.group(2)
        if meeting_date_from_id(mid)!=day:
            continue
        clean=urlunsplit(("https", "keirin.kdreams.jp",parsed.path,"",""))
        if mid in found and found[mid]!=clean:
            raise ValueError("conflicting venue result urls for meeting "+mid)
        found[mid]=clean
    if not found:
        raise ValueError("No published meeting-result links discovered on "+day.isoformat())
    return found


def cells_of(tr):
    return [c.get_text(" ",strip=True) for c in tr.find_all(["th","td"],recursive=False)]


def extract_podium(table):
    if not table or "order_table" not in table.get("class",[]):
        raise ValueError("no results order_table")
    trs=table.find_all("tr")
    if len(trs)<2:
        raise ValueError("empty results table")
    hdr=[unicodedata.normalize("NFKC",s).replace(" ","") for s in cells_of(trs[0])]
    if hdr[:3]!=["1着","2着","3着"]:
        raise ValueError("published first-second-third headers not present")
    top=cells_of(trs[1])[:3]
    if len(top)!=3:
        raise ValueError("top three race results missing")
    riders=[]
    for cell in top:
        m=re.match(r"^\s*([1-9])(?:\D|$)",cell)
        if not m:
            raise ValueError("published podium cell missing car number")
        riders.append(int(m.group(1)))
    if len(set(riders))!=3:
        raise ValueError("duplicate podium car number")
    return riders


def venue_podiums(html, venue_url, day):
    match=VENUE_RX.fullmatch(urlsplit(venue_url).path)
    if match is None:
        raise ValueError("venue result URL malformed")
    meeting_id=match.group(2)
    if meeting_date_from_id(meeting_id)!=day:
        raise ValueError("venue meeting date mismatch")
    soup=BeautifulSoup(html,"lxml")
    values={}
    cards=0
    girls=0
    invalid=[]
    for p in soup.select("p.race"):
        anchors=[a for a in p.find_all("a",href=True)
                 if RACE_RX.search(urlsplit(urljoin(venue_url,a["href"])).path)]
        if not anchors:
            continue
        a=anchors[0]
        rid=RACE_RX.search(urlsplit(urljoin(venue_url,a["href"])).path).group(1)
        # Strict meeting AND calendar-day identity check.
        if rid[:12]!=meeting_id[:12] or race_date_from_id(rid)!=day:
            invalid.append([rid,"race-ID does not belong to published meeting/date"])
            continue
        cards+=1
        if is_girls(p.get_text(" ",strip=True)):
            girls+=1
            continue
        # The closest order_table must be BETWEEN this race header and next
        # race header, never silently borrowed from a subsequent race.
        table=None
        for element in p.next_elements:
            if getattr(element,"name",None)=="p" and "race" in element.get("class",[]):
                break
            if getattr(element,"name",None)=="table" and "order_table" in element.get("class",[]):
                table=element
                break
        try:
            finish=extract_podium(table)
            record={"race_id":rid,"finish_order":finish,"source_url":venue_url}
            if rid in values and values[rid]!=record:
                raise ValueError("conflicting podium in same meeting")
            values[rid]=record
        except Exception as exc:
            invalid.append([rid,str(exc)[:140]])
    if cards==0:
        raise ValueError("no race cards on venue results page")
    if cards!=len(values)+girls+len(invalid):
        raise ValueError("published race cards not fully accounted for")
    # Preserve verifiable races even when one result is a dead heat or cancelled.
    # Never synthesize or silently classify an abnormal podium as complete.
    return values, {"cards":cards,"girls_excluded":girls,
                    "men_podiums":len(values),"unresolved_podiums":invalid}


def file_paths(day):
    return (DAY_DIR/str(day.year)/(day.isoformat()+".jsonl.gz"),
            LEDGER_DIR/str(day.year)/(day.isoformat()+".json"))


def load_json(p,default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def load_stored(day):
    f,_=file_paths(day)
    rows={}
    if f.exists():
        with gzip.open(f,"rt",encoding="utf-8") as src:
            for line in src:
                item=json.loads(line)
                rows[item["race_id"]]=item
    return rows


def save_deterministic_gzip(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw,mode="wb",filename="",mtime=0) as gz:
            for rid,item in sorted(rows.items()):
                gz.write((json.dumps(item,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8"))


def get_html(session,url,pause):
    time.sleep(pause)
    response=session.get(url,timeout=40)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower()=="iso-8859-1":
        response.encoding=response.apparent_encoding
    return response.text


def collect_day(session,day,pause):
    output,ledger_path=file_paths(day)
    state=load_json(ledger_path,{"schema":1,"day":day.isoformat(),"attempts":0,"venues_completed":[]})
    if state.get("complete") is True:
        return {"day":day.isoformat(),"skipped_complete":True,"results":state.get("podiums",0),"failed":0}
    state["attempts"]=int(state.get("attempts",0))+1
    errors=[]
    rows=load_stored(day)
    try:
        index_html=get_html(session,INDEX.format(year=day.year,month=day.month,day=day.day),pause)
        meetings=daily_meetings(index_html,day)
        state["index_verified"]=True
        state["published_venue_count"]=len(meetings)
    except Exception as exc:
        state["index_verified"]=False
        meetings={}
        errors.append({"stage":"daily_index","error":str(exc)[:300]})
    successes=set(state.get("venues_completed",[]))
    for mid,url in sorted(meetings.items()):
        if mid in successes:
            continue
        try:
            venue_rows, venue_stats=venue_podiums(get_html(session,url,pause),url,day)
            for rid,record in venue_rows.items():
                if rid in rows and rows[rid]["finish_order"]!=record["finish_order"]:
                    raise ValueError("archived-versus-publisher conflicting podium "+rid)
            rows.update(venue_rows)
            if venue_stats["unresolved_podiums"]:
                errors.append({"stage":"race","venue":mid,"url":url,
                               "error":"Unresolved published podiums "+
                               json.dumps(venue_stats["unresolved_podiums"][:6],ensure_ascii=False)})
            else:
                successes.add(mid)
            print("VENUE_RESULTS",day,mid,venue_stats,flush=True)
        except Exception as exc:
            errors.append({"stage":"venue","venue":mid,"url":url,"error":f"{type(exc).__name__}: {exc}"[:400]})
    state["venues_completed"]=sorted(successes)
    state["complete"]=bool(meetings) and set(meetings)==successes and not errors
    state["podiums"]=len(rows)
    state["girls_excluded_by_source_label"]=True
    state["results_are_target_labels_only"]=True
    state["last_checked_utc"]=datetime.now(timezone.utc).isoformat()
    state["recent_errors"]=errors[:12]
    if rows:
        save_deterministic_gzip(output,rows)
    ledger_path.parent.mkdir(parents=True,exist_ok=True)
    ledger_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"day":day.isoformat(),"results":len(rows),"complete":state["complete"],
            "venues_completed":len(successes),"venues_total":len(meetings),"failed":len(errors),
            "errors":errors[:2]}


def run(shard,shards,days,pause):
    if not 0<=shard<shards or not 1<=days<=12 or pause<2:
        raise ValueError("incorrect shards/days; polite request spacing requires >=2 seconds")
    CURSOR_DIR.mkdir(parents=True,exist_ok=True)
    p=CURSOR_DIR/f"shard_{shard:02d}.json"
    state=load_json(p,{"schema":1,"shard":shard,"shards":shards,"next_offset":shard})
    if state.get("shards")!=shards or state.get("shard")!=shard:
        raise ValueError("sharding parameters do not match persisted state")
    if state.get("paused_for_source_failure"):
        print("SOURCE_PAUSED",shard,flush=True)
        return
    ses=requests.Session()
    ses.headers.update(HEADERS)
    stats=[]
    consecutive_zero=0
    for _ in range(days):
        i=int(state["next_offset"])
        day=BEGIN+timedelta(days=i)
        if day>END:
            break
        result=collect_day(ses,day,pause)
        stats.append(result)
        print("DAY_RESULTS",json.dumps(result,ensure_ascii=False),flush=True)
        if not result.get("complete",result.get("skipped_complete",False)) and result.get("results",0)==0:
            consecutive_zero+=1
        else:
            consecutive_zero=0
        state["next_offset"]=i+shards
        if consecutive_zero>=2:
            state["paused_for_source_failure"]=True
            state["pause_reason"]="Two consecutive days with zero verified outcomes and incomplete sources"
            break
    state["last_run_utc"]=datetime.now(timezone.utc).isoformat()
    state["last_batch"]=stats
    state["finished_forward_pass"]=(BEGIN+timedelta(days=int(state["next_offset"])))>END
    p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RESULT_SHARD",json.dumps({
       "shard":shard,"processed_days":len(stats),
       "valid_podiums_in_processed_days":sum(x.get("results",0) for x in stats),
       "complete_days":sum(x.get("complete",False) for x in stats),
       "paused":bool(state.get("paused_for_source_failure")),
       "finished":state["finished_forward_pass"],
       "next":state["next_offset"],
    }),flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--shard",type=int,required=True)
    a.add_argument("--shards",type=int,default=8)
    a.add_argument("--days",type=int,default=2)
    a.add_argument("--pause",type=float,default=2.0)
    v=a.parse_args()
    run(v.shard,v.shards,v.days,v.pause)


if __name__=="__main__":
    main()

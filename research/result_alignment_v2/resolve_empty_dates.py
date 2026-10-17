#!/usr/bin/env python3
"""Mark only independently verified ZERO-meeting publisher days as completed.

Never interpret a networking failure as zero meetings. Require two independent
signals per date: exact on-page "開催情報はありません" published statement and no
encrypted seven-bet odds archive directory. Keep all real race labels untouched.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from collect_result_days import (
    ROOT, BEGIN, END, file_paths, load_json, INDEX, DAY_DIR, CURSOR_DIR,
    daily_meetings, HEADERS,
)

EMPTY_DATES = ("2022-04-05", "2022-04-06")


def verify_empty_day(session, day):
    url=INDEX.format(year=day.year,month=day.month,day=day.day)
    response=session.get(url,timeout=35)
    response.raise_for_status()
    soup=BeautifulSoup(response.text,"lxml")
    plain=soup.get_text(" ",strip=True)
    if "開催情報はありません。" not in plain:
        raise ValueError("Publisher did not explicitly confirm no meetings: "+url)
    # Distinguish absent meeting data from a broken parser.
    if soup.select("p.race") or re.search(r"/raceresult/\d{14}/",response.text):
        raise ValueError("Race or meeting hyperlinks exist despite no-meeting label")
    # If this exact date has archived markets it cannot be waived.
    odir=ROOT/"data/odds_ciphertext"/str(day.year)/day.isoformat()
    if odir.exists():
        raise ValueError("ARCHIVED_ODDS_PRESENT; no-meeting waiver prohibited: "+str(odir))
    stored_results,_=file_paths(day)
    if stored_results.exists():
        raise ValueError("STORED_RESULTS_PRESENT; no-meeting waiver prohibited")
    return url


def main():
    session=requests.Session()
    session.headers.update(HEADERS)
    completed=[]
    for raw in EMPTY_DATES:
        day=date.fromisoformat(raw)
        url=verify_empty_day(session,day)
        _,ledger=file_paths(day)
        state=load_json(ledger,{"schema":1,"day":raw,"attempts":0,"venues_completed":[]})
        if state.get("day")!=raw or int(state.get("podiums",0))!=0:
            raise ValueError("Existing ledger conflicts with zero-meeting evidence")
        state.update({
            "index_verified":True,
            "published_venue_count":0,
            "no_published_meetings":True,
            "no_meeting_evidence_url":url,
            "no_odds_archive_dir_confirmed":True,
            "venues_completed":[],
            "recent_errors":[],
            "podiums":0,
            "complete":True,
            "no_meeting_verified_utc":datetime.now(timezone.utc).isoformat(),
            "result_exclusion_reason":"verified_no_published_meetings",
        })
        ledger.parent.mkdir(parents=True,exist_ok=True)
        ledger.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        completed.append(raw)
        time.sleep(2)
    span=(END-BEGIN).days+1
    for shard in range(8):
        cursor=CURSOR_DIR/f"shard_{shard:02d}.json"
        if not cursor.exists():
            raise ValueError("Missing cursor "+str(cursor))
        state=json.loads(cursor.read_text(encoding="utf-8"))
        visited=min(span,int(state["next_offset"]))
        pending=[];exhausted=[];finished=[]
        for offset in range(shard,visited,8):
            day=BEGIN+timedelta(days=offset)
            _,ledger=file_paths(day)
            record=load_json(ledger,{"attempts":0,"complete":False})
            if record.get("complete"):
                finished.append(day.isoformat())
            elif int(record.get("attempts",0))<3:
                pending.append(day.isoformat())
            else:
                exhausted.append(day.isoformat())
        state["pending_repair_days"]=len(pending)
        state["exhausted_unresolved_days"]=len(exhausted)
        state["finished_all"]=bool(state["finished_forward_pass"]) and not pending and not exhausted
        cursor.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        if not state["finished_all"]:
            raise ValueError(f"Shard {shard} still has unresolved days: "+
                             json.dumps({"pending":pending,"exhausted":exhausted})[:1200])
    print("OFFICIAL_EMPTY_DAY_CLOSURE",json.dumps({
        "dates":completed,"all_shards_finished":True,"covered_calendar_days":span,
        "note":"These are zero published-meeting days, not missing race podiums",
    }),flush=True)


if __name__=="__main__":
    main()

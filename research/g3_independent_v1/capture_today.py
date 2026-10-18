#!/usr/bin/env python3
"""Snapshot today's G3 cards BEFORE their race start. Do not fetch outcomes.

Records snapshot UTC timestamps, event and published forecast lineup.
Today's observations are never fed into the 2022-2026H1 training fit.
"""
from __future__ import annotations
import csv,datetime,json,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from keirin_market_analysis.g3_history import (
    KDREAMS_DAILY,discover_g3_event_cards_from_daily_html,
    discover_races_from_g3_racecard_html,fetch_html,make_session,extract_entries
)
from keirin_market_analysis.line_formation_2025 import parse_line_formation_html
from keirin_market_analysis.s_yosen_2025 import RaceRef

DATE="2026-10-18"
OUT=ROOT/"research/g3_independent_v1/today_20261018"

def writecsv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader();w.writerows(rows)

def main():
    tokyo=ZoneInfo("Asia/Tokyo")
    now=datetime.datetime.now(tokyo)
    daily_url=KDREAMS_DAILY.format(year=2026,month=10,day=18)
    ses=make_session()
    daily=fetch_html(ses,daily_url)
    events=discover_g3_event_cards_from_daily_html(daily,daily_url,DATE)
    audit={"source":"KDreams live day G3 racecards",
           "date":DATE,"daily_url":daily_url,"snapshot_started_local":now.isoformat(),
           "event_cards":[x.url for x in events],"races":[],"errors":[]}
    cardrefs={}
    for event in events:
        try:
            html=fetch_html(ses,event.url)
            for race in discover_races_from_g3_racecard_html(html,event.url,DATE):
                cardrefs[race.url]=race
        except Exception as exc:
            audit["errors"].append({"event_url":event.url,"stage":"discovery","error":str(exc)})
    if not cardrefs:
        # Explicitly grounded event: KDreams Keio-kaku G3 started 2026-10-17,
        # Oct 18 is event day 02. Discover from per-race pages instead of
        # broad daily page whose G3 DOM may not match historical parser.
        for race_no in range(1,13):
            url=f"https://keirin.kdreams.jp/keiokaku/racedetail/272026101702{race_no:04d}/"
            cardrefs[url]=RaceRef(discovered_on=DATE,race_no=race_no,url=url)
        audit["fallback"]="12 explicitly constructed KDreams Keio-kaku event-day02 race URLs; each page still separately parsed and verified"
    rows=[];entries=[]
    for ref in sorted(cardrefs.values(),key=lambda r:(r.race_no,r.url)):
        try:
            html=fetch_html(ses,ref.url)
            card,people=extract_entries(html,ref)
            actual_utc=datetime.datetime.now(datetime.timezone.utc)
            start=card.get("start_time") or ""
            if start:
                hour,minute=map(int,start.split(":"))
                scheduled=datetime.datetime(2026,10,18,hour,minute,tzinfo=tokyo)
                if actual_utc.astimezone(tokyo)>=scheduled:
                    audit["errors"].append({"url":ref.url,"stage":"already_started_skip",
                                            "start_time":start})
                    continue
            formation=parse_line_formation_html(html)
            card["meeting_grade"]="G3"
            card["predicted_line_formation"]=formation["formation"]
            card["line_status"]=formation["status"]
            card["line_provider"]=formation["provider"]
            card["snapshot_captured_at_utc"]=actual_utc.isoformat()
            card["asof_before_start"]=True
            cars=[]
            for item in people:
                item["meeting_grade"]="G3"
                item["predicted_line_formation"]=formation["formation"]
                item["line_status"]=formation["status"]
                item["snapshot_captured_at_utc"]=actual_utc.isoformat()
                cars.append(int(item.get("car_no",0)))
                entries.append(item)
            rows.append(card)
            audit["races"].append({
                "race_id":card.get("race_id"),"race_no":card.get("race_no"),
                "start_time":start,"published_line":formation["formation"],
                "entry_count":len(people),"capture":actual_utc.isoformat(),
                "valid_line":set(int(x) for group in formation["formation"].split("/")
                                 if group for x in group.split("-"))==set(cars) if formation["formation"] else False
            })
            print("G3_TODAY_PRESTART",card.get("race_no"),card.get("race_id"),
                  len(people),formation["formation"],flush=True)
        except Exception as exc:
            audit["errors"].append({"url":ref.url,"stage":"racecard","error":str(exc)})
        time.sleep(.10)
    OUT.mkdir(parents=True,exist_ok=True)
    writecsv(OUT/"races.csv",rows)
    writecsv(OUT/"entries.csv",entries)
    audit["eligible_card_count"]=len(rows)
    audit["all_records_asof_before_start"]=all(x.get("valid_line") and x.get("entry_count")>=5 for x in audit["races"])
    (OUT/"snapshot_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_TODAY_CAPTURE_RESULT",json.dumps({
        "cards":len(rows),"events":len(events),"errors":len(audit["errors"]),
        "races":audit["races"],"details":audit["errors"][:8]
    },ensure_ascii=False),flush=True)
    if not rows:raise RuntimeError("No pre-start G3 entry cards captured")
if __name__=="__main__":main()

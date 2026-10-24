#!/usr/bin/env python3
"""Sixteen-request historical WINTICKET PRELOADED_STATE availability probe.

Uses exactly the URL + SSR JSON + tanStackQuery FETCH_KEIRIN_RACE path from
nyanta-kun/kiseki/keirin/src/scraper/winticket.py. NO bulk historical crawl.
Diagnostics only: does not treat 2026 re-reads as race-day as-of snapshots.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.winticket.jp"
CASES = (
    # 2023-10-19..22 Yahiko GI, verified independent historic event.
    ("2023-10-19","yahiko","2023101921",1,1),
    ("2023-10-19","yahiko","2023101921",1,12),
    ("2023-10-21","yahiko","2023101921",3,1),
    ("2023-10-21","yahiko","2023101921",3,11),
    ("2023-10-22","yahiko","2023101921",4,12),
    # 2023 KEIRIN GP, Tachikawa 28..30 December.
    ("2023-12-30","tachikawa","2023122828",3,11),
    # 2023 Takamatsunomiya Memorial GI, Kishiwada 13..18 June.
    ("2023-06-18","kishiwada","2023061356",6,12),
    # 2024 Yahiko GI 17..20 October.
    ("2024-10-17","yahiko","2024101721",1,1),
    ("2024-10-20","yahiko","2024101721",4,12),
    # 7-car A/S/Challenge from verified 2023-10-19 scheduled events.
    ("2023-10-19","kawasaki","2023101734",3,1),
    ("2023-10-19","kawasaki","2023101734",3,9),
    ("2023-10-19","kochi","2023101774",3,4),
    ("2023-10-19","kokura","2023101781",3,8),
    ("2023-10-19","beppu","2023101886",2,1),
    # 2024 Matsuyama F2, final day 2024-10-20.
    ("2024-10-20","matsuyama","2024101875",3,6),
    ("2024-10-20","matsuyama","2024101875",3,9),
)
HEADERS={
    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":"ja,en;q=0.5",
}
MARKER=re.compile(r"window\.__PRELOADED_STATE__\s*=\s*")

def get_state(html):
    m=MARKER.search(html)
    if not m:return None,"preloaded_state_marker_missing"
    try:return json.JSONDecoder().raw_decode(html[m.end():].lstrip())[0],None
    except (json.JSONDecodeError,ValueError,TypeError):
        return None,"preloaded_state_json_unparseable"

def query(state,fragment):
    for q in (state.get("tanStackQuery") or {}).get("queries") or []:
        if fragment in str(q.get("queryKey",[])):
            d=(q.get("state") or {}).get("data")
            if isinstance(d,dict):return d
    return None

def parse_lineup(line):
    result={}
    for ix,segment in enumerate((line or {}).get("lines") or [],1):
        entries=segment.get("entries") or []
        pos=1
        size=sum(len(i.get("numbers") or []) for i in entries)
        for member in entries:
            nums=member.get("numbers") or []
            for no in nums:
                if no in result:
                    result[no]={"conflict":True}
                else:
                    result[no]={"line_group":ix,"line_pos":pos,"line_size":size}
            pos+=len(nums) if len(nums)==1 else 1
    return result

def covered(v):
    return v is not None and v!=""

def inspect(d,case):
    date,venue,cup,day,race=case
    ents=[x for x in (d.get("entries") or []) if not x.get("absent")]
    players={str(p.get("id")):p for p in (d.get("players") or [])}
    records={str(p.get("playerId")):p for p in (d.get("records") or [])}
    lineup=parse_lineup(d.get("linePrediction"))
    counts=collections.Counter()
    no_set=[]
    for ent in ents:
        num=ent.get("number")
        player_id=ent.get("playerId")
        player=players.get(str(player_id)) or {}
        rec=records.get(str(player_id)) or {}
        line=lineup.get(num,{})
        e=rec.get("exLeftBehind")
        left=e.get("percentage") if isinstance(e,dict) else None
        obs={
            "official_player_id":covered(player_id),
            "name":covered(player.get("name")),
            "prefecture":covered(player.get("prefecture")),
            "term":covered(player.get("term")),
            "line_group":covered(line.get("line_group")),
            "line_pos":covered(line.get("line_pos")),
            "race_point":covered(rec.get("racePoint")),
            "style":covered(rec.get("style")),
            "ex_left_behind_pct":covered(left),
        }
        for field,ok in obs.items():
            if ok:counts[field]+=1
        if not all(obs[f] for f in ("line_group","line_pos","race_point","style","ex_left_behind_pct")):
            no_set.append(num)
    n=len(ents)
    raceinfo=d.get("race") or {}
    full_line=(n in (7,9) and counts["line_group"]==n and counts["line_pos"]==n and not any(s.get("conflict") for s in lineup.values()))
    full_arare=(full_line and all(counts[k]==n for k in
      ("race_point","style","ex_left_behind_pct")))
    return {
       "entries":n,
       "n_entries_page":raceinfo.get("entriesNumber"),
       "page_status":raceinfo.get("status"),
       "cancel":raceinfo.get("cancel"),
       "with_original_fields":dict(counts),
       "full_7_9_lineup":full_line,
       "full_7_9_six_type_inputs_available_now":full_arare,
       "unavailable_car_numbers":no_set,
       "line_groups":len((d.get("linePrediction") or {}).get("lines") or []),
       "asof_at_historical_race_start_verified":False,
       "has_official_player_id_all":n>0 and counts["official_player_id"]==n,
    }

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    session.headers.update(HEADERS)
    records=[]
    for case in CASES:
        date,venue,cup,day,race=case
        url=f"{BASE}/keirin/{venue}/racecard/{cup}/{day}/{race}"
        row={"race_date":date,"venue_slug":venue,
             "cup_id":cup,"day_index":day,"race_no":race,"url":url,
             "retrieved_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
             "used_original_winticket_preloaded_state_pipeline":True,
             "historical_asof_proven":False}
        try:
            resp=session.get(url,timeout=14,allow_redirects=True)
            row["http_status"]=resp.status_code
            row["redirected"]=resp.url!=url
            row["html_length"]=len(resp.content)
            if resp.status_code!=200:
                row["failure_stage"]="http_"+str(resp.status_code)
            else:
                state,err=get_state(resp.text)
                if err:row["failure_stage"]=err
                else:
                    d=query(state,"FETCH_KEIRIN_RACE")
                    if d is None:
                        row["failure_stage"]="fetch_keirin_race_query_missing"
                        # Diagnostics only, no query payloads or secrets.
                        row["tanstack_query_count"]=len((state.get("tanStackQuery") or {}).get("queries") or [])
                    else:
                        row["failure_stage"]=None
                        row.update(inspect(d,case))
        except requests.RequestException as exc:
            row["failure_stage"]="request_"+exc.__class__.__name__
        records.append(row)
        print("HISTORIC_WINTICKET_PROBE_CASE",json.dumps({
           "date":date,"venue":venue,"race_no":race,
           "http":row.get("http_status"),"stage":row.get("failure_stage"),
           "n":row.get("entries"),
           "line":row.get("with_original_fields",{}).get("line_group"),
           "behind":row.get("with_original_fields",{}).get("ex_left_behind_pct"),
           "full_type_input_now":row.get("full_7_9_six_type_inputs_available_now")
        },ensure_ascii=False),flush=True)
        time.sleep(1.2)  # 16 targeted requests, no search burst.
    successful=[x for x in records if x.get("failure_stage") is None]
    fields=("official_player_id","name","prefecture","term","line_group","line_pos","race_point","style","ex_left_behind_pct")
    report={
       "schema":"type_lab_2023_2024_winticket_historical_small_probe_v1",
       "source_code":"nyanta-kun/kiseki/keirin/src/scraper/winticket.py",
       "requests_attempted":len(records),"pages_200":sum(x.get("http_status")==200 for x in records),
       "successful_preloaded_state_race_data":len(successful),
       "full_seven_nine_six_type_input_now":sum(x.get("full_7_9_six_type_inputs_available_now") is True for x in successful),
       "full_official_player_ids_now":sum(x.get("has_official_player_id_all") is True for x in successful),
       "fields_covered":{"entries_total":sum(int(x["entries"]) for x in successful),
                         **{k:sum(x["with_original_fields"].get(k,0) for x in successful) for k in fields}},
       "observations":records,
       "not_proven":"Data fetched now does NOT certify its values at original 2023/2024 deadline. It cannot be used as a no-leak historical feature without vintage proof.",
       "not_a_bulk_collection":True,
       "source_repository_or_full_odds_archive_modified":False
    }
    (out/"historical_winticket_probe_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("HISTORIC_WINTICKET_PROBE_FINAL",json.dumps({k:report[k] for k in
      ("requests_attempted","pages_200","successful_preloaded_state_race_data",
      "full_seven_nine_six_type_input_now","fields_covered")},ensure_ascii=False),flush=True)
    return 0 if successful else 2

def test():
    sample={
       "tanStackQuery":{"queries":[{"queryKey":["FETCH_KEIRIN_RACE"],"state":{"data":{"entries":[{"number":1,"playerId":9}],
         "players":[{"id":9,"name":"tester","prefecture":"example","term":111}],
         "records":[{"playerId":9,"racePoint":93.2,"style":"逃","exLeftBehind":{"percentage":8.5}}],
         "linePrediction":{"lines":[{"entries":[{"numbers":[1]}]}]}}}}]}}
    data,err=get_state('abc;window.__PRELOADED_STATE__ = '+json.dumps(sample,ensure_ascii=False)+';')
    assert err is None and query(data,"FETCH_KEIRIN_RACE") is not None
    record=inspect(query(data,"FETCH_KEIRIN_RACE"),("2023-10-19","yahiko","2023101921",1,1))
    assert record["with_original_fields"]["ex_left_behind_pct"]==1
    assert not record["full_7_9_six_type_inputs_available_now"]
    print("HISTORIC_WINTICKET_PROBE_SELFTEST_OK")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="probe-output")
    ap.add_argument("--selftest",action="store_true")
    a=ap.parse_args()
    sys.exit(test() or 0 if a.selftest else run(Path(a.output)))

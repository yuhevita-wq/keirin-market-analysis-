#!/usr/bin/env python3
"""Read-only, capped 3連単 historical odds pilot; does not run a bulk crawler."""
import argparse
import gzip
import json
import math
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from research.odds_path_archive.collect import Source, matched

JST = timezone(timedelta(hours=9))
ALIASES = {"3rentan","sanrentan","sanren_tan","trifecta","tricast","tierce","santan","3tan","3連単","trio_ordered","triple_exacta"}
TICKET = re.compile(r"^([1-9])[-=]([1-9])[-=]([1-9])$")


def extract(r, provenance):
    info = r["race_info"]
    pools = r.get("odds_info", {}).get("time_pops") or {}
    stamps = r.get("odds_info", {}).get("time_odds_times") or {}
    if not isinstance(pools, dict):
        raise ValueError("Unexpected time_pops structure")
    candidates = [key for key in pools if key.lower().strip().replace("-", "_") in ALIASES]
    if len(candidates) != 1:
        raise ValueError("3連単 market identifier not confirmed; available keys: " + ",".join(sorted(pools)))
    key = candidates[0]
    frames, times = pools[key], stamps.get(key)
    if not isinstance(frames, list) or not isinstance(times, list) or len(frames) != len(times):
        raise ValueError("3連単 timepoint arrays do not align")
    cars = sorted(int(i) for i in r.get("horse_info", {}) if str(i).isdigit())
    if not 3 <= len(cars) <= 9 or len(cars) != len(set(cars)):
        raise ValueError("Invalid roster")
    full = len(cars) * (len(cars)-1) * (len(cars)-2)
    snapshots = []
    for stamp, frame in zip(times, frames):
        if not isinstance(stamp, str) or not isinstance(frame, dict):
            raise ValueError("Invalid snapshot")
        valid, invalid = {}, 0
        for selection, value in frame.items():
            m = TICKET.fullmatch(str(selection))
            if not m:
                raise ValueError("Not an ordered three-number ticket")
            triple = tuple(map(int, m.groups()))
            if len(set(triple)) != 3 or not set(triple).issubset(cars):
                raise ValueError("Impossible ticket")
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 1:
                invalid += 1
                continue
            name = "-".join(map(str, triple))
            if name in valid:
                raise ValueError("Duplicate ticket")
            valid[name] = value
        snapshots.append({"source_time_label": stamp, "is_source_final_label": stamp=="最終",
                          "valid_count": len(valid), "invalid_count":invalid,
                          "source_count":len(frame), "full_book":len(valid)==full,
                          "odds":valid})
    if not any(s["valid_count"] for s in snapshots):
        raise ValueError("No 3連単 odds observed")
    return {"race_date":info["race_date"],"track":info["track"],"race_no":info["race"],
            "source_race_id":info.get("id"),"bet_type":"3rentan","cars":cars,
            "expected_full_book":full,"source_pool_key":key,
            "provenance":provenance,"snapshots":snapshots,
            "note":"Unobserved timestamps and missing combinations are missing, not zero; full coverage is not guaranteed."}


def selftest():
    r={"race_info":{"race_date":"2026-11-08","track":"TEST","race":1},
       "horse_info":{str(i):{} for i in range(1,8)},
       "odds_info":{"time_pops":{"trifecta":[{"1-2-3":45.1,"2-1-3":36.4}]},
                    "time_odds_times":{"trifecta":["12:05"]}}}
    z=extract(r, {})
    assert z["expected_full_book"]==210 and len(z["snapshots"])==1
    assert z["snapshots"][0]["valid_count"]==2 and not z["snapshots"][0]["full_book"]
    try:
        extract({**r,"odds_info":{"time_pops":{"trio":[]},"time_odds_times":{"trio":[]}}}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("Mixed 3連複 with 3連単")
    print("PASS: synthetic 3連単 detection, timestamp alignment, full-book count, unknown-pool rejection")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--day",default="2026-11-08")
    p.add_argument("--out",default="work/trifecta_pilot.json.gz")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        selftest()
        return
    day=date.fromisoformat(args.day)
    if not date(2021,1,1)<=day<datetime.now(JST).date():
        p.error("Choose a past race date")
    src=Source(pause=2.0)
    catalog,_=src.api(args.day)
    tracks=catalog.get("kr_info",{}).get("track_info") or []
    if not isinstance(tracks,list) or not tracks:
        raise ValueError("No tracks for date")
    errors=[]
    for track in tracks[:2]:  # HARD CAP on live race fetches
        try:
            response,source=src.api(args.day,track,1)
            race=matched(response,args.day,track,1)
            if "ガール" in str(race["race_info"].get("syumoku","")):
                continue
            data=extract(race,source)
            output=Path(args.out)
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False,separators=(",",":")).encode()))
            print(json.dumps({"status":"SAVED","day":args.day,"track":track,
                              "frames":len(data["snapshots"]),
                              "observations":sum(x["valid_count"] for x in data["snapshots"]),
                              "max_frame":max(x["valid_count"] for x in data["snapshots"]),
                              "expected_full_book":data["expected_full_book"],
                              "output":str(output)},ensure_ascii=False))
            return
        except (ValueError,KeyError,RuntimeError) as exc:
            errors.append({"track":str(track),"reason":str(exc)[:450]})
    print(json.dumps({"status":"FAILED_TO_VERIFY_3RENTAN","attempted":len(errors),"errors":errors},ensure_ascii=False))
    sys.exit(2)

if __name__=="__main__":
    main()

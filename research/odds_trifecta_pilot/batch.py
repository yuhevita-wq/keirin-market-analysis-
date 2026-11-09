#!/usr/bin/env python3
"""Bounded ten-race verification of historic 3連単 timepoint availability."""
import argparse
import gzip
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from research.odds_path_archive.collect import Source, matched
from research.odds_trifecta_pilot.collect import extract

JST=timezone(timedelta(hours=9))

def sample(day,out):
    src=Source(pause=2.0)
    catalog,_=src.api(day)
    tracks=catalog.get("kr_info",{}).get("track_info") or []
    if not isinstance(tracks,list) or not tracks: raise ValueError("No meeting tracks")
    data=[];errors=[];calls=0
    for track in tracks[:2]:
        for number in range(1,11):
            if len(data)>=10: break
            calls+=1
            try:
                obj,provenance=src.api(day,track,number)
                race=matched(obj,day,track,number)
                if re.search("ガール|女子|L級|Ｌ級",str(race["race_info"].get("syumoku",""))):
                    continue
                data.append(extract(race,provenance))
            except (ValueError,RuntimeError,KeyError) as e:
                errors.append({"track":str(track),"race":number,"reason":str(e)[:160]})
        if len(data)>=10: break
    if not data: raise ValueError("No usable 3連単 records, errors="+str(errors))
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(gzip.compress(json.dumps({"day":day,
        "captured_at_jst":datetime.now(JST).isoformat(),
        "races":data,"errors":errors},ensure_ascii=False,separators=(",",":")).encode()))
    n=[len(r["snapshots"]) for r in data]
    print(json.dumps({"status":"SAVED","day":day,"races":len(data),
        "races_with_multiple_timepoints":sum(i>1 for i in n),
        "timepoints_total":sum(n),"timepoints_by_race":n,
        "observations_total":sum(sum(s["valid_count"] for s in r["snapshots"]) for r in data),
        "requests_to_races":calls,"errors":len(errors)},ensure_ascii=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--day",default="2026-11-08")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    sample(a.day,a.out)

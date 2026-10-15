"""Measure ACTUAL odds-only completeness against independently found race IDs.

Counts cover only days that have been enumerated from the public daily calendar.
This does NOT confuse the numerator (market books in odds-only files) with the
denominator (actual races found by the external calendar). It never assumes
uncrawled historical months have no races.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from collections import defaultdict
from datetime import date,timedelta,datetime,timezone
from pathlib import Path

from scope import in_scope,girls_race

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/audits/odds_census_coverage.json"


def market_ids():
    found=set()
    by_year=defaultdict(int)
    for path in sorted((ROOT/"data/odds_only").glob("odds_20??.jsonl.gz")):
        year=path.name[5:9]
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                row=json.loads(line)
                rid=row["race_id"]
                if rid in found:
                    raise ValueError("Duplicate market race ID across year files")
                found.add(rid)
                by_year[year]+=1
    return found,dict(by_year)


def census_indexes():
    dates=set()
    discovered={}
    summaries=[]
    for path in sorted((ROOT/"data/nationwide_calendar").glob("*/summary.json")):
        doc=json.loads(path.read_text(encoding="utf-8"))
        start=date.fromisoformat(doc["start"])
        end=date.fromisoformat(doc["end"])
        for n in range((end-start).days+1):
            day=start+timedelta(days=n)
            if in_scope(day.isoformat()):
                dates.add(day.isoformat())
        with gzip.open(path.parent/"race_index.csv.gz","rt",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid=row["race_id"]
                if not in_scope(row.get("race_date")):
                    continue
                # A later census may add an explicit race type to a formerly
                # unlabelled archived race; preserve the most informative label.
                if rid not in discovered or (row.get("race_type") and not discovered[rid].get("race_type")):
                    discovered[rid]=row
        summaries.append({"source":str(path.relative_to(ROOT)),
                          "status":doc["status"],
                          "days":doc["days_attempted"],
                          "race_ids":doc["unique_race_ids"]})
    return dates,discovered,summaries


def run():
    available,totals=market_ids()
    days,calendar,summaries=census_indexes()
    years=defaultdict(lambda:{"race_ids_found":0,"complete_numeric_market":0,
                                "no_complete_numeric_market":0,
                                "race_type_unlabelled":0})
    girls=0
    for rid,row in calendar.items():
        if girls_race(row.get("race_type")):
            girls+=1
            continue
        y=row["race_date"][:4]
        rec=years[y]
        rec["race_ids_found"]+=1
        if rid in available:
            rec["complete_numeric_market"]+=1
        else:
            rec["no_complete_numeric_market"]+=1
        if not row.get("race_type"):
            rec["race_type_unlabelled"]+=1
    data={}
    for y in sorted(set(totals)|set(years)):
        r=years[y]
        data[y]={"all_exported_full_market_races":totals.get(y,0),
                 **dict(r),
                 "observed_census_market_coverage_pct":
                     round(100*r["complete_numeric_market"]/r["race_ids_found"],2)
                     if r["race_ids_found"] else None}
    report={
        "status":"PARTIAL_NATIONWIDE_DATE_CENSUS_WITH_EXACT_MARKET_COVERAGE",
        "calculated_utc":datetime.now(timezone.utc).isoformat(),
        "covered_window":["2021-01-01","2026-06-30"],
        "only_men":True,
        "total_complete_market_races":len(available),
        "calendar_days_sampled":len(days),
        "calendar_races_with_girls_label_excluded":girls,
        "calendar_race_ids_unlabelled_may_include_girls":
            sum(v["race_type_unlabelled"] for v in years.values()),
        "year":data,
        "census_source_summaries":summaries,
        "critical_caveat":"Only race IDs from actually visited public daily-calendar days are in the denominator; uncrawled days are NOT counted as zero races. Unlabelled race types may include girls.",
        "market_qualifier":"Only complete numeric 3連単 + 3連複 price grids; 2車 and wide odds are not yet integrated",
        "no_result_or_player_features":True,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MARKET_VS_DISCOVERED_RACES",json.dumps(
        {"days":len(days),"by_year":data,"men_only_source_gap":girls},ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()

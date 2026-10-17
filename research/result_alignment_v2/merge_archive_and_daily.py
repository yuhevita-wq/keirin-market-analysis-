#!/usr/bin/env python3
"""Merge archived real finishing orders and newly published daily orders, by race_id.

Deduplicate by exact race ID; disagreeing outcome sources are quarantined,
never silently chosen. Result labels are not market features. Encrypted odds
cannot be matched without a private local key; no false matching claim.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/result_universe_v2"
OUT=BASE/"combined"
REPORT=ROOT/"data/audits/result_universe_merged_v2.json"

def lines(p):
    with gzip.open(p,"rt",encoding="utf-8") as f:
        for s in f:
            if s.strip():
                yield json.loads(s)


def merge():
    candidates=defaultdict(set)
    provenance=defaultdict(set)
    prior_ids=set()
    web_ids=set()
    for p in sorted(BASE.glob("results_20??.jsonl.gz")):
        for r in lines(p):
            rid=r["race_id"];finish=tuple(r["finish_order"])
            if len(finish)!=3 or len(set(finish))!=3:
                raise ValueError("invalid previously archived result "+rid)
            candidates[rid].add(finish)
            provenance[rid].add("archive")
            prior_ids.add(rid)
    day_files=list(sorted((BASE/"daily").glob("*/*.jsonl.gz")))
    for p in day_files:
        for r in lines(p):
            rid=r["race_id"];finish=tuple(r["finish_order"])
            if len(finish)!=3 or len(set(finish))!=3:
                raise ValueError("invalid published daily result "+rid)
            candidates[rid].add(finish)
            provenance[rid].add("published_day")
            web_ids.add(rid)
    conflicts={rid:[list(v) for v in sorted(vals)] for rid,vals in candidates.items() if len(vals)!=1}
    completed={rid:next(iter(vals)) for rid,vals in candidates.items() if len(vals)==1}
    OUT.mkdir(parents=True,exist_ok=True)
    years={}
    for year in range(2021,2027):
        # Meeting start year can differ from actual race year at New Year's.
        from datetime import date,timedelta
        ids=[rid for rid in completed
             if (date(int(rid[2:6]),int(rid[6:8]),int(rid[8:10]))+timedelta(days=int(rid[10:12])-1)).year==year]
        ids.sort()
        p=OUT/f"results_{year}.jsonl.gz"
        with p.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw,mode="wb",filename="",mtime=0) as gz:
                for rid in ids:
                    gz.write((json.dumps({"race_id":rid,"finish_order":completed[rid]},separators=(",",":"))+"\n").encode())
        years[str(year)]={"unique_verified_results":len(ids),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    report={
      "schema":"result_universe_v2_combined_exact_id_dedup",
      "generated_utc":datetime.now(timezone.utc).isoformat(),
      "verified_unique_results":len(completed),
      "previous_archived_results":len(prior_ids),
      "published_daily_result_ids":len(web_ids),
      "new_race_ids_not_in_old_results":len(web_ids-prior_ids),
      "same_race_id_confirmed_by_both":len(prior_ids & web_ids),
      "conflicting_race_ids":len(conflicts),
      "conflicts_sample":dict(list(sorted(conflicts.items()))[:15]),
      "daily_files_used":len(day_files),
      "years":years,
      "warning":"Comparison against 134,336+ encrypted seven-bet odds races NOT performed. Decrypt privately before exact-ID join.",
      "results_are_targets_never_odds_features":True,
      "no_fabricated_or_substituted_podiums":True
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RESULTS_UNIQUE_JOIN",json.dumps({
        "unique":len(completed),"daily_races":len(web_ids),
        "new":len(web_ids-prior_ids),"conflicts":len(conflicts),
        "daily_files":len(day_files)
    }),flush=True)
    return report

if __name__=="__main__":
    merge()

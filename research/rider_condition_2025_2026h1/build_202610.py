#!/usr/bin/env python3
"""Extend the canonical rider results dataset through 2026-10-31, without touching H1 outputs."""
from __future__ import annotations
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
import build as base

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"data/rider_condition_2025_202610"
prior=base.folders

def full_window_sources():
    # Canonical original H1 sources, with no all_classes shadow folders.
    yield from prior()
    root=ROOT/"data"
    extra=[root/"2026_future_block1/s_class_f1_20260701_20260830"]
    extra+=sorted((root/"a_class_12").glob("2026_07_08"))
    extra+=sorted((root/"a_class_12").glob("2026_09_1002"))
    extra+=sorted((root/"a3_challenge").glob("2026_07_08"))
    extra+=sorted((root/"a3_challenge").glob("2026_09_1007"))
    extra+=sorted((root/"rider_condition_gap_2026").glob("batch_*"))
    for folder in extra:
        if (folder/"races.csv").exists():
            yield folder

base.folders=full_window_sources
base.END="2026-10-31"
base.OUT=OUT

def main():
    base.run()
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    months=Counter()
    type_counts=Counter()
    for y in (2025,2026):
        p=OUT/f"races_{y}.csv.gz"
        with gzip.open(p,"rt",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                months[row["race_date"][:7]]+=1
                type_counts[(row["race_date"][:7],row["race_type"])]+=1
    gaps=[]
    for p in sorted((ROOT/"data/rider_condition_gap_2026").glob("batch_*/summary.json")):
        j=json.loads(p.read_text(encoding="utf-8"))
        gaps.append({"batch":str(p.parent.relative_to(ROOT)),
                     "period":j["period"],
                     "discovered":j["discovered_unique_men_race_ids_in_batch"],
                     "already_present_skipped":j["already_present_skipped"],
                     "new_full_result_races":j["new_full_result_races"],
                     "recovery_failures":j["recovery_failures"],
                     "daily_discovery_failures":len(j["daily_discovery_failures"]),
                     "unmapped_link_ids":j["unmapped_daily_link_id_count"]})
    extras={
        "window_requested":["2025-01-01","2026-10-31"],
        "unique_races_by_month":dict(sorted(months.items())),
        "race_types_by_month":[{"month":m,"race_type":k,"races":v}
                                for (m,k),v in sorted(type_counts.items())],
        "gap_batch_audits":gaps,
        "gap_batch_count":len(gaps),
        "covered_all_requested_date_batches":len(gaps)>=11,
        "nationwide_completeness_confirmed":False,
        "note":"No full nationwide race census exists for 2026-H2. This is a merged, audited collection of available sources, not a certified nationwide denominator."
    }
    summary.update(extras)
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("EXTENDED_WINDOW_RESULT",json.dumps({
         "races":summary["race_ids_with_entrant_data"],
         "rider_rows":summary["rider_race_rows"],
         "unresolved_results":summary["counters"].get("rider_status_unverified",0),
         "month_counts":extras["unique_races_by_month"],
         "batches":len(gaps)},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

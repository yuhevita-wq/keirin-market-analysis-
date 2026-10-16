#!/usr/bin/env python3
"""Count archived seven-bet odds races using immutable sealed-run manifests.

Manifest race totals alone cannot identify duplicated race IDs from retries.
Compute a rigorous lower bound by taking the maximum sealed-race count for
each calendar day; report raw run counts separately without presenting those
as unique races. Do not read, decrypt or modify any archived odds.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/audits/odds_seven_bet_runs"
CIPHERS = ROOT / "data/odds_ciphertext"
OUT = ROOT / "data/audits/odds_seven_bet_archive_count.json"

def audit():
    days = defaultdict(list)
    bad_manifests = []
    total_manifest_races = 0
    total_manifest_quote_rows = 0
    checked = 0
    for path in sorted(MANIFESTS.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc.get("dates", {}), dict):
            bad_manifests.append({"file": path.name, "reason": "dates not dict"})
            continue
        date_rows = doc.get("dates") or {}
        if not date_rows:
            continue
        by_path = {Path(f["path"]).parts[-2]: f for f in doc.get("encrypted_files", [])}
        if sum(int(v.get("complete_races", 0)) for v in date_rows.values()) != int(doc.get("complete_races", 0)):
            bad_manifests.append({"file": path.name, "reason": "run totals disagree with dates"})
            continue
        if len(by_path) != len(date_rows):
            bad_manifests.append({"file": path.name, "reason": "ciphertext file/date count mismatch"})
            continue
        # Check complete validity of each run before including its counts.
        ok = True
        for day, data in date_rows.items():
            f = by_path.get(day)
            if not f or not f["path"].startswith("data/odds_ciphertext/"):
                ok = False
                break
            sealed = ROOT / f["path"]
            if not sealed.is_file() or sealed.stat().st_size < 500:
                ok = False
                break
        if not ok:
            bad_manifests.append({"file": path.name, "reason": "archived encrypted file missing"})
            continue
        checked += 1
        total_manifest_races += int(doc["complete_races"])
        total_manifest_quote_rows += int(doc.get("quote_rows", 0))
        for day, v in date_rows.items():
            days[day].append({"complete_races": int(v.get("complete_races",0)),
                              "quote_rows": int(v.get("quote_rows",0)),
                              "run_id": doc["run_id"]})
    years = {}
    for year in range(2021,2027):
        selected = {d:records for d,records in days.items() if d.startswith(str(year))}
        max_counts = [max(r["complete_races"] for r in records) for records in selected.values()]
        max_quotes = [max(r["quote_rows"] for r in records) for records in selected.values()]
        years[str(year)] = {
            "distinct_days_with_verified_ciphertext": len(selected),
            "confirmed_distinct_race_lower_bound": sum(max_counts),
            "distinct_day_quote_rows_lower_bound": sum(max_quotes),
            "raw_complete_races_across_sealed_runs_with_retry_duplicates": sum(
                r["complete_races"] for records in selected.values() for r in records),
            "days_with_multiple_sealed_runs": sum(len(records)>1 for records in selected.values()),
        }
    report = {
        "schema": "seven_bet_archive_race_count_v1",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "scope": "separately archived seven-bet encrypted odds, 2021-01-01 to 2026-06-30",
        "verified_sealed_run_manifests": checked,
        "unverified_run_manifests": bad_manifests[:25],
        "unverified_run_manifest_count": len(bad_manifests),
        "distinct_days":len(days),
        "confirmed_distinct_race_lower_bound": sum(v["confirmed_distinct_race_lower_bound"] for v in years.values()),
        "raw_complete_races_across_sealed_runs_WITH_DUPLICATES":total_manifest_races,
        "raw_quote_rows_across_sealed_runs_WITH_DUPLICATES":total_manifest_quote_rows,
        "year": years,
        "dedup_caveat": (
            "Per-day MAX of fully verified sealed-run race counts guarantees a lower bound. "
            "Same race IDs may occur in multiple sealed runs. Separate reruns may contain "
            "different races, so an exact unique-race total requires decrypting the odds "
            "and counting distinct race_id, or an equivalent plaintext secure index."
        ),
        "does_not_mutate_original_odds":True,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SEVEN_BET_ODDS_ARCHIVE_COUNTS",json.dumps({
        "days":len(days),
        "distinct_races_at_least":report["confirmed_distinct_race_lower_bound"],
        "sum_with_retry_duplicates":total_manifest_races,
        "per_year":{k: v["confirmed_distinct_race_lower_bound"] for k,v in years.items()},
        "checked_manifests": checked, "rejected_manifests": len(bad_manifests)
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    audit()

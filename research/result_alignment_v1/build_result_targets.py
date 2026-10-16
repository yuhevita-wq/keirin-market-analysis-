#!/usr/bin/env python3
"""Align verified race results to the exact race-ID set in odds_only.

Outcomes are LABELS ONLY, never model features.  This script reads existing
archival data; it does not request external sites or modify odds.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ODDS = ROOT / "data/odds_only"
OUT = ROOT / "data/results_only"
REPORT = ROOT / "data/audits/results_to_odds_alignment.json"
ID_RX = re.compile(r"^\d{16}$")
YEAR_RX = re.compile(r"^odds_(20\d\d)\.jsonl\.gz$")


def odds_universe(root=ROOT):
    """Use year-partitioned *actual* saved odds, never a date estimate."""
    ids = {}
    for path in sorted((root / "data/odds_only").glob("odds_20??.jsonl.gz")):
        match = YEAR_RX.fullmatch(path.name)
        if match is None:
            continue
        year = match.group(1)
        with gzip.open(path, "rt", encoding="utf-8") as src:
            for line in src:
                row = json.loads(line)
                rid = str(row["race_id"])
                if not ID_RX.fullmatch(rid) or rid in ids:
                    raise ValueError(f"invalid/duplicate odds race ID: {rid}")
                tri = row["odds"]["3連単"]
                cars = {int(c) for combo in tri for c in combo.split("-")}
                if len(cars) < 3 or len(tri) != len(cars)*(len(cars)-1)*(len(cars)-2):
                    raise ValueError(f"invalid 3-way odds book: {rid}")
                ids[rid] = (year, cars)
    if not ids:
        raise ValueError("No source odds books found")
    return ids


def parse_podium(rows, allowed_cars):
    """A complete, non-ambiguous 1st/2nd/3rd among cars in its odds book."""
    podium = defaultdict(list)
    for row in rows:
        pos = str(row.get("finish_position", "")).strip()
        car = str(row.get("car_no", "")).strip()
        if pos in ("1", "2", "3"):
            if not re.fullmatch(r"[1-9]", car):
                return None
            podium[int(pos)].append(int(car))
    if set(podium) != {1, 2, 3} or any(len(podium[k]) != 1 for k in (1,2,3)):
        return None
    finish = tuple(podium[k][0] for k in (1,2,3))
    if len(set(finish)) != 3 or not set(finish).issubset(allowed_cars):
        return None
    return finish


def csv_records(handle):
    yield from csv.DictReader(handle)


def consume_records(rows, provenance, universe, proposals, stats):
    groups = defaultdict(list)
    for row in rows:
        rid = str(row.get("race_id", "")).strip()
        if rid in universe:
            groups[rid].append(row)
    stats["files_read"] += 1
    stats["candidate_races"] += len(groups)
    for rid, records in groups.items():
        outcome = parse_podium(records, universe[rid][1])
        if outcome is None:
            stats["invalid_podium_candidates"] += 1
            continue
        proposals[rid].add(outcome)
        stats["valid_podium_candidates"] += 1


def read_sources(root, universe):
    """Read independent full-source tables, recovered sharded labels and grade ZIPs."""
    proposals = defaultdict(set)
    stats = Counter()
    for path in sorted((root / "data").rglob("results.csv")):
        if "results_only" in path.parts:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            consume_records(csv_records(handle), str(path.relative_to(root)),
                            universe, proposals, stats)
    for path in sorted((root / "data").rglob("results.csv.gz")):
        if "results_only" in path.parts:
            continue
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            consume_records(csv_records(handle), str(path.relative_to(root)),
                            universe, proposals, stats)
    for path in sorted((root / "data/grade_races").rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.rsplit("/", 1)[-1] != "results.csv":
                    continue
                with archive.open(name) as raw:
                    with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
                        consume_records(csv_records(handle),
                                        f"{path.relative_to(root)}:{name}",
                                        universe, proposals, stats)
    return proposals, stats


def write_gzip_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic content across automatic reruns.
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            for line in lines:
                compressed.write((line + "\n").encode("utf-8"))


def build(root=ROOT):
    universe = odds_universe(root)
    proposals, source_stats = read_sources(root, universe)
    outputs = root / "data/results_only"
    report_path = root / "data/audits/results_to_odds_alignment.json"
    reports = {}
    missing = []
    conflicts = []
    complete_total = 0
    for year in sorted({year for year, _ in universe.values()}):
        output_rows = []
        for rid in sorted(k for k, (y, _) in universe.items() if y == year):
            matches = proposals.get(rid, set())
            if len(matches) == 1:
                rank = next(iter(matches))
                output_rows.append(json.dumps({
                    "race_id": rid, "finish_order": list(rank)
                }, ensure_ascii=False, separators=(",", ":")))
            else:
                reason = "CONFLICTING_ARCHIVES" if len(matches) > 1 else "MISSING_OR_INVALID_PODIUM"
                missing.append((rid, year, reason))
                if len(matches) > 1:
                    conflicts.append({"race_id": rid, "observed_orders": sorted(list(m) for m in matches)})
        path = outputs / f"results_{year}.jsonl.gz"
        write_gzip_lines(path, output_rows)
        n = sum(1 for y, _ in universe.values() if y == year)
        reports[year] = {"odds_races": n, "matched_results": len(output_rows),
                         "unmatched": n-len(output_rows),
                         "results_file": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        complete_total += len(output_rows)
    write_gzip_lines(outputs / "unmatched_race_ids.csv.gz",
                     ["race_id,year,reason"] + [",".join(item) for item in missing])
    assert complete_total + len(missing) == len(universe)
    assert sum(x["odds_races"] for x in reports.values()) == len(universe)
    data = {
        "schema": "result_targets_matched_to_odds_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "odds_race_count": len(universe),
        "matched_result_count": complete_total,
        "missing_result_count": len(missing),
        "conflicting_result_count": len(conflicts),
        "by_year": reports,
        "source_counts": dict(source_stats),
        "conflicts_sample": conflicts[:30],
        "results_are_labels_not_features": True,
        "source_odds_unchanged": True,
        "note": "Unmatched or conflicting races are NOT assigned a synthetic result."
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RESULT_ALIGNMENT", json.dumps({
        "odds": len(universe), "matched": complete_total, "missing": len(missing),
        "conflicts": len(conflicts), "by_year": reports
    }, ensure_ascii=False), flush=True)
    return data


if __name__ == "__main__":
    build()

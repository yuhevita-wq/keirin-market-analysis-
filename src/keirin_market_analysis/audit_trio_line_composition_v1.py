"""Verify the frozen 2023-2025 seven-rider 3-2-2 trio line-composition baseline.

Read-only by design. Never automatically updates the frozen v1 JSON.
Run at the repository root:
    python -m src.keirin_market_analysis.audit_trio_line_composition_v1
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

YEARS = (2023, 2024, 2025)
ROOT = Path("data")
FROZEN = ROOT / "probabilities" / "trio_line_composition_s_yosen_2023_2025_v1.json"


def records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pattern_counts(rows: list[dict[str, str]], entries: list[dict[str, str]], payouts: list[dict[str, str]]):
    by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        by_race[entry["race_id"]].append(entry)
    trio_paid = {
        item["race_id"]: item
        for item in payouts
        if item["ticket_type"] == "3連複" and item["status"] == "paid"
    }
    counts, shape_counts, shape_patterns = Counter(), Counter(), defaultdict(Counter)
    excluded = Counter()
    for race in rows:
        if int(race["entry_count"]) != 7:
            excluded["non_7_rider"] += 1
            continue
        es = by_race.get(race["race_id"], [])
        winner = trio_paid.get(race["race_id"])
        if len(es) != 7 or winner is None:
            raise ValueError(f"Missing seven entries or paid trio: {race['race_id']}")
        line_map: dict[str, list[dict[str, str]]] = defaultdict(list)
        for e in es:
            if not e["line_id"].isdigit() or not e["line_position"].isdigit():
                raise ValueError(f"Missing line field: {race['race_id']}")
            try:
                Decimal(e["score"])
            except (InvalidOperation, KeyError):
                raise ValueError(f"Missing or malformed score: {race['race_id']}")
            line_map[e["line_id"]].append(e)
        lines = [
            sorted(members, key=lambda z: int(z["line_position"]))
            for members in line_map.values()
        ]
        if len(lines) != 3 or sorted(map(len, lines)) != [2, 2, 3]:
            excluded["non_3_2_2_line_shape"] += 1
            continue
        ranked = sorted(
            ((Decimal(line[0]["score"]) + Decimal(line[1]["score"]), line) for line in lines),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked[0][0] == ranked[1][0] or ranked[1][0] == ranked[2][0]:
            excluded["equal_pair_score_tie"] += 1
            continue
        winners = [int(x) for x in re.findall(r"\d+", winner["combination"])]
        eligible = {int(e["car_no"]) for e in es}
        if len(winners) != 3 or len(set(winners)) != 3 or not set(winners) <= eligible:
            raise ValueError(f"Malformed paid trio: {race['race_id']}")
        pattern = "-".join(
            str(sum(int(e["car_no"]) in winners for e in line))
            for _, line in ranked
        )
        shape = "-".join(str(len(line)) for _, line in ranked)
        counts[pattern] += 1
        shape_counts[shape] += 1
        shape_patterns[shape][pattern] += 1
    return counts, shape_counts, shape_patterns, excluded


def main() -> int:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    aggregate = Counter()
    shape_aggregate = defaultdict(Counter)
    combined_shape_counts = Counter()
    for year in YEARS:
        directory = ROOT / str(year) / "s_class_yosen"
        rows = records(directory / "races.csv")
        es = records(directory / "entries.csv")
        pays = records(directory / "payouts.csv")
        patterns, shapes, shape_patterns, excluded = pattern_counts(rows, es, pays)
        year_record = frozen["by_year"][str(year)]
        expected_patterns = {x["pattern"]: x["count"] for x in year_record["patterns"]}
        if patterns != Counter(expected_patterns):
            raise AssertionError(f"Year {year} pattern counts differ from frozen v1: {patterns}")
        if dict(shapes) != year_record["line_size_shape_counts"]:
            raise AssertionError(f"Year {year} strength-ranked line sizes differ")
        if dict(excluded) != {
            k: v for k, v in frozen["eligibility_audit"]["by_year"][str(year)]["excluded"].items() if v
        }:
            raise AssertionError(f"Year {year} exclusions differ")
        if sum(patterns.values()) != year_record["races"]:
            raise AssertionError(f"Year {year} total differs")
        aggregate.update(patterns)
        combined_shape_counts.update(shapes)
        for shape, cnt in shape_patterns.items():
            expected = {
                row["pattern"]: row["count"]
                for row in frozen["by_line_size_shape"][shape]["by_year"][str(year)]["patterns"]
            }
            if cnt != Counter(expected):
                raise AssertionError(f"Year {year} line-size {shape} counts differ")
            shape_aggregate[shape].update(cnt)
    aggregate_expected = {row["pattern"]: row["count"] for row in frozen["aggregate"]["patterns"]}
    if aggregate != Counter(aggregate_expected):
        raise AssertionError("All-year counts differ")
    for shape, cnt in shape_aggregate.items():
        expected = {
            row["pattern"]: row["count"]
            for row in frozen["by_line_size_shape"][shape]["patterns"]
        }
        if cnt != Counter(expected):
            raise AssertionError(f"All-year line-size {shape} counts differ")
    if sum(aggregate.values()) != frozen["aggregate"]["races"]:
        raise AssertionError("All-year eligible race total differs")
    print(json.dumps({
        "status": "FROZEN_V1_VERIFIED",
        "eligible_races": sum(aggregate.values()),
        "patterns": dict(aggregate),
        "line_shapes": dict(combined_shape_counts),
        "frozen_file": str(FROZEN),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

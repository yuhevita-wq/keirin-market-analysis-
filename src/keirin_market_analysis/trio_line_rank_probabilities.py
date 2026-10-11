"""3連複: seven-rider 3-2-2 S-class preliminary race line-rank census.

Usage from repository root:
    python src/keirin_market_analysis/trio_line_rank_probabilities.py

Uses only published predicted lines and prerace score to rank lines.
Results are used only to label the three riders who finished on the podium.
No odds, payouts, or post-result features enter line-ranking.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research/trio_line_rank_2023_2026h1_v1.json"
PERIODS = {
    "2023": ROOT / "data/2023/s_class_yosen",
    "2024": ROOT / "data/2024/s_class_yosen",
    "2025": ROOT / "data/2025/s_class_yosen",
    "2026_h1": ROOT / "data/2026_h1/s_class_yosen",
}
PATTERNS = (
    "3-0-0", "2-1-0", "2-0-1", "1-2-0", "1-1-1",
    "1-0-2", "0-3-0", "0-2-1", "0-1-2", "0-0-3",
)


def read_rows(directory, filename):
    with (directory / filename).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def wilson95(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [round(100 * (center - margin), 3), round(100 * (center + margin), 3)]


def counter_report(counts, denominator, keys=None):
    labels = keys if keys is not None else sorted(counts)
    return {
        k: {
            "count": counts.get(k, 0),
            "rate_pct": round(100 * counts.get(k, 0) / denominator, 4) if denominator else None,
            "wilson95_pct": wilson95(counts.get(k, 0), denominator),
        }
        for k in labels
    }


def analyze_period(period, directory):
    races = read_rows(directory, "races.csv")
    entries = read_rows(directory, "entries.csv")
    payouts = read_rows(directory, "payouts.csv")

    by_race = defaultdict(list)
    for e in entries:
        by_race[e["race_id"]].append(e)
    trifecta_sets = {
        p["race_id"]: p
        for p in payouts
        if p["ticket_type"] == "3連複" and p["status"] == "paid"
    }
    record = []
    excluded = Counter()

    for race in races:
        if race.get("entry_count") != "7":
            excluded["not_7_riders"] += 1
            continue
        rider_rows = by_race.get(race["race_id"], [])
        paid = trifecta_sets.get(race["race_id"])
        if len(rider_rows) != 7 or paid is None:
            excluded["missing_entry_or_result"] += 1
            continue
        lines = defaultdict(list)
        invalid = False
        for e in rider_rows:
            try:
                lid = int(e["line_id"])
                pos = int(e["line_position"])
                score = float(e["score"])
                if not math.isfinite(score) or pos < 1:
                    raise ValueError("invalid score/position")
            except (TypeError, ValueError):
                invalid = True
                break
            lines[lid].append(e)
        if invalid:
            excluded["invalid_prerace_line_or_score"] += 1
            continue
        members = [sorted(line, key=lambda e: int(e["line_position"]))
                   for line in lines.values()]
        if len(members) != 3 or sorted(map(len, members)) != [2, 2, 3]:
            excluded["not_exact_3_2_2"] += 1
            continue
        if any([int(e["line_position"]) for e in line] != list(range(1, len(line) + 1))
               for line in members):
            excluded["invalid_line_positions"] += 1
            continue

        # Rank every line by the sum of its first two riders' competition scores.
        ranked = sorted(members,
                        key=lambda line: float(line[0]["score"]) + float(line[1]["score"]),
                        reverse=True)
        sums = [float(line[0]["score"]) + float(line[1]["score"]) for line in ranked]
        if abs(sums[0] - sums[1]) < 1e-8 or abs(sums[1] - sums[2]) < 1e-8:
            excluded["pair_score_tie"] += 1
            continue
        try:
            winners = [int(x) for x in paid["combination"].replace("-", "=").split("=")]
        except ValueError:
            winners = []
        all_cars = {int(e["car_no"]) for e in rider_rows}
        if len(winners) != 3 or len(set(winners)) != 3 or not set(winners) <= all_cars:
            excluded["invalid_trio_result"] += 1
            continue

        counts = [sum(int(e["car_no"]) in winners for e in line) for line in ranked]
        pattern = "-".join(map(str, counts))
        lengths = "-".join(str(len(line)) for line in ranked)
        roles = sorted(
            f"{'HML'[idx]}{int(e['line_position'])}"
            for idx, line in enumerate(ranked)
            for e in line if int(e["car_no"]) in winners
        )
        exact = "=".join(roles)
        record.append((pattern, lengths, exact))

    return record, excluded, len(races)


def summarize(records):
    total = len(records)
    pats = Counter(x[0] for x in records)
    by_length = {}
    for length in ("3-2-2", "2-3-2", "2-2-3"):
        sub = [r for r in records if r[1] == length]
        sizes = [int(x) for x in length.split("-")]
        role_options = [
            [f"{'HML'[i]}{j}" for j in range(1, sizes[i] + 1)] for i in range(3)
        ]
        all_codes = ["=".join(t) for t in itertools.combinations(
            sorted(itertools.chain.from_iterable(role_options)), 3)]
        by_length[length] = {
            "races": len(sub),
            "pattern_probabilities": counter_report(
                Counter(x[0] for x in sub), len(sub), PATTERNS),
            "exact_trio_probabilities": counter_report(
                Counter(x[2] for x in sub), len(sub), all_codes),
        }
        assert sum(x["count"] for x in by_length[length]["exact_trio_probabilities"].values()) == len(sub)
    assert sum(pats.values()) == total
    return {
        "races": total,
        "pattern_probabilities": counter_report(pats, total, PATTERNS),
        "top3_inclusion_by_line": {
            lab: {
                "count": sum(r[0].split("-")[i] != "0" for r in records),
                "rate_pct": round(100 * sum(r[0].split("-")[i] != "0" for r in records) / total, 4)
                if total else None,
            } for i, lab in enumerate(("strong", "middle", "weak"))
        },
        "by_line_size_rank": by_length,
    }


def main():
    parts = {}
    for period, directory in PERIODS.items():
        records, excluded, input_count = analyze_period(period, directory)
        parts[period] = {
            "input_races": input_count,
            "excluded": dict(excluded),
            "statistics": summarize(records),
            "_records": records,
        }
    development = [r for p in ("2023", "2024", "2025") for r in parts[p]["_records"]]
    result = {
        "title": "Three-line 3連複 top-three composition, 7 riders, S-class 予選",
        "version": "v1",
        "scope": "S級予選: 7-car races with exactly 3-2-2 predicted line formation",
        "line_strength_definition": "descending sum of competition scores of first two listed riders in each line",
        "line_role_definition": "H=strongest pair-score line, M=middle, L=weakest; digit=published line position",
        "outcome_definition": "counts or exact rider-role set of three winners, order ignored",
        "tie_policy": "exclude any pair-score ties; never resolve with race result",
        "odds_used": False,
        "payout_used_for_classification": False,
        "source": "data/{2023,2024,2025,2026_h1}/s_class_yosen/{races,entries,payouts}.csv",
        "limitations": [
            "The three-line composition probability is not an individual 3連複 ticket probability.",
            "Published predicted formations are not actual in-race positions.",
            "Periods contain the same S級予選 event type only, not all racing grades.",
            "2026_h1 is a later descriptive validation period, not used to select categories.",
            "Odds and price drift must be evaluated separately before any EV claim.",
        ],
        "development_2023_2025": summarize(development),
        "validation_2026_h1": parts["2026_h1"]["statistics"],
        "by_period": {
            p: {k: v for k, v in entry.items() if k != "_records"}
            for p, entry in parts.items()
        },
    }
    if [parts[p]["statistics"]["races"] for p in ("2023", "2024", "2025")] != [1029, 970, 1016]:
        raise RuntimeError("Historical race-count regression detected")
    if result["development_2023_2025"]["races"] != 3015:
        raise RuntimeError("Development population changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", OUT, "development", len(development),
          "validation", result["validation_2026_h1"]["races"])


if __name__ == "__main__":
    main()

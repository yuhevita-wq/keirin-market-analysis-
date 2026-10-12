"""Read-only, full-source reproducibility audit for final-back (B) observations.

Run: python -m keirin_market_analysis.audit_final_back_322_observations_v1
Does not edit historical baselines, observations, or probability models.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path("data")
YEARS = ("2023", "2024", "2025", "2026_h1")
OBS_PATH = ROOT / "observations/final_back_s_yosen_322_2023_2026h1_v1.csv"
SUMMARY_PATH = ROOT / "probabilities/final_back_s_yosen_322_observed_v1.json"
BASE_PATH = ROOT / "probabilities/trio_line_composition_s_yosen_2023_2025_v1.json"

FIELDS = (
    "race_id", "period", "race_date", "track", "race_no", "line_size_shape",
    "back_car_no", "back_sb", "back_line_rank", "back_line_position",
    "back_line_size", "back_line_head_car_no", "H_head_pre_b_count",
    "M_head_pre_b_count", "L_head_pre_b_count", "H_pair_score",
    "M_pair_score", "L_pair_score", "unique_highest_pre_b_line",
    "finish_status", "top3_car_nos", "top3_line_counts_HML",
    "top3_slots", "back_line_top3_count", "result_source_url",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_race(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r["race_id"]].append(r)
    return grouped


def reconstruct(year: str) -> dict[str, dict[str, str]]:
    folder = ROOT / year / "s_class_yosen"
    entries = group_by_race(read_csv(folder / "entries.csv"))
    results = group_by_race(read_csv(folder / "results.csv"))
    output: dict[str, dict[str, str]] = {}

    for rid, es in entries.items():
        if len(es) != 7:
            continue
        by_line: dict[str, list[dict[str, str]]] = defaultdict(list)
        for e in es:
            by_line[e["line_id"]].append(e)
        if len(by_line) != 3 or sorted(map(len, by_line.values())) != [2, 2, 3]:
            continue

        lines = [sorted(g, key=lambda x: int(x["line_position"])) for g in by_line.values()]
        if any([int(e["line_position"]) for e in g] != list(range(1, len(g) + 1)) for g in lines):
            continue
        ranked = sorted(
            [
                (Decimal(g[0]["score"]) + Decimal(g[1]["score"]), g)
                for g in lines
            ],
            key=lambda x: x[0],
            reverse=True,
        )
        if ranked[0][0] == ranked[1][0] or ranked[1][0] == ranked[2][0]:
            continue

        rr = results.get(rid, [])
        if len(rr) != 7 or len({r["car_no"] for r in rr}) != 7:
            raise AssertionError(f"Missing or duplicated race result: {rid}")
        back = [r for r in rr if "B" in r["sb"]]
        if len(back) != 1:
            raise AssertionError(f"Not exactly one observed B: {rid} -> {len(back)}")
        car = back[0]["car_no"]
        line_index = next(
            (i for i, (_, g) in enumerate(ranked) if any(e["car_no"] == car for e in g)),
            None,
        )
        if line_index is None:
            raise AssertionError(f"B car not in lineup: {rid}:{car}")
        b_line = ranked[line_index][1]
        b_pos = next(i + 1 for i, e in enumerate(b_line) if e["car_no"] == car)
        shape = "-".join(str(len(g)) for _, g in ranked)
        head_bs = [int(g[0]["b_count"]) for _, g in ranked]
        biggest = max(head_bs)
        max_indices = [i for i, n in enumerate(head_bs) if n == biggest]
        sole_biggest = "HML"[max_indices[0]] if len(max_indices) == 1 else ""

        top = [r for r in rr if r["finish_position"].isdigit()
               and 1 <= int(r["finish_position"]) <= 3]
        normal = len(top) == 3 and sorted(int(r["finish_position"]) for r in top) == [1, 2, 3]
        top_cars = slots = counts_str = b_count = ""
        if normal:
            top.sort(key=lambda r: int(r["finish_position"]))
            top_cars = "-".join(r["car_no"] for r in top)
            counts = [0, 0, 0]
            slots_list = []
            for rider in top:
                found = [
                    (i, pos + 1)
                    for i, (_, g) in enumerate(ranked)
                    for pos, e in enumerate(g)
                    if e["car_no"] == rider["car_no"]
                ]
                if len(found) != 1:
                    raise AssertionError(f"Invalid finish car: {rid}:{rider['car_no']}")
                li, slot = found[0]
                counts[li] += 1
                slots_list.append((li, slot))
            counts_str = "-".join(map(str, counts))
            b_count = str(counts[line_index])
            slots = "+".join("HML"[li] + str(slot) for li, slot in sorted(slots_list))

        def score(i: int) -> str:
            return format(ranked[i][0], ".2f")

        candidate = {
            "race_id": rid,
            "period": "holdout_2026_h1" if year == "2026_h1" else "train_2023_2025",
            "race_date": es[0]["race_date"],
            "track": es[0]["track"],
            "race_no": es[0]["race_no"],
            "line_size_shape": shape,
            "back_car_no": car,
            "back_sb": back[0]["sb"],
            "back_line_rank": "HML"[line_index],
            "back_line_position": str(b_pos),
            "back_line_size": str(len(b_line)),
            "back_line_head_car_no": b_line[0]["car_no"],
            "H_head_pre_b_count": str(head_bs[0]),
            "M_head_pre_b_count": str(head_bs[1]),
            "L_head_pre_b_count": str(head_bs[2]),
            "H_pair_score": score(0),
            "M_pair_score": score(1),
            "L_pair_score": score(2),
            "unique_highest_pre_b_line": sole_biggest,
            "finish_status": "normal" if normal else "tie_or_nonstandard",
            "top3_car_nos": top_cars,
            "top3_line_counts_HML": counts_str,
            "top3_slots": slots,
            "back_line_top3_count": b_count,
            "result_source_url": back[0]["result_source_url"],
        }
        output[rid] = candidate
    return output


def summary_of(rows: list[dict[str, str]]) -> dict[str, dict]:
    result = {}
    for period, subset in (
        ("train", [r for r in rows if r["period"] == "train_2023_2025"]),
        ("holdout", [r for r in rows if r["period"] == "holdout_2026_h1"]),
    ):
        leaders = Counter(r["back_line_rank"] for r in subset)
        finish = Counter(r["finish_status"] for r in subset)
        normal = [r for r in subset if r["finish_status"] == "normal"]
        conditional = Counter(r["back_line_rank"] + ":" + r["back_line_top3_count"] for r in normal)
        by_shape: dict[str, Counter] = defaultdict(Counter)
        tickets: dict[str, Counter] = defaultdict(Counter)
        for r in normal:
            sh = r["line_size_shape"]
            by_shape[sh][r["back_line_rank"] + ":" + r["back_line_top3_count"]] += 1
            tickets[sh][r["back_line_rank"] + ":" + r["top3_slots"]] += 1
        unique = [r for r in subset if r["unique_highest_pre_b_line"]]
        result[period] = {
            "races": len(subset),
            "leaders": {s: leaders[s] for s in "HML"},
            "finish_status": {s: finish[s] for s in ("normal", "tie_or_nonstandard")},
            "conditional": dict(conditional),
            "conditional_by_shape": {k: dict(v) for k, v in by_shape.items()},
            "exact_trio_by_shape": {k: dict(v) for k, v in tickets.items()},
            "front_b_rank": {
                "sole_max": len(unique),
                "correct": sum(r["unique_highest_pre_b_line"] == r["back_line_rank"] for r in unique),
            },
        }
    return result


def main() -> int:
    expected = {}
    source_counts = {}
    for year in YEARS:
        batch = reconstruct(year)
        source_counts[year] = len(batch)
        if expected.keys() & batch.keys():
            raise AssertionError(f"Duplicate race IDs across years: {year}")
        expected.update(batch)

    observed = read_csv(OBS_PATH)
    ids = [r["race_id"] for r in observed]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate observation race IDs")
    if len(ids) != len(expected):
        raise AssertionError(f"Observation count differs: {len(ids)} != {len(expected)}")
    for o in observed:
        e = expected.get(o["race_id"])
        if e is None:
            raise AssertionError(f"Unexpected race: {o['race_id']}")
        diff = [f for f in FIELDS if o.get(f) != e.get(f)]
        if diff:
            raise AssertionError(f"Observation mismatch {o['race_id']}: {diff}: "
                                 f"{[(f, o.get(f), e.get(f)) for f in diff][:3]}")

    frozen = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    assert [source_counts[y] for y in YEARS] == [1029, 970, 1016, 495]
    for year in YEARS[:3]:
        assert source_counts[year] == frozen["by_year"][year]["races"]
    manifest = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary_of(observed) == manifest["totals"], "Summary mismatch"
    assert len(observed) == 3510
    print(json.dumps({
        "status": "VERIFIED_SOURCE_REPRODUCIBILITY",
        "observed_races": len(observed),
        "by_year": source_counts,
        "train": manifest["totals"]["train"]["races"],
        "holdout": manifest["totals"]["holdout"]["races"],
        "normal_finish_train": manifest["totals"]["train"]["finish_status"]["normal"],
        "normal_finish_holdout": manifest["totals"]["holdout"]["finish_status"]["normal"],
        "back_marker": "B at final back (not continuous control)",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

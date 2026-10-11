"""Reproduce frozen v2 positional and exact trio-ticket probabilities.

Run from the repository root:
    python -m src.keirin_market_analysis.audit_trio_line_parts_v2

This verifier reads historical CSV and the immutable v1/v2 baselines; it writes nothing.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from .audit_trio_line_composition_v1 import records

ROOT = Path("data")
YEARS = (2023, 2024, 2025)
V1_PATH = ROOT / "probabilities" / "trio_line_composition_s_yosen_2023_2025_v1.json"
V2_PATH = ROOT / "probabilities" / "trio_line_parts_s_yosen_2023_2025_v2.json"
POSITION_KEYS = ("1-1-1", "1-1-2", "1-1-3", "1-2-2", "1-2-3", "2-2-2", "2-2-3")


def fresh():
    return {
        "races": 0,
        "pos": Counter(),
        "slot": Counter(),
        "mix": Counter(),
        "crosstab": defaultdict(Counter),
    }


def add(dst, src):
    dst["races"] += src["races"]
    for name in ("pos", "slot", "mix"):
        dst[name].update(src[name])
    for mix, vals in src["crosstab"].items():
        dst["crosstab"][mix].update(vals)


def compute(year):
    folder = ROOT / str(year) / "s_class_yosen"
    race_rows = records(folder / "races.csv")
    entries = records(folder / "entries.csv")
    payouts = records(folder / "payouts.csv")
    by_race = defaultdict(list)
    for e in entries:
        by_race[e["race_id"]].append(e)
    paid_trio = {
        p["race_id"]: p
        for p in payouts
        if p["ticket_type"] == "3連複" and p["status"] == "paid"
    }
    total, by_shape, exclusions = fresh(), defaultdict(fresh), Counter()
    for r in race_rows:
        if int(r["entry_count"]) != 7:
            exclusions["non_7_rider"] += 1
            continue
        rid = r["race_id"]
        es = by_race.get(rid, [])
        if len(es) != 7 or rid not in paid_trio:
            raise AssertionError(f"Missing source records: {rid}")
        line_map = defaultdict(list)
        for e in es:
            if not e.get("line_id", "").isdigit() or not e.get("line_position", "").isdigit():
                raise AssertionError(f"Missing line data: {rid}")
            line_map[e["line_id"]].append(e)
        lines = [sorted(g, key=lambda e: int(e["line_position"])) for g in line_map.values()]
        if len(lines) != 3 or sorted(len(g) for g in lines) != [2, 2, 3]:
            exclusions["non_3_2_2_line_shape"] += 1
            continue
        ranked = sorted(
            ((Decimal(line[0]["score"]) + Decimal(line[1]["score"]), line) for line in lines),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked[0][0] == ranked[1][0] or ranked[1][0] == ranked[2][0]:
            exclusions["equal_pair_score_tie"] += 1
            continue
        winners = [int(v) for v in re.findall(r"\d+", paid_trio[rid]["combination"])]
        if len(winners) != 3 or len(set(winners)) != 3:
            raise AssertionError(f"Bad paid trio: {rid}")
        slot_by_car = {}
        for i, (_, line) in enumerate(ranked):
            for j, e in enumerate(line, start=1):
                slot_by_car[int(e["car_no"])] = (i, j, "HML"[i] + str(j))
        if not all(w in slot_by_car for w in winners):
            raise AssertionError(f"Paid rider missing: {rid}")
        win_slots = [slot_by_car[w] for w in winners]
        pos = "-".join(str(v) for v in sorted(x[1] for x in win_slots))
        slot = "+".join(
            sorted((x[2] for x in win_slots),
                   key=lambda s: ("HML".index(s[0]), int(s[1:])))
        )
        line_counts = Counter(x[0] for x in win_slots)
        mix = "-".join(str(line_counts[i]) for i in range(3))
        shape = "-".join(str(len(line)) for _, line in ranked)
        for dest in (total, by_shape[shape]):
            dest["races"] += 1
            dest["pos"][pos] += 1
            dest["slot"][slot] += 1
            dest["mix"][mix] += 1
            dest["crosstab"][mix][pos] += 1
    return total, by_shape, exclusions, len(race_rows)


def check_position(g, frozen):
    assert g["races"] == frozen["races"]
    for row in frozen["positional_patterns"]:
        assert g["pos"][row["pattern"]] == row["count"]
        assert abs(row["empirical_probability"] - row["count"] / g["races"]) < 1e-8
    assert sum(g["pos"].values()) == g["races"]


def check_slots(g, frozen):
    candidates = frozen["each_of_35_exact_trio_slot_tickets"]
    assert len(candidates) == 35
    assert sum(g["slot"].values()) == g["races"]
    for c in candidates:
        assert c["count"] == g["slot"][c["ticket_slots"]]
        assert abs(c["empirical_probability"] - c["count"] / g["races"]) < 1e-8
    assert sum(c["count"] for c in candidates) == g["races"]


def check_crosstab(g, table):
    assert set(g["mix"]) == set(table)
    for k, block in table.items():
        assert g["mix"][k] == block["n"]
        assert dict(g["crosstab"][k]) == {p: n for p, n in block["joint"].items() if n}


def main() -> int:
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
    all_data, aggregate_shapes = fresh(), defaultdict(fresh)
    for year in YEARS:
        total, shapes, excluded, source_races = compute(year)
        v2_year = v2["by_year"][str(year)]
        assert source_races == v2["audit"]["by_year"][str(year)]["source_races"]
        assert dict(excluded) == {
            k: v for k, v in v2["audit"]["by_year"][str(year)]["excluded"].items() if v
        }
        check_position(total, v2_year)
        assert total["races"] == v1["by_year"][str(year)]["races"]
        assert total["slot"] == Counter(v2_year["ranked_slot_raw_counts"])
        for shape, g in shapes.items():
            shape_year = v2["by_line_size_shape"][shape]["by_year"][str(year)]
            check_position(g, shape_year)
            check_slots(g, shape_year)
            add(aggregate_shapes[shape], g)
        assert set(shapes) == set(v2_year["line_size_shape_races"])
        add(all_data, total)
    check_position(all_data, v2["aggregate"])
    assert all_data["races"] == v1["aggregate"]["races"] == 3015
    assert all_data["mix"] == Counter({
        x["pattern"]: x["count"] for x in v1["aggregate"]["patterns"]
    })
    assert all_data["slot"] == Counter(v2["aggregate"]["ranked_slot_raw_counts"])
    assert all_data["slot"]["H1+M1+L1"] == v2["aggregate"]["three_line_head_ticket"]["count"]
    check_crosstab(all_data, v2["aggregate"]["line_structure_position_crosstab"])
    assert set(aggregate_shapes) == set(v2["by_line_size_shape"])
    for shape, g in aggregate_shapes.items():
        v2_shape = v2["by_line_size_shape"][shape]
        check_position(g, v2_shape)
        check_slots(g, v2_shape)
        check_crosstab(g, v2_shape["line_structure_position_crosstab"])
    print(json.dumps({
        "status": "REPRODUCED",
        "source": str(V2_PATH),
        "races": all_data["races"],
        "pos": dict(all_data["pos"]),
        "shape_counts": {k: v["races"] for k, v in aggregate_shapes.items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

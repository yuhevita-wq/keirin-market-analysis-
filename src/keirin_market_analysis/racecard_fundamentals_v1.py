from __future__ import annotations

"""Deterministic race-card fundamentals for seven-rider keirin races.

This module contains no betting rule and no outcome-dependent tuning.
It converts historical race-card snapshot fields into one reproducible
within-race fundamental score F for each rider.
"""

from math import isfinite
from typing import Iterable


REQUIRED_NUMERIC_FIELDS = (
    "score",
    "top3_rate",
    "b_count",
    "nige_count",
    "makuri_count",
    "sashi_count",
    "mark_count",
)


def _pf(x):
    try:
        v = float(x)
        return v if isfinite(v) else None
    except Exception:
        return None


def _pi(x):
    try:
        return int(float(x))
    except Exception:
        return None


def rank01(values: dict[int, float]) -> dict[int, float]:
    """Within-race ordinal normalization. Best=1, worst=0; ties remain tied."""
    n = len(values)
    if n <= 1:
        return {k: 1.0 for k in values}
    raw = list(values.values())
    return {
        k: sum(1 for z in raw if z < v) / (n - 1)
        for k, v in values.items()
    }


def fundamental_scores(entry_rows: Iterable[dict]) -> dict[str, object]:
    rows = list(entry_rows)
    by_car: dict[int, dict[str, float | int]] = {}

    for row in rows:
        car = _pi(row.get("car_no"))
        if car is None or car in by_car:
            return {"ok": False, "reason": "FUNDAMENTAL_INVALID_CAR_ROWS"}

        vals = {field: _pf(row.get(field)) for field in REQUIRED_NUMERIC_FIELDS}
        pos = _pi(row.get("line_position"))
        size = _pi(row.get("line_size"))
        if (
            any(v is None for v in vals.values())
            or pos is None
            or size is None
            or pos < 1
            or size < 1
        ):
            return {"ok": False, "reason": "FUNDAMENTAL_INPUT_INCOMPLETE"}

        by_car[car] = {**vals, "line_position": pos, "line_size": size}

    if len(by_car) != 7:
        return {"ok": False, "reason": "FUNDAMENTAL_NOT_SEVEN_COMPLETE_ROWS"}

    def ranks(field: str) -> dict[int, float]:
        return rank01({car: float(v[field]) for car, v in by_car.items()})

    score_r = ranks("score")
    top3_r = ranks("top3_rate")
    b_r = ranks("b_count")
    nige_r = ranks("nige_count")
    makuri_r = ranks("makuri_count")
    sashi_r = ranks("sashi_count")
    mark_r = ranks("mark_count")

    f: dict[int, float] = {}
    components: dict[int, dict[str, float | int]] = {}

    for car, row in by_car.items():
        attack = (b_r[car] + nige_r[car] + makuri_r[car]) / 3.0
        follow = (sashi_r[car] + mark_r[car]) / 2.0

        if int(row["line_size"]) == 1:
            role_fit = (attack + follow) / 2.0
        elif int(row["line_position"]) == 1:
            role_fit = attack
        else:
            role_fit = follow

        fi = (score_r[car] + top3_r[car] + role_fit) / 3.0
        f[car] = fi
        components[car] = {
            "score_rank01": score_r[car],
            "top3_rank01": top3_r[car],
            "attack_rank01": attack,
            "follow_rank01": follow,
            "role_fit_rank01": role_fit,
            "F": fi,
            "line_position": int(row["line_position"]),
            "line_size": int(row["line_size"]),
        }

    order = tuple(sorted(f, key=lambda car: (-f[car], car)))
    return {
        "ok": True,
        "F": f,
        "components": components,
        "order": order,
    }

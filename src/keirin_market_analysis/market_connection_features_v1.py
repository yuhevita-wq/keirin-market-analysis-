from __future__ import annotations

"""Pure market-connection feature extraction.

Retained from the useful part of the rejected v12.0-N01 idea.
This module DOES NOT decide whether to bet and DOES NOT generate tickets.

It describes, for every possible head h and companion pair {a,b}:
- normalized trifecta head support H(h)
- head-conditioned pair attachment from the 210-way trifecta market
- matching three-rider set attachment from the 35-way trio market
- log cross-market connection residual
- exact tail-order asymmetry
- deterministic historical race-card fundamental strength

No result, payout, race-type branch, or line-based betting rule is present.
Line position/size can enter only through racecard_fundamentals_v1 role fit.
"""

from itertools import combinations, permutations
from math import isfinite, log
from typing import Iterable, Mapping

from racecard_fundamentals_v1 import fundamental_scores


def _positive_float(x) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if isfinite(v) and v > 0 else None


def _normalize_inverse_odds(odds: Mapping[tuple[int, ...], float]) -> dict[tuple[int, ...], float] | None:
    inv: dict[tuple[int, ...], float] = {}
    for key, value in odds.items():
        v = _positive_float(value)
        if v is None:
            return None
        inv[tuple(key)] = 1.0 / v
    z = sum(inv.values())
    if z <= 0:
        return None
    return {k: v / z for k, v in inv.items()}


def _top_block(values: Mapping[int, float]) -> tuple[int, ...]:
    """Descriptive natural tier only. It is not itself a betting gate."""
    order = sorted(values, key=lambda k: (-float(values[k]), int(k)))
    if len(order) <= 1:
        return tuple(order)
    ranked = [float(values[k]) for k in order]
    if max(ranked) - min(ranked) <= 1e-15:
        return tuple(order)
    ratios = [ranked[i] / ranked[i + 1] for i in range(len(ranked) - 1)]
    largest = max(ratios)
    eps = 1e-12
    cut = max(i + 1 for i, r in enumerate(ratios) if abs(r - largest) <= eps)
    return tuple(order[:cut])


def _fundamental_top_block(values: Mapping[int, float]) -> tuple[int, ...]:
    order = sorted(values, key=lambda k: (-float(values[k]), int(k)))
    if len(order) <= 1:
        return tuple(order)
    ranked = [float(values[k]) for k in order]
    gaps = [ranked[i] - ranked[i + 1] for i in range(len(ranked) - 1)]
    largest = max(gaps)
    if largest <= 1e-15:
        return tuple(order)
    eps = 1e-12
    cut = max(i + 1 for i, g in enumerate(gaps) if abs(g - largest) <= eps)
    return tuple(order[:cut])


def extract_market_connection_features(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    entry_rows: Iterable[dict],
) -> dict[str, object]:
    if len(trio_odds) != 35:
        return {"ok": False, "reason": "TRIO_NOT_COMPLETE_35"}
    if len(trifecta_odds) != 210:
        return {"ok": False, "reason": "TRIFECTA_NOT_COMPLETE_210"}

    cars = tuple(sorted({c for t in trifecta_odds for c in t}))
    if len(cars) != 7:
        return {"ok": False, "reason": "NOT_SEVEN_CARS"}

    expected_trio = {tuple(c) for c in combinations(cars, 3)}
    actual_trio = {tuple(sorted(t)) for t in trio_odds}
    if actual_trio != expected_trio:
        return {"ok": False, "reason": "TRIO_COMBINATION_SET_INVALID"}

    if {tuple(t) for t in trifecta_odds} != set(permutations(cars, 3)):
        return {"ok": False, "reason": "TRIFECTA_PERMUTATION_SET_INVALID"}

    p3 = _normalize_inverse_odds({tuple(sorted(k)): v for k, v in trio_odds.items()})
    q = _normalize_inverse_odds(trifecta_odds)
    if p3 is None or q is None:
        return {"ok": False, "reason": "ODDS_NORMALIZATION_FAILED"}

    fd = fundamental_scores(entry_rows)
    if not fd.get("ok"):
        return {"ok": False, "reason": str(fd.get("reason", "FUNDAMENTAL_INPUT_INVALID"))}
    f = {int(k): float(v) for k, v in fd["F"].items()}
    if set(f) != set(cars):
        return {"ok": False, "reason": "FUNDAMENTAL_CAR_SET_MISMATCH"}

    h = {car: sum(prob for t, prob in q.items() if t[0] == car) for car in cars}
    head_block = _top_block(h)
    f_head_block = _fundamental_top_block(f)

    connections: list[dict[str, object]] = []
    for head in cars:
        others = [c for c in cars if c != head]
        h_mass = h[head]
        if h_mass <= 0:
            continue
        pairs = list(combinations(others, 2))
        trio_raw = {pair: p3[tuple(sorted((head, pair[0], pair[1])))] for pair in pairs}
        trio_z = sum(trio_raw.values())
        for a, b in pairs:
            tf_ab = q[(head, a, b)]
            tf_ba = q[(head, b, a)]
            tf_pair = (tf_ab + tf_ba) / h_mass
            trio_pair = trio_raw[(a, b)] / trio_z
            residual = log(tf_pair / trio_pair)
            order_asymmetry = log(tf_ab / tf_ba)
            connections.append(
                {
                    "head": head,
                    "a": a,
                    "b": b,
                    "tf_head_pair": tf_pair,
                    "trio_set_pair": trio_pair,
                    "market_connection_log_residual": residual,
                    "tail_order_log_asymmetry": order_asymmetry,
                    "head_support": h[head],
                    "head_F": f[head],
                    "pair_F_sum": f[a] + f[b],
                }
            )

    return {
        "ok": True,
        "cars": list(cars),
        "head_support": h,
        "market_head_block": list(head_block),
        "fundamental_F": f,
        "fundamental_head_block": list(f_head_block),
        "connections": connections,
        "result_used": False,
        "payout_used": False,
        "betting_rule_present": False,
    }

from __future__ import annotations

"""v11.0-C01: v8.25 market psychology + race-card context.

This is the first safe reintegration of race-card data after rejecting v9/v10.
The v8.25-F26 market engine remains the complete betting decision and formation
engine. Race-card data do NOT veto, add, or prune tickets here.

Instead, every race receives deterministic context describing whether the
market's leading head and leading line are supported or challenged by the
independent race-card fundamentals. This lets us study the *meaning* of market
psychology without selecting winners after the fact.
"""

from typing import Iterable, Mapping

from racecard_fundamentals_v1 import fundamental_scores
from v7_0_f01_market_hierarchy import (
    implied_probabilities,
    parse_lines,
    positional_support,
    trio_implied_probabilities,
)
from v8_25_f26_final_set_lock_only import build_v8_25_f26

SCHEME_VERSION = "v11.0-C01"
BASE_SCHEME_VERSION = "v8.25-F26"
STATUS = "MARKET_PSYCHOLOGY_WITH_RACECARD_CONTEXT_NO_BET_MODIFICATION"


def _natural_fundamental_head_block(f: Mapping[int, float]) -> tuple[int, ...]:
    order = sorted(f, key=lambda car: (-float(f[car]), car))
    if len(order) <= 1:
        return tuple(order)
    gaps = [float(f[order[i]]) - float(f[order[i + 1]]) for i in range(len(order) - 1)]
    largest = max(gaps)
    if largest <= 0:
        return tuple(order)
    eps = 1e-12
    # Later equal-largest gap preserves the broader upper block.
    cut = max(i + 1 for i, gap in enumerate(gaps) if abs(gap - largest) <= eps)
    return tuple(order[:cut])


def racecard_market_context(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
    entry_rows: Iterable[dict],
) -> dict[str, object]:
    rows = list(entry_rows)
    fd = fundamental_scores(rows)
    if not fd.get("ok"):
        return {"ok": False, "reason": fd.get("reason", "FUNDAMENTAL_INPUT_INVALID")}

    lines = parse_lines(predicted_line_formation)
    if lines is None or len(lines) < 2:
        return {"ok": False, "reason": "INVALID_LINE_STRUCTURE"}

    cars = sorted({car for combo in trio_odds for car in combo})
    if len(cars) != 7 or {car for line in lines for car in line} != set(cars):
        return {"ok": False, "reason": "LINE_CAR_SET_MISMATCH"}

    q = implied_probabilities(trifecta_odds)
    h1, _, _ = positional_support(q, cars)
    h_order = tuple(sorted(cars, key=lambda car: (-h1[car], car)))
    market_h1 = h_order[0]
    market_h2 = h_order[1]
    h_state = "H_CONCENTRATED" if h1[market_h1] >= 2.0 * h1[market_h2] else "H_BALANCED"

    f = fd["F"]
    f_order = tuple(sorted(f, key=lambda car: (-float(f[car]), car)))
    f_head_block = _natural_fundamental_head_block(f)
    head_context = "HEAD_SUPPORTED" if market_h1 in set(f_head_block) else "HEAD_CHALLENGED"
    market_h1_f_rank = f_order.index(market_h1) + 1

    p3 = trio_implied_probabilities(trio_odds)
    rider_market = {
        car: sum(prob for combo, prob in p3.items() if car in combo) / 3.0
        for car in cars
    }
    market_line_support = [sum(rider_market[car] for car in line) for line in lines]
    fundamental_line_support = [sum(float(f[car]) for car in line) for line in lines]
    market_line_order = tuple(sorted(range(len(lines)), key=lambda idx: (-market_line_support[idx], idx)))
    fundamental_line_order = tuple(sorted(range(len(lines)), key=lambda idx: (-fundamental_line_support[idx], idx)))
    market_top_line = market_line_order[0]
    fundamental_top_line = fundamental_line_order[0]
    line_context = "LINE_SUPPORTED" if market_top_line == fundamental_top_line else "LINE_CHALLENGED"
    market_top_line_f_rank = fundamental_line_order.index(market_top_line) + 1

    context = f"{h_state}|{head_context}|{line_context}"
    return {
        "ok": True,
        "context": context,
        "h_state": h_state,
        "head_context": head_context,
        "line_context": line_context,
        "market_h1": market_h1,
        "market_h2": market_h2,
        "market_h1_support": h1[market_h1],
        "market_h2_support": h1[market_h2],
        "fundamental_order": list(f_order),
        "fundamental_head_block": list(f_head_block),
        "market_h1_fundamental_rank": market_h1_f_rank,
        "market_line_order": list(market_line_order),
        "fundamental_line_order": list(fundamental_line_order),
        "market_top_line_fundamental_rank": market_top_line_f_rank,
        "market_line_support": market_line_support,
        "fundamental_line_support": fundamental_line_support,
        "fundamental_F": {str(k): v for k, v in f.items()},
        "fundamental_components": {str(k): v for k, v in fd["components"].items()},
    }


def build_v11_0_c01(
    trio_odds,
    trifecta_odds,
    predicted_line_formation: str,
    race_type: str,
    entry_rows: Iterable[dict],
) -> dict[str, object]:
    base = dict(build_v8_25_f26(trio_odds, trifecta_odds, predicted_line_formation, race_type))
    original_scheme = base.get("scheme_version", BASE_SCHEME_VERSION)
    ctx = racecard_market_context(trio_odds, trifecta_odds, predicted_line_formation, entry_rows)

    base["base_scheme_version"] = original_scheme
    base["scheme_version"] = SCHEME_VERSION
    base["development_status"] = STATUS
    base["racecard_context_changes_bet"] = False

    if not ctx.get("ok"):
        base["racecard_context_ok"] = False
        base["racecard_context_reason"] = ctx.get("reason")
        return base

    base["racecard_context_ok"] = True
    for key, value in ctx.items():
        if key != "ok":
            base[f"racecard_{key}"] = value
    return base

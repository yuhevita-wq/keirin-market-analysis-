from __future__ import annotations

"""v9.2-F28: universal fundamental-vs-market line-set mispricing gate.

The v8.25-F26 market engine remains the complete candidate generator.
Race-card fundamentals are used only as an independent race-entry signal:
buy the unchanged market candidate only when the market's top-two line set
differs from the race-card fundamental top-two line set.

No fitted numeric threshold, race-type-specific fundamental weight, payout,
result, prediction mark, evaluation mark, or individual ticket pruning is used.
"""

from typing import Iterable, Mapping

from v7_0_f01_market_hierarchy import parse_lines, trio_implied_probabilities
from v8_25_f26_final_set_lock_only import build_v8_25_f26
from v9_0_f27_fundamental_first_branch import fundamental_scores

SCHEME_VERSION = "v9.2-F28"
BASE_SCHEME_VERSION = "v8.25-F26"
STATUS = "DEVELOPMENT_Q1Q2Q3_UNIVERSAL_LINE_SET_MISPRICING_GATE"


def line_set_state(
    trio_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
    entry_rows: Iterable[dict],
) -> dict[str, object]:
    lines = parse_lines(predicted_line_formation)
    if lines is None or len(lines) < 2:
        return {"ok": False, "reason": "FUNDAMENTAL_INVALID_LINE_STRUCTURE"}

    cars = sorted({car for combo in trio_odds for car in combo})
    flat = {car for line in lines for car in line}
    if len(cars) != 7 or flat != set(cars):
        return {"ok": False, "reason": "FUNDAMENTAL_LINE_CAR_MISMATCH"}

    fd = fundamental_scores(entry_rows)
    if not fd.get("ok"):
        return {"ok": False, "reason": fd.get("reason", "FUNDAMENTAL_INPUT_INVALID")}

    p3 = trio_implied_probabilities(trio_odds)
    rider_market = {
        car: sum(prob for combo, prob in p3.items() if car in combo) / 3.0
        for car in cars
    }
    market_line_support = [sum(rider_market[car] for car in line) for line in lines]
    fundamental_line_support = [sum(fd["F"][car] for car in line) for line in lines]

    market_order = tuple(
        sorted(range(len(lines)), key=lambda idx: (-market_line_support[idx], idx))
    )
    fundamental_order = tuple(
        sorted(range(len(lines)), key=lambda idx: (-fundamental_line_support[idx], idx))
    )

    market_top2 = tuple(market_order[:2])
    fundamental_top2 = tuple(fundamental_order[:2])
    top2_set_agree = set(market_top2) == set(fundamental_top2)

    return {
        "ok": True,
        "lines": lines,
        "market_line_support": market_line_support,
        "fundamental_line_support": fundamental_line_support,
        "market_line_order": market_order,
        "fundamental_line_order": fundamental_order,
        "market_top2_line_indices": market_top2,
        "fundamental_top2_line_indices": fundamental_top2,
        "top2_line_set_agree": top2_set_agree,
        "fundamental_F": fd["F"],
        "fundamental_components": fd["components"],
    }


def build_v9_2_f28(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
    race_type: str,
    entry_rows: Iterable[dict],
) -> dict[str, object]:
    base = dict(
        build_v8_25_f26(
            trio_odds,
            trifecta_odds,
            predicted_line_formation,
            race_type,
        )
    )
    base["base_scheme_version"] = base.get("scheme_version", BASE_SCHEME_VERSION)
    base["scheme_version"] = SCHEME_VERSION
    base["development_status"] = STATUS
    base["fundamental_policy"] = (
        "Universal market-vs-race-card top-two line-set mispricing gate. "
        "The original v8.25 candidate and ticket formation are kept unchanged "
        "only when the market top-two line set differs from the independent "
        "fundamental top-two line set."
    )

    if not base.get("buy"):
        base["fundamental_gate_action"] = "BASE_NO_BET_PASSTHROUGH"
        return base

    state = line_set_state(trio_odds, predicted_line_formation, entry_rows)
    if not state.get("ok"):
        return {
            **base,
            "buy": False,
            "tickets": [],
            "ticket_count": 0,
            "reason": state.get("reason", "FUNDAMENTAL_LINE_STATE_INVALID"),
            "fundamental_gate_action": "NO_BET_INVALID_FUNDAMENTAL_STATE",
        }

    meta = {
        "market_line_support": state["market_line_support"],
        "fundamental_line_support": state["fundamental_line_support"],
        "market_line_order": list(state["market_line_order"]),
        "fundamental_line_order": list(state["fundamental_line_order"]),
        "market_top2_line_indices": list(state["market_top2_line_indices"]),
        "fundamental_top2_line_indices": list(state["fundamental_top2_line_indices"]),
        "top2_line_set_agree": state["top2_line_set_agree"],
        "fundamental_F": {str(k): v for k, v in state["fundamental_F"].items()},
        "fundamental_components": {
            str(k): v for k, v in state["fundamental_components"].items()
        },
    }

    if state["top2_line_set_agree"]:
        return {
            **base,
            **meta,
            "buy": False,
            "tickets": [],
            "ticket_count": 0,
            "reason": "FUNDAMENTAL_MARKET_TOP2_LINE_SET_AGREE_NO_MISPRICING",
            "fundamental_gate_action": "NO_BET_TOP2_LINE_SET_AGREEMENT",
        }

    return {
        **base,
        **meta,
        "buy": True,
        "fundamental_gate_action": "BUY_TOP2_LINE_SET_DISAGREEMENT",
        "branch_policy": (
            f"{base.get('branch_policy', '')} | v9.2 universal fundamental "
            "top-two line-set mispricing gate"
        ),
    }

from __future__ import annotations

"""v10.0-F29: universal market x race-card value formation.

One rule for every eligible F1 S-class seven-rider race.

The scheme deliberately removes race-type-specific betting branches. It uses:
1) normalized trio market support for three-rider membership,
2) normalized trifecta market support for exact order,
3) an independent deterministic race-card fundamental distribution,
then combines the three views with an equal-weight geometric opinion pool.

The final formation is discovered from positional support cliffs. No individual
exact ticket is pruned. The complete rectangular formation is bought only when
its estimated equal-stake expected ROI exceeds the natural break-even level 1.

No result, payout, race-type-specific threshold, fitted coefficient, prediction
mark, evaluation mark, or historical hit/miss selector is used here.
"""

from math import isfinite
from typing import Iterable, Mapping

from racecard_fundamentals_v1 import fundamental_scores
from v7_0_f01_market_hierarchy import (
    generate_nested_tickets,
    implied_probabilities,
    market_top_cluster,
    parse_lines,
    positional_support,
    trio_implied_probabilities,
)

SCHEME_VERSION = "v10.0-F29"
STATUS = "DESIGN_FROZEN_PRE_SIMULATION"
STAKE_YEN_PER_TICKET = 100
BREAK_EVEN_ROI = 1.0


def _positive_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if isfinite(v) and v > 0 else None


def _int_value(x) -> int | None:
    try:
        return int(float(x))
    except Exception:
        return None


def _validate_complete_markets(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
) -> tuple[bool, str, tuple[int, ...]]:
    if len(trio_odds) != 35:
        return False, "TRIO_NOT_COMPLETE_35", ()
    if len(trifecta_odds) != 210:
        return False, "TRIFECTA_NOT_COMPLETE_210", ()

    if any(_positive_float(v) is None for v in trio_odds.values()):
        return False, "TRIO_HAS_INVALID_ODDS", ()
    if any(_positive_float(v) is None for v in trifecta_odds.values()):
        return False, "TRIFECTA_HAS_INVALID_ODDS", ()

    cars = tuple(sorted({int(car) for combo in trio_odds for car in combo}))
    tf_cars = tuple(sorted({int(car) for ticket in trifecta_odds for car in ticket}))
    if len(cars) != 7 or tf_cars != cars:
        return False, "MARKET_CAR_SET_NOT_COMPLETE_SEVEN", ()

    expected_trio = {
        tuple(sorted((a, b, c)))
        for i, a in enumerate(cars)
        for j, b in enumerate(cars)
        for c in cars
        if i < j and b < c
    }
    actual_trio = {tuple(sorted(map(int, combo))) for combo in trio_odds}
    if actual_trio != expected_trio:
        return False, "TRIO_COMBINATIONS_NOT_EXACT_COMPLETE_SET", ()

    expected_tf = {
        (a, b, c)
        for a in cars
        for b in cars
        for c in cars
        if len({a, b, c}) == 3
    }
    actual_tf = {tuple(map(int, ticket)) for ticket in trifecta_odds}
    if actual_tf != expected_tf:
        return False, "TRIFECTA_PERMUTATIONS_NOT_EXACT_COMPLETE_SET", ()

    return True, "OK", cars


def _validate_line_snapshot(
    predicted_line_formation: str,
    entry_rows: Iterable[dict],
    cars: tuple[int, ...],
) -> tuple[bool, str, tuple[tuple[int, ...], ...] | None]:
    lines = parse_lines(predicted_line_formation)
    if lines is None:
        return False, "INVALID_PREDICTED_LINE_FORMATION", None
    if {car for line in lines for car in line} != set(cars):
        return False, "LINE_CAR_SET_MISMATCH", None

    expected = {
        car: (position + 1, len(line))
        for line in lines
        for position, car in enumerate(line)
    }
    seen: dict[int, tuple[int, int]] = {}
    for row in entry_rows:
        car = _int_value(row.get("car_no"))
        pos = _int_value(row.get("line_position"))
        size = _int_value(row.get("line_size"))
        if car is None or pos is None or size is None or car in seen:
            return False, "ENTRY_LINE_METADATA_INVALID", None
        seen[car] = (pos, size)

    if set(seen) != set(cars):
        return False, "ENTRY_LINE_CAR_SET_MISMATCH", None
    if any(seen[car] != expected[car] for car in cars):
        return False, "ENTRY_LINE_METADATA_DISAGREES_WITH_PREDICTED_LINE", None

    return True, "OK", lines


def _fundamental_order_distribution(
    cars: tuple[int, ...],
    f: Mapping[int, float],
) -> dict[tuple[int, int, int], float]:
    """Convert ordinal F into a top-three ordered Plackett-Luce distribution.

    F is in [0,1]. In an N-rider race, weight = 1 + (N-1)*F, so the natural
    seven-rider scale is 1..7 without a fitted coefficient.
    """
    n = len(cars)
    weights = {car: 1.0 + (n - 1) * float(f[car]) for car in cars}
    total = sum(weights.values())

    out: dict[tuple[int, int, int], float] = {}
    for first in cars:
        p1 = weights[first] / total
        z2 = total - weights[first]
        for second in cars:
            if second == first:
                continue
            p2 = weights[second] / z2
            z3 = z2 - weights[second]
            for third in cars:
                if third == first or third == second:
                    continue
                p3 = weights[third] / z3
                out[(first, second, third)] = p1 * p2 * p3

    z = sum(out.values())
    return {ticket: prob / z for ticket, prob in out.items()}


def _equal_geometric_pool(
    trio_p: Mapping[tuple[int, int, int], float],
    trifecta_q: Mapping[tuple[int, int, int], float],
    fundamental_r: Mapping[tuple[int, int, int], float],
) -> dict[tuple[int, int, int], float]:
    """Equal log-weight pool of membership market, order market, fundamentals."""
    raw: dict[tuple[int, int, int], float] = {}
    for ticket, q in trifecta_q.items():
        combo = tuple(sorted(ticket))
        p = float(trio_p[combo])
        r = float(fundamental_r[ticket])
        raw[ticket] = (float(q) * p * r) ** (1.0 / 3.0)
    z = sum(raw.values())
    return {ticket: value / z for ticket, value in raw.items()}


def _nested_union(*groups: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(car) for group in groups for car in group}))


def _formation_display(first, second, third) -> str:
    def side(xs) -> str:
        return "".join(str(x) for x in sorted(xs))
    return f"{side(first)}-{side(second)}-{side(third)}"


def build_v10_0_f29(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
    race_type: str,
    entry_rows: Iterable[dict],
) -> dict[str, object]:
    """Build one universal equal-stake trifecta formation decision.

    `race_type` is recorded only as metadata. It does not alter the rule.
    """
    entry_rows = list(entry_rows)

    ok, reason, cars = _validate_complete_markets(trio_odds, trifecta_odds)
    if not ok:
        return {
            "scheme_version": SCHEME_VERSION,
            "status": STATUS,
            "buy": False,
            "reason": reason,
            "race_type": race_type,
        }

    line_ok, line_reason, lines = _validate_line_snapshot(
        predicted_line_formation, entry_rows, cars
    )
    if not line_ok:
        return {
            "scheme_version": SCHEME_VERSION,
            "status": STATUS,
            "buy": False,
            "reason": line_reason,
            "race_type": race_type,
        }

    fd = fundamental_scores(entry_rows)
    if not fd.get("ok"):
        return {
            "scheme_version": SCHEME_VERSION,
            "status": STATUS,
            "buy": False,
            "reason": fd.get("reason", "FUNDAMENTAL_INPUT_INVALID"),
            "race_type": race_type,
        }

    trio_p = trio_implied_probabilities(trio_odds)
    trifecta_q = implied_probabilities(trifecta_odds)
    fundamental_r = _fundamental_order_distribution(cars, fd["F"])
    blended = _equal_geometric_pool(trio_p, trifecta_q, fundamental_r)

    h1, h2, h3 = positional_support(blended, cars)
    b1 = market_top_cluster(h1)
    b2 = market_top_cluster(h2)
    b3 = market_top_cluster(h3)

    # Preserve the strongest successful structural lesson from the market work:
    # the formation is a whole rectangle discovered from support hierarchy.
    first = tuple(sorted(b1.selected))
    second = _nested_union(first, b2.selected)
    third = _nested_union(second, b3.selected)
    tickets = generate_nested_tickets(first, second, third)

    if not tickets:
        return {
            "scheme_version": SCHEME_VERSION,
            "status": STATUS,
            "buy": False,
            "reason": "NO_VALID_RECTANGULAR_TICKETS",
            "race_type": race_type,
        }

    odds = {ticket: float(trifecta_odds[ticket]) for ticket in tickets}
    expected_payout_multiple = sum(blended[ticket] * odds[ticket] for ticket in tickets)
    estimated_roi = expected_payout_multiple / len(tickets)
    estimated_edge = estimated_roi - BREAK_EVEN_ROI
    stake_yen = len(tickets) * STAKE_YEN_PER_TICKET
    formation_mass = sum(blended[ticket] for ticket in tickets)

    market_inverse_sum = sum(1.0 / float(v) for v in trifecta_odds.values())
    buy = estimated_roi > BREAK_EVEN_ROI

    common = {
        "scheme_version": SCHEME_VERSION,
        "status": STATUS,
        "race_type": race_type,
        "race_type_affects_decision": False,
        "predicted_lines": [list(line) for line in (lines or ())],
        "formation": _formation_display(first, second, third),
        "first": first,
        "second": second,
        "third": third,
        "tickets": tickets,
        "ticket_count": len(tickets),
        "stake_yen": stake_yen,
        "formation_mass": formation_mass,
        "estimated_roi": estimated_roi,
        "estimated_edge": estimated_edge,
        "break_even_roi": BREAK_EVEN_ROI,
        "trifecta_inverse_odds_sum": market_inverse_sum,
        "fundamental_F": {str(k): v for k, v in fd["F"].items()},
        "fundamental_components": {
            str(k): v for k, v in fd["components"].items()
        },
        "blended_position_support": {
            "first": {str(k): v for k, v in h1.items()},
            "second": {str(k): v for k, v in h2.items()},
            "third": {str(k): v for k, v in h3.items()},
        },
        "boundaries": {
            "first": {
                "selected": list(b1.selected),
                "cut_after_rank": b1.cut_after_rank,
                "boundary_ratio": b1.boundary_ratio,
            },
            "second": {
                "selected": list(b2.selected),
                "cut_after_rank": b2.cut_after_rank,
                "boundary_ratio": b2.boundary_ratio,
            },
            "third": {
                "selected": list(b3.selected),
                "cut_after_rank": b3.cut_after_rank,
                "boundary_ratio": b3.boundary_ratio,
            },
        },
        "policy": (
            "One universal rule: equal geometric pool of trio membership market, "
            "trifecta exact-order market, and deterministic race-card fundamentals; "
            "derive one complete rectangular formation from positional support cliffs; "
            "buy the whole formation only when its equal-stake estimated ROI exceeds 1."
        ),
        "guardrails": {
            "result_used": False,
            "payout_used": False,
            "race_type_specific_branch": False,
            "fitted_numeric_cutoff": False,
            "individual_ticket_pruning": False,
            "prediction_mark_used": False,
            "evaluation_mark_used": False,
        },
    }

    if not buy:
        return {
            **common,
            "buy": False,
            "reason": "FORMATION_ESTIMATED_ROI_NOT_ABOVE_BREAK_EVEN",
        }

    return {
        **common,
        "buy": True,
        "reason": "UNIVERSAL_FORMATION_ESTIMATED_ROI_ABOVE_BREAK_EVEN",
    }

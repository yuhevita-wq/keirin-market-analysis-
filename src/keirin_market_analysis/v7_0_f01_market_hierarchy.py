from __future__ import annotations

"""v7.0-F01: market-hierarchy trifecta formation engine.

Design principle
----------------
Do not decide the number of riders in 1st/2nd/3rd place in advance.
The market decides the boundary for each place from its own positional
support distribution.  The resulting candidate sets are then converted
into a clean nested formation F1-F2-F3.

No race result, payout, rider ability, score, style, or race-development
information is used by this module.
"""

import math
from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

SCHEME_VERSION = "v7.0-F01"
BASE_ENTRY_SCHEME_VERSION = "v6.1"
STATUS = "FORMATION_REWRITE_MARKET_HIERARCHY"
STAKE_YEN_PER_TICKET = 100
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False


@dataclass(frozen=True)
class ClusterBoundary:
    ranked_cars: tuple[int, ...]
    ranked_support: tuple[float, ...]
    cut_after_rank: int
    boundary_ratio: float
    selected: tuple[int, ...]


@dataclass(frozen=True)
class FormationResult:
    scheme_version: str
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    first_boundary: ClusterBoundary
    second_boundary: ClusterBoundary
    third_boundary: ClusterBoundary
    formation_mass: float

    @property
    def ticket_count(self) -> int:
        return len(self.tickets)

    @property
    def stake_yen(self) -> int:
        return self.ticket_count * STAKE_YEN_PER_TICKET

    @property
    def display(self) -> str:
        def side(xs: Sequence[int]) -> str:
            return "".join(str(x) for x in sorted(xs))

        return f"{side(self.first)}-{side(self.second)}-{side(self.third)}"


def _positive_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def implied_probabilities(
    trifecta_odds: Mapping[tuple[int, int, int], float],
) -> dict[tuple[int, int, int], float]:
    """Normalize 1/odds across the complete trifecta market."""
    inv: dict[tuple[int, int, int], float] = {}
    for ticket, raw_odds in trifecta_odds.items():
        odds = _positive_float(raw_odds)
        if odds is not None:
            inv[ticket] = 1.0 / odds
    z = sum(inv.values())
    if z <= 0:
        raise ValueError("trifecta odds contain no positive finite values")
    return {ticket: value / z for ticket, value in inv.items()}


def positional_support(
    q: Mapping[tuple[int, int, int], float],
    cars: Sequence[int],
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Return market support for each rider finishing 1st, 2nd, and 3rd."""
    h1 = {car: 0.0 for car in cars}
    h2 = {car: 0.0 for car in cars}
    h3 = {car: 0.0 for car in cars}
    for (first, second, third), prob in q.items():
        if first in h1:
            h1[first] += prob
        if second in h2:
            h2[second] += prob
        if third in h3:
            h3[third] += prob
    return h1, h2, h3


def market_top_cluster(scores: Mapping[int, float]) -> ClusterBoundary:
    """Select the upper market tier without fixing its rider count.

    Riders are ordered by support.  For every adjacent pair we calculate
    support(rank r) / support(rank r+1).  The largest adjacent drop is the
    boundary between the upper tier and the rest.

    The cut rank is therefore data-dependent.  If all supports are exactly
    equal, there is no hierarchy and every rider is retained.  If several
    boundaries have the same largest ratio, the later boundary is used so
    a tie never narrows the formation unnecessarily.
    """
    ranked = sorted(scores, key=lambda car: (-scores[car], car))
    if not ranked:
        raise ValueError("scores must not be empty")
    if len(ranked) == 1:
        only = ranked[0]
        return ClusterBoundary(
            ranked_cars=(only,),
            ranked_support=(float(scores[only]),),
            cut_after_rank=1,
            boundary_ratio=1.0,
            selected=(only,),
        )

    vals = [max(float(scores[car]), 0.0) for car in ranked]
    ratios: list[float] = []
    for left, right in zip(vals[:-1], vals[1:]):
        if right <= 0:
            ratio = math.inf if left > 0 else 1.0
        else:
            ratio = left / right
        ratios.append(ratio)

    largest = max(ratios)
    if largest == 1.0:
        cut = len(ranked)
    else:
        # Later tie wins: preserve the broader upper tier when evidence is tied.
        boundary_index = max(i for i, ratio in enumerate(ratios) if ratio == largest)
        cut = boundary_index + 1

    return ClusterBoundary(
        ranked_cars=tuple(ranked),
        ranked_support=tuple(vals),
        cut_after_rank=cut,
        boundary_ratio=largest,
        selected=tuple(ranked[:cut]),
    )


def _nested_union(*groups: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({car for group in groups for car in group}))


def generate_nested_tickets(
    first: Sequence[int],
    second: Sequence[int],
    third: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    """Generate every valid ticket represented by F1-F2-F3.

    There is deliberately no odds>N pruning or any other price cut.
    """
    tickets = {
        (a, b, c)
        for a, b, c in product(first, second, third)
        if len({a, b, c}) == 3
    }
    return tuple(sorted(tickets))


def build_market_hierarchy_formation(
    trifecta_odds: Mapping[tuple[int, int, int], float],
    cars: Sequence[int] | None = None,
) -> FormationResult:
    """Create the v7.0-F01 formation from the full trifecta market.

    Raw positional tiers are independently discovered from H1/H2/H3.
    They are converted to nested sets by closure:

        F1 = raw first-place upper tier
        F2 = F1 union raw second-place upper tier
        F3 = F2 union raw third-place upper tier

    Thus F1 subset F2 subset F3 always holds, while the number of riders
    in every place remains fully variable from race to race.
    """
    q = implied_probabilities(trifecta_odds)
    if cars is None:
        cars = sorted({car for ticket in trifecta_odds for car in ticket})
    else:
        cars = tuple(sorted(set(cars)))
    if len(cars) < 3:
        raise ValueError("at least three riders are required")

    h1, h2, h3 = positional_support(q, cars)
    b1 = market_top_cluster(h1)
    b2 = market_top_cluster(h2)
    b3 = market_top_cluster(h3)

    f1 = tuple(sorted(b1.selected))
    f2 = _nested_union(f1, b2.selected)
    f3 = _nested_union(f2, b3.selected)
    tickets = generate_nested_tickets(f1, f2, f3)
    mass = sum(q.get(ticket, 0.0) for ticket in tickets)

    return FormationResult(
        scheme_version=SCHEME_VERSION,
        first=f1,
        second=f2,
        third=f3,
        tickets=tickets,
        first_boundary=b1,
        second_boundary=b2,
        third_boundary=b3,
        formation_mass=mass,
    )


def parse_lines(text: str) -> tuple[tuple[int, ...], ...] | None:
    lines: list[tuple[int, ...]] = []
    for raw in (text or "").strip().split("/"):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("-")
        if not all(part.strip().isdigit() for part in parts):
            return None
        line = tuple(int(part.strip()) for part in parts)
        if not line:
            return None
        lines.append(line)
    flat = [car for line in lines for car in line]
    if not lines or len(flat) != len(set(flat)):
        return None
    return tuple(lines)


def trio_implied_probabilities(
    trio_odds: Mapping[tuple[int, int, int], float],
) -> dict[tuple[int, int, int], float]:
    inv: dict[tuple[int, int, int], float] = {}
    for combo, raw_odds in trio_odds.items():
        odds = _positive_float(raw_odds)
        if odds is not None:
            inv[tuple(sorted(combo))] = 1.0 / odds
    z = sum(inv.values())
    if z <= 0:
        raise ValueError("trio odds contain no positive finite values")
    return {combo: value / z for combo, value in inv.items()}


def v6_1_entry_gate(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
) -> dict[str, object]:
    """Preserve only the v6.1 pre-formation entry gate.

    M_pre, PruneDamage, the old conditional formation, and price pruning are
    intentionally absent because they depend on the retired v6 formation.
    """
    p3 = trio_implied_probabilities(trio_odds)
    q = implied_probabilities(trifecta_odds)
    cars = sorted({car for combo in trio_odds for car in combo})
    lines = parse_lines(predicted_line_formation)
    if lines is None or set(car for line in lines for car in line) != set(cars):
        return {"entry_pass": False, "reason": "INVALID_LINE"}

    s = {
        car: sum(prob for combo, prob in p3.items() if car in combo) / 3.0
        for car in cars
    }
    ls = [sum(s[car] for car in line) for line in lines]
    line_order = sorted(range(len(lines)), key=lambda idx: (-ls[idx], idx))
    if len(line_order) < 2:
        return {"entry_pass": False, "reason": "LT2_LINES"}
    ai, bi = line_order[:2]

    pair_rows: list[tuple[float, int, int, tuple[int, int]]] = []
    for li, line in enumerate(lines):
        for pos in range(len(line) - 1):
            pair = (line[pos], line[pos + 1])
            ps = sum(
                prob
                for combo, prob in p3.items()
                if pair[0] in combo and pair[1] in combo
            )
            pair_rows.append((ps, li, pos, pair))
    pair_rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    if len(pair_rows) < 2 or {pair_rows[0][1], pair_rows[1][1]} != {ai, bi}:
        return {"entry_pass": False, "reason": "PS_TOP2_NOT_AB"}

    h1, _, _ = positional_support(q, cars)
    ranked_head = sorted(cars, key=lambda car: (-h1[car], car))
    top2 = ranked_head[:2]
    line_of = {car: li for li, line in enumerate(lines) for car in line}
    if {line_of[top2[0]], line_of[top2[1]]} != {ai, bi}:
        return {"entry_pass": False, "reason": "H_TOP2_NOT_AB"}

    pos_of = {car: pos for line in lines for pos, car in enumerate(line)}
    if any(pos_of[car] > 1 for car in top2):
        return {"entry_pass": False, "reason": "H_TOP_NOT_HEAD2"}

    if not (h1[top2[0]] < 2.0 * h1[top2[1]]):
        return {"entry_pass": False, "reason": "H1_GE_2H2"}

    return {
        "entry_pass": True,
        "reason": "PASS",
        "A_line": lines[ai],
        "B_line": lines[bi],
        "head_rank": tuple(ranked_head),
    }


def build_v7_0_f01(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
) -> dict[str, object]:
    """Full deterministic v7.0-F01 decision for one race."""
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": gate["reason"],
        }

    cars = sorted({car for combo in trio_odds for car in combo})
    formation = build_market_hierarchy_formation(trifecta_odds, cars)
    if not formation.tickets:
        return {
            "scheme_version": SCHEME_VERSION,
            "buy": False,
            "reason": "NO_VALID_TICKETS",
        }

    return {
        "scheme_version": SCHEME_VERSION,
        "buy": True,
        "reason": "PASS",
        "formation": formation.display,
        "first": formation.first,
        "second": formation.second,
        "third": formation.third,
        "tickets": formation.tickets,
        "ticket_count": formation.ticket_count,
        "stake_yen": formation.stake_yen,
        "formation_mass": formation.formation_mass,
        "boundaries": {
            "first": formation.first_boundary,
            "second": formation.second_boundary,
            "third": formation.third_boundary,
        },
        "entry_gate": gate,
        "price_cut_enabled": PRICE_CUT_ENABLED,
        "fixed_place_counts": FIXED_PLACE_COUNTS,
    }

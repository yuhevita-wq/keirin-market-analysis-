from __future__ import annotations

"""v8.0-F01: clean rectangular market-hierarchy formation.

Correction from v7:
A clean formation does NOT require F1 subset F2 subset F3.  The user's own
examples such as 12-12-3 are rectangular position sets, not nested sets.

This engine therefore generates one branch-free F1-F2-F3, with each position
set free to differ.  Rider counts are never fixed.  No odds>N price pruning is
used.  The unchanged v6.1 entry gate and its A/B structural pools are retained.

For every non-empty subset of the structural 1st, 2nd and 3rd pools, the exact
trifecta market mass and valid ticket count are calculated.  The efficient
frontier (mass vs point count) is built and the parameter-free Pareto knee is
selected.  No race result or payout is an input.
"""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping, Sequence

from v7_0_f01_market_hierarchy import (
    implied_probabilities,
    trio_implied_probabilities,
    v6_1_entry_gate,
)

SCHEME_VERSION = "v8.0-F01"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
STAKE_YEN_PER_TICKET = 100


@dataclass(frozen=True)
class Candidate:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    market_mass: float

    @property
    def ticket_count(self) -> int:
        return len(self.tickets)

    @property
    def display(self) -> str:
        def show(xs: Sequence[int]) -> str:
            return "".join(map(str, sorted(xs)))
        return f"{show(self.first)}-{show(self.second)}-{show(self.third)}"


def _powerset_nonempty(xs: Sequence[int]):
    vals = tuple(sorted(set(xs)))
    for r in range(1, len(vals) + 1):
        yield from combinations(vals, r)


def _tickets(first, second, third):
    return tuple(sorted({
        (a, b, c)
        for a, b, c in product(first, second, third)
        if len({a, b, c}) == 3
    }))


def _s_scores(trio_odds: Mapping[tuple[int, int, int], float]) -> dict[int, float]:
    p3 = trio_implied_probabilities(trio_odds)
    cars = sorted({c for combo in trio_odds for c in combo})
    return {
        c: sum(prob for combo, prob in p3.items() if c in combo) / 3.0
        for c in cars
    }


def _structural_pools(trio_odds, gate):
    cars = sorted({c for combo in trio_odds for c in combo})
    s = _s_scores(trio_odds)
    A = tuple(gate["A_line"])
    B = tuple(gate["B_line"])
    first_pool = tuple(sorted(set(A[:2]) | set(B[:2])))
    second_pool = first_pool
    third_pool = set(A[:3]) | set(B[:3])
    outside = [c for c in cars if c not in A and c not in B]
    if outside:
        third_pool.add(max(outside, key=lambda c: (s[c], -c)))
    return first_pool, second_pool, tuple(sorted(third_pool))


def _pareto_knee(candidates: Sequence[Candidate]) -> Candidate:
    best_at_n: dict[int, Candidate] = {}
    for c in candidates:
        old = best_at_n.get(c.ticket_count)
        if old is None or c.market_mass > old.market_mass + 1e-15:
            best_at_n[c.ticket_count] = c

    frontier: list[Candidate] = []
    best_mass = -1.0
    for n in sorted(best_at_n):
        c = best_at_n[n]
        if c.market_mass > best_mass + 1e-15:
            frontier.append(c)
            best_mass = c.market_mass

    if not frontier:
        raise ValueError("empty rectangular frontier")
    if len(frontier) == 1:
        return frontier[0]

    n0, n1 = frontier[0].ticket_count, frontier[-1].ticket_count
    m0, m1 = frontier[0].market_mass, frontier[-1].market_mass
    if n1 <= n0 or m1 <= m0:
        return frontier[0]

    def score(c: Candidate):
        x = (c.ticket_count - n0) / (n1 - n0)
        y = (c.market_mass - m0) / (m1 - m0)
        return y - x

    return max(frontier, key=lambda c: (score(c), -c.ticket_count, c.market_mass))


def choose_rectangular_hierarchy(trio_odds, trifecta_odds, gate):
    q = implied_probabilities(trifecta_odds)
    p1, p2, p3 = _structural_pools(trio_odds, gate)
    candidates: list[Candidate] = []
    seen = set()

    for f1 in _powerset_nonempty(p1):
        for f2 in _powerset_nonempty(p2):
            for f3 in _powerset_nonempty(p3):
                tickets = _tickets(f1, f2, f3)
                if not tickets:
                    continue
                # Avoid decorative riders that create no valid ticket.
                used1 = {t[0] for t in tickets}
                used2 = {t[1] for t in tickets}
                used3 = {t[2] for t in tickets}
                if used1 != set(f1) or used2 != set(f2) or used3 != set(f3):
                    continue
                key = (tuple(f1), tuple(f2), tuple(f3))
                if key in seen:
                    continue
                seen.add(key)
                mass = sum(q.get(t, 0.0) for t in tickets)
                candidates.append(Candidate(tuple(f1), tuple(f2), tuple(f3), tickets, mass))

    return _pareto_knee(candidates)


def build_v8_0_f01(trio_odds, trifecta_odds, predicted_line_formation):
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}
    try:
        c = choose_rectangular_hierarchy(trio_odds, trifecta_odds, gate)
    except ValueError as exc:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": str(exc)}
    return {
        "scheme_version": SCHEME_VERSION,
        "buy": True,
        "reason": "PASS",
        "formation": c.display,
        "first": c.first,
        "second": c.second,
        "third": c.third,
        "tickets": c.tickets,
        "ticket_count": c.ticket_count,
        "market_mass": c.market_mass,
        "entry_gate": gate,
        "price_cut_enabled": False,
        "fixed_place_counts": False,
        "nested_required": False,
    }

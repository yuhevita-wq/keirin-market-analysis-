from __future__ import annotations

"""v7.3-F04: structural market frontier formation.

Keep the v6.1 entry gate and its two dominant A/B lines, but replace the old
branchy formation with a single clean nested F1-F2-F3.

No rider counts are fixed. No odds>N trimming is used. Candidate size is chosen
from a Pareto frontier of trifecta market mass versus ticket count.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from v7_0_f01_market_hierarchy import (
    generate_nested_tickets,
    implied_probabilities,
    positional_support,
    trio_implied_probabilities,
    v6_1_entry_gate,
)

SCHEME_VERSION = "v7.3-F04"
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
    k1: int
    k2: int
    k3: int

    @property
    def n(self) -> int:
        return len(self.tickets)

    @property
    def display(self) -> str:
        def s(xs: Sequence[int]) -> str:
            return "".join(map(str, sorted(xs)))
        return f"{s(self.first)}-{s(self.second)}-{s(self.third)}"


def _rank(scores: Mapping[int, float], pool: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(pool), key=lambda c: (-scores[c], c)))


def _union(*groups: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({c for g in groups for c in g}))


def _s_scores(trio_odds):
    p3 = trio_implied_probabilities(trio_odds)
    cars = sorted({c for combo in trio_odds for c in combo})
    return {c: sum(p for combo, p in p3.items() if c in combo) / 3.0 for c in cars}


def choose_structural_knee(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    gate: Mapping[str, object],
) -> Candidate:
    q = implied_probabilities(trifecta_odds)
    cars = sorted({c for combo in trio_odds for c in combo})
    h1, h2, h3 = positional_support(q, cars)
    s = _s_scores(trio_odds)

    A = tuple(gate["A_line"])
    B = tuple(gate["B_line"])
    head_pool = tuple(sorted(set(A[:2]) | set(B[:2])))
    third_pool = set(A[:3]) | set(B[:3])
    outside = [c for c in cars if c not in A and c not in B]
    if outside:
        third_pool.add(max(outside, key=lambda c: (s[c], -c)))
    third_pool = tuple(sorted(third_pool))

    r1 = _rank(h1, head_pool)
    r2 = _rank(h2, head_pool)
    r3 = _rank(h3, third_pool)

    unique: dict[tuple, Candidate] = {}
    for k1 in range(1, len(r1) + 1):
        f1 = tuple(sorted(r1[:k1]))
        for k2 in range(1, len(r2) + 1):
            f2 = _union(f1, r2[:k2])
            for k3 in range(1, len(r3) + 1):
                f3 = _union(f2, r3[:k3])
                tickets = generate_nested_tickets(f1, f2, f3)
                if not tickets:
                    continue
                key = (f1, f2, f3)
                if key in unique:
                    continue
                mass = sum(q.get(t, 0.0) for t in tickets)
                unique[key] = Candidate(f1, f2, f3, tickets, mass, k1, k2, k3)

    if not unique:
        raise ValueError("no valid structural formation")

    # Best market mass at each ticket count.
    by_n: dict[int, Candidate] = {}
    for c in unique.values():
        old = by_n.get(c.n)
        if old is None or (c.market_mass, tuple(-x for x in c.first + c.second + c.third)) > (
            old.market_mass, tuple(-x for x in old.first + old.second + old.third)
        ):
            by_n[c.n] = c

    # Pareto frontier: mass must strictly increase as ticket count grows.
    frontier: list[Candidate] = []
    best_mass = -1.0
    for n in sorted(by_n):
        c = by_n[n]
        if c.market_mass > best_mass + 1e-15:
            frontier.append(c)
            best_mass = c.market_mass

    if len(frontier) == 1:
        return frontier[0]

    n0, n1 = frontier[0].n, frontier[-1].n
    m0, m1 = frontier[0].market_mass, frontier[-1].market_mass
    if n1 == n0 or m1 <= m0:
        return frontier[0]

    # Knee = greatest vertical gain above the straight line joining the
    # smallest and largest efficient formations. This is parameter-free and
    # directly balances extra mass against extra tickets.
    def knee_score(c: Candidate) -> float:
        x = (c.n - n0) / (n1 - n0)
        y = (c.market_mass - m0) / (m1 - m0)
        return y - x

    return max(frontier, key=lambda c: (knee_score(c), -c.n, c.market_mass))


def build_v7_3_f04(trio_odds, trifecta_odds, predicted_line_formation):
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}
    c = choose_structural_knee(trio_odds, trifecta_odds, gate)
    return {
        "scheme_version": SCHEME_VERSION,
        "buy": True,
        "reason": "PASS",
        "formation": c.display,
        "first": c.first,
        "second": c.second,
        "third": c.third,
        "tickets": c.tickets,
        "ticket_count": c.n,
        "market_mass": c.market_mass,
        "raw_rank_depths": (c.k1, c.k2, c.k3),
        "entry_gate": gate,
        "price_cut_enabled": False,
        "fixed_place_counts": False,
    }

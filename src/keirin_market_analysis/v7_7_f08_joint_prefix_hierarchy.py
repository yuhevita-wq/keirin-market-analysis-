from __future__ import annotations

"""v7.7-F08: joint-prefix market hierarchy.

The race-entry gate remains exactly v6.1.  Formation construction is rewritten
as one clean nested F1-F2-F3 with no fixed rider counts and no price pruning.

Unlike earlier v7 attempts, 1st and 2nd place are not selected independently.
The complete trifecta market is first collapsed to ordered 1->2 prefixes.
All nested F1 subset F2 candidates inside the two dominant A/B line head-pools
are evaluated directly on joint prefix mass versus the number of ordered
prefixes.  A parameter-free Pareto knee chooses the 1st/2nd shape.

Then 3rd-place candidates are evaluated conditional on that selected prefix.
All F3 supersets are evaluated on exact trifecta mass versus ticket count and a
second Pareto knee chooses the final 3rd-place width.

No result/payout/rider-performance data is an input.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

from v7_0_f01_market_hierarchy import (
    generate_nested_tickets,
    implied_probabilities,
    trio_implied_probabilities,
    v6_1_entry_gate,
)

SCHEME_VERSION = "v7.7-F08"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
STAKE_YEN_PER_TICKET = 100


@dataclass(frozen=True)
class PrefixCandidate:
    first: tuple[int, ...]
    second: tuple[int, ...]
    prefix_mass: float
    prefix_count: int


@dataclass(frozen=True)
class FormationCandidate:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    market_mass: float
    prefix_mass: float
    prefix_count: int

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
        for comb in combinations(vals, r):
            yield comb


def _supersets(base: Sequence[int], universe: Sequence[int]):
    base_set = set(base)
    rest = tuple(sorted(set(universe) - base_set))
    for r in range(0, len(rest) + 1):
        for add in combinations(rest, r):
            yield tuple(sorted(base_set | set(add)))


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
    head_pool = tuple(sorted(set(A[:2]) | set(B[:2])))
    third_pool = set(A[:3]) | set(B[:3])
    outside = [c for c in cars if c not in A and c not in B]
    if outside:
        third_pool.add(max(outside, key=lambda c: (s[c], -c)))
    return head_pool, tuple(sorted(third_pool))


def _pareto_knee(items, cost_fn, mass_fn):
    """Choose a parameter-free knee on a strictly increasing mass/cost frontier."""
    best_at_cost = {}
    for item in items:
        cost = int(cost_fn(item))
        mass = float(mass_fn(item))
        old = best_at_cost.get(cost)
        if old is None or mass > float(mass_fn(old)) + 1e-15:
            best_at_cost[cost] = item

    frontier = []
    best_mass = -1.0
    for cost in sorted(best_at_cost):
        item = best_at_cost[cost]
        mass = float(mass_fn(item))
        if mass > best_mass + 1e-15:
            frontier.append(item)
            best_mass = mass

    if not frontier:
        raise ValueError("empty Pareto frontier")
    if len(frontier) == 1:
        return frontier[0]

    c0, c1 = float(cost_fn(frontier[0])), float(cost_fn(frontier[-1]))
    m0, m1 = float(mass_fn(frontier[0])), float(mass_fn(frontier[-1]))
    if c1 <= c0 or m1 <= m0:
        return frontier[0]

    def score(item):
        x = (float(cost_fn(item)) - c0) / (c1 - c0)
        y = (float(mass_fn(item)) - m0) / (m1 - m0)
        return y - x

    return max(
        frontier,
        key=lambda item: (score(item), -int(cost_fn(item)), float(mass_fn(item))),
    )


def _prefix_candidates(q, head_pool):
    out = []
    for f1 in _powerset_nonempty(head_pool):
        set1 = set(f1)
        for f2 in _supersets(f1, head_pool):
            set2 = set(f2)
            prefixes = {(a, b) for a in set1 for b in set2 if a != b}
            if not prefixes:
                continue
            mass = sum(prob for (a, b, _), prob in q.items() if (a, b) in prefixes)
            out.append(
                PrefixCandidate(
                    first=tuple(sorted(f1)),
                    second=tuple(sorted(f2)),
                    prefix_mass=mass,
                    prefix_count=len(prefixes),
                )
            )
    return out


def choose_joint_prefix_hierarchy(trio_odds, trifecta_odds, gate):
    q = implied_probabilities(trifecta_odds)
    head_pool, third_pool = _structural_pools(trio_odds, gate)

    prefix_candidates = _prefix_candidates(q, head_pool)
    prefix = _pareto_knee(
        prefix_candidates,
        cost_fn=lambda x: x.prefix_count,
        mass_fn=lambda x: x.prefix_mass,
    )

    formations = []
    for f3 in _supersets(prefix.second, third_pool):
        tickets = generate_nested_tickets(prefix.first, prefix.second, f3)
        if not tickets:
            continue
        mass = sum(q.get(t, 0.0) for t in tickets)
        formations.append(
            FormationCandidate(
                first=prefix.first,
                second=prefix.second,
                third=f3,
                tickets=tickets,
                market_mass=mass,
                prefix_mass=prefix.prefix_mass,
                prefix_count=prefix.prefix_count,
            )
        )

    return _pareto_knee(
        formations,
        cost_fn=lambda x: x.ticket_count,
        mass_fn=lambda x: x.market_mass,
    )


def build_v7_7_f08(trio_odds, trifecta_odds, predicted_line_formation):
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}
    try:
        c = choose_joint_prefix_hierarchy(trio_odds, trifecta_odds, gate)
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
        "prefix_mass": c.prefix_mass,
        "prefix_count": c.prefix_count,
        "entry_gate": gate,
        "price_cut_enabled": False,
        "fixed_place_counts": False,
    }

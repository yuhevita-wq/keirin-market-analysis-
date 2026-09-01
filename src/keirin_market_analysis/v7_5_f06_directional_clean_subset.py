from __future__ import annotations

"""v7.5-F06: one clean nested formation carved from the old v6 directional core.

Principles
----------
- Race-entry gate is unchanged v6.1.
- No fixed number of riders in any finishing position.
- No odds>N / price pruning.
- Preserve the useful directional information of the old v6 pre-formation,
  but never output its branchy ticket list directly.
- Build one clean nested F1-F2-F3 whose every ticket is already contained in
  that directional pre-formation.
- Choose its size mechanically from the Pareto knee of market mass vs tickets.

No results or payouts are inputs to this module.
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

SCHEME_VERSION = "v7.5-F06"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
STAKE_YEN_PER_TICKET = 100


@dataclass(frozen=True)
class CleanCandidate:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    market_mass: float

    @property
    def n(self) -> int:
        return len(self.tickets)

    @property
    def display(self) -> str:
        def show(xs: Sequence[int]) -> str:
            return "".join(map(str, sorted(xs)))
        return f"{show(self.first)}-{show(self.second)}-{show(self.third)}"


def _union(*groups: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({c for g in groups for c in g}))


def _rank(scores: Mapping[int, float], pool: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(pool), key=lambda c: (-scores.get(c, 0.0), c)))


def _s_scores(trio_odds: Mapping[tuple[int, int, int], float]) -> dict[int, float]:
    p3 = trio_implied_probabilities(trio_odds)
    cars = sorted({c for combo in trio_odds for c in combo})
    return {
        c: sum(prob for combo, prob in p3.items() if c in combo) / 3.0
        for c in cars
    }


def old_v6_preformation(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    gate: Mapping[str, object],
) -> tuple[set[tuple[int, int, int]], tuple[int, ...], tuple[int, ...]]:
    """Rebuild only the old v6 directional ticket universe BEFORE price cut."""
    q = implied_probabilities(trifecta_odds)
    cars = sorted({c for combo in trio_odds for c in combo})
    h1, _, _ = positional_support(q, cars)
    s = _s_scores(trio_odds)

    A = tuple(gate["A_line"])
    B = tuple(gate["B_line"])
    if len(A) < 2 or len(B) < 2:
        return set(), (), ()

    AH = max(A[:2], key=lambda c: (h1[c], -c))
    BH = max(B[:2], key=lambda c: (h1[c], -c))
    A_other = A[1] if AH == A[0] else A[0]
    B_other = B[1] if BH == B[0] else B[0]

    outside = [c for c in cars if c not in A and c not in B]
    ext = max(outside, key=lambda c: (s[c], -c)) if outside else None

    third_pool = set(A[:3]) | set(B[:3])
    if ext is not None:
        third_pool.add(ext)

    pre: set[tuple[int, int, int]] = set()
    for first, seconds in ((AH, (A_other, BH)), (BH, (B_other, AH))):
        for second in seconds:
            if second == first:
                continue
            for third in third_pool:
                if third not in (first, second):
                    pre.add((first, second, third))

    head_pool = tuple(sorted(set(A[:2]) | set(B[:2])))
    return pre, head_pool, tuple(sorted(third_pool))


def _directional_role_scores(
    q: Mapping[tuple[int, int, int], float],
    pre: set[tuple[int, int, int]],
    cars: Sequence[int],
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    a = {c: 0.0 for c in cars}
    b = {c: 0.0 for c in cars}
    d = {c: 0.0 for c in cars}
    for t in pre:
        prob = q.get(t, 0.0)
        a[t[0]] += prob
        b[t[1]] += prob
        d[t[2]] += prob
    return a, b, d


def choose_directional_clean_subset(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    gate: Mapping[str, object],
) -> CleanCandidate:
    q = implied_probabilities(trifecta_odds)
    pre, head_pool, third_pool = old_v6_preformation(trio_odds, trifecta_odds, gate)
    if not pre:
        raise ValueError("empty old directional pre-formation")

    cars = sorted({c for t in trifecta_odds for c in t})
    v1, v2, v3 = _directional_role_scores(q, pre, cars)
    r1 = _rank(v1, head_pool)
    r2 = _rank(v2, head_pool)
    r3 = _rank(v3, third_pool)

    unique: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], CleanCandidate] = {}
    for k1 in range(1, len(r1) + 1):
        f1 = tuple(sorted(r1[:k1]))
        for k2 in range(1, len(r2) + 1):
            f2 = _union(f1, r2[:k2])
            for k3 in range(1, len(r3) + 1):
                f3 = _union(f2, r3[:k3])
                tickets = generate_nested_tickets(f1, f2, f3)
                if not tickets:
                    continue
                # The defining constraint: a clean formation may contain no
                # ticket outside the old directional core.
                if any(t not in pre for t in tickets):
                    continue
                key = (f1, f2, f3)
                if key in unique:
                    continue
                mass = sum(q.get(t, 0.0) for t in tickets)
                unique[key] = CleanCandidate(f1, f2, f3, tickets, mass)

    if not unique:
        raise ValueError("no clean nested subset of directional core")

    # Best mass for each ticket count.
    by_n: dict[int, CleanCandidate] = {}
    for c in unique.values():
        old = by_n.get(c.n)
        if old is None or c.market_mass > old.market_mass + 1e-15:
            by_n[c.n] = c

    # Strict Pareto frontier.
    frontier: list[CleanCandidate] = []
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

    def knee(c: CleanCandidate) -> float:
        x = (c.n - n0) / (n1 - n0)
        y = (c.market_mass - m0) / (m1 - m0)
        return y - x

    return max(frontier, key=lambda c: (knee(c), -c.n, c.market_mass))


def build_v7_5_f06(trio_odds, trifecta_odds, predicted_line_formation):
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}
    try:
        c = choose_directional_clean_subset(trio_odds, trifecta_odds, gate)
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
        "ticket_count": c.n,
        "market_mass": c.market_mass,
        "entry_gate": gate,
        "price_cut_enabled": PRICE_CUT_ENABLED,
        "fixed_place_counts": FIXED_PLACE_COUNTS,
    }

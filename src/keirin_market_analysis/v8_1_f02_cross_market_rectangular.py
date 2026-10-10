from __future__ import annotations

"""v8.1-F02: clean rectangular formation selected by cross-market excess.

The v6.1 pre-formation entry gate and structural candidate pools are unchanged.
The output is always one branch-free F1-F2-F3 formation. Rider counts and point
count are not fixed. There is no odds>N price pruning and no race result/payout
is an input.

Cross-market idea
-----------------
For each unordered trio c:
  P3(c) = normalized 3-renpuku implied probability
  Q3(c) = sum of normalized 3-rentan probabilities over its six orders

Within a trio, the 3-rentan market's order share is
  R(t|c) = q(t) / Q3(c)

The 3-renpuku-implied exact-order mass is
  q_star(t) = P3(c) * R(t|c)

and the ticket's cross-market excess is
  E(t) = q_star(t) - q(t)
       = q(t) * (P3(c) / Q3(c) - 1).

So the sign is intentionally a trio-level disagreement, while q(t) allocates
that disagreement among exact orders. This avoids pretending that the simple
ratio q_star/q contains order information (it cancels within each trio).

All clean rectangular formations inside the unchanged A/B structural pools are
enumerated. For each point count, keep the formation with the greatest summed
cross-market excess. Build the efficient frontier of increasing excess versus
point count and choose its parameter-free knee. Thus extra points must buy
additional cross-market excess, not merely additional popularity mass.
"""

from dataclasses import dataclass
from itertools import combinations, product
from math import isfinite
from typing import Mapping, Sequence

from v7_0_f01_market_hierarchy import (
    implied_probabilities,
    trio_implied_probabilities,
    v6_1_entry_gate,
)
from v8_0_f01_rectangular_hierarchy import _structural_pools

SCHEME_VERSION = "v8.1-F02"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
STAKE_YEN_PER_TICKET = 100


@dataclass(frozen=True)
class Candidate:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    cross_excess: float
    q_mass: float
    q_star_mass: float
    positive_excess: float
    negative_excess: float

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


def cross_market_ticket_table(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
):
    p3 = trio_implied_probabilities(trio_odds)
    q = implied_probabilities(trifecta_odds)
    q3: dict[tuple[int, int, int], float] = {}
    for t, prob in q.items():
        c = tuple(sorted(t))
        q3[c] = q3.get(c, 0.0) + prob

    table = {}
    for t, prob in q.items():
        c = tuple(sorted(t))
        qc = q3.get(c, 0.0)
        pc = p3.get(c)
        if pc is None or qc <= 0.0:
            raise ValueError("incomplete cross-market trio mapping")
        ratio = pc / qc
        q_star = prob * ratio
        excess = q_star - prob
        if not all(isfinite(x) for x in (ratio, q_star, excess)):
            raise ValueError("non-finite cross-market value")
        table[t] = {
            "q": prob,
            "q_star": q_star,
            "excess": excess,
            "trio_ratio": ratio,
            "trio_p": pc,
            "trifecta_trio_q": qc,
        }
    return table


def _candidate(first, second, third, tickets, value_table):
    q_mass = sum(value_table[t]["q"] for t in tickets)
    q_star_mass = sum(value_table[t]["q_star"] for t in tickets)
    pos = sum(max(value_table[t]["excess"], 0.0) for t in tickets)
    neg = sum(min(value_table[t]["excess"], 0.0) for t in tickets)
    return Candidate(
        tuple(first), tuple(second), tuple(third), tuple(tickets),
        q_star_mass - q_mass, q_mass, q_star_mass, pos, neg,
    )


def _better_same_n(a: Candidate, b: Candidate) -> bool:
    """True when a deterministically beats b at identical point count."""
    ka = (a.cross_excess, a.positive_excess, -abs(a.negative_excess), a.q_star_mass,
          tuple(-x for x in a.first), tuple(-x for x in a.second), tuple(-x for x in a.third))
    kb = (b.cross_excess, b.positive_excess, -abs(b.negative_excess), b.q_star_mass,
          tuple(-x for x in b.first), tuple(-x for x in b.second), tuple(-x for x in b.third))
    return ka > kb


def _cross_excess_knee(candidates: Sequence[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("empty cross-market rectangular candidate set")

    best_at_n: dict[int, Candidate] = {}
    for c in candidates:
        old = best_at_n.get(c.ticket_count)
        if old is None or _better_same_n(c, old):
            best_at_n[c.ticket_count] = c

    # Efficient frontier: more points survive only when they buy more net excess.
    frontier: list[Candidate] = []
    best_excess = float("-inf")
    for n in sorted(best_at_n):
        c = best_at_n[n]
        if c.cross_excess > best_excess + 1e-15:
            frontier.append(c)
            best_excess = c.cross_excess

    if len(frontier) == 1:
        return frontier[0]

    # If the frontier never reaches positive excess, choose the least-bad point
    # on the frontier rather than inventing a result-driven skip rule.
    positive = [c for c in frontier if c.cross_excess > 1e-15]
    working = positive if positive else frontier
    if len(working) == 1:
        return working[0]

    n0, n1 = working[0].ticket_count, working[-1].ticket_count
    e0, e1 = working[0].cross_excess, working[-1].cross_excess
    if n1 <= n0 or e1 <= e0 + 1e-15:
        return max(working, key=lambda c: (c.cross_excess, -c.ticket_count))

    def knee_score(c: Candidate):
        x = (c.ticket_count - n0) / (n1 - n0)
        y = (c.cross_excess - e0) / (e1 - e0)
        return y - x

    return max(
        working,
        key=lambda c: (
            knee_score(c),
            c.cross_excess,
            -c.ticket_count,
            c.positive_excess,
            tuple(-x for x in c.first),
            tuple(-x for x in c.second),
            tuple(-x for x in c.third),
        ),
    )


def choose_cross_market_rectangular(trio_odds, trifecta_odds, gate):
    values = cross_market_ticket_table(trio_odds, trifecta_odds)
    p1, p2, p3 = _structural_pools(trio_odds, gate)
    candidates: list[Candidate] = []

    for f1 in _powerset_nonempty(p1):
        for f2 in _powerset_nonempty(p2):
            for f3 in _powerset_nonempty(p3):
                tickets = _tickets(f1, f2, f3)
                if not tickets:
                    continue
                # Do not display riders that generate no actual valid ticket.
                if {t[0] for t in tickets} != set(f1):
                    continue
                if {t[1] for t in tickets} != set(f2):
                    continue
                if {t[2] for t in tickets} != set(f3):
                    continue
                candidates.append(_candidate(f1, f2, f3, tickets, values))

    return _cross_excess_knee(candidates)


def build_v8_1_f02(trio_odds, trifecta_odds, predicted_line_formation):
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}

    try:
        c = choose_cross_market_rectangular(trio_odds, trifecta_odds, gate)
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
        "cross_excess": c.cross_excess,
        "positive_excess": c.positive_excess,
        "negative_excess": c.negative_excess,
        "q_mass": c.q_mass,
        "q_star_mass": c.q_star_mass,
        "entry_gate": gate,
        "price_cut_enabled": False,
        "fixed_place_counts": False,
        "nested_required": False,
        "formation_selector": "cross-market-excess Pareto knee",
    }

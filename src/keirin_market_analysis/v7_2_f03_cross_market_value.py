from __future__ import annotations

"""v7.2-F03: clean variable-size formation driven by cross-market value.

The 3連複 market supplies the probability of the unordered three-rider set.
The 3連単 market supplies the conditional ordering inside that set.
Those are combined into a market-only pseudo-fair probability for every
trifecta.  A ticket's modeled gross value is fair_prob * offered_odds.

Formation size is not fixed.  We choose the nested market-ranked formation
with the largest modeled total profit after charging one stake unit per ticket.
No result/payout information and no odds>N ticket pruning are used.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from v7_0_f01_market_hierarchy import (
    generate_nested_tickets,
    implied_probabilities,
    trio_implied_probabilities,
    v6_1_entry_gate,
)

SCHEME_VERSION = "v7.2-F03"
BASE_ENTRY_SCHEME_VERSION = "v6.1"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False
STAKE_YEN_PER_TICKET = 100


@dataclass(frozen=True)
class ValueFormation:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    modeled_profit_units: float
    modeled_gross_units: float
    raw_k1: int
    raw_k2: int
    raw_k3: int

    @property
    def ticket_count(self) -> int:
        return len(self.tickets)

    @property
    def modeled_roi(self) -> float:
        return self.modeled_gross_units / self.ticket_count if self.ticket_count else 0.0

    @property
    def display(self) -> str:
        def show(xs: Sequence[int]) -> str:
            return "".join(str(x) for x in sorted(xs))
        return f"{show(self.first)}-{show(self.second)}-{show(self.third)}"


def _set3(t: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sorted(t))


def _rank(scores: Mapping[int, float]) -> tuple[int, ...]:
    return tuple(sorted(scores, key=lambda c: (-scores[c], c)))


def _union(*groups: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({c for g in groups for c in g}))


def cross_market_ticket_values(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int, int], float]]:
    """Return pseudo-fair probability and gross-value multiplier per trifecta."""
    p3 = trio_implied_probabilities(trio_odds)
    q = implied_probabilities(trifecta_odds)
    q_set = defaultdict(float)
    for t, prob in q.items():
        q_set[_set3(t)] += prob

    fair: dict[tuple[int, int, int], float] = {}
    gross: dict[tuple[int, int, int], float] = {}
    for t, prob in q.items():
        s = _set3(t)
        denom = q_set[s]
        if denom <= 0 or s not in p3:
            continue
        fair_t = p3[s] * prob / denom
        fair[t] = fair_t
        gross[t] = fair_t * float(trifecta_odds[t])
    return fair, gross


def choose_cross_market_value_formation(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    cars: Sequence[int] | None = None,
) -> ValueFormation:
    fair, gross = cross_market_ticket_values(trio_odds, trifecta_odds)
    if cars is None:
        cars = tuple(sorted({c for t in trifecta_odds for c in t}))
    else:
        cars = tuple(sorted(set(cars)))

    # Positional value contribution.  A rider ranks highly in a place when
    # tickets using that rider in that place contribute more modeled profit.
    v1 = {c: 0.0 for c in cars}
    v2 = {c: 0.0 for c in cars}
    v3 = {c: 0.0 for c in cars}
    for t, g in gross.items():
        edge = g - 1.0
        v1[t[0]] += edge
        v2[t[1]] += edge
        v3[t[2]] += edge

    r1, r2, r3 = _rank(v1), _rank(v2), _rank(v3)
    ncar = len(cars)
    unique: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ValueFormation] = {}

    for k1 in range(1, ncar + 1):
        raw1 = r1[:k1]
        for k2 in range(1, ncar + 1):
            f1 = tuple(sorted(raw1))
            f2 = _union(f1, r2[:k2])
            for k3 in range(1, ncar + 1):
                f3 = _union(f2, r3[:k3])
                tickets = generate_nested_tickets(f1, f2, f3)
                if not tickets:
                    continue
                key = (f1, f2, f3)
                if key in unique:
                    continue
                gross_units = sum(gross.get(t, 0.0) for t in tickets)
                profit_units = gross_units - len(tickets)
                unique[key] = ValueFormation(
                    first=f1,
                    second=f2,
                    third=f3,
                    tickets=tickets,
                    modeled_profit_units=profit_units,
                    modeled_gross_units=gross_units,
                    raw_k1=k1,
                    raw_k2=k2,
                    raw_k3=k3,
                )

    if not unique:
        raise ValueError("no valid formation")

    # Max modeled profit.  Ties favor fewer tickets, then higher modeled ROI.
    return max(
        unique.values(),
        key=lambda f: (
            f.modeled_profit_units,
            -f.ticket_count,
            f.modeled_roi,
            -(len(f.first) + len(f.second) + len(f.third)),
        ),
    )


def build_v7_2_f03(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
    predicted_line_formation: str,
) -> dict[str, object]:
    gate = v6_1_entry_gate(trio_odds, trifecta_odds, predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version": SCHEME_VERSION, "buy": False, "reason": gate["reason"]}

    cars = sorted({c for s in trio_odds for c in s})
    f = choose_cross_market_value_formation(trio_odds, trifecta_odds, cars)
    return {
        "scheme_version": SCHEME_VERSION,
        "buy": True,
        "reason": "PASS",
        "formation": f.display,
        "first": f.first,
        "second": f.second,
        "third": f.third,
        "tickets": f.tickets,
        "ticket_count": f.ticket_count,
        "stake_yen": f.ticket_count * STAKE_YEN_PER_TICKET,
        "modeled_profit_units": f.modeled_profit_units,
        "modeled_roi": f.modeled_roi,
        "raw_rank_depths": (f.raw_k1, f.raw_k2, f.raw_k3),
        "entry_gate": gate,
        "price_cut_enabled": PRICE_CUT_ENABLED,
        "fixed_place_counts": FIXED_PLACE_COUNTS,
    }

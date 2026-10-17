from __future__ import annotations

"""v8.3-F04: clean rectangular formation chosen by direct modeled profit.

Unchanged v6.1 entry gate and A/B structural pools. One branch-free F1-F2-F3.
No fixed rider counts, no fixed point count, no odds>N price pruning, and no
race result/payout input.

For exact ticket t, q_star(t) is the 3-renpuku-implied exact-order probability
from v8.1. With archived final 3-rentan decimal odds O(t), the modeled gross
return per one-unit stake is q_star(t)*O(t). For an equal-stake rectangular
formation T:

    modeled_profit(T) = sum_{t in T} [q_star(t)*O(t) - 1]

The -1 is the explicit cost of every additional ticket, so rectangular filler
points are penalized directly rather than indirectly by an endpoint knee.
Among all clean rectangles in the structural pools, choose maximum modeled
profit. Ties prefer greater q mass, then fewer tickets, then deterministic car
number order. This is entirely pre-result market logic.
"""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

from v7_0_f01_market_hierarchy import v6_1_entry_gate
from v8_0_f01_rectangular_hierarchy import _structural_pools
from v8_1_f02_cross_market_rectangular import cross_market_ticket_table

SCHEME_VERSION = "v8.3-F04"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False


@dataclass(frozen=True)
class Candidate:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    modeled_profit: float
    modeled_gross: float
    modeled_roi: float
    q_mass: float
    q_star_mass: float

    @property
    def ticket_count(self): return len(self.tickets)

    @property
    def display(self):
        show=lambda xs:"".join(map(str,sorted(xs)))
        return f"{show(self.first)}-{show(self.second)}-{show(self.third)}"


def _powerset(xs: Sequence[int]):
    vals=tuple(sorted(set(xs)))
    for r in range(1,len(vals)+1):
        yield from combinations(vals,r)


def _tickets(f1,f2,f3):
    return tuple(sorted({(a,b,c) for a,b,c in product(f1,f2,f3) if len({a,b,c})==3}))


def choose_modeled_profit(trio_odds,trifecta_odds,gate):
    values=cross_market_ticket_table(trio_odds,trifecta_odds)
    p1,p2,p3=_structural_pools(trio_odds,gate)
    best=None;best_key=None
    for f1 in _powerset(p1):
        for f2 in _powerset(p2):
            for f3 in _powerset(p3):
                ts=_tickets(f1,f2,f3)
                if not ts:continue
                if {t[0] for t in ts}!=set(f1) or {t[1] for t in ts}!=set(f2) or {t[2] for t in ts}!=set(f3):continue
                gross=sum(values[t]["q_star"]*float(trifecta_odds[t]) for t in ts)
                profit=gross-len(ts)
                qmass=sum(values[t]["q"] for t in ts)
                qsmass=sum(values[t]["q_star"] for t in ts)
                c=Candidate(tuple(f1),tuple(f2),tuple(f3),ts,profit,gross,gross/len(ts),qmass,qsmass)
                key=(profit,qmass,-len(ts),qsmass,tuple(-x for x in f1),tuple(-x for x in f2),tuple(-x for x in f3))
                if best is None or key>best_key:
                    best=c;best_key=key
    if best is None:raise ValueError("empty modeled-profit candidate set")
    return best


def build_v8_3_f04(trio_odds,trifecta_odds,predicted_line_formation):
    gate=v6_1_entry_gate(trio_odds,trifecta_odds,predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":gate["reason"]}
    try:c=choose_modeled_profit(trio_odds,trifecta_odds,gate)
    except ValueError as exc:return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":str(exc)}
    return {"scheme_version":SCHEME_VERSION,"buy":True,"reason":"PASS","formation":c.display,
            "first":c.first,"second":c.second,"third":c.third,"tickets":c.tickets,
            "ticket_count":c.ticket_count,"modeled_profit":c.modeled_profit,"modeled_gross":c.modeled_gross,
            "modeled_roi":c.modeled_roi,"q_mass":c.q_mass,"q_star_mass":c.q_star_mass,"entry_gate":gate,
            "price_cut_enabled":False,"fixed_place_counts":False,"fixed_point_count":False,
            "nested_required":False,"formation_selector":"max direct cross-market modeled profit"}

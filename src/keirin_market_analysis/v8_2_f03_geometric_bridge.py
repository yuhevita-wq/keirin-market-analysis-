from __future__ import annotations

"""v8.2-F03: geometric bridge between trifecta support and cross-market value.

v8.1 overweights disagreement and can collapse into sparse longshot formations.
This version keeps the unchanged v6.1 entry gate, the same A/B structural pools,
one clean branch-free F1-F2-F3, variable rider counts, no fixed point count and
no price pruning.

For each exact trifecta ticket t:
  q(t)      = normalized 3-rentan market support
  q_star(t) = 3-renpuku-implied exact-order mass from v8.1

Define the parameter-free bridge weight
  G(t) = sqrt(q(t) * q_star(t)) = q(t) * sqrt(P_trio/Q_trifecta_trio).

This is the geometric midpoint of the two market views: a ticket must retain
real 3-rentan support while receiving an uplift when the 3-renpuku market is
stronger for its unordered trio. No fitted blending coefficient is introduced.
The clean rectangular formation is chosen at the Pareto knee of summed G mass
versus point count.
"""

from dataclasses import dataclass
from itertools import combinations, product
from math import sqrt
from typing import Sequence

from v7_0_f01_market_hierarchy import v6_1_entry_gate
from v8_0_f01_rectangular_hierarchy import _structural_pools
from v8_1_f02_cross_market_rectangular import cross_market_ticket_table

SCHEME_VERSION = "v8.2-F03"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False


@dataclass(frozen=True)
class Candidate:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    bridge_mass: float
    q_mass: float
    q_star_mass: float
    cross_excess: float

    @property
    def ticket_count(self):
        return len(self.tickets)

    @property
    def display(self):
        show = lambda xs: "".join(map(str, sorted(xs)))
        return f"{show(self.first)}-{show(self.second)}-{show(self.third)}"


def _powerset(xs: Sequence[int]):
    vals = tuple(sorted(set(xs)))
    for r in range(1, len(vals) + 1):
        yield from combinations(vals, r)


def _tickets(f1, f2, f3):
    return tuple(sorted({(a,b,c) for a,b,c in product(f1,f2,f3) if len({a,b,c}) == 3}))


def _knee(candidates):
    best_at_n = {}
    for c in candidates:
        old = best_at_n.get(c.ticket_count)
        key = (c.bridge_mass, c.cross_excess, c.q_mass, tuple(-x for x in c.first), tuple(-x for x in c.second), tuple(-x for x in c.third))
        if old is None:
            best_at_n[c.ticket_count] = c
        else:
            old_key = (old.bridge_mass, old.cross_excess, old.q_mass, tuple(-x for x in old.first), tuple(-x for x in old.second), tuple(-x for x in old.third))
            if key > old_key:
                best_at_n[c.ticket_count] = c

    frontier=[]; best=-1.0
    for n in sorted(best_at_n):
        c=best_at_n[n]
        if c.bridge_mass > best + 1e-15:
            frontier.append(c); best=c.bridge_mass
    if not frontier:
        raise ValueError("empty geometric bridge frontier")
    if len(frontier)==1:
        return frontier[0]
    n0,n1=frontier[0].ticket_count,frontier[-1].ticket_count
    m0,m1=frontier[0].bridge_mass,frontier[-1].bridge_mass
    if n1<=n0 or m1<=m0:
        return frontier[0]
    def score(c):
        x=(c.ticket_count-n0)/(n1-n0)
        y=(c.bridge_mass-m0)/(m1-m0)
        return y-x
    return max(frontier,key=lambda c:(score(c),c.cross_excess,-c.ticket_count,c.bridge_mass))


def choose_geometric_bridge(trio_odds,trifecta_odds,gate):
    values=cross_market_ticket_table(trio_odds,trifecta_odds)
    p1,p2,p3=_structural_pools(trio_odds,gate)
    candidates=[]
    for f1 in _powerset(p1):
        for f2 in _powerset(p2):
            for f3 in _powerset(p3):
                ts=_tickets(f1,f2,f3)
                if not ts: continue
                if {t[0] for t in ts}!=set(f1) or {t[1] for t in ts}!=set(f2) or {t[2] for t in ts}!=set(f3):
                    continue
                q=sum(values[t]["q"] for t in ts)
                qs=sum(values[t]["q_star"] for t in ts)
                g=sum(sqrt(values[t]["q"]*values[t]["q_star"]) for t in ts)
                candidates.append(Candidate(tuple(f1),tuple(f2),tuple(f3),ts,g,q,qs,qs-q))
    return _knee(candidates)


def build_v8_2_f03(trio_odds,trifecta_odds,predicted_line_formation):
    gate=v6_1_entry_gate(trio_odds,trifecta_odds,predicted_line_formation)
    if not gate["entry_pass"]:
        return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":gate["reason"]}
    try:
        c=choose_geometric_bridge(trio_odds,trifecta_odds,gate)
    except ValueError as exc:
        return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":str(exc)}
    return {
        "scheme_version":SCHEME_VERSION,"buy":True,"reason":"PASS","formation":c.display,
        "first":c.first,"second":c.second,"third":c.third,"tickets":c.tickets,
        "ticket_count":c.ticket_count,"bridge_mass":c.bridge_mass,"q_mass":c.q_mass,
        "q_star_mass":c.q_star_mass,"cross_excess":c.cross_excess,"entry_gate":gate,
        "price_cut_enabled":False,"fixed_place_counts":False,"fixed_point_count":False,
        "nested_required":False,"formation_selector":"geometric support-value Pareto knee",
    }

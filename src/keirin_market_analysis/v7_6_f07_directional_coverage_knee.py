from __future__ import annotations

"""v7.6-F07: clean nested formation chosen by directional-core coverage per point.

The old v6 directional pre-formation is used only as a market/line-derived
reference set.  Candidate output is always one hole-free nested F1-F2-F3.
Extra closure tickets are allowed, but they contribute zero directional
coverage and still cost a ticket.  This prevents both exact-subset overpruning
and broad market-mass expansion.

No fixed place counts. No price cut. No results/payouts.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from v7_0_f01_market_hierarchy import generate_nested_tickets, implied_probabilities, v6_1_entry_gate
from v7_5_f06_directional_clean_subset import old_v6_preformation, _directional_role_scores

SCHEME_VERSION = "v7.6-F07"
PRICE_CUT_ENABLED = False
FIXED_PLACE_COUNTS = False


@dataclass(frozen=True)
class Candidate:
    first: tuple[int, ...]
    second: tuple[int, ...]
    third: tuple[int, ...]
    tickets: tuple[tuple[int, int, int], ...]
    covered_core_mass: float
    covered_core_tickets: int

    @property
    def n(self): return len(self.tickets)
    @property
    def display(self):
        s=lambda xs:"".join(map(str,sorted(xs)))
        return f"{s(self.first)}-{s(self.second)}-{s(self.third)}"


def _rank(scores,pool): return tuple(sorted(set(pool),key=lambda c:(-scores.get(c,0.0),c)))
def _union(*groups): return tuple(sorted({c for g in groups for c in g}))


def choose(trio_odds,trifecta_odds,gate):
    q=implied_probabilities(trifecta_odds)
    pre,head_pool,third_pool=old_v6_preformation(trio_odds,trifecta_odds,gate)
    if not pre: raise ValueError("empty directional core")
    cars=sorted({c for t in trifecta_odds for c in t})
    v1,v2,v3=_directional_role_scores(q,pre,cars)
    r1,r2,r3=_rank(v1,head_pool),_rank(v2,head_pool),_rank(v3,third_pool)
    candidates={}
    for k1 in range(1,len(r1)+1):
        f1=tuple(sorted(r1[:k1]))
        for k2 in range(1,len(r2)+1):
            f2=_union(f1,r2[:k2])
            for k3 in range(1,len(r3)+1):
                f3=_union(f2,r3[:k3])
                tickets=generate_nested_tickets(f1,f2,f3)
                if not tickets: continue
                key=(f1,f2,f3)
                if key in candidates: continue
                inter=[t for t in tickets if t in pre]
                covered=sum(q.get(t,0.0) for t in inter)
                candidates[key]=Candidate(f1,f2,f3,tickets,covered,len(inter))
    if not candidates: raise ValueError("no clean candidate")

    # At each cost keep the candidate that covers most q-mass of the directional core.
    by_n={}
    for c in candidates.values():
        old=by_n.get(c.n)
        if old is None or (c.covered_core_mass,c.covered_core_tickets)>(old.covered_core_mass,old.covered_core_tickets): by_n[c.n]=c

    # Strict Pareto frontier of coverage versus total ticket cost.
    frontier=[];best=-1.0
    for n in sorted(by_n):
        c=by_n[n]
        if c.covered_core_mass>best+1e-15:
            frontier.append(c);best=c.covered_core_mass
    if len(frontier)==1:return frontier[0]
    n0,n1=frontier[0].n,frontier[-1].n;m0,m1=frontier[0].covered_core_mass,frontier[-1].covered_core_mass
    if n1==n0 or m1<=m0:return frontier[0]
    def score(c):
        x=(c.n-n0)/(n1-n0);y=(c.covered_core_mass-m0)/(m1-m0)
        return y-x
    return max(frontier,key=lambda c:(score(c),-c.n,c.covered_core_mass))


def build_v7_6_f07(trio_odds,trifecta_odds,predicted_line_formation):
    gate=v6_1_entry_gate(trio_odds,trifecta_odds,predicted_line_formation)
    if not gate["entry_pass"]:return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":gate["reason"]}
    try:c=choose(trio_odds,trifecta_odds,gate)
    except ValueError as exc:return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":str(exc)}
    pre,_,_=old_v6_preformation(trio_odds,trifecta_odds,gate);q=implied_probabilities(trifecta_odds);pre_mass=sum(q.get(t,0.0) for t in pre)
    return {"scheme_version":SCHEME_VERSION,"buy":True,"reason":"PASS","formation":c.display,"first":c.first,"second":c.second,"third":c.third,"tickets":c.tickets,"ticket_count":c.n,"covered_core_mass":c.covered_core_mass,"core_mass":pre_mass,"core_mass_coverage":c.covered_core_mass/pre_mass if pre_mass else 0.0,"covered_core_tickets":c.covered_core_tickets,"core_ticket_count":len(pre),"price_cut_enabled":False,"fixed_place_counts":False,"entry_gate":gate}

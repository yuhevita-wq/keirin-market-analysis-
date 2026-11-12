from __future__ import annotations

"""v7.4-F05: nested formation from cumulative positional-support elbows.

Entry is unchanged v6.1. Candidate universes retain the A/B structure.
No place has a fixed rider count and there is no price trimming.
"""
from typing import Mapping, Sequence
from v7_0_f01_market_hierarchy import implied_probabilities, positional_support, trio_implied_probabilities, generate_nested_tickets, v6_1_entry_gate

SCHEME_VERSION="v7.4-F05"

def _rank(scores,pool): return tuple(sorted(set(pool),key=lambda c:(-scores[c],c)))
def _union(*gs): return tuple(sorted({c for g in gs for c in g}))

def _elbow(scores:Mapping[int,float],pool:Sequence[int]):
    ranked=_rank(scores,pool); vals=[max(0.0,float(scores[c])) for c in ranked]; total=sum(vals)
    if not ranked:return ()
    if total<=0:return (ranked[0],)
    cum=0.0; best=None
    for i,v in enumerate(vals,1):
        cum+=v
        concentration=cum/total-i/len(ranked)
        key=(concentration,-i)
        if best is None or key>best[0]:best=(key,i)
    return tuple(ranked[:best[1]])

def _s(trio):
    p=trio_implied_probabilities(trio); cars=sorted({c for s in trio for c in s})
    return {c:sum(x for comb,x in p.items() if c in comb)/3 for c in cars}

def build_v7_4_f05(trio,trifecta,line_text):
    gate=v6_1_entry_gate(trio,trifecta,line_text)
    if not gate["entry_pass"]:return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":gate["reason"]}
    cars=sorted({c for s in trio for c in s});q=implied_probabilities(trifecta);h1,h2,h3=positional_support(q,cars);s=_s(trio)
    A=tuple(gate["A_line"]);B=tuple(gate["B_line"]);head=tuple(sorted(set(A[:2])|set(B[:2])));third=set(A[:3])|set(B[:3]);outside=[c for c in cars if c not in A and c not in B]
    if outside:third.add(max(outside,key=lambda c:(s[c],-c)))
    r1=_elbow(h1,head);r2=_elbow(h2,head);r3=_elbow(h3,tuple(sorted(third)))
    f1=tuple(sorted(r1));f2=_union(f1,r2);f3=_union(f2,r3);tickets=generate_nested_tickets(f1,f2,f3)
    if not tickets:return {"scheme_version":SCHEME_VERSION,"buy":False,"reason":"NO_VALID_TICKETS"}
    mass=sum(q.get(t,0.0) for t in tickets)
    def show(xs):return ''.join(map(str,sorted(xs)))
    return {"scheme_version":SCHEME_VERSION,"buy":True,"reason":"PASS","formation":f"{show(f1)}-{show(f2)}-{show(f3)}","first":f1,"second":f2,"third":f3,"tickets":tickets,"ticket_count":len(tickets),"market_mass":mass,"price_cut_enabled":False,"fixed_place_counts":False,"entry_gate":gate}

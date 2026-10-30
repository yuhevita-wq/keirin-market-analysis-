#!/usr/bin/env python3
"""EXPERIMENTAL independent V6-C1 purchase admission gate; odds and races only.

Freeze original V6 and V6-C1; evaluate explicit single/multi blocks from manual-based formation engine.
Do not choose a threshold from realized profit on Feb-Apr 2023.
Only propose a purchase when BOTH frozen models support *conditional finishing
order* mispricing inside each unordered finishing-three set, while broad
3renpuku (unordered trio) probabilities remain approximately market-consistent.
This is a falsifiable pre-specified hypothesis, not a proven market edge.
The rule allows some ticket-level negative EV inside a formation.

NO result, winning payout, or market odds appears in model training.
All source prices must be complete, positive, same-race, same-cars,
with both 3連単 and 3連複. Skip on any invalid component.
"""
from __future__ import annotations
import itertools,math
from collections import defaultdict

TRIO_MIN_RATIO=.75
TRIO_MAX_RATIO=1.35
ORDER_MIN_MODEL_OVER_MARKET=1.5
MIN_SUPPORTED_TICKETS=2
MIN_DISTINCT_SUPPORTED_TRIOS=2
MIN_CROSS_MODEL_FLAT_ROI=1.20

def normalize_market(three,unordered):
    if not three or not unordered:raise ValueError("Missing market")
    cars={c for t in three for c in t}
    if len(cars)<5 or len(cars)>9:raise ValueError("Invalid car count")
    all_ordered=set(itertools.permutations(cars,3))
    if set(three)!=all_ordered:raise ValueError("Incomplete trifecta prices")
    all_trios=set(itertools.combinations(sorted(cars,key=int),3))
    if set(unordered)!=all_trios:raise ValueError("Incomplete trio prices")
    if not all(math.isfinite(p) and p>1 for p in list(three.values())+list(unordered.values())):
        raise ValueError("Invalid odds")
    inv_t={t:1/p for t,p in three.items()}
    inv_u={k:1/p for k,p in unordered.items()}
    total_u=sum(inv_u.values())
    q_u={k:v/total_u for k,v in inv_u.items()}
    cond_den=defaultdict(float)
    for t,v in inv_t.items():
        cond_den[tuple(sorted(t,key=int))]+=v
    return q_u,{t:inv_t[t]/cond_den[tuple(sorted(t,key=int))] for t in inv_t}

def decide(formation,p_v6,p_c1,three,unordered):
    """Evaluate one already-frozen candidate formation. No outcome input accepted.

    formation: original select() result: tickets=list[str] and points=int.
    p_v6/p_c1: full order-triple distributions summing to one.
    three: dict[(car,car,car)]->final odds
    unordered: dict[(car,car,car) sorted]->final 3renpuku odds

    Returns (adopt:bool, diagnostics:dict). Skips when market grid invalid.
    """
    if not formation:return False,{"reason":"no_original_candidate"}
    if set(p_v6)!=set(three) or set(p_c1)!=set(three):
        return False,{"reason":"prediction_market_universe_mismatch"}
    if abs(sum(p_v6.values())-1)>1e-8 or abs(sum(p_c1.values())-1)>1e-8:
        return False,{"reason":"probability_not_normalized"}
    # Whole-purchase audit: one or several independent Cartesian blocks,
    # with duplicate 3連単 outcomes charged ONCE (manual p.33).
    blocks=formation.get("blocks") or [formation]
    expected=set()
    for block in blocks:
        if not all(col in block for col in ("first","second","third")):
            return False,{"reason":"missing_formation_block"}
        expected|={"-".join(x) for x in itertools.product(
            block["first"],block["second"],block["third"]) if len(set(x))==3}
    if set(formation.get("tickets",[]))!=expected:
        return False,{"reason":"formation_incomplete"}
    if len(formation["tickets"])!=len(expected):
        return False,{"reason":"duplicate_formation_ticket"}
    tickets={tuple(k.split("-")) for k in formation["tickets"]}
    if len(tickets)!=formation["points"]:
        return False,{"reason":"point_count_mismatch"}
    try:
        q_trio,q_order=normalize_market(three,unordered)
    except (TypeError,ValueError) as exc:
        return False,{"reason":"invalid_full_odds","details":str(exc)}
    p_trio={key:defaultdict(float) for key in ("v6","c1")}
    for tag,dist in (("v6",p_v6),("c1",p_c1)):
        for t,v in dist.items():
            p_trio[tag][tuple(sorted(t,key=int))]+=v
    flat_v6=sum(p_v6[t]*three[t] for t in tickets)/len(tickets)
    flat_c1=sum(p_c1[t]*three[t] for t in tickets)/len(tickets)
    supported=[]
    near_consensus=[]
    for t in tickets:
        u=tuple(sorted(t,key=int))
        model_u6=p_trio["v6"][u];model_uc=p_trio["c1"][u]
        q_u=q_trio[u]
        ratio6=model_u6/q_u; ratioc=model_uc/q_u
        if not(TRIO_MIN_RATIO<=ratio6<=TRIO_MAX_RATIO and TRIO_MIN_RATIO<=ratioc<=TRIO_MAX_RATIO):
            continue
        near_consensus.append(t)
        market_cond=q_order[t]
        cond6=(p_v6[t]/model_u6)/market_cond
        condc=(p_c1[t]/model_uc)/market_cond
        if min(cond6,condc)>=ORDER_MIN_MODEL_OVER_MARKET:
            supported.append((t,round(cond6,4),round(condc,4)))
    ndistinct=len({tuple(sorted(t,key=int)) for t,_,_ in supported})
    adopt=(flat_v6>=MIN_CROSS_MODEL_FLAT_ROI and flat_c1>=MIN_CROSS_MODEL_FLAT_ROI
           and len(supported)>=MIN_SUPPORTED_TICKETS and ndistinct>=MIN_DISTINCT_SUPPORTED_TRIOS)
    diag={"reason":"robust_two_pool_order_disagreement" if adopt else "insufficient_robust_market_disagreement",
          "supported_tickets":len(supported),
          "distinct_supported_trios":ndistinct,
          "trio_consensus_tickets":len(near_consensus),
          "full_formation_points":len(tickets),
          "flat_roi_v6_estimate":round(flat_v6,5),
          "flat_roi_c1_estimate":round(flat_c1,5),
          "minimum_model_agreement_roi_threshold":MIN_CROSS_MODEL_FLAT_ROI,
          "support_examples":[{"ticket":"-".join(t),"order_ratio_v6":a,"order_ratio_c1":b}
                              for t,a,b in sorted(supported)[:5]]}
    return adopt,diag

def _test():
    cars=tuple("12345")
    triples=list(itertools.permutations(cars,3))
    trios=list(itertools.combinations(cars,3))
    base={t:1/len(triples) for t in triples}
    tri_odds={t:80. for t in triples}
    trio_odds={u:15. for u in trios}
    form={"first":["1","2"],"second":["1","2"],"third":["3","4"],
          "tickets":["1-2-3","1-2-4","2-1-3","2-1-4"],"points":4}
    # Uniform model matches uniform market conditional -> NO admission.
    adopt,detail=decide(form,base,base,tri_odds,trio_odds)
    assert not adopt and detail["supported_tickets"]==0,detail
    # At least two different trios need support; an implausible isolated
    # longshot cannot pass the decision gate by high P*O alone.
    q=dict(base)
    q[("1","2","3")]*=2
    z=sum(q.values());q={t:p/z for t,p in q.items()}
    adopt,detail=decide(form,q,q,tri_odds,trio_odds)
    assert not adopt and detail["distinct_supported_trios"]<=1,detail
    assert decide(form,base,base,{},trio_odds)[0] is False
    assert decide(form,base,base,tri_odds,{})[0] is False
    multi={"blocks":[{"first":["1","2"],"second":["1","2"],"third":["3","4"]},
                     {"first":["1"],"second":["3","4"],"third":["2"]}],
           "tickets":["1-2-3","1-2-4","2-1-3","2-1-4","1-3-2","1-4-2"],
           "points":6}
    no,diag=decide(multi,base,base,tri_odds,trio_odds)
    assert not no and diag["full_formation_points"]==6,diag
    bad=dict(multi);bad["points"]=7
    assert decide(bad,base,base,tri_odds,trio_odds)[1]["reason"]=="point_count_mismatch"
    print("MANUAL_MULTIBLOCK_GATE_DEDUP_AUDIT_AND_MISPRICING_TEST_PASS")

if __name__=="__main__":_test()

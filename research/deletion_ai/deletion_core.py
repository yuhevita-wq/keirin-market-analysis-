"""2023-from-zero keirin deletion learners. No inherited trained weights.

Each learner starts with EVERY 3連単 and 3連複 ticket, ranks tickets by
estimated payout-per-yen, and REMOVES the least promising tickets until its
precommitted probability-coverage constraint would be violated.

Features are observable ODDS ONLY. Winners are labels after pre-result seal.
No rider strength stories, head/box templates, or arbitrary ticket count caps.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np

BRANCHES=("OLD-R","NEW-R","V3-R","V3-ME-R")
MARKETS=("3連単","3連複")
FEATURES=(
    "relative_log_price_popularity",
    "within_market_log_odds_rank",
    "cross_market_price_gap",
    "within_trio_order_imbalance",
    "price_gap_x_popularity",
    "order_imbalance_x_rank",
    "relative_trio_order_concentration",
    "trio_order_concentration_x_price_gap",
)
# Different inductive biases, but ALL zero-initialized and independently trained.
ACTIVE={
    "OLD-R":(0,1),
    "NEW-R":(0,1,4),
    "V3-R":(0,1,2,3,4,5),
    "V3-ME-R":tuple(range(len(FEATURES))),
}
COVERAGES=(0.25,0.4,0.55,0.7,0.8,0.9,0.95,1.0)
LR=0.017
RIDGE_STEP=0.00035
MAX_WEIGHT=1.5
UNIT=100


def stable_hash(obj):
    return hashlib.sha256(json.dumps(
        obj,sort_keys=True,ensure_ascii=False,separators=(",",":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def softmax(a):
    a=np.asarray(a,dtype=np.float64)
    z=np.exp(a-np.max(a))
    return z/z.sum()


def rank_feature(probabilities):
    """Race-local odds rank, tie averaged, no outcome information."""
    p=np.asarray(probabilities,dtype=np.float64)
    levels=np.unique(p)
    positions=np.searchsorted(levels,p)
    return 2.0*positions/max(len(levels)-1,1)-1.0


def books(pre):
    """Full-market ordered and unordered ticket alignment & covariates."""
    odds=pre["odds"]
    entries=pre["entries"]
    cars=tuple(sorted(int(c) for c in entries))
    if not 5<=len(cars)<=9:raise ValueError("Outside 5..9 cars")
    keys3=tuple(sorted(odds["3連単"],key=lambda x:tuple(map(int,x.split("-")))))
    keys2=tuple(sorted(odds["3連複"],key=lambda x:tuple(map(int,x.split("=")))))
    expected3={"-".join(map(str,v)) for v in itertools.permutations(cars,3)}
    expected2={"=".join(map(str,v)) for v in itertools.combinations(cars,3)}
    if set(keys3)!=expected3 or set(keys2)!=expected2:
        raise ValueError("Incomplete actual odds book")
    price3=np.array([float(odds["3連単"][k]) for k in keys3])
    price2=np.array([float(odds["3連複"][k]) for k in keys2])
    if not(np.all(np.isfinite(price3)) and np.all(np.isfinite(price2)) and
           np.min(price3)>=1 and np.min(price2)>=1):
        raise ValueError("Invalid price")
    q3=softmax(-np.log(price3))
    q2=softmax(-np.log(price2))
    tix={key:i for i,key in enumerate(keys2)}
    group=np.array([tix["=".join(sorted(k.split("-"),key=int))] for k in keys3])
    qsum=np.bincount(group,weights=q3,minlength=len(keys2))
    if np.min(qsum)<=0:raise AssertionError("Missing unordered permutation group")
    pgap=np.log(np.maximum(q2,1e-100)/np.maximum(qsum,1e-100))
    order=np.log(np.maximum(q3,1e-100)/(qsum[group]/6.0))
    imbalance=np.zeros(len(keys2))
    for j in range(len(keys2)):
        mask=group==j
        if np.sum(mask)!=6:raise AssertionError("Three riders must have six permutations")
        imbalance[j]=math.log(max(q3[mask]/qsum[j])*6.0)
    gap3=pgap[group]
    gap2=-pgap
    conc3=imbalance[group]
    conc2=imbalance
    p3=np.clip(np.log(q3*len(q3)),-5,5)
    p2=np.clip(np.log(q2*len(q2)),-5,5)
    r3=rank_feature(q3)
    r2=rank_feature(q2)
    # Trio entry has no within-six order, but does have the observed concentration.
    order3=np.clip(order,-4,4)
    order2=np.clip(imbalance,-4,4)
    x3=np.column_stack((p3,r3,np.clip(gap3,-4,4),order3,
         np.clip(p3*gap3,-5,5),np.clip(r3*order3,-4,4),
         np.clip(conc3,-4,4),np.clip(conc3*gap3,-5,5)))
    x2=np.column_stack((p2,r2,np.clip(gap2,-4,4),order2,
         np.clip(p2*gap2,-5,5),np.clip(r2*order2,-4,4),
         np.clip(conc2,-4,4),np.clip(conc2*gap2,-5,5)))
    return {
        "3連単":{"keys":keys3,"prices":price3,"prior":q3,"x":x3},
        "3連複":{"keys":keys2,"prices":price2,"prior":q2,"x":x2},
        "cars":cars,
    }


def new_weights():
    return {name:{market:np.zeros(len(FEATURES),dtype=np.float64)
                  for market in MARKETS} for name in BRANCHES}


def probabilities(book,weight,branch):
    keep=np.zeros(len(FEATURES),dtype=np.float64)
    for k in ACTIVE[branch]:keep[k]=float(weight[k])
    prior=book["prior"]
    return softmax(np.log(np.maximum(prior,1e-100))+book["x"]@keep)


def full_sheet_score(book_map,finish):
    """Buy all tickets at JPY100; 1 hit in each market."""
    t="-".join(map(str,finish))
    c="=".join(map(str,sorted(finish)))
    spent=sum(len(book_map[m]["keys"]) for m in MARKETS)*UNIT
    paid=int(round(UNIT*(
         float(book_map["3連単"]["prices"][book_map["3連単"]["keys"].index(t)])+
         float(book_map["3連複"]["prices"][book_map["3連複"]["keys"].index(c)]))))
    return {"stake_yen":spent,"payout_yen":paid,"net_yen":paid-spent,
            "total_marks":spent//UNIT,"winning_marks":2}


def deletion_plan(book,prob,coverage):
    """Begin with EVERY ticket and remove one at a time by weakest model EV.

    In the exact-market-baseline (all equal EV), do not feign an edge:
    keep every ticket. No fixed mark count; the coverage level is a separate
    2023-validation policy parameter.
    """
    keys=book["keys"]
    odds=book["prices"]
    p=np.asarray(prob,dtype=np.float64)
    if not 0<coverage<=1:raise ValueError("Coverage out of range")
    if not math.isclose(float(sum(p)),1.,rel_tol=0,abs_tol=1e-8):
        raise AssertionError("Invalid probability mass")
    multiplier=p*odds
    if float(np.max(multiplier)-np.min(multiplier))<1e-8:
        return {"kept":list(keys),"deleted":[],"prob_mass_kept":1.,
                "removed_value_ev_sum":0., "no_evidence_no_deletion":True}
    # Deterministic, market-by-market elimination; no favorite/formation axis.
    order=np.argsort(multiplier,kind="stable")
    is_kept=np.ones(len(keys),dtype=bool)
    retained=1.
    removed_value=0.
    for i in order:
        q=float(p[i])
        if retained-q+1e-12<coverage:
            continue
        if np.sum(is_kept)<=1:
            break
        is_kept[i]=False
        retained-=q
        removed_value+=float(multiplier[i]-1.)
    keep=[k for i,k in enumerate(keys) if is_kept[i]]
    deleted=[k for i,k in enumerate(keys) if not is_kept[i]]
    assert keep and len(keep)+len(deleted)==len(keys)
    return {"kept":keep,"deleted":deleted,
            "prob_mass_kept":float(retained),
            "removed_value_ev_sum":float(removed_value),
            "no_evidence_no_deletion":False}


def evaluate_plan(books_dict,plans,finish):
    """Paired whole-sheet comparison: savings and forfeited real payouts."""
    wins={"3連単":"-".join(map(str,finish)),
          "3連複":"=".join(map(str,sorted(finish)))}
    full=full_sheet_score(books_dict,finish)
    retained=[];excluded_hits=0;forfeited=0
    for market in MARKETS:
        book=books_dict[market]
        plan=plans[market]
        win=wins[market]
        assert win in book["keys"]
        kept=plan["kept"]
        assert len(set(kept))==len(kept)
        assert set(plan["deleted"])==set(book["keys"])-set(kept)
        assert len(set(kept))==len(kept) and len(kept)>0
        retained.extend((market,t) for t in kept)
        if win not in kept:
            excluded_hits+=1
            price=float(book["prices"][book["keys"].index(win)])
            forfeited+=int(round(UNIT*price))
    stake=len(retained)*UNIT
    paid=full["payout_yen"]-forfeited
    saved=full["stake_yen"]-stake
    net=paid-stake
    assert net-full["net_yen"]==saved-forfeited
    return {"stake_yen":stake,"payout_yen":paid,"net_yen":net,
            "retained_marks":len(retained),"deleted_marks":full["total_marks"]-len(retained),
            "saved_stake_yen":saved,"forfeited_payout_yen":forfeited,
            "deleted_winning_tickets":excluded_hits,
            "retained_winning_tickets":2-excluded_hits,
            "delta_net_vs_all_marks":saved-forfeited}


def update(weights,books_dict,branch,finish):
    """Outcome only AFTER the pre-result decision was sealed.

    Full-sheet supervision: each market's winner vs ALL losing ticket outcomes.
    Loss is market log loss, not arbitrarily penalizing every losing mark
    with an extra incorrect independent binary classification term.
    """
    truth={"3連単":"-".join(map(str,finish)),
           "3連複":"=".join(map(str,sorted(finish)))}
    changes={}
    for market in MARKETS:
        book=books_dict[market]
        q=probabilities(book,weights[branch][market],branch)
        idx=book["keys"].index(truth[market])
        grad=book["x"][idx]-q@book["x"]
        theta=weights[branch][market]
        for k in ACTIVE[branch]:
            theta[k]=(1.-RIDGE_STEP)*theta[k]+LR*float(grad[k])
            theta[k]=np.clip(theta[k],-MAX_WEIGHT,MAX_WEIGHT)
        assert np.all(np.isfinite(theta))
        changes[market]={"observed_logloss":-math.log(max(float(q[idx]),1e-50)),
                         "outcome":truth[market]}
    return changes


def model_digest(weights,branch):
    return hashlib.sha256(b"".join(weights[branch][m].astype("<f8").tobytes()
         for m in MARKETS)).hexdigest()

"""GEN-008 V3: market-aware, coherent, deterministic portfolio candidate.

V3 does not gamble via random enter/stop or seed choices. It uses the frozen
NEW predictor, learns a *single* market/predictor blending parameter on
Sep 15-22 (2023), audits it on Sep 23-30, then locks before November.

For every finish order w with probability p(w), and 100-yen increment x,
choose the increment which maximizes:
  sum_w p(w) log((wealth_w + x * [payoff_w - 1]) / wealth_w)
Stop when every feasible increment has non-positive expected log growth.
3連単 and 3連複 are jointly optimized over the SAME possible finish orders.

No result is used before a decision is sealed. No October outcomes used in
fitting, parameter selection or V3 portfolio construction.
Historical FINAL odds, not real-time price.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np

from train_week import feature_matrix
from gen008_experiment import coherent_probabilities, model_digest

MARKETS=("3連単","3連複")
UNIT_YEN=100
BANK_YEN=10000
ALPHAS=tuple(i/10 for i in range(11))
TRAIN_FIRST,TRAIN_LAST="2023-09-15","2023-09-22"
VALID_FIRST,VALID_LAST="2023-09-23","2023-09-30"
TEST_FIRST,TEST_LAST="2023-11-01","2023-11-14"


def matrix(pre, weights, temperature):
    """Return normalized ordered probabilities and market price probabilities."""
    arms=feature_matrix(pre["entries"],pre["odds"])
    _,q_model=coherent_probabilities(arms,weights,temperature)
    keys=tuple(q_model["3連単"])
    model=np.asarray([q_model["3連単"][k] for k in keys],dtype=float)
    price=np.asarray([float(pre["odds"]["3連単"][k]) for k in keys])
    market_raw=1/price
    market=market_raw/market_raw.sum()
    if np.any(model<=0) or abs(model.sum()-1)>1e-7:
        raise ValueError("Bad predictor mass")
    return keys,model,market


def blend(model,market,alpha):
    if not (0<=alpha<=1):
        raise ValueError("alpha outside [0,1]")
    logits=alpha*np.log(np.maximum(model,1e-100))+(1-alpha)*np.log(np.maximum(market,1e-100))
    logits-=logits.max()
    exp=np.exp(logits)
    return exp/exp.sum()


def trio_probabilities(keys,q):
    out={}
    for k,p in zip(keys,q):
        combo="=".join(sorted(k.split("-"),key=int))
        out[combo]=out.get(combo,0.)+float(p)
    if abs(sum(out.values())-1)>1e-8:raise AssertionError("unordered distribution mass")
    return out


def score(keys,q,outcome):
    win="-".join(map(str,outcome))
    tri="=".join(map(str,sorted(outcome)))
    j=keys.index(win)
    trio=trio_probabilities(keys,q)
    return {"trifecta_logloss":-math.log(max(float(q[j]),1e-30)),
            "trio_logloss":-math.log(max(trio[tri],1e-30))}


def prepare_features(rows,weights,temperature):
    """Gather predictions independent of the actual finish."""
    items=[]
    for row in rows:
        pre={"race_id":row["race_id"],"race":row["race"],
             "entries":row["entries"],"odds":row["odds"]}
        keys,m,q=matrix(pre,weights,temperature)
        outcome=tuple(row["outcome"])
        win="-".join(map(str,outcome))
        if win not in keys:raise ValueError("Missing winning ordered tuple")
        ix=keys.index(win)
        trio_target="=".join(map(str,sorted(outcome)))
        trio_ix=np.asarray([i for i,k in enumerate(keys)
                       if "=".join(sorted(k.split("-"),key=int))==trio_target])
        items.append((m,q,ix,trio_ix))
    return items


def calibration_loss(cache,alpha):
    losses=np.zeros(2,dtype=float)
    for model,market,ordered,trio in cache:
        q=blend(model,market,alpha)
        losses+=(-math.log(max(float(q[ordered]),1e-30)),
                 -math.log(max(float(q[trio].sum()),1e-30)))
    return {"trifecta":float(losses[0]/len(cache)),
            "trio":float(losses[1]/len(cache)),
            "mean":float(losses.sum()/(2*len(cache)))}


def calibrate(train,validation,weights,temperature):
    if not train or not validation:raise ValueError("Empty train/validation calibration period")
    assert all(TRAIN_FIRST<=r["race"]["race_date"]<=TRAIN_LAST for r in train)
    assert all(VALID_FIRST<=r["race"]["race_date"]<=VALID_LAST for r in validation)
    train_ids={r["race_id"] for r in train}
    assert not train_ids.intersection(r["race_id"] for r in validation)
    tt=prepare_features(train,weights,temperature)
    vv=prepare_features(validation,weights,temperature)
    table=[{"alpha":a,"calibration_train":calibration_loss(tt,a),
            "chronological_validation":calibration_loss(vv,a)} for a in ALPHAS]
    best=min(table,key=lambda z:(z["calibration_train"]["mean"],z["alpha"]))
    return {"alpha":best["alpha"],"alpha_selection_rule":"minimize both-market mean log loss on Sep 15-22 only",
            "selected_train":best["calibration_train"],
            "selected_validation_no_retuning":best["chronological_validation"],
            "all_candidate_scores":table,
            "train_races":len(train),"validation_races":len(validation),
            "train_dates":[TRAIN_FIRST,TRAIN_LAST],
            "validation_dates":[VALID_FIRST,VALID_LAST],
            "trained_predictor_weights":False,
            "train_race_ids_sha256":hashlib.sha256("\n".join(r["race_id"] for r in train).encode()).hexdigest(),
            "validation_race_ids_sha256":hashlib.sha256("\n".join(r["race_id"] for r in validation).encode()).hexdigest()}


def joint_portfolio(pre,weights,temperature,alpha,bank=BANK_YEN,unit=UNIT_YEN):
    """Discrete coordinate search on an exact 3連単 outcome lattice.

    Every 100-yen increase must strictly improve the same expected-log
    objective over all final results. No fixed mark count, formation, bet type,
    favorite axis, arbitrary ROI or odds threshold.
    """
    keys,model,market=matrix(pre,weights,temperature)
    probs=blend(model,market,alpha)
    key_index={k:i for i,k in enumerate(keys)}
    unordered=trio_probabilities(keys,probs)
    trio_keys=sorted(unordered, key=lambda s:tuple(map(int,s.split("="))))
    tri_indices=np.asarray([
         [i for i,k in enumerate(keys)
          if "=".join(sorted(k.split("-"),key=int))==tri]
         for tri in trio_keys],dtype=np.int32)
    if any(len(x)!=6 for x in tri_indices):
        raise AssertionError("Trio does not map to six ordered results")
    odds3=np.asarray([float(pre["odds"]["3連単"][k]) for k in keys])
    odds2=np.asarray([float(pre["odds"]["3連複"][k]) for k in trio_keys])
    if not np.all(np.isfinite(odds3)) or not np.all(np.isfinite(odds2)):
        raise ValueError("Non-finite odds")
    wealth=np.full(len(keys),float(bank),dtype=float)
    allocations={}
    total=0
    trace=[]
    # Each increment is admitted ONLY if its exact expected marginal log
    # utility is positive. Prevent zero/negative wealth for any outcome.
    max_steps=(bank-unit)//unit
    for _ in range(max_steps):
        if min(wealth)<=unit:
            break
        spend=wealth-unit
        base=float(probs@np.log(spend/wealth))
        d3=base+probs*np.log((spend+unit*odds3)/spend)
        win_wealth=spend[tri_indices]
        d2=base+np.sum(probs[tri_indices]*np.log(
            (win_wealth+unit*odds2[:,None])/win_wealth),axis=1)
        i3=int(np.argmax(d3))
        i2=int(np.argmax(d2))
        if float(d3[i3])>=float(d2[i2]):
            market_name,key,odd,winning,delta="3連単",keys[i3],float(odds3[i3]),(i3,),float(d3[i3])
        else:
            market_name,key,odd,winning,delta="3連複",trio_keys[i2],float(odds2[i2]),tuple(int(x) for x in tri_indices[i2]),float(d2[i2])
        if delta<=1e-12:
            break
        allocations[(market_name,key)]=allocations.get((market_name,key),0)+unit
        wealth-=unit
        wealth[list(winning)]+=unit*odd
        total+=unit
        trace.append({"market":market_name,"ticket":key,"price":odd,
                      "marginal_expected_log_gain":delta})
    marks=[{"type":market,"selection":selection,
            "stake_yen":amount,"odds":float(pre["odds"][market][selection]),
            "estimated_win_probability":float(
                probs[key_index[selection]] if market=="3連単" else unordered[selection]),
            "ev_per_yen":float(
                (probs[key_index[selection]] if market=="3連単" else unordered[selection])*
                float(pre["odds"][market][selection])-1)}
            for (market,selection),amount in sorted(allocations.items())]
    assert sum(x["stake_yen"] for x in marks)==total
    assert total<bank and np.min(wealth)>0
    assert not marks or trace[-1]["marginal_expected_log_gain"]>0
    return {"branch":"v3","race_id":pre["race_id"],
            "predictor_sha256":model_digest(weights),
            "alpha":float(alpha),"bank_yen":bank,"min_stake_increment_yen":unit,
            "selected":marks,"actual_paper_stake_yen":total,
            "actual_paper_marks":total//unit,
            "market_probability_sums":{"3連単":float(probs.sum()),"3連複":float(sum(unordered.values()))},
            "last_positive_increment_expected_log_gain":trace[-1]["marginal_expected_log_gain"] if trace else None,
            "stopped_by":"all remaining increments nonpositive" if len(trace)<max_steps else "bankroll solvency constraint",
            "no_fixed_ticket_cap":True,
            "no_result_used":True}


def settle_v3(chosen,outcome):
    ordered="-".join(map(str,outcome))
    unordered="=".join(map(str,sorted(outcome)))
    stake=int(chosen["actual_paper_stake_yen"])
    payout=int(round(sum(
        item["stake_yen"]*item["odds"]
        for item in chosen["selected"]
        if (item["type"]=="3連単" and item["selection"]==ordered) or
           (item["type"]=="3連複" and item["selection"]==unordered))))
    return {"stake_yen":stake,"payout_yen":payout,"net_yen":payout-stake,
            "marks":int(chosen["actual_paper_marks"]),
            "hit":bool(payout>0)}


def fit_signature(checkpoint,alpha):
    payload={"parent_new_prediction":checkpoint["weights_sha256"],
             "parent_new_policy":checkpoint["policy_sha256"],
             "parent_new_temperature":checkpoint["temperature"],
             "alpha":alpha,
             "decision_policy":"coherent geometric blend, marginal expected log portfolio, integer 100-yen increments",
             "schema":1}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"id":"GEN-008-V3","algorithm_digest":digest,**payload}

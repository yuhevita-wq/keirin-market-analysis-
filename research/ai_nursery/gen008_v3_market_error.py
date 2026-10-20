"""GEN-008 V3-ME: empirically learn repeatable market-pricing residuals.

Observed finish orders are the SUPERVISION for relative errors in the FINAL
3連単 and 3連複 market books. Never identify mispricing merely because
a market gives an outsider a high price; the conditional corrections are fit
to realized orders by regularized multinomial maximum likelihood.

Dataset:
  Fit: 2023-09-15..2023-10-07 (Oct is formerly examined, now *development*).
  Diagnostics only: 2023-10-08..2023-10-14 (formerly examined).
  New untouched forward exam: 2023-12-01..2023-12-14.

The model learns from outcome-vs-market disagreement; no rules for rider
strength, favorite axes, fixed bet types, or fixed formation templates.
The original GEN-008 V3 remains unmodified.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from gen008_v3 import matrix,blend,trio_probabilities

FIRST_FIT,LAST_FIT="2023-09-15","2023-10-07"
FIRST_DIAG,LAST_DIAG="2023-10-08","2023-10-14"
FIRST_TEST,LAST_TEST="2023-12-01","2023-12-14"
FEATURE_NAMES=(
 "trio_vs_trifecta_log_market_gap",
 "frozen_NEW_vs_trifecta_market_log_gap",
 "trifecta_relative_log_popularity",
 "within_trio_order_log_concentration",
 "trio_gap_x_relative_popularity",
)
RIDGE=5.0
MAX_ABS_COEFF=1.5
BANK=10000
TICKET_UNIT=100
MARKETS=("3連単","3連複")


def market_features(pre,weights,temp):
    """The only inputs are pre-result 3連単/3連複 quotes + frozen predictor.

    q3 is normalized inverse odds. qT is normalized inverse unordered odds.
    A result labels a winning order only AFTER features are constructed.
    """
    keys,q_model,q3=matrix(pre,weights,temp)
    trio_names=tuple(sorted(pre["odds"]["3連複"],key=lambda c:tuple(int(i) for i in c.split("="))))
    tprice=np.asarray([float(pre["odds"]["3連複"][k]) for k in trio_names])
    invt=1/tprice
    qt=invt/invt.sum()
    tindex={c:i for i,c in enumerate(trio_names)}
    tgroup=np.asarray([tindex["=".join(sorted(key.split("-"),key=int))] for key in keys],dtype=int)
    q3set=np.bincount(tgroup,weights=q3,minlength=len(trio_names))
    if any(q3set<=0) or len(q3)!=len(keys):raise AssertionError("invalid ordered/unordered book")
    dis=np.log(qt[tgroup]/q3set[tgroup])
    model=np.log(np.maximum(q_model,1e-100)/np.maximum(q3,1e-100))
    rank=np.log(q3*len(keys))
    order=np.log(q3/(q3set[tgroup]/6))
    dis_rank=dis*rank
    x=np.column_stack([
       np.clip(dis,-4,4),np.clip(model,-4,4),
       np.clip(rank,-4,4),np.clip(order,-4,4),
       np.clip(dis_rank,-4,4)
    ])
    x=np.asarray(x,dtype=np.float64)
    if not np.all(np.isfinite(x)):raise ValueError("Non-finite market residual feature")
    assert abs(q3.sum()-1)<1e-8 and abs(qt.sum()-1)<1e-8
    return {"keys":keys,"baseline":q3,"x":x,"trio_names":trio_names,
            "trio_group":tgroup,"new":q_model,
            "trio_price":tprice}


def outcome_index(features,finish):
    key="-".join(map(str,finish))
    try:return features["keys"].index(key)
    except ValueError:raise ValueError("Winning ordered result absent from full final book")


def corrected_probability(feat,coef):
    theta=np.asarray(coef,dtype=np.float64)
    if theta.shape!=(len(FEATURE_NAMES),):raise ValueError("Wrong coefficient dimension")
    logits=np.log(np.maximum(feat["baseline"],1e-100))+feat["x"]@theta
    logits-=logsumexp(logits)
    probability=np.exp(logits)
    assert abs(float(probability.sum())-1)<1e-8
    return probability


def gather(rows,weights,temp):
    obj=[]
    for row in rows:
        pre={"race_id":row["race_id"],"race":row["race"],
             "entries":row["entries"],"odds":row["odds"]}
        f=market_features(pre,weights,temp)
        win=outcome_index(f,row["outcome"])
        obj.append((f,win))
    return obj


def loss_race(feat,idx,coef):
    p=corrected_probability(feat,coef)
    return -math.log(max(float(p[idx]),1e-50)),p


def loss_and_grad(theta,examples):
    """Negative true-result loglikelihood relative to observed public market.

    Sum log score over races; ridge shrinks corrections toward honest market
    odds, rather than forcing discovered positive EV.
    """
    value=0.
    grad=np.zeros(len(FEATURE_NAMES),dtype=float)
    for f,index in examples:
        logits=np.log(np.maximum(f["baseline"],1e-100))+f["x"]@theta
        z=logsumexp(logits)
        q=np.exp(logits-z)
        value+=z-logits[index]
        grad+=q@f["x"]-f["x"][index]
    value+=0.5*RIDGE*float(theta@theta)
    grad+=RIDGE*theta
    return value,grad


def fit(rows,weights,temp):
    if len(rows)<100:raise ValueError("Insufficient independent historical results to identify market mistakes")
    assert all(FIRST_FIT<=r["race"]["race_date"]<=LAST_FIT for r in rows)
    inputs=gather(rows,weights,temp)
    m=len(FEATURE_NAMES)
    outcome=minimize(
       fun=lambda z:loss_and_grad(z,inputs),
       x0=np.zeros(m,dtype=float),jac=True,method="L-BFGS-B",
       bounds=[(-MAX_ABS_COEFF,MAX_ABS_COEFF)]*m,
       options={"maxiter":130,"ftol":1e-10,"gtol":1e-7}
    )
    if not outcome.success and float(np.linalg.norm(outcome.jac))>0.015:
        raise RuntimeError("Market-residual learning did not converge: "+outcome.message)
    coefs=[float(v) for v in outcome.x]
    before=loss_and_grad(np.zeros(m),inputs)[0]/len(inputs)
    after=loss_and_grad(np.asarray(coefs),inputs)[0]/len(inputs)
    return {"model":"GEN-008-V3-ME","status":"FIT_COMPLETE",
            "feature_names":list(FEATURE_NAMES),"coefficients":coefs,
            "ridge":RIDGE,"coefficient_bounds":[-MAX_ABS_COEFF,MAX_ABS_COEFF],
            "optimizer":"L-BFGS-B analytical gradient","optimizer_success":bool(outcome.success),
            "optimizer_message":str(outcome.message),
            "train_races":len(rows),"train_dates":[FIRST_FIT,LAST_FIT],
            "train_market_logloss_per_race":before,
            "train_fitted_penalized_logloss_per_race":after,
            "race_ids_sha256":hashlib.sha256("\n".join(r["race_id"] for r in rows).encode()).hexdigest(),
            "model_uses_real_finish_labels_to_fit":True,
            "market_inputs":["3連単 final full price grid","3連複 final full price grid",
                             "frozen NEW model probabilities"],
            "no_october_holdout_claim":True,
            "future_december_labels_used_in_fit":False}


def logloss_report(rows,coefs,weights,temp,initial_v3_alpha=.2):
    examples=gather(rows,weights,temp)
    totals={"market_trifecta":0.,"market_trio":0.,
            "frozen_v3_trifecta":0.,"frozen_v3_trio":0.,
            "corrected_trifecta":0.,"corrected_trio":0.}
    gains=[]
    totals_positive=0
    for f,ix in examples:
        p0=f["baseline"]
        pv=blend(f["new"],p0,initial_v3_alpha)
        pc=corrected_probability(f,coefs)
        target=f["trio_group"][ix]
        for prefix,p in (("market",p0),("frozen_v3",pv),("corrected",pc)):
            totals[prefix+"_trifecta"]-=math.log(max(float(p[ix]),1e-50))
            totals[prefix+"_trio"]-=math.log(max(float(np.bincount(f["trio_group"],weights=p,minlength=len(f["trio_names"]))[target]),1e-50))
        gains.append(math.log(max(float(pc[ix]),1e-50)/max(float(p0[ix]),1e-50)))
        if float(pc[ix]/max(p0[ix],1e-50))>1:totals_positive+=1
    return {"races":len(examples),
            "mean_logloss":{k:v/len(examples) for k,v in totals.items()},
            "corrected_winner_loglik_gain_over_market_per_race":sum(gains)/len(gains),
            "true_winners_given_greater_probability_than_market":totals_positive,
            "no_refit_on_this_period":True}


def best_expected_value_stats(pre,f,prob):
    odds3=np.asarray([float(pre["odds"]["3連単"][k]) for k in f["keys"]])
    probs_trio=np.bincount(f["trio_group"],weights=prob,minlength=len(f["trio_names"]))
    ev3=prob*odds3-1
    evt=probs_trio*f["trio_price"]-1
    return {"max_3rentan_ev_per_yen":float(max(ev3)),
            "max_3renpuku_ev_per_yen":float(max(evt)),
            "positive_3rentan_tickets":int(np.sum(ev3>0)),
            "positive_3renpuku_tickets":int(np.sum(evt>0)),
            "best_3rentan_ticket":f["keys"][int(np.argmax(ev3))],
            "best_3renpuku_ticket":f["trio_names"][int(np.argmax(evt))]}


def portfolio(pre,feat,prob,bank=BANK,unit=TICKET_UNIT):
    """Optimize coherent 3連単+3連複 portfolio over every finish order.

    Greedily take the best positive *marginal* expected log-wealth increase
    across all full-book tickets, 100 JPY increments. An unprofitable race
    yields an empty set (abstention), never synthetic positive wagers.
    """
    keys=feat["keys"]
    n=len(keys)
    odds3=np.asarray([float(pre["odds"]["3連単"][k]) for k in keys])
    odds2=feat["trio_price"]
    group=feat["trio_group"]
    groups=[np.flatnonzero(group==j) for j in range(len(feat["trio_names"]))]
    assert all(len(c)==6 for c in groups)
    group_arr=np.stack(groups)
    wealth=np.full(n,float(bank))
    amounts={}
    steps=0
    for _ in range((bank-unit)//unit):
        if min(wealth)<=unit:break
        spent=wealth-unit
        common=float(prob@np.log(spent/wealth))
        gains3=common+prob*np.log1p(unit*odds3/spent)
        gains2=common+np.sum(
            prob[group_arr]*np.log1p(unit*odds2[:,None]/spent[group_arr]),axis=1)
        j3=int(np.argmax(gains3))
        j2=int(np.argmax(gains2))
        if gains3[j3]>=gains2[j2]:
            choice=("3連単",keys[j3])
            winners=[j3]
            odd=float(odds3[j3])
            gain=float(gains3[j3])
        else:
            choice=("3連複",feat["trio_names"][j2])
            winners=groups[j2]
            odd=float(odds2[j2])
            gain=float(gains2[j2])
        if gain<=1e-12:break
        amounts[choice]=amounts.get(choice,0)+unit
        wealth-=unit
        wealth[winners]+=unit*odd
        steps+=1
    by_type={"3連単":0,"3連複":0}
    marks=[]
    for (market,ticket),yen in sorted(amounts.items()):
        by_type[market]+=1
        q=float(prob[keys.index(ticket)]) if market=="3連単" else float(
             np.bincount(group,weights=prob,minlength=len(feat["trio_names"]))[
               feat["trio_names"].index(ticket)])
        odd=float(pre["odds"][market][ticket])
        marks.append({"type":market,"selection":ticket,
                      "stake_yen":yen,"price":odd,
                      "estimated_probability":q,
                      "estimated_ev_per_yen":q*odd-1})
    total=steps*unit
    assert total<bank and sum(m["stake_yen"] for m in marks)==total
    return {"branch":"v3_market_error","race_id":pre["race_id"],
            "selected":marks,"stake_yen":total,"marks":steps,
            "selected_by_type":by_type,"bankroll_yen":bank,
            "all_full_market_tickets_evaluated":True,
            "no_market_disagreement_assumed_profitable":True,
            "no_random_entry_or_seed":True,
            "pre_result_only":True}


def settle(choice,finish):
    t="-".join(map(str,finish))
    c="=".join(map(str,sorted(finish)))
    stake=int(choice["stake_yen"])
    payout=int(round(sum(x["stake_yen"]*x["price"] for x in choice["selected"]
        if (x["type"]=="3連単" and x["selection"]==t) or
           (x["type"]=="3連複" and x["selection"]==c))))
    return {"stake_yen":stake,"payout_yen":payout,"net_yen":payout-stake,
            "marks":choice["marks"],"hit":payout>0}

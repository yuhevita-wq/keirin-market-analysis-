"""Benter (1994) second-stage combination, adapted to Keirin G3.

Original Benter: c_i ∝ f_i^alpha * p_i^beta, with public win
probabilities inferred from *win* tote prices, parameters fitted to actual
winners and historical OUT-OF-SAMPLE fundamental f.

Keirin does not offer a normal win pool. THIS RESEARCH ADAPTATION derives a
first-place public proxy by summing a COMPLETE 2車単 joint inverse-odds
book. It is NOT a method claimed to have been used by Benter.
The public proxy must be tested against historical actual win frequencies.
"""
from __future__ import annotations
from datetime import datetime
from math import exp,log,isfinite
from itertools import permutations
from scipy.optimize import minimize
from scipy.special import logsumexp
import numpy as np

def require_probability_list(items):
    if not isinstance(items,list) or not 5<=len(items)<=9:
        raise ValueError("Expected 5-9 G3 first-win probabilities")
    probs={}
    for item in items:
        if not isinstance(item,dict) or set(item)!={"car_no","probability"}:
            raise ValueError("first_win requires car_no and probability only")
        car=int(item["car_no"]);v=float(item["probability"])
        if car in probs or not 1<=car<=9 or not isfinite(v) or v<=0 or v>=1:
            raise ValueError("Invalid probability or duplicate car")
        probs[car]=v
    if abs(sum(probs.values())-1)>1e-7:
        raise ValueError("G3 first-win probabilities must sum to one")
    return probs

def public_first_from_complete_exacta(first_q,odds):
    """Normalize reciprocal 2車単 odds -> joint market proxy -> first marginal.

    Overround/takeout is removed by normalization; NOT a direct winning
    probability from a single-car market. Every possible ordered pair must
    have an actual observed decimal quote; no synthetic odds or pooling.
    """
    cars=sorted(first_q)
    expected=set(permutations(cars,2))
    seen={}
    if not isinstance(odds,list):raise ValueError("exacta_odds must be a list")
    for row in odds:
        if set(row)!={"selection","decimal_odds"}:
            raise ValueError("Unexpected exacta odds format")
        try:
            key=tuple(int(v) for v in row["selection"])
        except Exception as exc:
            raise ValueError("Invalid exacta selection") from exc
        if len(key)!=2 or key not in expected or key in seen:
            raise ValueError("2車単 book contains invalid or duplicate pair")
        value=float(row["decimal_odds"])
        if not isfinite(value) or value<=1:
            raise ValueError("No positive valid decimal exacta odds")
        seen[key]=value
    if set(seen)!=expected:
        raise ValueError(f"Incomplete 2車単 market: expected {len(expected)}, got {len(seen)}")
    shares={key:1/value for key,value in seen.items()}
    denom=sum(shares.values())
    if denom<=0:raise ValueError("Market implied distribution degenerate")
    q={car:sum(val for (first,second),val in shares.items() if first==car)/denom
       for car in cars}
    if abs(sum(q.values())-1)>1e-10:
        raise AssertionError("Public first proxy not normalized")
    return q,{
        "method":"NORMALIZED_COMPLETE_2CAR_TAN_INVERSE_ODDS_AS_MARKET_PROXY",
        "benter_original_method":False,
        "normalization_reciprocal_sum":denom,
        "available_exacta_pairs":len(expected),
        "limitations":[
            "2車単 marginal is a proxy for market first probability, not a standalone win pool",
            "Inverse-odds pooling assumes a single internally comparable 2車単 pool",
            "Price movement and private bettors' information can change the model after snapshot",
        ]
    }

def blended_first(fundamental,public,alpha,beta):
    if set(fundamental)!=set(public):
        raise ValueError("G3 independent and public proxy participants disagree")
    if not all(isfinite(float(x)) for x in (alpha,beta)):
        raise ValueError("Nonfinite public/fundamental coefficients")
    cars=sorted(fundamental)
    logits=np.array([alpha*log(fundamental[i])+beta*log(public[i]) for i in cars],dtype=float)
    probs=np.exp(logits-logsumexp(logits))
    return {i:float(p) for i,p in zip(cars,probs)}

def validate_historical_row(r):
    if r.get("meeting_grade")!="G3":
        raise ValueError("Only G3 eligible for market fit")
    if not str(r.get("race_type","")).startswith(("S級","Ｓ級")):
        raise ValueError("Only G3 S-class")
    if r.get("first_win_training_window")!="prior_year_only_G3":
        raise ValueError("Fundamental probabilities must be earlier-period G3 only")
    date=r.get("race_date")
    if not isinstance(date,str) or len(date)!=10:
        raise ValueError("Bad research date")
    if r.get("odds_phase") not in ("final","T-15"):
        raise ValueError("Odds phase must be final or T-15; do not mix silently")
    if "race_id" not in r or "actual_first" not in r:
        raise ValueError("Missing G3 historical outcome label")
    f=require_probability_list(r["first_win"])
    p,_=public_first_from_complete_exacta(f,r["exacta_odds"])
    winner=int(r["actual_first"])
    if winner not in f:raise ValueError("Actual G3 winner not among starters")
    return f,p,winner

def fit_combination(rows,phase):
    """Fit alpha beta to historical G3 outcomes; no win-pool fiction."""
    if not rows:raise ValueError("No G3 historical rows to combine")
    data=[]
    ids=set()
    for r in rows:
        if r.get("odds_phase")!=phase:
            raise ValueError("Mixing different odds phases in one second stage forbidden")
        race_id=r["race_id"]
        if race_id in ids:raise ValueError("Repeated race id")
        ids.add(race_id)
        f,p,w=validate_historical_row(r)
        cars=sorted(f)
        logf=np.array([log(f[i]) for i in cars])
        logp=np.array([log(p[i]) for i in cars])
        data.append((logf,logp,cars.index(w)))
    def objective(co):
        alpha,beta=co
        total=0.0;deriv=np.zeros(2)
        for logf,logp,win in data:
            z=alpha*logf+beta*logp
            norm=logsumexp(z)
            predicted=np.exp(z-norm)
            total+=norm-z[win]
            deriv+=np.array([predicted@logf-logf[win],
                             predicted@logp-logp[win]])
        return total/len(data),deriv/len(data)
    optimized=minimize(objective,[1.,1.],jac=True,
                       bounds=[(0,5),(0,5)],method="L-BFGS-B",
                       options={"maxiter":1200,"gtol":1e-9,"ftol":1e-13})
    if not optimized.success and np.max(np.abs(optimized.jac))>1.e-5:
        raise RuntimeError("Benter-style public and fundamental MLE failed")
    return {"alpha":float(optimized.x[0]),"beta":float(optimized.x[1]),
            "races":len(data),"train_log_loss":float(optimized.fun),
            "odds_phase":phase,"fit_status":"G3_HISTORICAL_MLE_ONLY_NOT_YET_MARKET_EDGE"}

def score_combination(rows,phase,co):
    if not rows:raise ValueError("No later G3 evaluation races")
    all_loss={"fundamental":[],"public_proxy":[],"combined":[],"uniform":[]}
    correct=0
    for r in rows:
        if r.get("odds_phase")!=phase:raise ValueError("Odds snapshot type mismatch")
        f,p,w=validate_historical_row(r)
        c=blended_first(f,p,co["alpha"],co["beta"])
        correct+=int(max(c,key=c.get)==w)
        for label,probs in (("fundamental",f),("public_proxy",p),("combined",c)):
            all_loss[label].append(-log(probs[w]))
        all_loss["uniform"].append(log(len(f)))
    means={key:sum(vals)/len(vals) for key,vals in all_loss.items()}
    return {"races":len(rows),"mean_log_losses":means,
            "combined_top1_accuracy":correct/len(rows),
            "combined_improvement_over_public_proxy":means["public_proxy"]-means["combined"],
            "combined_improvement_over_fundamental":means["fundamental"]-means["combined"],
            "note":"Positive improvement in chronological evaluation is evidence of incremental accuracy, NOT proof of profitable tickets."}

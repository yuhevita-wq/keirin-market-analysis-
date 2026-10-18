#!/usr/bin/env python3
"""Fit Benter log-linear public+G3 probabilities from ALIGNED historical outcomes.

The fitting method is from Benter 1994:
 P(outcome) proportional to f(outcome)^alpha * public(outcome)^beta.
G3 uses mutually exclusive ordered bet books in lieu of horse WIN pool. This is
an EXTENSION to race outcome markets, not a claim Benter fitted keirin tickets.

Input: one complete market record per G3 race; no overlap in train/test.
Outputs are explicitly marked as historical FINAL-quote models when appropriate;
FINAL odds never stand in for predeadline 15-min quotes.

No automatic Kelly or stake recommendation: pool size/own price impact unknown.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from engine import EXCLUSIVE, inverse_price_public_probability, validate_quotes, fuse

HERE=Path(__file__).resolve().parent
SINGLE_WIN_PROXY="1着市場代理_3連単"
ALLOWED=EXCLUSIVE+(SINGLE_WIN_PROXY,)

def _probs(nums):
    if not isinstance(nums,dict) or not nums:raise ValueError("Complete probabilities required")
    v={str(k):float(x) for k,x in nums.items()}
    if not all(math.isfinite(x) and x>0 and x<1 for x in v.values()):
        raise ValueError("Probability nonfinite/out of unit interval")
    if abs(sum(v.values())-1)>1e-7:raise ValueError("Model probability book does not sum to 1")
    return v

def first_proxy_from_3rentan(trifecta_odds,car_numbers):
    cars={int(c) for c in car_numbers}
    if len(cars) not in range(5,10):raise ValueError("Invalid race field size")
    expected={f"{i}-{j}-{k}" for i in cars for j in cars for k in cars if len({i,j,k})==3}
    prices=validate_quotes(trifecta_odds,expected,"3連単")
    raw,raw_mass=inverse_price_public_probability(prices,"3連単")
    first={str(k):0.0 for k in cars}
    for key,p in raw.items():first[key.split("-")[0]]+=p
    if abs(sum(first.values())-1)>1e-8:raise RuntimeError("First marginal not normalized")
    return first,raw_mass

def parse_record(row,market):
    if row.get("scope")!="G3 S級":
        raise ValueError("Train ONLY G3-S級, no other grades or race types")
    if row.get("market")!=market:raise ValueError("Cannot mix different pools in one coefficient fit")
    if row.get("fundamental_provenance")!="G3_only_prior_period_independent":
        raise ValueError("Fundamental probabilities must come from G3-only prior-period forecasts")
    if row.get("price_phase")!="final":
        raise ValueError("Historical fitting requires explicitly labeled finalized odds")
    date=str(row.get("race_date",""))
    try:day=dt.date.fromisoformat(date)
    except ValueError as ex:raise ValueError("Missing or invalid race date") from ex
    race_id=str(row.get("race_id",""))
    if not race_id:raise ValueError("Missing immutable race_id")
    f=_probs(row.get("fundamental_probabilities"))
    if market==SINGLE_WIN_PROXY:
        odds=row.get("odds")
        first,inv_mass=first_proxy_from_3rentan(odds,row.get("entrant_cars",[]))
        if set(first)!=set(f):raise ValueError("First-book cars differ from G3 independent model")
        p=first
    else:
        odds=validate_quotes(row.get("odds"),f,market)
        p,inv_mass=inverse_price_public_probability(odds,market)
    actual=str(row.get("actual_winning_selection",""))
    if actual not in f:raise ValueError("Missing actual G3 outcome, or outcome outside supported book")
    return {
      "id":race_id,"date":day.isoformat(),"f":f,"p":p,
      "winner":actual,"quotes_inverse_mass":inv_mass,
      "model_source_certified_predeadline":row.get("original_asof_card_certified") is True,
      "odds_snapshot_type":"final",
    }

def _log_prob(rec,coef):
    keys=sorted(rec["f"])
    a,b=map(float,coef)
    logf=np.array([math.log(rec["f"][k]) for k in keys])
    logp=np.array([math.log(rec["p"][k]) for k in keys])
    z=a*logf+b*logp
    return float(z[keys.index(rec["winner"])]-logsumexp(z))

def fit(records):
    X=[]
    for x in records:
        fkeys=sorted(x["f"])
        X.append((np.array([[math.log(x["f"][k]),math.log(x["p"][k])] for k in fkeys]),
                  fkeys.index(x["winner"])))
    def obj(theta):
        ll=0.0;grad=np.zeros(2,dtype=float)
        for feats,actual in X:
            z=feats@theta
            normal=logsumexp(z)
            pp=np.exp(z-normal)
            ll+=normal-z[actual]
            grad+=pp@feats-feats[actual]
        return ll/len(X),grad/len(X)
    # Coefficients are DATA-ESTIMATED. No arbitrary alpha+beta=1 or nonnegative rule.
    result=minimize(obj,np.array([1.,1.]),jac=True,method="L-BFGS-B",
                    options={"maxiter":800,"ftol":1e-13,"gtol":1e-9})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError("Nonfinite fitted coefficients")
    if (not result.success) and np.max(abs(result.jac))>1e-5:
        raise RuntimeError("Public+fundamental fusion fit failed: "+str(result.message))
    return result.x,{"optimizer_success":bool(result.success),"iterations":int(result.nit),
                     "score_gradient":list(map(float,result.jac))}

def evaluation(records,coef):
    combo=np.array([-_log_prob(r,coef) for r in records])
    public=np.array([-_log_prob(r,[0,1]) for r in records])
    fundamental=np.array([-_log_prob(r,[1,0]) for r in records])
    uniform=np.array([math.log(len(r["f"])) for r in records])
    return {"races":len(records),
            "mean_log_loss_combined":float(combo.mean()),
            "mean_log_loss_public":float(public.mean()),
            "mean_log_loss_fundamental":float(fundamental.mean()),
            "mean_log_loss_uniform":float(uniform.mean()),
            "pseudo_r2_combined":float(1-combo.mean()/uniform.mean()),
            "pseudo_r2_public":float(1-public.mean()/uniform.mean()),
            "added_log_loss_improvement_over_market":float((public-combo).mean()),
            "race_days":len({r["date"] for r in records}),
            "original_predeadline_entry_certified_races":sum(r["model_source_certified_predeadline"] for r in records)}

def day_cluster_ci(records,coef,reps=1300):
    byday=collections.defaultdict(list)
    for r in records:
        byday[r["date"]].append(float(-_log_prob(r,[0,1])+_log_prob(r,coef)))
    days=sorted(byday)
    sums=np.array([sum(byday[d]) for d in days])
    counts=np.array([len(byday[d]) for d in days])
    rng=np.random.default_rng(20261018)
    ix=rng.integers(len(days),size=(reps,len(days)))
    vals=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return {"delta_loss_public_minus_combined":float(sums.sum()/counts.sum()),
            "day_cluster_bootstrap_95":[float(q) for q in np.quantile(vals,[.025,.975])],
            "n_days":len(days)}

def run(input_path, market, training_before, evaluation_from, result_path,source_sha=None):
    if market not in ALLOWED:raise ValueError("Market not exclusive; ワイド requires separate methodology")
    records=[]
    seen=set()
    with Path(input_path).open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():continue
            val=parse_record(json.loads(raw),market)
            if val["id"] in seen:raise ValueError("Duplicate market race_id: "+val["id"])
            seen.add(val["id"]);records.append(val)
    train=[r for r in records if r["date"]<training_before]
    test=[r for r in records if r["date"]>=evaluation_from]
    if not training_before<=evaluation_from:
        raise ValueError("Chronological evaluation must begin no earlier than train cutoff")
    if set(x["id"] for x in train)&set(x["id"] for x in test):
        raise ValueError("Train and validation races overlapped")
    if len(train)<50 or len(test)<50:
        raise ValueError(f"Insufficient aligned G3+odds races (min 50 for exploratory fit): train {len(train)} test {len(test)}")
    params,optimizer=fit(train)
    in_sample=evaluation(train,params)
    out_of_period=evaluation(test,params)
    ci=day_cluster_ci(test,params)
    source_bytes=Path(input_path).read_bytes()
    report={
        "schema":"G3_BENTER_LIKE_MARKET_FUSION_PER_EXCLUSIVE_POOL_V1",
        "status":"CHRONOLOGICAL_MARKET_FIT_COMPLETED",
        "scope":"G3 S級",
        "market":market,
        "market_kind":"public WIN proxy from 3連単" if market==SINGLE_WIN_PROXY else "exclusive payout book",
        "fundamental_model_frozen":True,
        "method":"Benter-style second-stage multinomial logit on log fundamental probability and log public implied probability",
        "coefficients":{"alpha_fundamental":float(params[0]),"beta_public":float(params[1])},
        "training_cutoff_exclusive":training_before,
        "evaluation_start_inclusive":evaluation_from,
        "training_races":len(train),"evaluation_races":len(test),
        "at_least_3198_races_each_partition":len(train)>=3198 and len(test)>=3198,
        "training_metrics_in_sample":in_sample,
        "chronological_evaluation_metrics":out_of_period,
        "extra_information_vs_market_day_bootstrap":ci,
        "optimizer":optimizer,
        "source_jsonl_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "source_data_commit":source_sha,
        "price_phase":"final",
        "historical_outcomes_used_only_as_fit_labels":True,
        "approved_for_15min_runtime":False,
        "approved_for_automatic_wager_amounts":False,
        "caveats":[
            "Historical finalized odds are not 15-min-before-close market snapshots.",
            "Source historical entrant/line features were retrospectively obtained and may contain as-of drift.",
            "Positive estimated value on an unvalidated quote phase does not establish realizable advantage.",
            "This fit works market by market on mutually exclusive alternatives; wide cannot use one-winner normalization.",
            "An apparent improvement without a positive out-of-period bootstrap interval is not confirmed incremental value.",
            "In a parimutuel pool prices vary with all bets including your own; no stakes estimated."
        ]
    }
    Path(result_path).parent.mkdir(parents=True,exist_ok=True)
    Path(result_path).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--market",required=True,choices=ALLOWED)
    p.add_argument("--training-before",required=True)
    p.add_argument("--evaluation-from",required=True)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    report=run(args.input,args.market,args.training_before,args.evaluation_from,args.output)
    print(json.dumps({k:report[k] for k in ("status","market","training_races","evaluation_races","coefficients","at_least_3198_races_each_partition","chronological_evaluation_metrics","extra_information_vs_market_day_bootstrap")},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Re-study FIRST-place probability using V2 plus foreign term=999 categorical encoding.

No first-win or V7 coefficients are overwritten. Compare simple 1st calibration
candidates on chronological development data and inspect later holdout once.
This is retrospective time-ordered research, not an untouched live blind test.
"""
from __future__ import annotations
import csv, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from freeze_prior_win import PERIODS,sources,rows

OUT=HERE/"first_win_research_term_category"
AUDIT=HERE/"frozen_sample_audit.json"
FROZEN=HERE/"frozen_term_category"
FIT_LIMIT="2025-04-01"
VALID_START="2025-01-01"
METHODS=("raw","temperature","high_hinge","combined")
BINS=[0,.01,.025,.05,.075,.10,.15,.20,.25,.30,.4,.5,.6,.7,.8,1.00000001]

def read_frozen():
    forecasts={}
    for period in PERIODS:
        p=FROZEN/f"independent_win_{period}.csv"
        m=FROZEN/f"manifest_{period}.json"
        manifest=json.loads(m.read_text(encoding="utf-8"))
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest["prediction_sha256"],period
        n=0
        for row in rows(p):
            n+=1
            rid=row["race_id"]
            ent=forecasts.setdefault(rid,{
                "date":row["race_date"],"period":period,"prob":{} })
            if ent["date"]!=row["race_date"] or ent["period"]!=period:
                raise RuntimeError("Period collision: "+rid)
            car=int(row["car_no"]);pr=float(row["independent_win_probability"])
            if car in ent["prob"] or not (0<pr<1):
                raise RuntimeError("Invalid or repeated prediction: "+rid)
            ent["prob"][car]=pr
        if n!=manifest["target_rider_rows"]:
            raise RuntimeError("Rider count mismatch: "+period)
    return forecasts

def load_races():
    forecasts=read_frozen()
    podium=defaultdict(lambda:defaultdict(list))
    for folder in sources():
        p=folder/"results.csv"
        if not p.exists():continue
        for row in rows(p):
            rid=row.get("race_id")
            if rid in forecasts and str(row.get("finish_position","")).strip() in ("1","2","3"):
                podium[rid][str(row["finish_position"]).strip()].append(row.get("car_no"))
    good=[];excluded=Counter()
    for rid,rec in forecasts.items():
        places=podium.get(rid,{})
        if any(len(places.get(str(k),[]))!=1 for k in (1,2,3)):
            excluded["ambiguous_or_missing_places"]+=1
            continue
        try:
            finish=[int(places[str(k)][0]) for k in (1,2,3)]
        except (ValueError,TypeError):
            excluded["invalid_finish_car"]+=1;continue
        if len(set(finish))!=3 or not set(finish).issubset(rec["prob"]):
            excluded["finish_not_in_roster"]+=1;continue
        cars=sorted(rec["prob"]);p=np.array([rec["prob"][c] for c in cars],dtype=float)
        if abs(p.sum()-1)>1e-8:
            raise RuntimeError("Non-normalized original: "+rid)
        good.append({"race_id":rid,"date":rec["date"],"period":rec["period"],
                     "cars":cars,"p":p,"winner":cars.index(finish[0])})
    good.sort(key=lambda r:(r["date"],r["race_id"]))
    previous=json.loads(AUDIT.read_text(encoding="utf-8"))
    if len(good)!=sum(x["eligible_races"] for x in previous["partition"].values()):
        raise RuntimeError("First stage differs from fixed podium audit")
    for split in ("fit","chronological_check"):
        ids=[r["race_id"] for r in good if (r["date"]<FIT_LIMIT)==(split=="fit")]
        old=previous["partition"][split]
        sha=hashlib.sha256(("\n".join(sorted(ids))+"\n").encode()).hexdigest()
        if sha!=old["race_ids_sha256"]:
            raise RuntimeError("Frozen race IDs differ for "+split)
    return good,dict(excluded)

def logits(p,method,parameters):
    z=np.log(np.maximum(p,1e-15))
    if method=="raw":
        return z
    if method=="temperature":
        return parameters[0]*z
    if method=="high_hinge":
        return z - parameters[0]*np.maximum(p-.4,0)
    if method=="combined":
        return parameters[0]*z-parameters[1]*np.maximum(p-.4,0)
    raise ValueError(method)

def calibrated(p,method,parameters):
    z=logits(p,method,parameters)
    return np.exp(z-logsumexp(z))

def fit_cal(races,method):
    if method=="raw":return []
    initial={"temperature":[1.0],"high_hinge":[0.0],"combined":[1.0,0.0]}[method]
    bounds={"temperature":[(.25,3.0)],
            "high_hinge":[(-6,6)],
            "combined":[(.25,3.0),(-6,6)]}[method]
    def nll(par):
        loss=0
        for r in races:
            z=logits(r["p"],method,par)
            loss+=logsumexp(z)-z[r["winner"]]
        return loss/len(races)
    res=minimize(nll,initial,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":150,"ftol":1e-12})
    if not res.success:raise RuntimeError(f"Fit failed {method}: {res.message}")
    return [float(x) for x in res.x]

def metrics(races,method,params,details=False):
    losses=[];briers=[];wins=[];top2=[];unif=[];allpred=[];alltrue=[];byperiod=defaultdict(list);detailsrows=[]
    for r in races:
        p=calibrated(r["p"],method,params)
        y=r["winner"]
        ll=-math.log(max(float(p[y]),1e-15))
        losses.append(ll)
        briers.append(float(np.square(p-(np.arange(len(p))==y)).sum()))
        hit=int(np.argmax(p)==y);wins.append(hit)
        top2.append(int(y in np.argsort(-p)[:2]))
        unif.append(math.log(len(p)))
        byperiod[r["period"]].append((ll,hit))
        for j,x in enumerate(p):
            allpred.append(float(x));alltrue.append(int(j==y))
        if details:
            detailsrows.append([r["race_id"],r["date"],r["period"],r["cars"][y],
                                len(p),f"{ll:.12g}",f"{p[y]:.12g}",
                                f"{float(r['p'][y]):.12g}",hit])
    binned=[]
    pp=np.array(allpred);yy=np.array(alltrue)
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        idx=(pp>=lo)&(pp<hi)
        if int(idx.sum()):
            binned.append({"range":[lo,min(hi,1.0)],"riders":int(idx.sum()),
                           "predicted_mean":float(pp[idx].mean()),
                           "actual_win_rate":float(yy[idx].mean()),
                           "expected_winners":float(pp[idx].sum()),
                           "actual_winners":int(yy[idx].sum())})
    return {
        "races":len(races),"rider_candidate_rows":len(allpred),
        "mean_log_loss":float(np.mean(losses)),
        "uniform_mean_log_loss":float(np.mean(unif)),
        "benter_pseudo_r2_vs_uniform":float(1-np.mean(losses)/np.mean(unif)),
        "mean_brier":float(np.mean(briers)),
        "top1_accuracy":float(np.mean(wins)),
        "top2_includes_winner":float(np.mean(top2)),
        "periods":{k:{"races":len(v),
                        "log_loss":float(np.mean([x[0] for x in v])),
                        "top1_accuracy":float(np.mean([x[1] for x in v]))}
                   for k,v in sorted(byperiod.items())},
        "calibration":binned,
    },detailsrows

def boot_improvement(races,method,params,reps=2500):
    # Day-cluster bootstrap; positive means method has lower log loss than raw
    group=defaultdict(list)
    for r in races:
        raw=-math.log(max(float(r["p"][r["winner"]]),1e-15))
        cal=calibrated(r["p"],method,params)
        new=-math.log(max(float(cal[r["winner"]]),1e-15))
        group[r["date"]].append(raw-new)
    days=sorted(group)
    sums=np.array([sum(group[d]) for d in days])
    counts=np.array([len(group[d]) for d in days])
    obs=float(sums.sum()/counts.sum())
    rng=np.random.default_rng(20261018)
    samples=rng.integers(len(days),size=(reps,len(days)))
    gains=sums[samples].sum(axis=1)/counts[samples].sum(axis=1)
    return {"improvement_positive_is_better":obs,
            "day_cluster_95_interval":[float(x) for x in np.quantile(gains,[.025,.975])],
            "days":len(days)}

def main():
    races,exclusion=load_races()
    chronological_train=[r for r in races if r["date"]<VALID_START]
    chronological_valid=[r for r in races if VALID_START<=r["date"]<FIT_LIMIT]
    fit_period=[r for r in races if r["date"]<FIT_LIMIT]
    test_period=[r for r in races if r["date"]>=FIT_LIMIT]
    if not chronological_train or not chronological_valid or not test_period:
        raise RuntimeError("Missing chronological subpartition")
    experiments={}
    for method in METHODS:
        p=fit_cal(chronological_train,method)
        valid_result,_=metrics(chronological_valid,method,p)
        experiments[method]={"selection_training_coefficients":p,
                             "development_validation_loss":valid_result["mean_log_loss"]}
    # No model selection using the evaluation period.
    choice=min(METHODS,key=lambda m:experiments[m]["development_validation_loss"])
    chosen=fit_cal(fit_period,choice)
    raw_fit,_=metrics(fit_period,"raw",[])
    raw_eval,raw_rows=metrics(test_period,"raw",[],details=True)
    selected_fit,_=metrics(fit_period,choice,chosen)
    selected_eval,chosen_rows=metrics(test_period,choice,chosen,details=True)
    all_eval={}
    for method in METHODS:
        # Diagnostics only; do NOT choose alternative by its evaluation-period score.
        q=fit_cal(fit_period,method)
        sc,_=metrics(test_period,method,q)
        all_eval[method]={"coefficients_fitted_before_eval":q,
                          "evaluation_log_loss_exploratory":sc["mean_log_loss"],
                          "calibration_candidate_only":method!=choice}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"heldout_first_by_race.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["race_id","date","period","winner","n","raw_log_loss","selected_log_loss",
                    "raw_true_winner_p","selected_true_winner_p","top1_hit"])
        for raw,sel in zip(raw_rows,chosen_rows):
            if raw[:5]!=sel[:5]:raise RuntimeError("Race order changed")
            w.writerow(raw[:6]+[sel[5],raw[7],sel[6],raw[8]])
    report={
      "status":"FIRST_WIN_CATEGORICAL_999_CORRECTED_WALKFORWARD_REESTIMATE",
      "model_scope":"F1 S級 only",
      "input":"V2 full_plus_line plus categorical term999 foreign indicator, strictly past-only quarterly forecasts",
      "excluded_podium_reasons":exclusion,
      "subpartitions":{"2024_development_training":len(chronological_train),
                       "2025q1_development_validation":len(chronological_valid),
                       "fit_calibrator_until_2025_03":len(fit_period),
                       "evaluation_from_2025_04":len(test_period)},
      "constraints":{"no_market_odds":True,"no_future_outcome_in_individual_forecast":True,
                     "V7_gamma_delta_a_b_modified":False,"pre_race_asof_verified":False,
                     "untouched_global_blind_period":False},
      "calibration_candidates":experiments,
      "selection":"Lowest 2025Q1 validation log loss only, trained initially on 2024",
      "selected_method":choice,
      "selected_parameters_from_development":chosen,
      "raw_fit_summary":{k:raw_fit[k] for k in ("races","mean_log_loss","mean_brier","top1_accuracy")},
      "selected_fit_summary":{k:selected_fit[k] for k in ("races","mean_log_loss","mean_brier","top1_accuracy")},
      "raw_evaluation":raw_eval,
      "selected_evaluation":selected_eval,
      "paired_day_bootstrap":boot_improvement(test_period,choice,chosen),
      "other_methods_evaluation_diagnostic_only":all_eval,
      "provenance":"Historical racecards are dated to races, but are not certified original live pre-race captures",
      "caveat":"Term999 correction was prompted by inspection of older model evaluation failures; therefore the comparison with that evaluation period is exploratory, NOT a virgin holdout. Original V7 q/2nd/3rd coefficients remain untouched.",
    }
    (OUT/"first_win_report.json").write_text(
         json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FIRST_RESEARCH_COMPLETE",json.dumps({
      "split":report["subpartitions"],"selection":experiments,
      "chosen":choice,"params":chosen,
      "raw_eval":{k:raw_eval[k] for k in ("races","mean_log_loss","mean_brier","top1_accuracy","benter_pseudo_r2_vs_uniform")},
      "selected_eval":{k:selected_eval[k] for k in ("races","mean_log_loss","mean_brier","top1_accuracy","benter_pseudo_r2_vs_uniform")},
      "paired":report["paired_day_bootstrap"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

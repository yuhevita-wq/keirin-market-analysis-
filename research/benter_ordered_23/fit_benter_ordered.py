#!/usr/bin/env python3
"""Benter (1994) corrected-Harville 1st/2nd/3rd study for Japanese keirin.

Models:
  1. Harville conditional: p_j / sum_{j != winner} p_j; same for third.
  2. Historical Benter exponents .81 / .65.
  3. Keirin MLE exponents, using ONLY July-Aug 2026 outcomes.

Evaluation: all eligible September 2026 F1 S-class races already forecast in
V4's GitHub-frozen pre-results CSV. NEVER use September results to pick
models/parameters. Outcomes only for scoring after train_*.json is committed.
All files for July/Aug model forecasts and raw source were independently frozen.

  P(i,j,k)=p_i * p_j^gamma/sum_{a != i}p_a^gamma
                      * p_k^delta/sum_{b not i,j}p_b^delta.

The line interactions already embedded in V2 p_i; conditional finish-order
residual dependence is not modeled here. DNF/ties/ambiguous ranks excluded.
"""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[2]
THIS=Path(__file__).resolve().parent
H2=ROOT/"research/benter_score_challenge_v3/frozen_2026h2_probabilities.csv"
H2RESULT=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/results.csv"
SEP=ROOT/"research/benter_calibration_v4/sep_predictions_frozen.csv"
SEPRESULT=ROOT/"research/benter_calibration_v4/sep_holdout/results.csv"
FIT=THIS/"parameters_frozen.json"
RESULT=THIS/"september_evaluation.json"
DETAIL=THIS/"september_by_race.csv"
BENTER={"gamma":0.81,"delta":0.65}
BOUNDS=(0.15,2.5)

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)

def calibrate_h2(p):
    logits=np.log(np.maximum(p,1e-15))-np.maximum(p-.4,0)
    logits-=np.max(logits)
    a=np.exp(logits)
    return a/a.sum()

def group_probability(path):
    groups=defaultdict(list)
    for rec in read(path):
        groups[rec["race_id"]].append(rec)
    return groups

def actual_rankings(path):
    outcomes=defaultdict(lambda:defaultdict(list))
    for r in read(path):
        try: rank=int(r["finish_position"])
        except (ValueError,TypeError):continue
        if rank in (1,2,3):
            outcomes[r["race_id"]][rank].append(r["car_no"])
    return outcomes

def collect(predictions,outcomes,origin):
    scores=group_probability(predictions)
    finish=actual_rankings(outcomes)
    valid=[]
    rejected=Counter()
    for rid,rs in scores.items():
        rank=finish.get(rid,{})
        if any(len(rank.get(i,[]))!=1 for i in (1,2,3)):
            rejected["tie_or_missing_finish_1_2_3"]+=1
            continue
        rs.sort(key=lambda r:int(r["car_no"]))
        cars=[r["car_no"] for r in rs]
        if any(rank[i][0] not in cars for i in (1,2,3)):
            rejected["rank_not_present_in_entries"]+=1
            continue
        n=len(cars)
        if len(set(cars))!=n or len(set(rank[i][0] for i in (1,2,3)))!=3 or n<5:
            rejected["invalid_distinct_cars"]+=1
            continue
        p=np.array([float(r["v2_line_model"] if origin=="h2" else r["original_probability"]) for r in rs],dtype=float)
        if np.any(p<=0) or abs(p.sum()-1)>1e-7:
            rejected["invalid_probabilities"]+=1
            continue
        corrected=calibrate_h2(p) if origin=="h2" else np.array([float(r["calibrated_probability"]) for r in rs],dtype=float)
        if abs(corrected.sum()-1)>1e-7:
            raise RuntimeError("Non-normalized calibrated probability")
        indexes=[cars.index(rank[i][0]) for i in (1,2,3)]
        valid.append({
            "date":rs[0]["race_date"],
            "track":rs[0]["track"],
            "race_id":rid,
            "ranks":indexes,
            "cars":cars,
            "p":p,"q":corrected
        })
    return valid,{"forecast_races":len(scores),"eligible":len(valid),"excluded":dict(rejected)}

def conditional(p, excluded, exponent):
    scores=np.power(np.clip(p,1e-15,1),exponent)
    scores[np.asarray(excluded,dtype=int)]=0
    return scores/scores.sum()

def ll_array(races,gamma,delta,series="p"):
    losses=[]
    for r in races:
        p=r[series];i,j,k=r["ranks"]
        pj=conditional(p,[i],gamma)[j]
        pk=conditional(p,[i,j],delta)[k]
        losses.append((-math.log(pj),-math.log(pk),-math.log(p[i])-math.log(pj)-math.log(pk)))
    return np.asarray(losses)

def fit(races,series):
    def evaluate(exponent,which):
        total=0.
        for r in races:
            p=r[series];i,j,k=r["ranks"]
            if which=="second":obs,excluded=j,[i]
            else:obs,excluded=k,[i,j]
            total-=math.log(max(conditional(p,excluded,exponent)[obs],1e-300))
        return total/len(races)
    parameters={}
    for which,name in (("second","gamma"),("third","delta")):
        res=minimize_scalar(lambda theta:evaluate(theta,which),
                            bounds=BOUNDS,method="bounded",options={"xatol":1e-6})
        if not res.success:
            raise RuntimeError(f"Optimizer failed {which}")
        parameters[name]=float(res.x)
        parameters[which+"_loss"]=float(res.fun)
        parameters[which+"_harville"]=float(evaluate(1.,which))
        parameters[which+"_benter"]=float(evaluate(BENTER[name],which))
    return parameters

def fit_stage():
    train,audit=collect(H2,H2RESULT,"h2")
    if len(train)<775:raise RuntimeError(f"Unexpected training sample {len(train)}")
    fitted={kind:fit(train,kind) for kind in ("p","q")}
    bymonth={}
    for month in ("2026-07","2026-08"):
        subset=[r for r in train if r["date"].startswith(month)]
        bymonth[month]={"n":len(subset),"raw":fit(subset,"p"),"calibrated":fit(subset,"q")}
    frozen={
        "model":"Benter 1994 rank-exponent, trained on independent V2 1st place probabilities",
        "formula":"p(i,j,k) = p_i*(p_j^gamma / sum_{a != i} p_a^gamma)*(p_k^delta / sum_{b != i,j} p_b^delta)",
        "training_period":"2026-07-02 to 2026-08-30",
        "holdout_period":"2026-09-01 to 2026-09-30",
        "training_audit":audit,
        "historical_benter_parameters":BENTER,
        "parameter_bounds":list(BOUNDS),
        "fitted_parameters":fitted,
        "monthly_fits":bymonth,
        "target_results_used_for_training":False,
        "input_models_frozen":True,
        "calibration":"Prior lambda=1, hinge threshold 0.4 fixed on July-Aug; no parameter refitting on September",
    }
    FIT.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("TRAINING_ONLY",json.dumps({"audit":audit,"fitted_parameters":fitted,"monthly_fits":bymonth},ensure_ascii=False),flush=True)

def candidate_metrics(races,params,series):
    losses=ll_array(races,params["gamma"],params["delta"],series)
    n=len(races)
    true_second=sum(np.argmax(conditional(r[series],[r["ranks"][0]],params["gamma"]))==r["ranks"][1] for r in races)
    true_third=sum(np.argmax(conditional(r[series],r["ranks"][:2],params["delta"]))==r["ranks"][2] for r in races)
    return {
        "races":n,
        "conditional_second_mean_logloss":float(losses[:,0].mean()),
        "conditional_third_mean_logloss":float(losses[:,1].mean()),
        "joint_first_second_third_mean_logloss":float(losses[:,2].mean()),
        "conditional_second_top1_hits":int(true_second),
        "conditional_third_top1_hits":int(true_third),
        "mean_uniform_ordered_logloss":float(np.mean([math.log(len(r["cars"])*(len(r["cars"])-1)*(len(r["cars"])-2)) for r in races])),
        "ordered_pseudo_R2_vs_uniform":1-float(losses[:,2].mean())/float(np.mean([math.log(len(r["cars"])*(len(r["cars"])-1)*(len(r["cars"])-2)) for r in races]))
    },losses

def conditional_calibration(races,params,series,stage):
    data=[]
    for r in races:
        p=r[series];i,j,k=r["ranks"]
        excludes=[i] if stage=="second" else [i,j]
        probs=conditional(p,excludes,params["gamma"] if stage=="second" else params["delta"])
        target=j if stage=="second" else k
        for car,prob in enumerate(probs):
            if car in excludes:continue
            data.append((prob,int(car==target)))
    cuts=[0,.025,.05,.1,.15,.2,.25,.3,.4,.5,1.000000001]
    bins=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        arr=[(p,y) for p,y in data if lo<=p<hi]
        if not arr:continue
        bins.append({"bin":[lo,hi],"n":len(arr),
                     "average_predicted":float(np.mean([p for p,y in arr])),
                     "actual":float(np.mean([y for p,y in arr]))})
    return bins

def score_stage():
    if not FIT.exists():
        raise RuntimeError("Parameter file must be persisted on GitHub before evaluating September.")
    train=json.loads(FIT.read_text(encoding="utf-8"))
    assert not train["target_results_used_for_training"]
    test,audit=collect(SEP,SEPRESULT,"sep")
    if len(test)<350:raise RuntimeError(f"Too few eligible September races: {audit}")
    models={
        "raw_harville":("p",{"gamma":1.,"delta":1.}),
        "raw_benter_1994":("p",BENTER),
        "raw_keirin_mle":("p",train["fitted_parameters"]["p"]),
        "calibrated_harville":("q",{"gamma":1.,"delta":1.}),
        "calibrated_benter_1994":("q",BENTER),
        "calibrated_keirin_mle":("q",train["fitted_parameters"]["q"]),
    }
    comparisons={}
    perrace={}
    for name,(series,theta) in models.items():
        metrics,losses=candidate_metrics(test,theta,series)
        comparisons[name]=metrics
        perrace[name]=losses
    daily=defaultdict(list)
    for k,r in enumerate(test):daily[r["date"]].append(k)
    dates=list(daily.values())
    def bootstrap_gain(a,b,col):
        x=perrace[b][:,col]-perrace[a][:,col]
        rng=np.random.default_rng(20261018)
        z=[]
        for _ in range(3500):
            idx=np.concatenate([np.array(dates[j]) for j in rng.integers(len(dates),size=len(dates))])
            z.append(float(x[idx].mean()))
        return {"gain_positive_better":float(x.mean()),"date_bootstrap_95pct":[float(v) for v in np.quantile(z,[.025,.975])]}
    pair={}
    for a,b in [
        ("raw_benter_1994","raw_harville"),
        ("raw_keirin_mle","raw_harville"),
        ("raw_keirin_mle","raw_benter_1994"),
        ("calibrated_benter_1994","calibrated_harville"),
        ("calibrated_keirin_mle","calibrated_harville"),
        ("calibrated_keirin_mle","calibrated_benter_1994"),
        ("calibrated_keirin_mle","raw_keirin_mle"),
    ]:
        pair[f"{a}_versus_{b}"]={
            "conditional_second":bootstrap_gain(a,b,0),
            "conditional_third":bootstrap_gain(a,b,1),
            "first_second_third_ordered":bootstrap_gain(a,b,2)
        }
    calibration={}
    for name in ("raw_harville","raw_benter_1994","raw_keirin_mle","calibrated_keirin_mle"):
        series,theta=models[name]
        calibration[name]={
            "second":conditional_calibration(test,theta,series,"second"),
            "third":conditional_calibration(test,theta,series,"third")
        }
    record=[]
    for idx,r in enumerate(test):
        row={"date":r["date"],"track":r["track"],"race_id":r["race_id"],
             "first_car":r["cars"][r["ranks"][0]],"second_car":r["cars"][r["ranks"][1]],
             "third_car":r["cars"][r["ranks"][2]],"car_count":len(r["cars"])}
        for m in models:
            row[f"{m}_second_logloss"]=float(perrace[m][idx,0])
            row[f"{m}_third_logloss"]=float(perrace[m][idx,1])
            row[f"{m}_ordered_logloss"]=float(perrace[m][idx,2])
        record.append(row)
    with DETAIL.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=record[0].keys());w.writeheader();w.writerows(record)
    report={
        "status":"FIXED_MODEL_INDEPENDENT_OUT_OF_PERIOD_TEST",
        "scope":"F1 S級 2026年9月, 1st-2nd-3rd uniquely ranked finishers only",
        "source":"Benter 1994 First Second and Third section pp192-194; Harville, and corrected exponents 0.81,0.65",
        "train_period":train["training_period"],
        "test_period":train["holdout_period"],
        "train_audit":train["training_audit"],
        "test_audit":audit,
        "fitted_exponents":train["fitted_parameters"],
        "benter_exponents":BENTER,
        "comparisons":comparisons,
        "pairwise":pair,
        "conditional_calibration":calibration,
        "days":len(dates),
        "limitation":"Line-dependent second/third correlations beyond first win probability are not modeled.",
        "no_market_odds_used":True,
        "forecasts_frozen_before_september_outcomes":True
    }
    RESULT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SEPTEMBER_TEST",json.dumps({k:report[k] for k in ("train_audit","test_audit","fitted_exponents","comparisons","pairwise")},ensure_ascii=False,indent=2),flush=True)

def main():
    import sys
    if len(sys.argv)!=2 or sys.argv[1] not in ("fit","score"):
        raise SystemExit("Usage: python fit_benter_ordered.py fit|score")
    if sys.argv[1]=="fit":fit_stage()
    else:score_stage()
if __name__=="__main__":main()

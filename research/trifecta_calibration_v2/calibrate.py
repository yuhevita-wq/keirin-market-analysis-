#!/usr/bin/env python3
"""Independent odds-free 3連単 probability calibration v2.

The raw 2023 model is immutable. 2024 races and results train ONLY a
low-dimensional post-calibration function; original rider/race forecasting
weights are never refitted. 2024 Jan-Jun estimates parameters; Jul-Dec is
temporal development validation. After that, a separate full-2024 post-
calibration refit is saved specifically for a still-sealed 2025 test.

Input archive is exactly the 2024 full-ticket prediction artifact from
workflow run 40807880319; 2025 data is not accessed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import calibration_audit as audit  # noqa: E402

BASE_ARCHIVE = ROOT/"data/audits/independent_trifecta_all_tickets_2024.jsonl.gz"
OUT_DIR = ROOT/"data/audits"
REPORT = OUT_DIR/"independent_trifecta_calibrator_v2_2024_report.json"
WEIGHTS = OUT_DIR/"independent_trifecta_calibrator_v2_for_2025.json"
EXAMPLE = OUT_DIR/"independent_trifecta_calibrator_v2_example_2024.json"

FEATURES=("log_baseline_probability",
          "same_line_12","same_line_13","same_line_23",
          "same_line_123","three_distinct_lines",
          "first_is_line_leader","first_is_score_rank_1")
# This model specification and its regularization were frozen prior to
# looking at the Jul-Dec 2024 probability scores in this experiment.
L2_STRENGTH = 50.0
LIMITS=[(0.5,2.0)]+[(-1.0,1.0)]*7
SPLIT_DAY="2024-06-30"
BANDS=("0–0.1%","0.1–0.5%","0.5–1%","1–2%","2–5%","5–10%","10–20%","20–100%")
UPPER=np.array([.001,.005,.01,.02,.05,.1,.2])
STRUCTURE_FEATURES=FEATURES[1:7]
EXPECTED_RACES=25348
EXPECTED_TICKETS=5820462
# H1 predictive development is already part of an examined 2024 calendar.
# True external unseen evaluation requires 2025 AFTER freezing weights.

def band_index(p):
    return np.searchsorted(UPPER,p,side="right")

@dataclass
class Packed:
    x:np.ndarray
    starts:np.ndarray
    counts:np.ndarray
    truth_idx:np.ndarray
    dates:list
    ids:list
    categories:list
    line_shapes:list
    riders:list

    @property
    def n(self):return len(self.dates)

def pack(items):
    mats=[];starts=[];counts=[];truth=[];dates=[];ids=[];cats=[];shapes=[];riders=[]
    offset=0
    for (rid,e,mat,tidx) in items:
        mats.append(mat)
        starts.append(offset)
        counts.append(len(mat))
        truth.append(offset+tidx)
        offset+=len(mat)
        dates.append(e["date"]);ids.append(rid)
        cats.append(e["race_class"])
        shapes.append("-".join(map(str,sorted([len(g) for g in e["groups"]],reverse=True))))
        riders.append(len(e["scores"]))
    return Packed(np.concatenate(mats),np.asarray(starts,np.int64),
        np.asarray(counts,np.int32),np.asarray(truth,np.int64),
        dates,ids,cats,shapes,riders)

def feature_matrix(e,probabilities,true_ticket):
    n=len(probabilities)
    expected=len(e["scores"])*(len(e["scores"])-1)*(len(e["scores"])-2)
    if n!=expected:raise ValueError(f"Trifecta grid incomplete: {n} != {expected}")
    x=np.empty((n,len(FEATURES)),dtype=np.float32)
    ln={car:v[0] for car,v in e["positions"].items()}
    lead={car:v[1]==1 for car,v in e["positions"].items()}
    rank1={car:v==1 for car,v in e["ranks"].items()}
    truth_index=None;acc=0.
    for ix,(k,p) in enumerate(probabilities.items()):
        a,b,c=k.split("-")
        p=float(p)
        if not(0<p<=1 and math.isfinite(p)):raise ValueError("Invalid baseline probability")
        v1,v2,v3=ln[a],ln[b],ln[c]
        z12=(v1==v2);z13=(v1==v3);z23=(v2==v3)
        x[ix]=[math.log(p),int(z12),int(z13),int(z23),
               int(z12 and z13),int(len({v1,v2,v3})==3),
               int(lead[a]),int(rank1[a])]
        acc+=p
        if k==true_ticket:truth_index=ix
    if abs(acc-1)>2e-7:raise ValueError(f"Baseline probabilities do not sum to 1: {acc}")
    if truth_index is None:raise ValueError("Actual 3連単 is absent from predicted grid")
    return x,truth_index

def load():
    print("READING_2024_PREDICTED_GRIDS",flush=True)
    events,coverage=audit.ingest_2024()
    h1=[];h2=[];seen=set();pred_count=0
    with gzip.open(BASE_ARCHIVE,"rt",encoding="utf-8") as f:
        for count,line in enumerate(f,1):
            r=json.loads(line);rid=str(r["race_id"])
            if rid in seen:raise RuntimeError("Duplicate prediction race ID")
            seen.add(rid)
            e=events.get(rid)
            if e is None:raise RuntimeError(f"Prediction/entry mismatch {rid}")
            true="-".join(e["order"])
            if true!=r["true_result"]:raise RuntimeError(f"Result mismatch {rid}")
            if r["date"]!=e["date"]:raise RuntimeError("Race date mismatch")
            if len(e["scores"])!=r["riders"]:raise RuntimeError("Rider count mismatch")
            X,tidx=feature_matrix(e,r["probabilities"],true)
            pred_count+=len(X)
            (h1 if e["date"]<=SPLIT_DAY else h2).append((rid,e,X,tidx))
            if count%5000==0:print("PACKED_GRIDS",count,flush=True)
    if len(seen)!=len(events) or len(seen)!=EXPECTED_RACES or pred_count!=EXPECTED_TICKETS:
        raise RuntimeError(f"Archive coverage mismatch: {len(seen)} / {len(events)}; tickets {pred_count}")
    print("ARCHIVE_VERIFIED",len(seen),pred_count,flush=True)
    t1=pack(h1);t2=pack(h2)
    del h1,h2,events
    return t1,t2,coverage,pred_count

def objective(packed,columns):
    # Stable grouped softmax for each race, all tickets, no odds whatsoever.
    X=packed.x[:,columns].astype(np.float64,copy=False)
    starts=packed.starts;counts=packed.counts;true_idx=packed.truth_idx
    n=packed.n
    true_feat=X[true_idx]
    truth_sum=true_feat.sum(axis=0)
    prior=np.array([1.0 if col==0 else 0.0 for col in columns])
    bounds=[LIMITS[col] for col in columns]
    def calc(coefs):
        logits=X@coefs
        maxs=np.maximum.reduceat(logits,starts)
        weights=np.exp(logits-np.repeat(maxs,counts))
        normalizers=np.add.reduceat(weights,starts)
        z=np.log(normalizers)+maxs
        loss=(z.sum()-logits[true_idx].sum())/n
        weights/=np.repeat(normalizers,counts)
        gradient=(X.T@weights-truth_sum)/n
        deviation=coefs-prior
        loss+=0.5*L2_STRENGTH*(deviation@deviation)/n
        gradient+=L2_STRENGTH*deviation/n
        return float(loss),gradient
    return calc,prior,bounds

def fit(packed,columns,name):
    fn,prior,bounds=objective(packed,columns)
    initial=fn(prior)[0]
    o=minimize(fn,prior,jac=True,method="L-BFGS-B",bounds=bounds,
               options={"maxiter":100,"ftol":1e-11,"gtol":1e-7})
    if not o.success and np.linalg.norm(o.jac)>1e-3:
        raise RuntimeError(f"Optimizer failed: {name} {o.message}")
    params=np.zeros(len(FEATURES),np.float64)
    params[0]=1.0
    params[columns]=o.x
    print("CALIBRATION_FIT",name,"races",packed.n,
          "nll_init",round(initial,7),"final_regularized_loss",round(float(o.fun),7),
          "parameters",np.round(params,6).tolist(),"iterations",o.nit,flush=True)
    return params,{
        "model":name,"trained_races":packed.n,
        "training_first_date":packed.dates[0],"training_last_date":packed.dates[-1],
        "columns":[FEATURES[k] for k in columns],
        "initial_regularized_objective":round(initial,9),
        "optimized_regularized_objective":round(float(o.fun),9),
        "iterations":int(o.nit),"success":bool(o.success),
        "raw_parameters":dict(zip(FEATURES,np.round(params,9).tolist())),
        "parameter_bounds":bounds,
    }

def softmax(scores):
    z=scores-np.max(scores)
    v=np.exp(z)
    return v/v.sum()

def setup_stat():
    return {"races":0,"sum_nll":0.,"sum_brier":0.,"top1":0,"top3":0,
            "truth_p_sum":0.}

def add_stat(s,q,tidx):
    s["races"]+=1
    s["sum_nll"]+=-math.log(max(float(q[tidx]),1e-20))
    s["sum_brier"]+=float(1.0-2.0*q[tidx]+q@q)
    order=np.argsort(-q,kind="stable")
    s["top1"]+=int(order[0]==tidx)
    s["top3"]+=int(tidx in order[:3])
    s["truth_p_sum"]+=float(q[tidx])

def finish_stat(s):
    n=s["races"]
    return {
        "races":n,
        "nll":round(s["sum_nll"]/n,8),
        "brier":round(s["sum_brier"]/n,8),
        "top1_hit_rate_pct":round(s["top1"]*100/n,5),
        "top3_hit_rate_pct":round(s["top3"]*100/n,5),
        "mean_probability_actual_pct":round(s["truth_p_sum"]*100/n,6)
    }

def make_stats():
    return {
      "overall":setup_stat(),
      "bands":{b:{"ticket_count":0,"expected_hits":0.,"actual_hits":0,"sum_race_m2":0.} for b in BANDS},
      "structures":{f:{"races":0,"actual":0,"expected":0.,"sum_variance":0.} for f in STRUCTURE_FEATURES},
      "by_class":defaultdict(setup_stat),
      "by_line_shape":defaultdict(setup_stat),
      "by_field":defaultdict(setup_stat),
      "by_month":defaultdict(setup_stat),
      "daily_nll":defaultdict(lambda:[0.,0]),
    }

def analyze(packed,coefs,stats,example=None):
    for i,(start, count, tidx) in enumerate(zip(packed.starts,packed.counts,packed.truth_idx)):
        xx=packed.x[start:start+count].astype(np.float64,copy=False)
        q=softmax(xx@coefs)
        local_true=int(tidx-start)
        add_stat(stats["overall"],q,local_true)
        add_stat(stats["by_class"][packed.categories[i]],q,local_true)
        add_stat(stats["by_line_shape"][packed.line_shapes[i]],q,local_true)
        add_stat(stats["by_field"][str(packed.riders[i])],q,local_true)
        add_stat(stats["by_month"][packed.dates[i][:7]],q,local_true)
        nll=-math.log(max(float(q[local_true]),1e-20))
        date=packed.dates[i]
        stats["daily_nll"][date][0]+=nll
        stats["daily_nll"][date][1]+=1
        bands=band_index(q)
        for bidx,label in enumerate(BANDS):
            chosen=bands==bidx
            n=int(np.sum(chosen))
            if not n:continue
            mass=float(np.sum(q[chosen]))
            h=int(bands[local_true]==bidx)
            s=stats["bands"][label]
            s["ticket_count"]+=n;s["expected_hits"]+=mass
            s["actual_hits"]+=h
            s["sum_race_m2"]+=mass*(1-mass)
        for ci,f in enumerate(STRUCTURE_FEATURES,start=1):
            fvals=xx[:,ci]
            predicted=float(q@fvals)
            actual=int(fvals[local_true])
            s=stats["structures"][f]
            s["races"]+=1;s["actual"]+=actual;s["expected"]+=predicted
            s["sum_variance"]+=max(0.,predicted*(1-predicted))
        if example is not None and packed.riders[i]==7 and len(example)==0:
            example.update({
                "race_id":packed.ids[i],"date":date,
                "true_result_ticket_index":local_true,
                "total_probability":float(q.sum()),
                "top10_probs_pct":list(np.round(np.sort(q)[-10:][::-1]*100,8)),
                "actual_ticket_probability_pct":round(float(q[local_true]*100),8),
                "all_210_probs_pct_ordered_by_baseline_ticket_key":np.round(q*100,10).tolist()
            })
    return stats

def summary(stats):
    results={"overall":finish_stat(stats["overall"])}
    total_tickets=0;abs_error=0;hi_tickets=0;hi_abs=0
    bins={}
    for label,s in stats["bands"].items():
        n=s["ticket_count"]
        if not n:continue
        exp=s["expected_hits"];obs=s["actual_hits"]
        se=math.sqrt(max(0,s["sum_race_m2"]))
        record={
          "ticket_count":n,
          "mean_predicted_pct":round(100*exp/n,7),
          "actual_frequency_pct":round(100*obs/n,7),
          "expected_hits":round(exp,3),"actual_hits":obs,
          "observed_expected_ratio":round(obs/exp,5) if exp else None,
          "z_score_if_well_calibrated":round((obs-exp)/se,4) if se else None,
          "reference_95pct_hits":[round(exp-1.96*se,2),round(exp+1.96*se,2)]
        }
        bins[label]=record
        total_tickets+=n;abs_error+=abs(exp-obs)
        if label in BANDS[5:]:
            hi_tickets+=n;hi_abs+=abs(exp-obs)
    results["bands"]=bins
    results["ticket_weighted_ECE_pp"]=round(100*abs_error/total_tickets,7)
    results["above_5pct_weighted_ECE_pp"]=round(100*hi_abs/hi_tickets,7) if hi_tickets else None
    struct={}
    for label,s in stats["structures"].items():
        n=s["races"];exp=s["expected"];obs=s["actual"]
        se=math.sqrt(s["sum_variance"])
        struct[label]={
          "races":n,"expected_frequency_pct":round(100*exp/n,6),
          "actual_frequency_pct":round(100*obs/n,6),
          "actual_minus_expected_pp":round(100*(obs-exp)/n,6),
          "z_score":round((obs-exp)/se,5) if se else None
        }
    results["structure"]=struct
    for key in ("by_class","by_field","by_month"):
        results[key]={label:finish_stat(s) for label,s in sorted(stats[key].items())}
    frequent=sorted(stats["by_line_shape"],
                    key=lambda k:stats["by_line_shape"][k]["races"],reverse=True)[:12]
    results["by_line_shape_top12"]={k:finish_stat(stats["by_line_shape"][k]) for k in frequent}
    return results

def daily_delta_ci(first,other):
    days=sorted(set(first["daily_nll"]) & set(other["daily_nll"]))
    total=sum(first["daily_nll"][d][1] for d in days)
    raw=sum((first["daily_nll"][d][0]-other["daily_nll"][d][0]) for d in days)
    mean=raw/total  # positive means other calibration has lower NLL
    se=math.sqrt(sum((first["daily_nll"][d][0]-other["daily_nll"][d][0] -
                      mean*first["daily_nll"][d][1])**2 for d in days))/total
    return {
        "days":len(days),"nll_baseline_minus_calibrated":round(mean,9),
        "day_cluster_standard_error":round(se,9),
        "approx_95pct_interval":[round(mean-1.96*se,9),round(mean+1.96*se,9)]
    }

def main():
    if not BASE_ARCHIVE.is_file():
        raise RuntimeError("2024 raw probability artifact must be downloaded first.")
    sha=hashlib.sha256(BASE_ARCHIVE.read_bytes()).hexdigest()
    H1,H2,coverage,pred_count=load()
    print("TEMPORAL_SPLIT_2024","H1",H1.n,"H2",H2.n,flush=True)
    theta_base=np.array([1.0]+[0.0]*7)
    theta_temp,temp_fit=fit(H1,[0],"temperature_only")
    theta_struct,struct_fit=fit(H1,list(range(len(FEATURES))),"temperature_plus_line_role")
    trials={
      "baseline_2023":theta_base,
      "temperature_only":theta_temp,
      "temperature_plus_line_role":theta_struct
    }
    test_stats={};test_results={};example={}
    for name,coef in trials.items():
        s=analyze(H2,coef,make_stats(),example if name=="temperature_plus_line_role" else None)
        test_stats[name]=s
        test_results[name]=summary(s)
        print("SECOND_HALF_VALIDATION",name,test_results[name]["overall"],
              "high5_ece",test_results[name]["above_5pct_weighted_ECE_pp"],flush=True)
    comparisons={
      k:daily_delta_ci(test_stats["baseline_2023"],test_stats[k])
      for k in ("temperature_only","temperature_plus_line_role")
    }
    primary=test_results["temperature_plus_line_role"]
    baseline=test_results["baseline_2023"]
    primary_nll_better=primary["overall"]["nll"]<baseline["overall"]["nll"]
    primary_brier_better=primary["overall"]["brier"]<=baseline["overall"]["brier"]
    primary_high_ece_better=primary["above_5pct_weighted_ECE_pp"]<=baseline["above_5pct_weighted_ECE_pp"]
    # 2024 H2 is a DEVELOPMENT check, not a final genuinely unseen test:
    # Earlier 2024-wide audit was reviewed before selecting this model family.
    gate=bool(primary_nll_better and primary_brier_better and primary_high_ece_better)
    print("2024_H2_PRIMARY_GATE",gate,"nll",primary_nll_better,
          "brier",primary_brier_better,"high5_ece",primary_high_ece_better,flush=True)
    # Explicit final refit: ALL of 2024 may be used to fit ONLY the correction
    # layer, now that the H1 vs H2 temporal comparison has been saved.
    together=Packed(
      np.concatenate((H1.x,H2.x),axis=0),
      np.concatenate((H1.starts,H2.starts+len(H1.x))),
      np.concatenate((H1.counts,H2.counts)),
      np.concatenate((H1.truth_idx,H2.truth_idx+len(H1.x))),
      H1.dates+H2.dates,H1.ids+H2.ids,H1.categories+H2.categories,
      H1.line_shapes+H2.line_shapes,H1.riders+H2.riders)
    if gate:
        final_theta,full_fit=fit(together,list(range(len(FEATURES))),"full2024_structural")
        final_type="temperature_plus_line_role"
    else:
        temp_results=test_results["temperature_only"]
        temp_gate=bool(temp_results["overall"]["nll"]<baseline["overall"]["nll"] and
                      temp_results["overall"]["brier"]<=baseline["overall"]["brier"] and
                      temp_results["above_5pct_weighted_ECE_pp"]<=baseline["above_5pct_weighted_ECE_pp"])
        if temp_gate:
            final_theta,full_fit=fit(together,[0],"full2024_temperature_only")
            final_type="temperature_only"
        else:
            final_theta,full_fit=theta_base,{"model":"unchanged_2023_base","trained_races":0}
            final_type="no_change"
    report={
      "state":"CALIBRATOR_V2_2024_COMPLETE",
      "2023_base_archive_run":40807880319,
      "2023_probability_archived_source_sha256":sha,
      "raw_probability_archive":"data/audits/independent_trifecta_all_tickets_2024.jsonl.gz",
      "2024_labels_only_for_calibration":True,"market_odds_seen":False,
      "2025_test_labels_seen":False,
      "train_period":["2024-01-01","2024-06-30"],
      "temporal_development_validation_period":["2024-07-01","2024-12-31"],
      "note_on_independence":"2024 all-year aggregate had already been examined before this experiment, so H2 is chronological development validation, not a genuinely sealed test. 2025 must remain sealed.",
      "first_half_races":H1.n,"second_half_races":H2.n,
      "full_year_races":together.n,"full_year_ticket_predictions":pred_count,
      "data_coverage":coverage,
      "frozen_2023_base_model_not_modified":True,
      "features":FEATURES,
      "log_linear_probability_formula":"P_v2(t|X) = exp(sum_j(theta_j * feature_j(t,X))) / sum_u exp(sum_j(theta_j * feature_j(u,X))), where feature_0=ln(P_v1(t|X))",
      "regularization_strength":L2_STRENGTH,"training_method":"L-BFGS-B penalized multiclass negative log-likelihood on 2024 H1",
      "first_half_fit":{"temperature_only":temp_fit,"temperature_plus_line_role":struct_fit},
      "second_half_temporal_validation":test_results,
      "second_half_nll_improvement":comparisons,
      "research_primary_gate":{
        "passes_all_three":gate,"nll_better":primary_nll_better,
        "brier_better":primary_brier_better,"above_5pct_ece_better":primary_high_ece_better
      },
      "final_selected_type_for_sealed_2025":final_type,
      "full_2024_correction_fit":full_fit,
      "final_for_2025_sha256_fields":dict(zip(FEATURES,np.round(final_theta,11).tolist())),
      "limitations":[
        "The baseline 2023 formula was learned on historical archived inputs, whose actual pre-race availability has not been independently certified.",
        "The probabilities from 2024 were already inspected in an all-year audit before this correction design; only the 2025 result can establish a new untouched-period improvement.",
        "2024 H2 figures are based ONLY on correction coefficients fitted on 2024 H1; do not compare 2024 full-year refit against 2024 in-sample scores as independent validation.",
        "2025 and 2026 not read, no market odds, no payouts or ROI.",
        "Class-specific structures are reported but coefficients are global to minimize overfitting.",
        "Probability mass always sums to one over all exact trifecta permutations per race."
      ]
    }
    weights={
      "schema":"frozen_keirin_3trifecta_no_odds_postcalibrator_v2",
      "model_type":final_type,
      "source_2023_formula":"research/independent_trifecta_2023/observe.py",
      "base_probability_artifact_2024_run_id":40807880319,
      "input":"P_v1 for every exact ordered podium outcome plus the pre-race line and score ranks",
      "feature_order":FEATURES,
      "coefficients":[float(round(x,11)) for x in final_theta],
      "formula":"softmax(theta_0 * log(P_v1) + theta_1 * I_12 + theta_2 * I_13 + theta_3 * I_23 + theta_4 * I_123 + theta_5 * I_all_distinct + theta_6 * I_winner_leader + theta_7 * I_winner_highest_score)",
      "normalization":"whole-race exact-trifecta grid; sum=1",
      "fit_period":["2024-01-01","2024-12-31"],
      "2024_split_validation_must_use_first_half_coefficients_only":True,
      "holdout_2025_not_used":True,
      "no_odds_features":True,
      "selection":"Full-year fit is created only after temporal development validation; no in-sample 2024 performance claim",
      "limits":"Retrospective archive availability was not independently certified"
    }
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    WEIGHTS.write_text(json.dumps(weights,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    EXAMPLE.write_text(json.dumps(example,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V2_CALIBRATION_FINISHED",json.dumps({
      "H1":H1.n,"H2":H2.n,"selected":final_type,
      "H2_baseline":baseline["overall"],"H2_corrected":primary["overall"],
      "high5_ece_baseline_pp":baseline["above_5pct_weighted_ECE_pp"],
      "high5_ece_corrected_pp":primary["above_5pct_weighted_ECE_pp"],
      "delta":comparisons["temperature_plus_line_role"],
      "frozen_for_2025":weights["coefficients"]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()

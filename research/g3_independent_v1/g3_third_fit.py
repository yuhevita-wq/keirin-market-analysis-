#!/usr/bin/env python3
"""G3-only third-place conditional law discovery AFTER third result census.

3rd model input: fixed prior-year G3 independent win probabilities q_k, plus
predicted line structure and the actual first and second finishers as conditioning
variables ONLY in historical conditional-likelihood study.

2022: in-sample first win q, included in empirical discovery only.
2023-24: conditional third likelihood fit (G3 first q from prior years).
2025: candidate mathematical family selection, by min conditional log-loss.
2023-25: refit fixed family, no 2026 outcomes used.
2026 H1: chronological evaluation, check line position/role calibration.
Neither first/second G3 model coefficients nor F1 models are changed.
"""
from __future__ import annotations
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from g3_third_empirical import HERE,STAGE,OUT,FEATURES,compute_census

FAMILIES={
    "harville":("log_q",),
    "benter_horse_delta_065":("log_q",),
    "g3_delta_only":("log_q",),
    "g3_delta_plus_line_membership":(
        "log_q","owner_both_line","owner_first_line_only","owner_second_line_only"),
    "g3_delta_plus_rider_role":(
        "log_q","role_second","role_third_or_later","role_solo"),
    "g3_delta_plus_membership_and_role":(
        "log_q","owner_both_line","owner_first_line_only","owner_second_line_only",
        "role_second","role_third_or_later","role_solo"),
    "g3_delta_plus_directed_position":(
        "log_q","direction_ahead_of_both","direction_between_first_second",
        "direction_behind_both","direction_ahead_of_first","direction_directly_behind_first",
        "direction_further_behind_first","direction_ahead_of_second",
        "direction_directly_behind_second","direction_further_behind_second"),
    "g3_delta_plus_direction_and_role":(
        "log_q","direction_ahead_of_both","direction_between_first_second",
        "direction_behind_both","direction_ahead_of_first","direction_directly_behind_first",
        "direction_further_behind_first","direction_ahead_of_second",
        "direction_directly_behind_second","direction_further_behind_second",
        "role_second","role_third_or_later","role_solo"),
}
BINS=(0,.01,.025,.05,.075,.1,.15,.2,.25,.3,.4,.5,.6,.7,1.00000001)

def feature_indices(name):
    return [FEATURES.index(k) for k in FAMILIES[name]]

def conditional_distribution(name,weights,X,M):
    sub=(X[:,:,0]*weights[0] if name in ("harville","benter_horse_delta_065")
         else X[:,:,feature_indices(name)]@weights)
    sub=np.where(M,sub,-1e8)
    p=np.exp(sub-logsumexp(sub,axis=1,keepdims=True))
    if not np.isfinite(p).all() or np.max(np.abs(p.sum(axis=1)-1))>1e-9:
        raise RuntimeError("Not normalized conditional third predictions")
    return p

def estimate(name,X,M,y):
    if name=="harville":return np.array([1.0])
    if name=="benter_horse_delta_065":return np.array([0.65])
    selected=X[:,:,feature_indices(name)]
    nn=len(y);positions=np.arange(nn)
    def objective(beta):
        z=np.where(M,selected@beta,-1e8)
        lz=logsumexp(z,axis=1)
        probs=np.exp(z-lz[:,None])
        nll=np.mean(lz-z[positions,y])
        grad=np.mean(
            np.einsum("rnf,rn->rf",selected,probs,optimize=True)-selected[positions,y],
            axis=0
        )
        return nll,grad
    initial=np.zeros(len(feature_indices(name)));initial[0]=1.
    bounds=[(.05,2.5)]+[(-6.,6.)]*(len(initial)-1)
    fit=minimize(objective,initial,method="L-BFGS-B",jac=True,bounds=bounds,
                 options={"maxiter":900,"gtol":1e-9,"ftol":1e-13})
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise RuntimeError("G3 3rd MLE did not produce finite result for "+name)
    if not fit.success and np.max(np.abs(fit.jac))>.0002:
        raise RuntimeError(f"G3 3rd MLE not converged for {name}: {fit.message}")
    return fit.x

def metric(name,co,X,M,y,detailed=False):
    pp=conditional_distribution(name,co,X,M)
    idx=np.arange(len(y))
    ll=-np.log(np.maximum(pp[idx,y],1e-300))
    obs=np.zeros_like(pp,dtype=bool);obs[idx,y]=True
    uniform=np.mean(np.log(M.sum(axis=1)))
    out={
        "races":len(y),
        "third_candidate_rows":int(M.sum()),
        "log_loss":float(ll.mean()),
        "uniform_baseline_log_loss":float(uniform),
        "pseudo_r2_against_conditional_uniform":float(1-ll.mean()/uniform),
        "top_predicted_third_given_actual_first_second_accuracy":
            float((pp.argmax(axis=1)==y).mean()),
        "multiclass_brier":float(np.square(pp-obs.astype(float)).sum(axis=1).mean())
    }
    if detailed:
        groups={}
        for label in list(FEATURES[1:]):
            ix=FEATURES.index(label)
            which=M&(X[:,:,ix]>.5)
            count=int(which.sum())
            groups[label]={
                "candidate_rows":count,
                "expected_third":float(pp[which].sum()),
                "actual_third":int(obs[which].sum()),
                "calibration_error_actual_minus_predicted":
                    float(obs[which].sum()-pp[which].sum())
            }
        for kind,start,end in (("other_line",1,4),("neither_line",4,13),
                               ("role_leader",13,16)):
            if kind=="role_leader":
                # Role-leader among multirider lines; feature baseline includes role_leader.
                # Since no unique solo? solo is recorded as explicit role and leader is remainder.
                which=M&(~(X[:,:,start:end]>.5).any(axis=2))
            else:
                which=M&(~(X[:,:,start:end]>.5).any(axis=2))
            groups[kind]={
                "candidate_rows":int(which.sum()),
                "expected_third":float(pp[which].sum()),
                "actual_third":int(obs[which].sum())
            }
        out["line_group_calibration"]=groups
        out["probability_bins"]=[]
        p=pp[M];actual=obs[M]
        for low,high in zip(BINS[:-1],BINS[1:]):
            take=(p>=low)&(p<high)
            if take.any():
                out["probability_bins"].append({
                    "range":[low,min(high,1.)],
                    "candidate_rows":int(take.sum()),
                    "predicted":float(p[take].sum()),
                    "actual":int(actual[take].sum())
                })
    return out,ll

def day_block_bootstrap(loss_base,loss_new,dates,reps=2500):
    groups=collections.defaultdict(list)
    for a,b,day in zip(loss_base,loss_new,dates):
        groups[day].append(float(a-b))
    keys=sorted(groups)
    sums=np.array([sum(groups[k]) for k in keys])
    nums=np.array([len(groups[k]) for k in keys])
    rng=np.random.default_rng(20261018)
    picks=rng.integers(len(keys),size=(reps,len(keys)))
    vals=sums[picks].sum(axis=1)/nums[picks].sum(axis=1)
    return {
        "point_improvement_positive_is_new_better":float(sums.sum()/nums.sum()),
        "95pct_day_cluster_bootstrap":[float(x) for x in np.quantile(vals,[.025,.975])],
        "days":len(keys)
    }

def fit_third():
    observed_path=OUT/"g3_third_empirical.json"
    if not observed_path.exists():raise RuntimeError("Must observe actual third results before fitting")
    empirical=json.loads(observed_path.read_text(encoding="utf-8"))
    if empirical["status"]!="G3_THIRD_RESULT_FIRST_EMPIRICAL_BEFORE_FITTING":
        raise RuntimeError("Incompatible G3 third census")
    index=json.loads((STAGE/"g3_third_index.json").read_text(encoding="utf-8"))
    npz=np.load(STAGE/"g3_third_matrix.npz")
    X=npz["X"];M=npz["mask"];y=npz["y"]
    if len(index)!=empirical["eligible_unique_first_second_third_races"] or int(M.sum())!=empirical["candidate_third_rows"]:
        raise RuntimeError("Empirical cohort mismatch")
    dev=np.array([i for i,r in enumerate(index) if "2023-01-01"<=r["date"]<"2025-01-01"],dtype=int)
    val=np.array([i for i,r in enumerate(index) if "2025-01-01"<=r["date"]<"2026-01-01"],dtype=int)
    check=np.array([i for i,r in enumerate(index) if "2026-01-01"<=r["date"]<"2026-07-01"],dtype=int)
    if min(len(dev),len(val),len(check))<900:raise RuntimeError("Insufficient chronological G3 third cohort")
    report={
        "status":"G3_INDEPENDENT_THIRD_CONDITIONAL_PROBABILITY_MODEL_DISCOVERED",
        "scope":"G3 S級 only; no F1 1st/2nd/3rd coefficients",
        "empirical_first":observed_path.name,
        "full_valid_real_first_second_third_races":len(index),
        "full_third_candidate_rows":int(M.sum()),
        "fit_2023_2024":len(dev),
        "model_select_2025":len(val),
        "chronological_evaluation_2026h1":len(check),
        "prior_year_only_G3_first_probabilities":True,
        "2022_first_q_in_sample_used_only_in_census":True,
        "conditional_third_given_actual_first_and_second":True,
        "inference_without_known_finish":"P(i,j,k)=q_i P_G3_second(j|i) P_G3_third(k|i,j), enumerate all ordered distinct triples",
        "equation":"P(k third|i first,j second) = q_k^delta exp(D_line_position(i,j,k)+R_line_role(k)) / sum_{h not in (i,j)} q_h^delta exp(D_line_position(i,j,h)+R_line_role(h))",
        "family_columns":{name:list(cols) for name,cols in FAMILIES.items()},
        "G3_first_model_unchanged":True,"G3_second_model_unchanged":True,
        "no_F1_parameters":True,"no_market_odds":True,
        "model_selection":"Best 2025 conditional-third log loss among all G3-sourced families, refit on G3 2023-2025, evaluate 2026H1",
        "families":{},
        "day_cluster_pairwise":{},
        "predeadline_historical_snapshots_verified":False
    }
    for name in FAMILIES:
        beta=estimate(name,X[dev],M[dev],y[dev])
        val_metric,_=metric(name,beta,X[val],M[val],y[val])
        report["families"][name]={
            "names":list(FAMILIES[name]),
            "dev_2023_24_coefficients":list(map(float,beta)),
            "validation_2025":val_metric
        }
        print("G3_THIRD_VALIDATION",name,"loss",round(val_metric["log_loss"],9),
              "coefficients",list(zip(FAMILIES[name],np.round(beta,5))),flush=True)
    ranking=sorted(FAMILIES,key=lambda n:report["families"][n]["validation_2025"]["log_loss"])
    selected=min((n for n in FAMILIES if n not in ("harville","benter_horse_delta_065")),
                 key=lambda n:report["families"][n]["validation_2025"]["log_loss"])
    report["ranking_validation_2025"]=ranking
    report["selected_family"]=selected
    refit=np.concatenate([dev,val])
    losses={}
    for name in FAMILIES:
        beta=estimate(name,X[refit],M[refit],y[refit])
        check_metric,loss=metric(name,beta,X[check],M[check],y[check],detailed=True)
        report["families"][name]["refit_2023_25_coefficients"]=list(map(float,beta))
        report["families"][name]["evaluation_2026h1"]=check_metric
        report["families"][name]["development_descriptive_log_loss"]=metric(name,beta,X[refit],M[refit],y[refit])[0]["log_loss"]
        losses[name]=loss
        print("G3_THIRD_LATER_EVALUATION",name,"loss",round(check_metric["log_loss"],9),
              "top1",round(check_metric["top_predicted_third_given_actual_first_second_accuracy"],6),
              flush=True)
    report["selected_evaluation_2026h1"]=report["families"][selected]["evaluation_2026h1"]
    dates=[index[i]["date"] for i in check]
    for base,new in (
        ("harville",selected),
        ("g3_delta_only",selected),
        ("g3_delta_plus_line_membership",selected),
        ("g3_delta_plus_directed_position",selected),
        ("g3_delta_plus_directed_position","g3_delta_plus_direction_and_role"),
        ("g3_delta_plus_line_membership","g3_delta_plus_directed_position")
    ):
        if base==new:continue
        report["day_cluster_pairwise"][new+"_OVER_"+base]=day_block_bootstrap(
            losses[base],losses[new],dates)
    report["rank_by_2026h1_evaluation"]=sorted(
        FAMILIES,key=lambda n:report["families"][n]["evaluation_2026h1"]["log_loss"])
    report["research_vs_prospective_warning"]="Historical cards are retrospectively downloaded. The 2026H1 conditional evaluation assumes actual first+second known, not 3連単 pre-race log loss. Role effect may be unproven; inspect day-block bootstrap."
    out=OUT/"g3_third_model_report.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_THIRD_COMPLETE",json.dumps({
        "races":len(index),"candidates":int(M.sum()),
        "dev":len(dev),"val":len(val),"test":len(check),
        "chosen":selected,"val_ranking":ranking,
        "test_rank":report["rank_by_2026h1_evaluation"],
        "test":report["selected_evaluation_2026h1"],
        "paired":report["day_cluster_pairwise"]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    fit_third()

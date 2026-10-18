#!/usr/bin/env python3
"""G3-S級 exclusive second-place probability from REAL finishes, then fitted rules.

Phase 1 observes EVERY valid G3 actual second finisher against G3-only first
probabilities and freezes that empirical census. Phase 2 estimates conditional
probability from those observed residues; the F1 second-place model is NEVER used.

G3 first probabilities are re-estimated by prior year for 2023-26. The 2022
probabilities are necessarily in-sample (no 2021 G3), retained ONLY for empirical
census, never as 2023-25 second-model training cases. Development: 2023-24,
selection: 2025, chronological evaluation: 2026H1.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from train_g3_first import load,FEATURE_NAMES,select,fit,transform_for,matrix,OUT

HERE=Path(__file__).resolve().parent
STAGE=HERE/"stage"
FEATURES=("log_q","same_line","immediately_behind_first","ahead_of_first",
          "further_behind_first","role_second","role_third_or_later","role_solo")
MODELS={
 "harville":("log_q",),
 "benter_horse_gamma_081":("log_q",),
 "g3_gamma_only":("log_q",),
 "g3_gamma_plus_same_line":("log_q","same_line"),
 "g3_gamma_plus_directed_position":("log_q","immediately_behind_first","ahead_of_first","further_behind_first"),
 "g3_gamma_plus_position_and_role":("log_q","immediately_behind_first","ahead_of_first","further_behind_first",
                                   "role_second","role_third_or_later","role_solo"),
}
REF_FIRST_ALPHA=0.0001
REF_FIRST_FEATURES=tuple(FEATURE_NAMES)

def first_by_calendar(races):
    """Refit same G3 first *specification* for each target year using earlier G3 only."""
    q={}
    fits=[]
    for year in (2022,2023,2024,2025,2026):
        target=[r for r in races if r["date"].startswith(str(year))]
        if not target:continue
        old=([r for r in races if r["date"][:4]<str(year)] if year>2022
             else [r for r in races if r["date"].startswith("2022")])
        if len(old)<500:raise RuntimeError("G3-first warmup population insufficient")
        oldview=select(old,REF_FIRST_FEATURES)
        tr=transform_for(oldview,REF_FIRST_FEATURES)
        beta,opt=fit(oldview,tr,REF_FIRST_ALPHA)
        X,M,y=matrix(select(target,REF_FIRST_FEATURES),tr)
        logits=np.einsum("rnf,f->rn",X,beta,optimize=True)
        logits=np.where(M,logits,-1.e8)
        pp=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        for n,r in enumerate(target):
            p=pp[n,:len(r["cars"])]
            if not np.isfinite(p).all() or abs(p.sum()-1)>1e-9:
                raise ValueError("Invalid first forecast")
            if r["id"] in q:raise ValueError("Duplicate G3 race")
            q[r["id"]]=p
        fits.append({
            "prediction_year":year,"training_races":len(old),
            "target_races":len(target),
            "trained_on_years":sorted(set(r["date"][:4] for r in old)),
            "target_is_in_sample":year==2022,
            "optimizer_converged":opt["success"],
            "coefficient_sha256":hashlib.sha256(beta.tobytes()).hexdigest()
        })
        print("G3_FIRST_FOR_SECOND",year,"trained",len(old),"predicted",len(target),
              "in_sample_reference_only",year==2022,flush=True)
    return q,fits

def row_for_candidate(i,j,car,qi,positions):
    wi,wp=positions[i]
    candidate_line,cp=positions[car]
    same=candidate_line==wi
    n=len(candidate_line)
    return [
        math.log(max(qi,1e-15)),float(same),
        float(same and cp==wp+1),float(same and cp<wp),
        float(same and cp>wp+1),
        float(n>=2 and cp==1),float(cp>=2),float(n==1),
    ]

def prepare():
    races,source=load()
    OUT.mkdir(parents=True,exist_ok=True)
    STAGE.mkdir(parents=True,exist_ok=True)
    q,fits=first_by_calendar(races)
    eligible=[]
    omitted=collections.Counter()
    for r in races:
        car=r.get("actual_second_car")
        if car is None or car not in r["cars"] or car==r["cars"][r["winner"]]:
            omitted["no_unique_valid_second"]+=1
            continue
        eligible.append(r)
    n=len(eligible)
    if n<8500:raise RuntimeError("Unexpected G3 second cohort loss")
    XX=np.zeros((n,9,len(FEATURES)),dtype=float)
    M=np.zeros((n,9),dtype=bool)
    Y=np.zeros(n,dtype=np.int64)
    sample=[];obs=[]
    for ix,r in enumerate(eligible):
        i=r["cars"][r["winner"]]
        j=r["actual_second_car"]
        positions={}
        for line in r["line"]:
            for k,c in enumerate(line):
                if c in positions:raise ValueError("Duplicate line role")
                positions[c]=(tuple(line),k)
        if set(positions)!=set(r["cars"]):raise ValueError("Published lineup incomplete")
        probs=q[r["id"]]
        for h,car in enumerate(r["cars"]):
            XX[ix,h]=row_for_candidate(i,j,car,probs[h],positions)
            M[ix,h]=car!=i
        Y[ix]=r["cars"].index(j)
        if not M[ix,Y[ix]]:raise ValueError("1st cannot also be 2nd")
        baseline=np.exp(np.where(M[ix],XX[ix,:,0],-1e8)-logsumexp(np.where(M[ix],XX[ix,:,0],-1e8)))
        obs.append({"id":r["id"],"year":r["date"][:4],"first":i,"second":j,
                    "winner_role":("solo" if len(positions[i][0])==1 else
                                   "leader" if positions[i][1]==0 else
                                   "second" if positions[i][1]==1 else "third_or_later"),
                    "features":XX[ix].tolist(),"cars":r["cars"],
                    "baseline":baseline.tolist()})
        sample.append({"id":r["id"],"date":r["date"],"cars":r["cars"],
                       "first":i,"second":j,"first_source_in_sample":r["date"].startswith("2022")})
    np.savez_compressed(STAGE/"g3_second_matrix.npz",X=XX,mask=M,y=Y)
    (STAGE/"g3_second_index.json").write_text(json.dumps(sample,ensure_ascii=False)+"\n",encoding="utf-8")
    def subset_summary(which):
        candidates=sum(int(M[ix].sum()) for ix in which)
        actual=sum(1 for _ in which)
        expected=collections.Counter()
        observed=collections.Counter()
        expected_position=collections.Counter()
        observed_position=collections.Counter()
        for ix in which:
            row=obs[ix]
            features=XX[ix]
            baseline=np.array(row["baseline"])
            for col,c in enumerate(row["cars"]):
                if not M[ix,col]:continue
                where="same" if features[col,1]>.5 else "other"
                direction=("immediately_behind" if features[col,2]>.5
                           else "ahead_of_first" if features[col,3]>.5
                           else "further_behind" if features[col,4]>.5
                           else "other")
                expected[where]+=float(baseline[col])
                expected_position[direction]+=float(baseline[col])
                if c==row["second"]:
                    observed[where]+=1
                    observed_position[direction]+=1
        return {
            "races":actual,"candidate_rows":candidates,
            "same_line_expected":expected["same"],"same_line_observed":observed["same"],
            "other_line_expected":expected["other"],"other_line_observed":observed["other"],
            "direction_expected":dict(expected_position),
            "direction_observed":dict(observed_position)
        }
    groups={
        "all_eligible_G3":subset_summary(range(n)),
        **{f"year_{year}":subset_summary([ix for ix,r in enumerate(eligible) if r["date"].startswith(str(year))])
           for year in range(2022,2027)},
    }
    report={
        "phase":"OBSERVE_ACTUAL_G3_SECOND_BEFORE_COEFFICIENT_FITTING",
        "scope":"G3 S級 only. No F1 rows, features coefficients or fit.",
        "all_G3_first_eligible_races":len(races),
        "all_unique_actual_G3_second_races":n,
        "all_second_candidate_rows":int(M.sum()),
        "exclusions":dict(omitted),
        "first_win_g3_prior_year_fits":fits,
        "same_line_and_position_real_result_census":groups,
        "baseline":"Harville q_second/(1-q_actual_first) using G3 independent first probabilities",
        "unique_result_order_second_used_as_target_only":True,
        "no_second_coefficient_fitted_yet":True,
        "date_design":"2022 first-q in-sample empirical reference; 2023-24 second fit; 2025 model selection; 2026H1 chronological model evaluation",
        "historical_entry_snapshots_pre_deadline_certified":False,
        "source_race_ids_sha256":source["source_race_ids_sha256"]
    }
    (OUT/"g3_second_empirical.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_2ND_RESULT_FIRST_CENSUS",json.dumps({k:report[k] for k in
          ("all_unique_actual_G3_second_races","all_second_candidate_rows","exclusions")},ensure_ascii=False),flush=True)
    print("G3_2ND_SAME_LINE",json.dumps(groups,ensure_ascii=False),flush=True)

def idx(name):return [FEATURES.index(x) for x in MODELS[name]]
def prob(name,coef,X,M):
    z=X[:,:,0]*coef[0] if name in ("harville","benter_horse_gamma_081") else X[:,:,idx(name)]@coef
    z=np.where(M,z,-1e8)
    return np.exp(z-logsumexp(z,axis=1,keepdims=True))

def estimate(name,X,M,y):
    if name=="harville":return np.array([1.])
    if name=="benter_horse_gamma_081":return np.array([.81])
    sub=X[:,:,idx(name)]
    rows=np.arange(len(y))
    def loss_grad(b):
        z=np.where(M,sub@b,-1e8)
        lm=logsumexp(z,axis=1)
        p=np.exp(z-lm[:,None])
        return np.mean(lm-z[rows,y]),np.mean(
            np.einsum("rnk,rn->rk",sub,p,optimize=True)-sub[rows,y],axis=0)
    start=np.zeros(len(idx(name)));start[0]=1.
    bounds=[(.05,2.5)]+[(-5,5)]*(len(start)-1)
    res=minimize(loss_grad,start,jac=True,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":1000,"gtol":1e-9,"ftol":1e-12})
    if not np.isfinite(res.fun) or not np.isfinite(res.x).all():
        raise RuntimeError("Second conditional MLE failed for "+name)
    return res.x

def metrics(name,b,X,M,y,more=False):
    p=prob(name,b,X,M)
    rows=np.arange(len(y))
    loss=-np.log(np.maximum(p[rows,y],1e-15))
    ans={
        "races":len(y),
        "conditional_second_log_loss":float(loss.mean()),
        "top_predicted_second_given_real_first_accuracy":float((p.argmax(axis=1)==y).mean()),
        "uniform_log_loss":float(np.mean(np.log(M.sum(axis=1))))
    }
    if more:
        ans["line_calibration"]={}
        observed=np.zeros_like(M,dtype=bool);observed[rows,y]=True
        for col,label in ((1,"same_line"),(2,"directly_behind_first"),(3,"ahead_of_first"),
                          (4,"further_behind_first")):
            take=M&(X[:,:,col]>.5)
            ans["line_calibration"][label]={
                "candidate_rows":int(take.sum()),
                "expected_second":float(p[take].sum()),
                "observed_second":int(observed[take].sum())
            }
    return ans,loss

def day_bootstrap(base,new,dates):
    groups=collections.defaultdict(list)
    for a,b,d in zip(base,new,dates):groups[d].append(float(a-b))
    days=sorted(groups)
    values=np.array([sum(groups[x]) for x in days])
    counts=np.array([len(groups[x]) for x in days])
    gen=np.random.default_rng(20261018)
    select=gen.integers(len(days),size=(1200,len(days)))
    sampled=values[select].sum(axis=1)/counts[select].sum(axis=1)
    return {"mean_logloss_improvement_positive_is_better":float(values.sum()/counts.sum()),
            "day_block_bootstrap95":[float(x) for x in np.quantile(sampled,[.025,.975])],
            "days":len(days)}

def fit_second():
    if not (OUT/"g3_second_empirical.json").exists():
        raise RuntimeError("Must observe real 2nd finish before fitting")
    a=json.loads((OUT/"g3_second_empirical.json").read_text(encoding="utf-8"))
    samples=json.loads((STAGE/"g3_second_index.json").read_text(encoding="utf-8"))
    npz=np.load(STAGE/"g3_second_matrix.npz")
    X=npz["X"];M=npz["mask"];y=npz["y"]
    if len(samples)!=a["all_unique_actual_G3_second_races"] or int(M.sum())!=a["all_second_candidate_rows"]:
        raise RuntimeError("Empirical census changed after first phase")
    fitidx=np.array([k for k,s in enumerate(samples) if "2023-01-01"<=s["date"]<"2025-01-01"],dtype=int)
    validx=np.array([k for k,s in enumerate(samples) if "2025-01-01"<=s["date"]<"2026-01-01"],dtype=int)
    testidx=np.array([k for k,s in enumerate(samples) if "2026-01-01"<=s["date"]<"2026-07-01"],dtype=int)
    if min(len(fitidx),len(validx),len(testidx))<900:
        raise RuntimeError("Too few 2nd cases in some G3 chronological cohort")
    report={
        "status":"G3_INDEPENDENT_SECOND_CONDITIONAL_MODEL_ESTIMATED_AND_EVALUATED",
        "scope":"G3 S級 only. NOT F1.",
        "formulas":{
            "general":"P(j second|i first)=q_j^gamma * exp(line_role_offset(i,j)) / sum_{h!=i} q_h^gamma * exp(line_role_offset(i,h))",
            "pre_race_joint":"P(i first,j second)=q_i P(j second|i first), for each ordered distinct pair",
            "same_line_is_sum_of_mutually_exclusive_directions":"same=directly_behind + ahead_of_first + further_behind; do not fit both together"
        },
        "empirical_reference_file":"g3_second_empirical.json",
        "all_eligible_g3_races":a["all_G3_first_eligible_races"],
        "eligible_2nd_races":a["all_unique_actual_G3_second_races"],
        "candidate_second_rows":a["all_second_candidate_rows"],
        "cohort_for_conditional_second_fit_2023_24":len(fitidx),
        "cohort_for_family_selection_2025":len(validx),
        "cohort_for_chronology_evaluation_2026h1":len(testidx),
        "first_probabilities_are_from_g3_only_expanding_prior_years":True,
        "2022_in_sample_first_predictions_excluded_from_second_fitting":True,
        "results_prior_to_2023_only_used_for_first_probability_warmup":True,
        "model_families":{},
        "model_selection":"Lowest conditional second log-loss on 2025; then refit 2023-25 and evaluate on 2026H1",
        "old_F1_coefficients_reused":False,
        "source_snapshots_predeadline_verified":False,
    }
    winner_first_known_test_losses={}
    for name in MODELS:
        b=estimate(name,X[fitidx],M[fitidx],y[fitidx])
        val,_=metrics(name,b,X[validx],M[validx],y[validx])
        report["model_families"][name]={
            "feature_names":list(MODELS[name]),
            "fitted_on_2023_2024":list(map(float,b)),
            "validation_2025":val,
        }
        print("G3_SECOND_CANDIDATE",name,"validationLL",round(val["conditional_second_log_loss"],8),
              "params",list(zip(MODELS[name],map(lambda x:round(x,5),b))),flush=True)
    eligible=[m for m in MODELS if m not in ("harville","benter_horse_gamma_081")]
    selected=min(eligible,key=lambda m:report["model_families"][m]["validation_2025"]["conditional_second_log_loss"])
    trained=np.concatenate((fitidx,validx))
    ranking=sorted(MODELS,key=lambda m:report["model_families"][m]["validation_2025"]["conditional_second_log_loss"])
    report["validation_ranking"]=ranking
    report["selected_family"]=selected
    for name in MODELS:
        # Refit using all available pre-2026H1 independent q; validation model
        # selection fixed before looking at the 2026H1 results.
        bb=estimate(name,X[trained],M[trained],y[trained])
        chk,ll=metrics(name,bb,X[testidx],M[testidx],y[testidx],more=True)
        report["model_families"][name]["fit_on_2023_2025"]=list(map(float,bb))
        report["model_families"][name]["evaluation_2026h1"]=chk
        winner_first_known_test_losses[name]=ll
    report["selected_2026h1_evaluation"]=report["model_families"][selected]["evaluation_2026h1"]
    report["daily_cluster_comparisons"]={
        selected+"_OVER_harville":day_bootstrap(
            winner_first_known_test_losses["harville"],winner_first_known_test_losses[selected],
            [samples[k]["date"] for k in testidx]),
        selected+"_OVER_g3_gamma_only":day_bootstrap(
            winner_first_known_test_losses["g3_gamma_only"],winner_first_known_test_losses[selected],
            [samples[k]["date"] for k in testidx])
    }
    if selected != "g3_gamma_plus_same_line":
        report["daily_cluster_comparisons"][selected+"_OVER_g3_gamma_plus_same_line"]=day_bootstrap(
            winner_first_known_test_losses["g3_gamma_plus_same_line"],winner_first_known_test_losses[selected],
            [samples[k]["date"] for k in testidx])
    report["conditional_2nd_given_actual_1st_is_not_unconditional_exacta_accuracy"]=True
    report["pre_race_no_actual_first_or_second_inputs"]=True
    (OUT/"g3_second_model_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_SECOND_FINAL",json.dumps({
        "selected_family":selected,
        "cohorts":{k:report[k] for k in
                  ("eligible_2nd_races","candidate_second_rows","cohort_for_conditional_second_fit_2023_24",
                   "cohort_for_family_selection_2025","cohort_for_chronology_evaluation_2026h1")},
        "selected_coefficients":report["model_families"][selected]["fit_on_2023_2025"],
        "validation_ranking":ranking,
        "holdout":report["selected_2026h1_evaluation"],
        "improvements":report["daily_cluster_comparisons"]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["empirical","fit"],required=True)
    opts=parser.parse_args()
    if opts.phase=="empirical":prepare()
    else:fit_second()

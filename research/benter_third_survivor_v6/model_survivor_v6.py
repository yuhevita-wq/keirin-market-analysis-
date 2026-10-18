#!/usr/bin/env python3
"""Discover and model keirin 3rd-place residual-line structure.

Locked source models: V2 first-win forecasts, V4 calibrated q, V5 conditional
second/third probabilities. Candidate third features depend on hypothetical
1st+2nd, never on future results. Fit candidate residual line effects on
July 2026, validate on August, freeze the chosen penalty/features and
refit July-Aug. Score September, then historical 2026 H1 with NO refit.
Note global Sept is not a pristine test because V5 already analyzed it.

Run: python model_survivor_v6.py fit | score-sep | score-h1
"""
from __future__ import annotations
import csv,json,math,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"research/benter_line_conditional_v5"))
from model_line_conditional import get_races,cprob,calibrate,parse_line
V5=ROOT/"research/benter_line_conditional_v5/parameters_pretest.json"
H2=ROOT/"research/benter_score_challenge_v3/frozen_2026h2_probabilities.csv"
H2RACES=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/races.csv"
H2RESULT=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/results.csv"
SEP=ROOT/"research/benter_calibration_v4/sep_predictions_frozen.csv"
SEPRACES=ROOT/"research/benter_calibration_v4/sep_holdout/races.csv"
SEPRESULT=ROOT/"research/benter_calibration_v4/sep_holdout/results.csv"
H1=ROOT/"research/benter_structural_discovery_v2/output/test_full_plus_line_win_probability.csv"
H1RACES=ROOT/"data/2026_h1/s_class_f1_all/races.csv"
H1RESULT=ROOT/"data/2026_h1/s_class_f1_all/results.csv"
PREFIT=HERE/"frozen_3rd_survivor_parameters.json"
COARSE=[
    "same_pair__third_teammate",
    "same_pair__third_outside",
    "diff_pair__third_first_line",
    "diff_pair__third_second_line",
    "diff_pair__third_other_line",
    "remaining_teammates_in_candidate_line",
    "candidate_only_survivor_in_line",
    "candidate_has_two_plus_further_survivors",
]
DETAILED=COARSE+[
    "same_pair__third_teammate_tail",
    "same_pair__third_outside_leader",
    "same_pair__third_outside_bante",
    "diff_pair__third_first_line_leader",
    "diff_pair__third_second_line_leader",
    "diff_pair__third_other_line_leader",
    "diff_pair__third_first_line_bante",
    "diff_pair__third_second_line_bante",
    "diff_pair__third_other_line_bante",
    "diff_pair__third_first_line_tail",
    "diff_pair__third_second_line_tail",
    "diff_pair__third_other_line_tail",
    "winner_leader_candidate_behind",
    "second_leader_candidate_behind",
    "same_pair__front_bante_tail_finished",
]
MODELS={"coarse":COARSE,"detailed":DETAILED}
PENALTIES=[.01,.05,.2,.75,3,12]

def row_features(r,i,j):
    """One fixed feature vector per *candidate* third car given hypothesized i,j."""
    n=len(r["cars"]);line=r["line_idx"];pos=r["pos_idx"];size=r["size"]
    same=(line[i]==line[j])
    feat=np.zeros((n,len(DETAILED)),dtype=float)
    for k in range(n):
        if k in (i,j):continue
        s1=(line[k]==line[i]);s2=(line[k]==line[j])
        other=not(s1 or s2)
        stage3_candidates=sum(int(line[t]==line[k]) for t in range(n) if t not in(i,j))
        companions=stage3_candidates-1
        role=pos[k]
        front=(role==0);bante=(role==1);tail=(role>=2)
        data=[
            same and s1,
            same and other,
            (not same) and s1,
            (not same) and s2,
            (not same) and other,
            companions,
            companions==0,
            companions>=2,
            same and s1 and tail,
            same and other and front,
            same and other and bante,
            (not same) and s1 and front,
            (not same) and s2 and front,
            (not same) and other and front,
            (not same) and s1 and bante,
            (not same) and s2 and bante,
            (not same) and other and bante,
            (not same) and s1 and tail,
            (not same) and s2 and tail,
            (not same) and other and tail,
            s1 and pos[i]==0 and pos[k]>pos[i],
            s2 and pos[j]==0 and pos[k]>pos[j],
            same and s1 and pos[i]==0 and pos[j]==1 and pos[k]==2,
        ]
        feat[k]=data
    return feat

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def get_h1():
    p=defaultdict(list)
    for r in read(H1):p[r["race_id"]].append(r)
    racecards={r["race_id"]:r for r in read(H1RACES)}
    res=defaultdict(lambda:defaultdict(list))
    for r in read(H1RESULT):
        try:place=int(r["finish_position"]);car=int(r["car_no"])
        except(ValueError,KeyError):continue
        if 1<=place<=3:res[r["race_id"]][place].append(car)
    ready=[];skips=Counter()
    for rid,rs in p.items():
        if rid not in racecards:skips["missing_card"]+=1;continue
        if any(len(res[rid].get(pos,[]))!=1 for pos in(1,2,3)):
            skips["tie_missing"]+=1;continue
        rs.sort(key=lambda x:int(x["car_no"]))
        cars=[int(x["car_no"]) for x in rs]
        lines=parse_line(racecards[rid].get("predicted_line_formation",""),cars)
        if lines is None:skips["bad_line"]+=1;continue
        if any(res[rid][z][0] not in cars for z in(1,2,3)):
            skips["missing_finisher"]+=1;continue
        vals=np.array([float(r["estimated_win_probability"]) for r in rs])
        if abs(vals.sum()-1)>1e-5:skips["bad_p"]+=1;continue
        q=calibrate(vals)
        loc={car:(idx,rank) for idx,arr in enumerate(lines) for rank,car in enumerate(arr)}
        ready.append({"race_id":rid,"date":rs[0]["date"],"cars":cars,
                      "track":racecards[rid]["track"],"win_idx":[cars.index(res[rid][z][0]) for z in(1,2,3)],
                      "prob":q,"line_idx":np.array([loc[c][0] for c in cars]),
                      "pos_idx":np.array([loc[c][1] for c in cars]),
                      "size":np.array([len(lines[loc[c][0]]) for c in cars]),
                      "lineup":racecards[rid]["predicted_line_formation"]})
    return ready,{"forecast_count":len(p),"n":len(ready),"skipped":dict(skips)}

def v5_param():
    old=json.loads(V5.read_text(encoding="utf-8"))
    return np.asarray(old["parameters"]["second"]["theta"]),np.asarray(old["parameters"]["third"]["theta"])

def third_probs(r,i,j,theta,fields,base3):
    original=cprob(r,3,i,j,base3)
    F=row_features(r,i,j)[:,:len(fields)]
    logits=np.log(np.maximum(original,1e-100))+F@theta
    mask=np.ones(len(r["cars"]),dtype=bool)
    mask[[i,j]]=False
    z=np.zeros(len(mask))
    z[mask]=np.exp(logits[mask]-logsumexp(logits[mask]))
    return z

def create_choices(races,base3,fields):
    out=[]
    for r in races:
        i,j,k=r["win_idx"]
        base=cprob(r,3,i,j,base3)
        alt=[t for t in range(len(r["cars"])) if t not in(i,j)]
        out.append((np.log(base[alt]),row_features(r,i,j)[:,:len(fields)][alt],alt.index(k)))
    return out

def objective(beta,choices,l2):
    loss=0.
    grad=np.zeros(len(beta))
    for offset,F,target in choices:
        logits=offset+F@beta
        s=np.exp(logits-logsumexp(logits))
        loss+=logsumexp(logits)-logits[target]
        grad+=s@F-F[target]
    n=len(choices)
    loss/=n;grad/=n
    return loss+0.5*l2*float(beta@beta),grad+l2*beta

def fit(choices,l2):
    d=choices[0][1].shape[1]
    result=minimize(lambda z:objective(z,choices,l2),np.zeros(d),
                    jac=True,method="L-BFGS-B",options={"maxiter":220,"ftol":1e-12})
    if not result.success and np.linalg.norm(result.jac)>1e-5:
        raise RuntimeError(result.message)
    return result.x
def mean_loss(choices,beta):
    return float(np.mean([logsumexp(offset+F@beta)-(offset+F@beta)[y] for offset,F,y in choices]))
def fit_stage():
    source,audit=get_races(H2,H2RACES,H2RESULT,"h2")
    assert len(source)==867
    base2,base3=v5_param()
    bymonth={mo:[r for r in source if r["date"].startswith(mo)] for mo in("2026-07","2026-08")}
    comparisons=[]
    for kind,fields in MODELS.items():
        july=create_choices(bymonth["2026-07"],base3,fields)
        aug=create_choices(bymonth["2026-08"],base3,fields)
        for penalty in PENALTIES:
            fitted=fit(july,penalty)
            comparisons.append({"family":kind,"penalty":penalty,
                                "july_loss":mean_loss(july,fitted),
                                "aug_validation_loss":mean_loss(aug,fitted)})
    winner=min(comparisons,key=lambda x:x["aug_validation_loss"])
    fields=MODELS[winner["family"]]
    trainset=create_choices(source,base3,fields)
    final=fit(trainset,winner["penalty"])
    julyfit=fit(create_choices(bymonth["2026-07"],base3,fields),winner["penalty"])
    augfit=fit(create_choices(bymonth["2026-08"],base3,fields),winner["penalty"])
    baseline_train=mean_loss(trainset,np.zeros(len(fields)))
    final_train=mean_loss(trainset,final)
    rec={
        "stage":"frozen_without_September_or_2026H1_outcomes",
        "train_period":"2026-07-02 to 2026-08-30",
        "calibration_samples":len(source),"source_audit":audit,
        "selection_protocol":"July train; August candidate family and L2 penalty validation; refit July+August",
        "candidate_grid":comparisons,"selected":winner,
        "feature_names":fields,"beta":final.tolist(),"beta_july":julyfit.tolist(),
        "beta_august":augfit.tolist(),
        "baseline_train_loss":baseline_train,"updated_train_loss":final_train,
        "v5_base2":base2.tolist(),"v5_base3":base3.tolist(),
        "history_leakage_warning":"September & H1 were used earlier in non-residual model studies; this is transfer replication, not a fully untouched blind test.",
        "formula":"P3(k|i,j) prop V5_P3(k|i,j) * exp(residual_survivor_features(i,j,k) dot beta)",
        "no_market_odds":True,"fitted_on_sept_or_h1":False}
    PREFIT.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FIT",json.dumps({"train_n":len(source),"selected":winner,"baseline_train":baseline_train,
                             "new_train":final_train,"features":dict(zip(fields,final.tolist()))},ensure_ascii=False),flush=True)

def origin(r,i,j,k):
    line=r["line_idx"]
    if line[i]==line[j]:
        return "paired_same__third_teammate" if line[k]==line[i] else "paired_same__third_external"
    if line[k]==line[i]:return "paired_different__third_first_line"
    if line[k]==line[j]:return "paired_different__third_second_line"
    return "paired_different__third_other_line"

def pair_case(r):
    i,j,_=r["win_idx"]
    return "pair_same" if r["line_idx"][i]==r["line_idx"][j] else "pair_different"

def evaluate_probs(r,p2,p3,theta,fields,top10=False):
    i,j,k=r["win_idx"]
    P2=cprob(r,2,i,None,p2)
    P30=cprob(r,3,i,j,p3)
    P31=third_probs(r,i,j,theta,fields,p3)
    o=-math.log(P30[k])
    n=-math.log(P31[k])
    result={"v5_third_nll":o,"v6_third_nll":n}
    if top10:
        base=[];novel=[]
        NN=len(r["cars"])
        for a in range(NN):
            pb=cprob(r,2,a,None,p2)
            for b in range(NN):
                if a==b:continue
                c0=cprob(r,3,a,b,p3)
                c1=third_probs(r,a,b,theta,fields,p3)
                for c in range(NN):
                    if c in(a,b):continue
                    base.append((r["prob"][a]*pb[b]*c0[c],(a,b,c)))
                    novel.append((r["prob"][a]*pb[b]*c1[c],(a,b,c)))
        base.sort(key=lambda x:-x[0]);novel.sort(key=lambda x:-x[0])
        truth=(i,j,k)
        result["old_top1"]=int(base[0][1]==truth)
        result["new_top1"]=int(novel[0][1]==truth)
        result["old_top10"]=int(truth in [x[1] for x in base[:10]])
        result["new_top10"]=int(truth in [x[1] for x in novel[:10]])
    return result,P30,P31

def diagnostics(races,theta,fields,p2,p3,include_top10):
    metrics=Counter();grouped=defaultdict(Counter);bydate=defaultdict(list);race_rows=[]
    group_rider=defaultdict(Counter)
    stage3_orig=Counter()
    for r in races:
        i,j,k=r["win_idx"]
        res,p0,p1=evaluate_probs(r,p2,p3,theta,fields,include_top10)
        typ=origin(r,i,j,k);case=pair_case(r)
        metrics["n"]+=1
        for key,val in res.items():metrics[key]+=val
        metrics["pair_same"]+=int(case=="pair_same")
        grouped[typ]["n"]+=1
        grouped[typ]["gain_sum"]+=res["v5_third_nll"]-res["v6_third_nll"]
        grouped[case]["n"]+=1
        grouped[case]["gain_sum"]+=res["v5_third_nll"]-res["v6_third_nll"]
        bydate[r["date"]].append(res["v5_third_nll"]-res["v6_third_nll"])
        d={"date":r["date"],"race_id":r["race_id"],"track":r["track"],"lineup":r["lineup"],
           "first_car":r["cars"][i],"second_car":r["cars"][j],"third_car":r["cars"][k],
           "actual_third_origin":typ,**res}
        race_rows.append(d)
        all_orig=Counter()
        for cand in range(len(r["cars"])):
            if cand in(i,j):continue
            g=origin(r,i,j,cand)
            all_orig[g]+=1
            agg=group_rider[(case,g)]
            agg["available_riders"]+=1
            agg["v5_predicted_wins"]+=p0[cand]
            agg["v6_predicted_wins"]+=p1[cand]
            agg["actual_wins"]+=int(cand==k)
        stage3_orig.update({typ:1})
        group_rider[(case,"SUM")]["n"]+=1
    n=metrics["n"]
    for s in grouped.values():
        s["mean_improvement"]=s["gain_sum"]/s["n"]
    rng=np.random.default_rng(20261018)
    days=list(bydate.values())
    boot=[]
    for _ in range(2500):
        data=[v for z in rng.integers(len(days),size=len(days)) for v in days[z]]
        boot.append(float(np.mean(data)))
    bootrange=np.quantile(boot,[.025,.975])
    summary={
        "races":n,
        "average_loss_old":metrics["v5_third_nll"]/n,
        "average_loss_new":metrics["v6_third_nll"]/n,
        "gain":(metrics["v5_third_nll"]-metrics["v6_third_nll"])/n,
        "cluster_bootstrap_days_95pct":[float(z) for z in bootrange],
        "pair_same_races":metrics["pair_same"],
        "pair_different_races":n-metrics["pair_same"],
        "origin_breakdown":{key:dict(val) for key,val in grouped.items()},
        "origin_probabilities":{
            f"{case}::{label}":{
                "available_candidates":int(c["available_riders"]),
                "actual_wins":int(c["actual_wins"]),
                "v5_predicted_wins":float(c["v5_predicted_wins"]),
                "v6_predicted_wins":float(c["v6_predicted_wins"]),
                "actual_share_per_race":float(c["actual_wins"]/group_rider[(case,"SUM")]["n"]),
                "v5_predicted_share_per_race":float(c["v5_predicted_wins"]/group_rider[(case,"SUM")]["n"]),
                "v6_predicted_share_per_race":float(c["v6_predicted_wins"]/group_rider[(case,"SUM")]["n"]),
            }
            for case,label in group_rider if label!="SUM"
            for c in [group_rider[(case,label)]]
        },
        "days":len(days)}
    if include_top10:
        summary.update({
            "v5_trifecta_top1_hits":metrics["old_top1"],
            "v6_trifecta_top1_hits":metrics["new_top1"],
            "v5_trifecta_top10_hits":metrics["old_top10"],
            "v6_trifecta_top10_hits":metrics["new_top10"]})
    return summary,race_rows

def score(stage):
    params=json.loads(PREFIT.read_text(encoding="utf-8"))
    assert params["fitted_on_sept_or_h1"] is False
    if stage=="score-sep":
        races,audit=get_races(SEP,SEPRACES,SEPRESULT,"sep")
        assert len(races)==419
    else:
        races,audit=get_h1()
        assert len(races)>2400
    beta=np.array(params["beta"])
    fields=params["feature_names"]
    p2=np.array(params["v5_base2"]);p3=np.array(params["v5_base3"])
    summary,rows=diagnostics(races,beta,fields,p2,p3,include_top10=(stage=="score-sep"))
    report={"stage":stage,"fit_source":"2026 July-Aug only","training_n":params["calibration_samples"],
            "evaluation_audit":audit,
            "selected":params["selected"],
            "parameters_locked":True,
            "numerics":summary,
            "warning":"This is conditional third given true 1st/2nd; separately count genuinely pre-race trifecta rank hits.",
            "data_not_globally_unseen":True}
    name="september" if stage=="score-sep" else "h1"
    (HERE/f"{name}_survivor_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if stage=="score-sep":
        with (HERE/"september_survivor_by_race.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print("RESULT",name,json.dumps({"aud":audit,
                    "avg_gain":summary["gain"],
                    "gain95":summary["cluster_bootstrap_days_95pct"],
                    "origin":summary["origin_probabilities"],
                    "top10_old":summary.get("v5_trifecta_top10_hits"),
                    "top10_new":summary.get("v6_trifecta_top10_hits")},ensure_ascii=False),flush=True)

if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in("fit","score-sep","score-h1"):
        raise SystemExit("fit|score-sep|score-h1")
    if sys.argv[1]=="fit":fit_stage()
    else:score(sys.argv[1])

#!/usr/bin/env python3
"""Line-conditional ordered finishing probabilities.

Benter/Harville baseline:
  P(i,j,k)=q_i * softmax_j(gamma*log(q_j) | j!=i)
                * softmax_k(delta*log(q_k) | k notin(i,j))
Upgrade:
  P(i,j,k)=q_i * softmax_j(gamma*log(q_j)+B2.x(i,j) | j!=i)
                * softmax_k(delta*log(q_k)+B3.z(i,j,k) | k notin(i,j))
B2.x and B3.z are only from pre-race predicted line formation.

July/Aug: select L2 penalty on July=>Aug, refit July+Aug.
September: already-frozen win probability and original pre-race lineups; held
out from parameter fitting. Model chosen solely by prior-period validation.
September had previously been used to evaluate no-line model: exploratory
relative to the larger research project, not an untouched global test.
"""
from __future__ import annotations
import csv, json, math, sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
H2=ROOT/"research/benter_score_challenge_v3/frozen_2026h2_probabilities.csv"
H2_RACES=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/races.csv"
H2_OUTCOMES=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/results.csv"
SEP=ROOT/"research/benter_calibration_v4/sep_predictions_frozen.csv"
SEP_RACES=ROOT/"research/benter_calibration_v4/sep_holdout/races.csv"
SEP_OUTCOMES=ROOT/"research/benter_calibration_v4/sep_holdout/results.csv"
BASE=ROOT/"research/benter_ordered_23/parameters_frozen.json"
CONFIG=HERE/"parameters_pretest.json"
REPORT=HERE/"holdout_score.json"
RACES=HERE/"holdout_by_race.csv"
FEAT2=[
 "same_first_line","winner_leader_candidate_bante","winner_bante_candidate_leader",
 "candidate_behind_winner","candidate_bante","candidate_tail",
]
FEAT3=[
 "same_first_line","same_second_line","all_three_same_line",
 "head_bante_tail_completion","candidate_bante","candidate_tail",
 "candidate_behind_first","candidate_behind_second","unselected_teammates",
]
PENALTIES=(0.05,0.5,2.0,10.0)
CAL_LAMBDA=0.9999940391390134

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def parse_line(text,cars):
    if not text:return None
    try:
        out=[[int(v) for v in bit.strip().split("-")] for bit in text.split("/")]
    except ValueError:return None
    flat=[c for line in out for c in line]
    return out if len(flat)==len(set(flat)) and set(flat)==set(cars) else None

def calibrate(p):
    z=np.log(np.maximum(p,1e-15))-CAL_LAMBDA*np.maximum(p-.4,0)
    return np.exp(z-logsumexp(z))

def get_races(probs,racecards,results,origin):
    ps=defaultdict(list)
    for p in read(probs):ps[p["race_id"]].append(p)
    rmap={r["race_id"]:r for r in read(racecards)}
    winners=defaultdict(lambda:defaultdict(list))
    for t in read(results):
        try:place=int(t["finish_position"])
        except(ValueError,KeyError):continue
        if 1<=place<=3:winners[t["race_id"]][place].append(int(t["car_no"]))
    eligible=[]
    audit=Counter()
    coocc2=Counter()
    coocc3=Counter()
    for rid,group in ps.items():
        if rid not in rmap:audit["missing_lineup_meta"]+=1;continue
        rank=winners.get(rid,{})
        if any(len(rank.get(pos,[]))!=1 for pos in (1,2,3)):
            audit["tie_or_missing_places"]+=1;continue
        group.sort(key=lambda x:int(x["car_no"]))
        cars=[int(x["car_no"]) for x in group]
        lines=parse_line(rmap[rid].get("predicted_line_formation",""),cars)
        if lines is None:audit["unusable_lineup"]+=1;continue
        try:
            ix=[cars.index(rank[z][0]) for z in (1,2,3)]
        except ValueError:audit["result_not_in_card"]+=1;continue
        if len(set(ix))<3:audit["duplicated_finishing_car"]+=1;continue
        try:
            orig=np.array([float(x["v2_line_model"] if origin=="h2" else x["original_probability"]) for x in group])
            q=calibrate(orig) if origin=="h2" else np.array([float(x["calibrated_probability"]) for x in group])
        except(ValueError,KeyError):audit["invalid_probability"]+=1;continue
        if min(q)<=0 or abs(q.sum()-1)>1e-7:audit["invalid_probability"]+=1;continue
        position={car:(lineidx,pos) for lineidx,line in enumerate(lines) for pos,car in enumerate(line)}
        line_idx=np.array([position[c][0] for c in cars])
        pos_idx=np.array([position[c][1] for c in cars])
        sizes=np.array([len(lines[position[c][0]]) for c in cars])
        r={
            "race_id":rid,"date":group[0]["race_date"],"track":group[0]["track"],
            "cars":cars,"win_idx":ix,"prob":q,
            "line_idx":line_idx,"pos_idx":pos_idx,"size":sizes,
            "lineup":rmap[rid].get("predicted_line_formation",""),
            "num_lines":len(lines)
        }
        i,j,k=ix
        coocc2["same_line" if line_idx[i]==line_idx[j] else "different_line"]+=1
        if line_idx[i]==line_idx[j] and line_idx[j]==line_idx[k]:coocc3["all_same"]+=1
        elif len({line_idx[i],line_idx[j],line_idx[k]})==3:coocc3["all_different"]+=1
        elif line_idx[i]==line_idx[j]:coocc3["first_second_same_third_diff"]+=1
        elif line_idx[i]==line_idx[k]:coocc3["first_third_same_second_diff"]+=1
        else:coocc3["second_third_same_first_diff"]+=1
        eligible.append(r)
    return eligible,{"forecast_count":len(ps),"n":len(eligible),
                     "excluded":dict(audit),"second_relationships":dict(coocc2),
                     "third_relationships":dict(coocc3)}

def features(r,stage,first,second=None):
    n=len(r["cars"])
    lines=r["line_idx"];pos=r["pos_idx"];size=r["size"]
    a=np.arange(n)
    if stage==2:
        return np.column_stack((
           (lines==lines[first]).astype(float),
           ((lines==lines[first]) & (pos[first]==0) & (pos==1)).astype(float),
           ((lines==lines[first]) & (pos[first]==1) & (pos==0)).astype(float),
           ((lines==lines[first]) & (pos>pos[first])).astype(float),
           (pos==1).astype(float),
           (pos>=2).astype(float)
        ))
    if stage==3:
        assert second is not None
        same1=(lines==lines[first])
        same2=(lines==lines[second])
        allsame=same1 & (lines[first]==lines[second])
        direct=(allsame & (pos[first]==0) & (pos[second]==1) & (pos==2))
        remaining=size-1-(same1.astype(int)+same2.astype(int))
        return np.column_stack((
           same1.astype(float),same2.astype(float),allsame.astype(float),
           direct.astype(float),(pos==1).astype(float),(pos>=2).astype(float),
           (same1&(pos>pos[first])).astype(float),
           (same2&(pos>pos[second])).astype(float),
           np.maximum(remaining,0).astype(float)
        ))
    raise ValueError(stage)

def choice_sets(races,stage):
    out=[]
    for r in races:
        i,j,k=r["win_idx"]
        excluded=[i] if stage==2 else [i,j]
        alts=[m for m in range(len(r["cars"])) if m not in excluded]
        target=j if stage==2 else k
        feat=features(r,stage,i,j if stage==3 else None)[alts]
        prob=np.log(r["prob"][alts])
        y=alts.index(target)
        out.append((np.column_stack((prob,feat)),y,r["date"],r["race_id"]))
    return out

def nll_grad(params,choices,penalty,with_features=True):
    value=0.
    grad=np.zeros(len(params),dtype=float)
    for x,y,date,rid in choices:
        scores=x[:,:len(params)]@params if with_features else x[:,0]*params[0]
        z=np.exp(scores-logsumexp(scores))
        value+=-(scores[y]-logsumexp(scores))
        grad+=z@x[:,:len(params)]-x[y,:len(params)]
    value/=len(choices);grad/=len(choices)
    if with_features and len(params)>1:
        value+=0.5*penalty*np.sum(params[1:]**2)
        grad[1:]+=penalty*params[1:]
    return value,grad

def train(choices,penalty):
    d=choices[0][0].shape[1]
    init=np.zeros(d);init[0]=.5
    res=minimize(lambda z:nll_grad(z,choices,penalty),
                 init,method="L-BFGS-B",jac=True,
                 bounds=[(.05,3.0)]+[(None,None)]*(d-1),
                 options={"maxiter":180,"ftol":1e-11,"gtol":1e-8})
    if not res.success and np.linalg.norm(res.jac)>1e-5:raise RuntimeError((res.message,res.jac))
    return res.x

def mean_loss(choices,theta):
    return float(np.mean([logsumexp(x@theta)-(x@theta)[y] for x,y,date,rid in choices]))

def choose_and_fit(trainraces,stage):
    july=choice_sets([r for r in trainraces if r["date"].startswith("2026-07")],stage)
    august=choice_sets([r for r in trainraces if r["date"].startswith("2026-08")],stage)
    choices=july+august
    v=[]
    for l2 in PENALTIES:
        param=train(july,l2)
        v.append({"lambda":l2,"validation_logloss":mean_loss(august,param),
                  "july_params":[float(x) for x in param]})
    picked=min(v,key=lambda d:d["validation_logloss"])["lambda"]
    final=train(choices,picked)
    # fit separately by month for interpretability, NOT selection
    jul=train(july,picked);aug=train(august,picked)
    return {"penalty":picked,"theta":final.tolist(),"validation_candidates":v,
            "july_theta":jul.tolist(),"august_theta":aug.tolist(),
            "train_loss":mean_loss(choices,final),"train_size":len(choices),
            "param_names":["probability_exponent"]+(FEAT2 if stage==2 else FEAT3)}

def fit_stage():
    trainr,stats=get_races(H2,H2_RACES,H2_OUTCOMES,"h2")
    if len(trainr)<850:raise RuntimeError(stats)
    fitted={"second":choose_and_fit(trainr,2),"third":choose_and_fit(trainr,3)}
    baseline=json.loads(BASE.read_text(encoding="utf-8"))
    out={
        "model":"Line-conditional Benter correction for 2nd and 3rd",
        "formula":"P(i,j,k)=q_i*softmax_{j!=i}(gamma log q_j + X2(i,j) @ beta2)*softmax_{k!=i,j}(delta log q_k + X3(i,j,k) @ beta3)",
        "feature_spec":{"second":FEAT2,"third":FEAT3},
        "training":"2026-07-02..2026-08-30",
        "validation":"July-only fit, August selection, refit both",
        "evaluation":"2026-09-01..2026-09-30; observed in prior no-line tests",
        "training_stats":stats,"parameters":fitted,
        "baseline_without_line_frozen":{"gamma":baseline["fitted_parameters"]["q"]["gamma"],
                                        "delta":baseline["fitted_parameters"]["q"]["delta"]},
        "calibration_lambda":CAL_LAMBDA,"target_results_read_to_fit":False,
        "market_odds_used":False,
        "note":"September had already been used for earlier no-line analysis, so exploratory replication, not globally untouched blind validation."
    }
    CONFIG.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FITTED",json.dumps({"n":len(trainr),"selected_second":fitted["second"]["penalty"],
        "selected_third":fitted["third"]["penalty"],"coefficients":{k:v["theta"] for k,v in fitted.items()},
        "train_structure":stats},ensure_ascii=False),flush=True)

def cprob(r,stage,first,second,params):
    X=features(r,stage,first,second if stage==3 else None)
    scores=params[0]*np.log(np.maximum(r["prob"],1e-15))+X@params[1:]
    mask=np.ones(len(scores),dtype=bool)
    mask[first]=False
    if stage==3:mask[second]=False
    z=np.zeros(len(scores),dtype=float)
    z[mask]=np.exp(scores[mask]-logsumexp(scores[mask]))
    return z

def evaluate(races,p2,p3):
    records=[]
    total=Counter()
    second_yes=0;third_yes=0;highest_yes=0;top10_yes=0
    for r in races:
        i,j,k=r["win_idx"]
        second=cprob(r,2,i,None,p2)
        third=cprob(r,3,i,j,p3)
        secondprob=second[j];thirdprob=third[k]
        loss2=-math.log(secondprob)
        loss3=-math.log(thirdprob)
        loss=-math.log(r["prob"][i])-math.log(secondprob)-math.log(thirdprob)
        second_yes+=int(np.argmax(second)==j)
        third_yes+=int(np.argmax(third)==k)
        # All combinations are valid ordered 3-car tickets, and normalize.
        n=len(r["cars"]);tickets=[]
        for a in range(n):
            ps=cprob(r,2,a,None,p2)
            for b in range(n):
                if a==b:continue
                pt=cprob(r,3,a,b,p3)
                for c in range(n):
                    if c==a or c==b:continue
                    tickets.append((r["prob"][a]*ps[b]*pt[c],a,b,c))
        totalmass=sum(z[0] for z in tickets)
        if abs(totalmass-1)>1e-9:raise AssertionError((r["race_id"],totalmass))
        tickets.sort(key=lambda x:-x[0])
        highest_yes+=int(tickets[0][1:]==(i,j,k))
        top10_yes+=int((i,j,k) in [t[1:] for t in tickets[:10]])
        record={"date":r["date"],"track":r["track"],"race_id":r["race_id"],
                "first":r["cars"][i],"second":r["cars"][j],"third":r["cars"][k],
                "second_logloss":loss2,"third_logloss":loss3,"ordered_logloss":loss,
                "second_top1_hit":int(np.argmax(second)==j),
                "third_top1_hit":int(np.argmax(third)==k),
                "trifecta_top1_hit":int(tickets[0][1:]==(i,j,k)),
                "trifecta_top10_hit":int((i,j,k) in [t[1:] for t in tickets[:10]]),
                "winning_ticket_prob":r["prob"][i]*secondprob*thirdprob,
                "winner_same_line_as_second":int(r["line_idx"][i]==r["line_idx"][j]),
                "winner_same_line_as_third":int(r["line_idx"][i]==r["line_idx"][k]),
                "second_same_line_as_third":int(r["line_idx"][j]==r["line_idx"][k])
        }
        records.append(record)
    n=len(races)
    metrics={"n":n,"conditional_second_logloss":float(np.mean([v["second_logloss"] for v in records])),
             "conditional_third_logloss":float(np.mean([v["third_logloss"] for v in records])),
             "joint_trifecta_logloss":float(np.mean([v["ordered_logloss"] for v in records])),
             "second_conditional_top1_hit":second_yes,"third_conditional_top1_hit":third_yes,
             "trifecta_top1_hit":highest_yes,"trifecta_top10_hit":top10_yes}
    return metrics,records

def bootstrap(comparisons,models):
    dated=defaultdict(list)
    for i,r in enumerate(models["baseline"]["records"]):
        dated[r["date"]].append(i)
    group=list(dated.values())
    out={}
    rng=np.random.default_rng(20261018)
    for a,b in comparisons:
        results={}
        for field in ("second_logloss","third_logloss","ordered_logloss"):
            diff=np.array([r[field] for r in models[b]["records"]])-np.array([r[field] for r in models[a]["records"]])
            ixs=rng.integers(len(group),size=(2500,len(group)))
            vals=[]
            for row in ixs:
                ix=np.concatenate([np.array(group[z],dtype=int) for z in row])
                vals.append(float(diff[ix].mean()))
            results[field]={"improvement_positive_better":float(diff.mean()),
                            "date_cluster_bootstrap95":[float(z) for z in np.quantile(vals,[.025,.975])]}
        out[f"{a}_over_{b}"]=results
    return out

def score_stage():
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["target_results_read_to_fit"] is False
    races,stats=get_races(SEP,SEP_RACES,SEP_OUTCOMES,"sep")
    if len(races)<400:raise RuntimeError(stats)
    gamma=config["baseline_without_line_frozen"]["gamma"]
    delta=config["baseline_without_line_frozen"]["delta"]
    params2=np.array(config["parameters"]["second"]["theta"])
    params3=np.array(config["parameters"]["third"]["theta"])
    p2=np.array([gamma]+[0.]*len(FEAT2));p3=np.array([delta]+[0.]*len(FEAT3))
    choices={
        "baseline":(p2,p3),
        "line_second_only":(params2,p3),
        "line_third_only":(p2,params3),
        "line_both":(params2,params3)
    }
    models={}
    for name,ab in choices.items():
        met,recs=evaluate(races,*ab)
        models[name]={"metrics":met,"records":recs}
        print("TEST",name,json.dumps(met,ensure_ascii=False),flush=True)
    checks=bootstrap([
        ("line_both","baseline"),("line_second_only","baseline"),
        ("line_third_only","baseline"),("line_both","line_second_only"),
        ("line_both","line_third_only")],models)
    n=len(races)
    null_loss=float(np.mean([math.log(len(r["cars"])*(len(r["cars"])-1)*(len(r["cars"])-2)) for r in races]))
    metrics={m:{**x["metrics"],
                "ordered_pseudo_R2":1-x["metrics"]["joint_trifecta_logloss"]/null_loss}
             for m,x in models.items()}
    raw=[]
    for i,r in enumerate(races):
        d={"date":r["date"],"track":r["track"],"race_id":r["race_id"],
           "lineup":r["lineup"],
           "actual_first":r["cars"][r["win_idx"][0]],
           "actual_second":r["cars"][r["win_idx"][1]],
           "actual_third":r["cars"][r["win_idx"][2]]}
        for name in choices:
            for field in ("second_logloss","third_logloss","ordered_logloss",
                          "trifecta_top1_hit","trifecta_top10_hit"):
                d[f"{name}_{field}"]=models[name]["records"][i][field]
        raw.append(d)
    with RACES.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    output={
        "scope":"F1 S級, 2026年9月, all uniquely completed ordered finishes with valid line",
        "train_period":config["training"],"test_period":config["evaluation"],
        "fit_audit":config["training_stats"],"test_audit":stats,
        "parameters":config["parameters"],
        "fixed_baseline":config["baseline_without_line_frozen"],
        "models":metrics,"comparison":checks,
        "uniform_ordered_logloss":null_loss,
        "interpretation":"Improved conditional probabilities and ordered 3-car joint likelihood do NOT themselves prove betting profitability",
        "selection_integrity":"July->August penalty choice, July+August full training; no Sept model/feature/penalty tuning",
        "global_test_integrity":"September outcomes already used in previous no-line tests; followup prospective holdout still needed",
    }
    REPORT.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("COMPARISON",json.dumps(checks,ensure_ascii=False),flush=True)

if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ("fit","score"):
        raise SystemExit("Usage: python model_line_conditional.py fit|score")
    (fit_stage if sys.argv[1]=="fit" else score_stage)()

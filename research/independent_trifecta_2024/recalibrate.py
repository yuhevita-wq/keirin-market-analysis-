#!/usr/bin/env python3
"""Strict, odds-free 2024 H1-fit / H2-evaluate trifecta calibration experiment.

Input: archived 2024 race-level ORIGINAL model probabilities from GitHub
Actions run 40807880319, trained ONLY on 2023 Jan--Oct. These are NEVER refit.

Candidate calibrators, all with per-race sum exactly 1:
  unchanged     q = p
  temperature   q = p**alpha / sum(p**alpha)
  line_structure q = p*exp(beta dot same-line flags) / sum(...)
  joint         q = p**alpha*exp(beta dot same-line flags) / sum(...)

Fit alpha and beta ONLY using 2024 Jan-Jun observed outcomes.
Evaluate ONCE on 2024 Jul-Dec; never use Jul-Dec for tuning.
No 2025/2026 data or any odds/payouts are read. Probability bins, line
event calibration, full multiclass NLL/Brier and day-block paired CIs.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import calibration_audit as prior

IN=ROOT/"data/audits/independent_trifecta_all_tickets_2024.jsonl.gz"
PRIOR=ROOT/"data/audits/independent_trifecta_calibration_2024.json"
OUT=ROOT/"data/audits/independent_trifecta_recalibration_2024.json"
SAMPLE=ROOT/"data/audits/independent_trifecta_recalibration_sample_2024.json"
PROBS=ROOT/"data/audits/independent_trifecta_recalibrated_2024.jsonl.gz"
H1_END="2024-06-30"
EPS=1e-15
BUCKETS=[
 ("0-0.1%",0.,.001),("0.1-0.5%",.001,.005),("0.5-1%",.005,.01),
 ("1-2%",.01,.02),("2-5%",.02,.05),("5-10%",.05,.1),
 ("10-20%",.1,.2),("20-100%",.2,1.000001),
]
FLAGS=["same_line_12","same_line_13","same_line_23","same_line_123"]
STRUCT_L2=0.002
MODEL_NAMES=("unchanged","temperature","line_structure","joint")

def open_2024_inputs():
    report=json.loads(PRIOR.read_text(encoding="utf-8"))
    if report["state"]!="COMPLETED_2024_NO_ODDS_FROZEN_2023":raise ValueError("Wrong audit source")
    if report["train_races"]!=19248 or report["test_races"]!=25348:raise ValueError("2023/24 universe mismatch")
    if report["market_odds_features"] or report["prices_read"] or report["outcomes_used_in_2024_training"]:
        raise ValueError("Prior experiment violates independent model requirements")
    data,coverage=prior.ingest_2024()
    if len(data)!=25348: raise ValueError("2024 exact results coverage changed")
    return data,report

def build_arrays(events,report):
    expected=int(report["number_of_3trifecta_ticket_predictions"])
    logp=np.empty(expected,dtype=np.float64)
    f=np.empty((expected,4),dtype=np.uint8)
    race_meta=[]
    seen=set();cursor=0
    dig=hashlib.sha256()
    with gzip.open(IN,"rt",encoding="utf-8") as source:
        for line in source:
            dig.update(line.encode("utf-8"))
            item=json.loads(line)
            rid=item["race_id"]
            if rid in seen:raise ValueError("Duplicate 2024 race in archived predictions")
            seen.add(rid)
            race=events.get(rid)
            if race is None:raise ValueError("Archived prediction missing 2024 racecard "+rid)
            day=race["date"]
            if item["date"]!=day or "-".join(race["order"])!=item["true_result"]:
                raise ValueError("2024 labels mismatch "+rid)
            entries=item["probabilities"]
            start=cursor; actual=None;total=0.
            n_riders=len(race["scores"])
            n=n_riders*(n_riders-1)*(n_riders-2)
            if len(entries)!=n:raise ValueError("Incomplete triple grid "+rid)
            g={car:race["positions"][car][0] for car in race["scores"]}
            true_key=item["true_result"]
            for idx,(ticket,p) in enumerate(entries.items()):
                if cursor>=expected:raise ValueError("Too many predictions")
                a,b,c=ticket.split("-")
                if len({a,b,c})!=3 or not all(k in g for k in (a,b,c)):
                    raise ValueError("Invalid ticket "+rid+" "+ticket)
                v=float(p)
                if v<=0. or v>=1. or not math.isfinite(v):raise ValueError("Invalid p")
                logp[cursor]=math.log(v)
                aa,bb,cc=g[a],g[b],g[c]
                f[cursor,0]=aa==bb
                f[cursor,1]=aa==cc
                f[cursor,2]=bb==cc
                f[cursor,3]=(aa==bb==cc)
                if ticket==true_key:actual=cursor
                total+=v
                cursor+=1
            if actual is None or abs(total-1)>1e-7:
                raise ValueError("Race not normalized or actual missing "+rid+" "+str(total))
            race_meta.append({
                "id":rid,"date":day,"start":start,"size":n,
                "truth_index":actual,"class":race["race_class"],
                "field":n_riders,
                "shape":"-".join(map(str,sorted(map(len,race["groups"]),reverse=True))),
                "true_line_flags":[int(f[actual,k]) for k in range(4)]
            })
            if len(race_meta)%5000==0:print("READ_PREDICTIONS",len(race_meta),flush=True)
    if cursor!=expected or len(seen)!=len(events):raise ValueError("Archived file incomplete")
    print("2024_ORIGINAL_PROBABILITIES_LOADED",len(seen),cursor,flush=True)
    return logp,f,race_meta,dig.hexdigest()

def to_slice(meta,predicate):
    groups=[r for r in meta if predicate(r)]
    if not groups:raise ValueError("Empty split")
    all_starts=np.array([r["start"] for r in groups],dtype=np.int64)
    all_sizes=np.array([r["size"] for r in groups],dtype=np.int64)
    targets=np.array([r["truth_index"] for r in groups],dtype=np.int64)
    start=int(groups[0]["start"]);end=int(groups[-1]["start"]+groups[-1]["size"])
    if end-start!=int(all_sizes.sum()):raise ValueError("Year not sorted by date or split fragmented")
    return {
        "groups":groups,"start":start,"end":end,
        "starts":all_starts-start,"sizes":all_sizes,"targets":targets-start,
        "n":len(groups)
    }

def make_weights(theta,kind,logs,flags,block):
    beg,end=block["start"],block["end"]
    a=float(theta[0]) if kind in ("temperature","joint") else 1.
    betas=(np.asarray(theta[1:],dtype=np.float64) if kind=="joint"
           else np.asarray(theta,dtype=np.float64) if kind=="line_structure"
           else np.zeros(4,dtype=np.float64))
    scores=a*logs[beg:end].copy()
    if kind in ("line_structure","joint"):
        # Avoid large intermediate dense Nx4 float arrays.
        for k in range(4):
            scores+=betas[k]*flags[beg:end,k]
    mm=np.maximum.reduceat(scores,block["starts"])
    weights=np.exp(scores-np.repeat(mm,block["sizes"]))
    denom=np.add.reduceat(weights,block["starts"])
    weights/=np.repeat(denom,block["sizes"])
    if not np.all(np.isfinite(weights)):raise ValueError("Nonfinite probabilities")
    if np.max(np.abs(np.add.reduceat(weights,block["starts"])-1))>1e-9:
        raise ValueError("Not normalized")
    return weights, scores,mm,denom

def objective(x,kind,logs,flags,block):
    weights,scores,mm,denom=make_weights(x,kind,logs,flags,block)
    target=block["targets"]
    val=np.mean(mm+np.log(denom)-scores[target])
    # Analytic gradient of mean multiclass proper log-loss.
    need_alpha=kind in ("temperature","joint")
    need_structure=kind in ("line_structure","joint")
    grad=[]
    sl=slice(block["start"],block["end"])
    if need_alpha:
        pred=np.add.reduceat(weights*logs[sl],block["starts"])
        grad.append(np.mean(pred-logs[sl][target]))
    if need_structure:
        for k in range(4):
            feature=flags[sl,k]
            pred=np.add.reduceat(weights*feature,block["starts"])
            grad.append(np.mean(pred-feature[target]))
    grad=np.asarray(grad,dtype=np.float64)
    # Small, prespecified regularizer is paid only by structural coefficients.
    if need_structure:
        betas=x[1:] if kind=="joint" else x
        val+=0.5*STRUCT_L2*np.dot(betas,betas)
        if kind=="joint":grad[1:]+=STRUCT_L2*betas
        else:grad+=STRUCT_L2*betas
    return float(val),grad

def fit(kind,logs,flags,block):
    if kind=="unchanged":return {"alpha":1.,"line_coefs":[0.]*4,
                                  "params":[],"train_penalized_nll":None,
                                  "converged":True,"iterations":0}
    if kind=="temperature":
        x0=[1.];bounds=[(.25,3.0)]
    elif kind=="line_structure":
        x0=[0.]*4;bounds=[(-1.5,1.5)]*4
    else:
        x0=[1.]+[0.]*4;bounds=[(.25,3.0)]+[(-1.5,1.5)]*4
    res=minimize(lambda z:objective(z,kind,logs,flags,block),
                 x0=np.asarray(x0,dtype=np.float64),jac=True,
                 bounds=bounds,method="L-BFGS-B",
                 options={"maxiter":80,"ftol":1e-11,"gtol":1e-8})
    if not res.success:raise RuntimeError("Calibration fitting failed "+kind+": "+str(res.message))
    alpha=float(res.x[0]) if kind in ("temperature","joint") else 1.
    beta=(res.x[1:].tolist() if kind=="joint" else
          res.x.tolist() if kind=="line_structure" else [0.]*4)
    print("FITTED",kind,"alpha",alpha,"betas",beta,"nll+penalty",res.fun,flush=True)
    return {"alpha":alpha,"line_coefs":beta,
            "params":[float(x) for x in res.x],
            "train_penalized_nll":float(res.fun),"converged":True,
            "iterations":int(res.nit)}

def per_race_weights(r,logs,flags,fit):
    s=r["start"];e=s+r["size"]
    score=fit["alpha"]*logs[s:e]
    if any(fit["line_coefs"]):
        score=score.copy()
        for k,beta in enumerate(fit["line_coefs"]):
            score+=beta*flags[s:e,k]
    score=score-np.max(score)
    w=np.exp(score)
    w/=np.sum(w)
    return w

def data_container():
    return {"races":0,"nll":0.,"brier":0.,"ptrue":0.,"top1":0,"top3":0}

def metric(m):
    n=m["races"]
    if not n:return None
    return {
      "races":n,"nll":round(m["nll"]/n,8),
      "brier":round(m["brier"]/n,8),
      "mean_probability_of_realized_pct":round(100*m["ptrue"]/n,6),
      "top1_hit_pct":round(100*m["top1"]/n,5),
      "top3_hit_pct":round(100*m["top3"]/n,5)
    }

def calc_metrics(races,logs,flags,fit,save_file=None):
    m=data_container()
    by_class=defaultdict(data_container)
    by_shape=defaultdict(data_container)
    per_day=defaultdict(lambda:[0.,0])
    # Calibration bin counts and expected hits; racing is the independent cluster.
    band={label:{"tickets":0,"expected":0.,"actual":0,"null_var":0.} for label,_,_ in BUCKETS}
    events={k:{"observed":0,"expected":0.,"null_var":0.} for k in FLAGS}
    for i,r in enumerate(races):
        w=per_race_weights(r,logs,flags,fit)
        s=r["start"];e=s+r["size"]
        local_target=r["truth_index"]-s
        ptrue=float(w[local_target])
        nll=-math.log(max(EPS,ptrue))
        brier=1.-2.*ptrue+float(np.dot(w,w))
        ix=np.argsort(-w,kind="stable")[:3]
        t1=int(local_target==ix[0]);t3=int(local_target in ix)
        for obj in (m,by_class[r["class"]],by_shape[r["shape"]]):
            obj["races"]+=1;obj["nll"]+=nll;obj["brier"]+=brier;obj["ptrue"]+=ptrue
            obj["top1"]+=t1;obj["top3"]+=t3
        per_day[r["date"]][0]+=nll
        per_day[r["date"]][1]+=1
        block_flags=flags[s:e,:]
        for k,label in enumerate(FLAGS):
            pp=float(np.dot(w,block_flags[:,k]))
            z=events[label]
            z["observed"]+=r["true_line_flags"][k]
            z["expected"]+=pp
            z["null_var"]+=max(0.,pp*(1.-pp))
        for label,lo,hi in BUCKETS:
            mask=(w>=lo)&(w<hi)
            if not mask.any():continue
            z=band[label];mass=float(w[mask].sum())
            z["tickets"]+=int(np.count_nonzero(mask))
            z["expected"]+=mass
            z["actual"]+=int(mask[local_target])
            z["null_var"]+=max(0.,mass*(1.-mass))
        if (i+1)%5000==0:print("EVAL",i+1,"/",len(races),flush=True)
    out_bands={}
    for label,z in band.items():
        if not z["tickets"]:continue
        total=z["tickets"];e=z["expected"];a=z["actual"]
        sd=math.sqrt(max(0.,z["null_var"]))
        out_bands[label]={
          "tickets":total,"expected_hits":round(e,4),"actual_hits":a,
          "mean_predicted_pct":round(100*e/total,6),
          "empirical_pct":round(100*a/total,6),
          "ratio_actual_to_expected":round(a/e,5) if e else None,
          "z_under_calibrated_null":round((a-e)/sd,4) if sd else None
        }
    out_events={}
    n=len(races)
    for label,z in events.items():
        e=z["expected"];a=z["observed"];sd=math.sqrt(z["null_var"])
        out_events[label]={
          "races":n,"expected_pct":round(100*e/n,5),
          "empirical_pct":round(100*a/n,5),
          "delta_pct_point":round(100*(a-e)/n,5),
          "null_z":round((a-e)/sd,4) if sd else None}
    ece_pp=100*sum(abs(z["actual"]-z["expected"]) for z in band.values())/sum(
        z["tickets"] for z in band.values())
    return {
      "overall":metric(m),"by_class":{k:metric(z) for k,z in sorted(by_class.items())},
      "by_line_shape_top_8":{k:metric(by_shape[k]) for k in sorted(by_shape,
                                                key=lambda key:-by_shape[key]["races"])[:8]},
      "calibration_bands":out_bands,"structural":out_events,
      "ticket_weighted_ece_pp":round(ece_pp,8),
      "daily_nll":{k:tuple(v) for k,v in sorted(per_day.items())},
    }

def paired_day_ci(base,other):
    days=sorted(set(base["daily_nll"])&set(other["daily_nll"]))
    n=sum(other["daily_nll"][d][1] for d in days)
    value=sum(base["daily_nll"][d][0]-other["daily_nll"][d][0] for d in days)/n
    variance=sum((base["daily_nll"][d][0]-other["daily_nll"][d][0]-value*other["daily_nll"][d][1])**2
                 for d in days)
    se=math.sqrt(variance)/n
    return {"nll_improvement":round(value,8),"date_cluster_se":round(se,8),
            "approx_95pct_ci":[round(value-1.96*se,8),round(value+1.96*se,8)],
            "days":len(days)}

def compact(r):
    if not r:return None
    return {k:v for k,v in r.items() if k!="daily_nll"}

def main():
    if not IN.is_file():raise RuntimeError("Missing archived frozen 2023 baseline 2024 predictions")
    events,source_report=open_2024_inputs()
    logs,flags,meta,archive_digest=build_arrays(events,source_report)
    h1=to_slice(meta,lambda x:x["date"]<=H1_END)
    h2=to_slice(meta,lambda x:x["date"]>H1_END)
    print("FIXED_SPLIT",h1["n"],h2["n"],"tickets",len(logs),flush=True)
    fitted={k:fit(k,logs,flags,h1) for k in MODEL_NAMES}
    train={}
    for model in MODEL_NAMES:
        train[model]=calc_metrics(h1["groups"],logs,flags,fitted[model])
    # Predeclare H1 proper log score as selection criterion. Never choose on H2.
    winner=min(MODEL_NAMES,key=lambda k:(train[k]["overall"]["nll"],MODEL_NAMES.index(k)))
    print("2024_H1_SELECTED",winner,flush=True)
    tests={}
    for model in MODEL_NAMES:
        tests[model]=calc_metrics(h2["groups"],logs,flags,fitted[model])
        print("H2_COMPLETE",model,tests[model]["overall"],flush=True)
    base=tests["unchanged"]
    comparisons={k:paired_day_ci(base,tests[k]) for k in MODEL_NAMES[1:]}
    obs_baseline=source_report["overall"]["line_role_score_rank"]["mean_nll"]
    complete=calc_metrics(meta,logs,flags,fitted["unchanged"])
    if abs(complete["overall"]["nll"]-obs_baseline)>1e-5:
        raise ValueError("2024 frozen baseline no longer reproduces prior report")
    # Predeclared acceptance: 2024 H2 must improve both proper log loss and
    # fixed-bin overall calibration on untouched 2024 H2. Day-cluster CI for
    # NLL should additionally have lower limit > 0; else keep unchanged.
    selected_ci=comparisons.get(winner)
    accept=(winner!="unchanged" and selected_ci["approx_95pct_ci"][0]>0
            and tests[winner]["ticket_weighted_ece_pp"]<base["ticket_weighted_ece_pp"])
    adopted=winner if accept else "unchanged"
    # Export chosen model probabilities for entire 2024, but H1 probabilities
    # are fitted/in-sample; H2 are independent. Outputs never feed training.
    with gzip.open(PROBS,"wt",encoding="utf-8",compresslevel=6) as f:
        for r in meta:
            p=per_race_weights(r,logs,flags,fitted[adopted])
            ref=events[r["id"]]
            # Ticket order in archived file is sorted numerical strings; since
            # car numbers are 1..9, lexicographic and numeric order coincide.
            cars=sorted(ref["scores"],key=int)
            import itertools
            tickets=sorted("-".join(t) for t in itertools.permutations(cars,3))
            if len(tickets)!=len(p):raise ValueError("Ticket alignment invalid")
            out={"race_id":r["id"],"date":r["date"],
                 "source_2023_baseline":"2024 original frozen artifact",
                 "calibration_fit":"2024-01 to 2024-06",
                 "evaluation_partition":"FIT_H1" if r["date"]<=H1_END else "TEST_H2",
                 "selected_variant":adopted,
                 "trifecta_probabilities":{"ticket":None} if False else
                   {t:round(float(q),12) for t,q in zip(tickets,p)}}
            f.write(json.dumps(out,ensure_ascii=False,separators=(",",":"))+"\n")
    seven=next((r for r in h2["groups"] if r["field"]==7),None)
    example=None
    if seven:
        ss=seven["start"];orig=np.exp(logs[ss:ss+seven["size"]])
        old=orig/orig.sum()
        new=per_race_weights(seven,logs,flags,fitted[adopted])
        cars=sorted(events[seven["id"]]["scores"],key=int)
        import itertools
        tickets=sorted("-".join(t) for t in itertools.permutations(cars,3))
        example={"race_id":seven["id"],"date":seven["date"],
                 "actual_trifecta":"-".join(events[seven["id"]]["order"]),
                 "adopted":adopted,"probability_sum":float(new.sum()),
                 "all_tickets":[{"ticket":t,"before_pct":round(float(a*100),8),
                                 "after_pct":round(float(b*100),8)}
                                 for t,a,b in zip(tickets,old,new)]}
    report={
       "state":"COMPLETE_FROZEN_2023_CALIBRATED_2024_H1_TEST_2024_H2",
       "original_2023_model_changed":False,
       "2023_base_observation_training":"2023-01-01 through 2023-10-31",
       "2024_calibrator_fit":["2024-01-01","2024-06-30"],
       "2024_unseen_assessment":["2024-07-01","2024-12-31"],
       "2025_or_later_accessed":False,"odds_or_market_data_accessed":False,
       "source_baseline_run":40807880319,
       "source_baseline_archive_sha256_uncompressed":archive_digest,
       "source_2024_verified_baseline_nll":obs_baseline,
       "baseline_nll_reproduced_all_2024":complete["overall"]["nll"],
       "races":{"calibration_fit":h1["n"],"strict_test":h2["n"],"entire_2024":len(meta)},
       "total_2024_ticket_probabilities":len(logs),
       "smoothing_unchanged":{"alpha_temperature":"fitted only Jan-Jun",
           "line_coefficient_regularizer":STRUCT_L2},
       "fitted_coefficients":fitted,
       "model_selected_using_2024_H1_only":winner,
       "selection_rule":"Minimum 2024 Jan-Jun multiclass NLL, ties in model order",
       "untouched_2024_H2_test":{
         "models":{k:compact(v) for k,v in tests.items()},
         "nll_improvement_vs_unchanged_daily_cluster_95pct_ci":comparisons
       },
       "calibration_fit_H1":{"models":{k:compact(v) for k,v in train.items()}},
       "acceptance_rule":"H1-selected model must improve H2 NLL with day-cluster 95% CI lower bound above zero AND reduce fixed-bin ticket ECE; else original unchanged",
       "accepted":bool(accept),
       "model_adopted":adopted,
       "all_year_recalibrated_probabilities":str(PROBS.relative_to(ROOT)),
       "all_year_export_note":"Jan-Jun probabilities are in-sample for calibrator; only Jul-Dec are strictly held out. Full archive is an Actions artifact, not tracked Git.",
       "sample":str(SAMPLE.relative_to(ROOT)),
       "limitations":[
         "2024 model selection uses Jan-Jun only; Jul-Dec used once to assess and accept, not refit or retune.",
         "2025 data is reserved for final independent validation; no 2025 predictions or results opened.",
         "Predictive pre-race archival timestamps of rider scores and published line forecasts are retrospectively recorded and not independently proven.",
         "Ticket-level calibration bins group dependent mutually exclusive outcomes; null variance computed at race-group level.",
         "Data-driven choice between multiple calibrators introduces selection optimism in H1, avoided in H2.",
         "Calibration ECE over 5.8m tickets can be misleadingly small; focus on NLL, fixed probability bands and structural residuals.",
         "Original baseline ticket probabilities were rounded to 12 decimals in frozen artifact; verified mean NLL matches baseline audit.",
       ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    SAMPLE.write_text(json.dumps(example,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RECALIBRATION_2024_COMPLETE",json.dumps({
       "races":report["races"],"winner_H1":winner,"adopted":adopted,
       "accepted":bool(accept),"fitted":fitted,
       "H2":{k:{"nll":v["overall"]["nll"],"ece":v["ticket_weighted_ece_pp"],
                 "5-10":v["calibration_bands"].get("5-10%"),
                 "10-20":v["calibration_bands"].get("10-20%")}
             for k,v in tests.items()},
       "CI":comparisons
    },ensure_ascii=False),flush=True)
if __name__=="__main__": main()

#!/usr/bin/env python3
"""Benter-first minimal line model. NOT a keirin-specific structural model.

Anchor is the independently frozen 1st-win fundamental q_i, plus fitted
Benter-style positive exponents gamma/delta. Add only minimal binary line
pair dependence as corrections to CONDITIONAL finishing probability.

Second: gamma * log q_j + a * 1(same_line(i,j))
Third:  delta * log q_k + b * 1(same_line(i,k)) + c * 1(same_line(j,k))
All normalized within eligible runners. Additions tested, not assumed.

Select L2 and ONE vs TWO 3rd-place line indicators using July -> August.
Freeze final coefficients (refit July+Aug) before scoring Sept and H1.
Sept and H1 have both been used in earlier research; these tests are
external to V7 parameter fitting, but not research-wide virgin holdouts.
"""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/"research/benter_line_conditional_v5"))
sys.path.insert(0,str(ROOT/"research/benter_third_survivor_v6"))
from model_line_conditional import get_races, cprob
from model_survivor_v6 import get_h1,third_probs
H2=ROOT/"research/benter_score_challenge_v3/frozen_2026h2_probabilities.csv"
H2R=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/races.csv"
H2O=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/results.csv"
SEP=ROOT/"research/benter_calibration_v4/sep_predictions_frozen.csv"
SEPR=ROOT/"research/benter_calibration_v4/sep_holdout/races.csv"
SEPO=ROOT/"research/benter_calibration_v4/sep_holdout/results.csv"
BASE=ROOT/"research/benter_ordered_23/parameters_frozen.json"
V5=ROOT/"research/benter_line_conditional_v5/parameters_pretest.json"
V6=ROOT/"research/benter_third_survivor_v6/frozen_3rd_survivor_parameters.json"
FROZEN=HERE/"parameters_frozen.json"
SEPT_REPORT=HERE/"september_comparison.json"
H1_REPORT=HERE/"h1_comparison.json"
GRID=(.01,.05,.2,1.0,5.0)

def init_constants():
    orig=json.loads(BASE.read_text(encoding="utf-8"))["fitted_parameters"]["q"]
    return float(orig["gamma"]),float(orig["delta"])

def extras(r,i,j,kind):
    l=r["line_idx"]
    if kind=="second": return (l==l[i]).astype(float).reshape(-1,1)
    if kind=="third_union":return ((l==l[i])|(l==l[j])).astype(float).reshape(-1,1)
    if kind=="third_split":return np.column_stack(((l==l[i]).astype(float),(l==l[j]).astype(float)))
    raise ValueError(kind)

def conditional(r,which,i,j,params,exponent):
    n=len(r["cars"]);eligible=np.ones(n,dtype=bool);eligible[i]=False
    if which!="second":eligible[j]=False
    X=extras(r,i,j,which)
    logits=exponent*np.log(np.maximum(r["prob"],1e-15)) + X@params
    mass=np.zeros(n)
    mass[eligible]=np.exp(logits[eligible]-logsumexp(logits[eligible]))
    return mass

def choices(races,which,exponent):
    records=[]
    for r in races:
        i,j,k=r["win_idx"]
        valid=[t for t in range(len(r["cars"])) if t!=i and (which=="second" or t!=j)]
        F=extras(r,i,j,which)[valid]
        p=exponent*np.log(np.maximum(r["prob"][valid],1e-15))
        winner=j if which=="second" else k
        records.append((p,F,valid.index(winner)))
    return records

def loss_grad(beta,rows,penalty):
    n=len(rows);loss=0.;g=np.zeros(len(beta))
    for offset,F,target in rows:
        z=offset+F@beta
        q=np.exp(z-logsumexp(z))
        loss+=logsumexp(z)-z[target]
        g+=q@F-F[target]
    loss/=n;g/=n
    return loss+0.5*penalty*float(beta@beta),g+penalty*beta

def fit(rows,penalty):
    d=rows[0][1].shape[1]
    res=minimize(lambda b:loss_grad(b,rows,penalty),np.zeros(d),
                 jac=True,method="L-BFGS-B",options={"ftol":1e-13,"maxiter":150})
    if not res.success and np.linalg.norm(res.jac)>1e-5:raise RuntimeError(res.message)
    return res.x
def meanloss(rows,coef):
    return float(np.mean([logsumexp(off+F@coef)-(off+F@coef)[y] for off,F,y in rows]))

def fit_stage():
    gamma,delta=init_constants()
    races,audit=get_races(H2,H2R,H2O,"h2")
    assert len(races)==867
    juli=[r for r in races if r["date"].startswith("2026-07")]
    aug=[r for r in races if r["date"].startswith("2026-08")]
    sweep=[]
    for which in ("second","third_union","third_split"):
        expo=gamma if which=="second" else delta
        train=choices(juli,which,expo)
        test=choices(aug,which,expo)
        for l2 in GRID:
            co=fit(train,l2)
            sweep.append({"which":which,"penalty":l2,"july_loss":meanloss(train,co),
                          "august_loss":meanloss(test,co),"july_coefficients":co.tolist()})
    winner2=min((z for z in sweep if z["which"]=="second"),key=lambda z:z["august_loss"])
    winner3=min((z for z in sweep if z["which"]!="second"),key=lambda z:z["august_loss"])
    params={}
    for target,winner in (("second",winner2),("third",winner3)):
        kind=winner["which"]
        expo=gamma if target=="second" else delta
        both=choices(races,kind,expo)
        final=fit(both,winner["penalty"])
        params[target]={"kind":kind,"penalty":winner["penalty"],"coefficients":final.tolist(),
                        "training_conditional_loss":meanloss(both,final),
                        "baseline_loss":meanloss(both,np.zeros(len(final)))}
    frozen={
        "title":"Benter independent 1st win + conditional exponents + minimum empirical line indicators",
        "fit_period":"2026-07-02..2026-08-30",
        "train_races":len(races),"training_audit":audit,
        "choose":"July for candidate coefficients; August for selecting 3rd feature count and L2 penalty; Jul+Aug refit",
        "gamma":gamma,"delta":delta,
        "feature_candidates":{"second":"1(same i/j line)",
                              "third_union":"1(k in line of i OR j)",
                              "third_split":["1(k in line i)","1(k in line j)"]},
        "selected":{"second":winner2,"third":winner3},
        "fitted":params,
        "sweep":sweep,
        "sept_or_h1_outcome_used_to_fit":False,
        "v5_v6_status":"exploratory challenger models; not promoted on basis of greater feature complexity",
        "note":"No race-type dummies, rider role dummies, line size, third-survivor counts, or 322-specific formula",
    }
    FROZEN.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FROZEN",json.dumps({"n":len(races),"gamma":gamma,"delta":delta,
        "final":params,"selected":frozen["selected"]},ensure_ascii=False),flush=True)

def scoring(races,configs,with_top):
    rows=[]
    for r in races:
        i,j,k=r["win_idx"]
        m={}
        for label,mode in configs.items():
            if mode=="benter":
                p2=conditional(r,"second",i,None,np.zeros(1),mode_gamma)
                p3=conditional(r,"third_union",i,j,np.zeros(1),mode_delta)
            elif mode=="simple":
                p2=conditional(r,"second",i,None,parameters2,mode_gamma)
                p3=conditional(r,third_kind,i,j,parameters3,mode_delta)
            elif mode=="v5":
                p2=cprob(r,2,i,None,v5_2)
                p3=cprob(r,3,i,j,v5_3)
            elif mode=="v6":
                p2=cprob(r,2,i,None,v5_2)
                p3=third_probs(r,i,j,v6_3,v6_fields,v5_3)
            sec=-math.log(p2[j]);third=-math.log(p3[k])
            m[label]={"second":sec,"third":third,"joint":-math.log(r["prob"][i])+sec+third}
        rows.append({"date":r["date"],"race_id":r["race_id"],"models":m,"race":r})
    if with_top:
        # true PRE-RACE ranking over all possible ordered tickets, not
        # "conditionally predicted third" with actual first/second supplied.
        for rec in rows:
            r=rec["race"];truth=tuple(r["win_idx"]);n=len(r["cars"])
            for label,mode in configs.items():
                tickets=[]
                for a in range(n):
                    if mode in ("v5","v6"):m2=cprob(r,2,a,None,v5_2)
                    elif mode=="simple":m2=conditional(r,"second",a,None,parameters2,mode_gamma)
                    else:m2=conditional(r,"second",a,None,np.zeros(1),mode_gamma)
                    for b in range(n):
                        if a==b:continue
                        if mode=="v5":m3=cprob(r,3,a,b,v5_3)
                        elif mode=="v6":m3=third_probs(r,a,b,v6_3,v6_fields,v5_3)
                        elif mode=="simple":m3=conditional(r,third_kind,a,b,parameters3,mode_delta)
                        else:m3=conditional(r,"third_union",a,b,np.zeros(1),mode_delta)
                        for c in range(n):
                            if c in (a,b):continue
                            tickets.append((r["prob"][a]*m2[b]*m3[c],(a,b,c)))
                tickets.sort(reverse=True,key=lambda t:t[0])
                rec["models"][label]["top1"]=int(tickets[0][1]==truth)
                rec["models"][label]["top10"]=int(truth in [v[1] for v in tickets[:10]])
                if abs(sum(p for p,_ in tickets)-1)>1e-8:raise AssertionError("sum tickets != 1")
    return rows

def bootstrap(rows,labels):
    groups=defaultdict(list)
    for n,r in enumerate(rows):groups[r["date"]].append(n)
    days=list(groups.values())
    rng=np.random.default_rng(20261018)
    outs={}
    for a,b in labels:
        key=f"{a}_vs_{b}"
        outs[key]={}
        for item in ("second","third","joint"):
            diff=np.array([r["models"][b][item]-r["models"][a][item] for r in rows])
            values=[]
            for _ in range(1900):
                ix=np.concatenate([np.array(days[z]) for z in rng.integers(len(days),size=len(days))])
                values.append(float(diff[ix].mean()))
            outs[key][item]={"gain":float(diff.mean()),
                              "day_cluster_bootstrap_95":[float(x) for x in np.quantile(values,[.025,.975])]}
    return outs

def test(stage):
    global parameters2,parameters3,third_kind,mode_gamma,mode_delta,v5_2,v5_3,v6_3,v6_fields
    fixed=json.loads(FROZEN.read_text(encoding="utf-8"))
    assert fixed["sept_or_h1_outcome_used_to_fit"] is False
    mode_gamma=fixed["gamma"];mode_delta=fixed["delta"]
    parameters2=np.array(fixed["fitted"]["second"]["coefficients"])
    parameters3=np.array(fixed["fitted"]["third"]["coefficients"])
    third_kind=fixed["fitted"]["third"]["kind"]
    v5=json.loads(V5.read_text(encoding="utf-8"))
    v5_2=np.array(v5["parameters"]["second"]["theta"])
    v5_3=np.array(v5["parameters"]["third"]["theta"])
    v6=json.loads(V6.read_text(encoding="utf-8"))
    v6_3=np.array(v6["beta"]);v6_fields=v6["feature_names"]
    if stage=="sept":races,audit=get_races(SEP,SEPR,SEPO,"sep")
    else:races,audit=get_h1()
    assert len(races)==(419 if stage=="sept" else 2538),len(races)
    modelnames={"benter":"benter","simple":"simple","v5":"v5","v6":"v6"}
    results=scoring(races,modelnames,with_top=(stage=="sept"))
    metrics={}
    for name in modelnames:
        z={item:float(np.mean([r["models"][name][item] for r in results]))
           for item in ("second","third","joint")}
        if stage=="sept":
            z["top1_count"]=sum(r["models"][name]["top1"] for r in results)
            z["top10_count"]=sum(r["models"][name]["top10"] for r in results)
        metrics[name]=z
    comparisons=bootstrap(results,[("simple","benter"),("simple","v5"),("simple","v6"),
                                    ("v5","benter"),("v6","v5")])
    output={"stage":stage,"train":"2026-07-02..2026-08-30 only",
            "n":len(results),"days":len({r["date"] for r in results}),"audit":audit,
            "parameters":fixed["fitted"],
            "comparators":metrics,"paired_improvement":comparisons,
            "interpretation":"Benter-first minimal residual line model is preferred only if validated; higher accuracy ≠ profitable betting.",
            "caveat":"Earlier research already examined both September and H1 outcomes. Not a globally virgin holdout."}
    path=HERE/("september_comparison.json" if stage=="sept" else "h1_comparison.json")
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RESULT",stage,json.dumps({"n":len(results),"comparators":metrics,
          "simple_benter":comparisons["simple_vs_benter"],
          "simple_v5":comparisons["simple_vs_v5"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ("fit","sept","h1"):
        raise SystemExit("Usage: minimal_line.py fit|sept|h1")
    {"fit":fit_stage,"sept":lambda:test("sept"),"h1":lambda:test("h1")}[sys.argv[1]]()

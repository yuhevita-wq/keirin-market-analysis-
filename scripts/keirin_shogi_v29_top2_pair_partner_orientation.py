#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean
import numpy as np

spec=importlib.util.spec_from_file_location("v26","scripts/keirin_shogi_v26_top2_pair_blind_first.py")
v26=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v26)
v25=v26.v25

OUT=Path("results/keirin_shogi/v29_top2_pair_partner_orientation")
OUT.mkdir(parents=True,exist_ok=True)

# v29
# Preserve the unordered Top2 pair posterior from v26.
# Do NOT train another 1st/2nd orientation classifier.
# For each unordered pair {a,b}, allocate its pair probability by the
# frozen v21 first-probability ratio inside that pair:
#   P(2nd=b | pair={a,b}) ~= P_v21(first=a | a,b)
#   P(2nd=a | pair={a,b}) ~= P_v21(first=b | a,b)
# This keeps pair structure intact and uses v21 only after pair formation.

CAL_A_START,CAL_A_END="2025-10-27","2025-11-30"
CAL_B_START,CAL_B_END="2025-12-01","2025-12-28"

def pair_first_share(vr,a,b,alpha):
    eps=1e-6
    pa=max(0.0,float(vr["p1_map"].get(a,0.0)))
    pb=max(0.0,float(vr["p1_map"].get(b,0.0)))
    aa=(pa+eps)**alpha
    bb=(pb+eps)**alpha
    z=aa+bb
    return aa/z if z>0 else 0.5

def predict(base,vr,pm,alpha):
    nos=sorted(base)
    pairs=[(nos[i],nos[j]) for i in range(len(nos)) for j in range(i+1,len(nos))]
    X=[]
    for a,b in pairs:
        x=v26.pair_features_blind(base,a,b,vr["p1_map"])
        X.append([x[f] for f in pm["features"]])
    raw=np.clip(pm["model"].predict_proba(np.asarray(X,float))[:,1],1e-12,None)
    pairp=raw/raw.sum()

    p2=defaultdict(float)
    prows=[]
    for (a,b),pp in zip(pairs,pairp):
        qa=pair_first_share(vr,a,b,alpha)
        # if a is first, b is second; if b is first, a is second.
        p2[b]+=float(pp)*qa
        p2[a]+=float(pp)*(1.0-qa)
        prows.append((a,b,float(pp),float(qa)))
    z=sum(p2.values()) or 1.0
    rank=sorted(((n,p/z) for n,p in p2.items()),key=lambda kv:(-kv[1],kv[0]))
    prows.sort(key=lambda x:-x[2])
    return rank,prows

def choose(rank,t,maxk):
    out=[];cum=0.0
    for n,p in rank[:maxk]:
        out.append(n);cum+=p
        if cum>=t:break
    return out

def evaluate(ds,pm,alpha,t,maxk):
    logs=[]
    for rid,dt,base,order,vr in ds:
        if not vr.get("v21_participate"):continue
        rank,pairs=predict(base,vr,pm,alpha)
        c=choose(rank,t,maxk)
        f,s=order[:2]
        fc=list(map(v25.ino,vr["candidates"]))
        relation="same_line" if base[f]["line_id"]==base[s]["line_id"] else "different_line"
        truepair=frozenset((f,s))
        prank=next((i+1 for i,(a,b,pp,qa) in enumerate(pairs) if frozenset((a,b))==truepair),None)
        logs.append({
          "race_id":rid,"race_date":dt,
          "actual_first":f,"actual_second":s,
          "first_candidates":fc,
          "first_hit":int(f in fc),
          "second_in_first_candidates":int(s in fc),
          "relation":relation,
          "second_candidates":c,
          "second_hit":int(s in c),
          "complete_top2_hit":int(f in fc and s in c),
          "second_rank":next((i+1 for i,(n,p) in enumerate(rank) if n==s),None),
          "pair_rank":prank,
          "second_candidate_count":len(c),
          "top_pairs":[{"a":a,"b":b,"pair_prob":pp,"p_a_first_within_pair":qa} for a,b,pp,qa in pairs[:5]],
          "second_ranking":[{"no":n,"prob":p} for n,p in rank],
        })
    return logs

def safe_rate(rows,key):
    return sum(x[key] for x in rows)/len(rows) if rows else 0.0

def metrics(logs):
    n=len(logs)
    if not n:return {}
    first=[x for x in logs if x["first_hit"]]
    inside=[x for x in first if x["second_in_first_candidates"]]
    outside=[x for x in first if not x["second_in_first_candidates"]]
    hard=[x for x in outside if x["relation"]=="different_line"]
    return {
      "races":n,
      "second_capture":safe_rate(logs,"second_hit"),
      "complete_top2_capture":safe_rate(logs,"complete_top2_hit"),
      "avg_second_candidates":mean(x["second_candidate_count"] for x in logs),
      "second_given_first_capture":safe_rate(first,"second_hit"),
      "inside_capture_given_first":safe_rate(inside,"second_hit"),
      "outside_capture_given_first":safe_rate(outside,"second_hit"),
      "hard_diff_line_outside_capture_given_first":safe_rate(hard,"second_hit"),
      "hard_races":len(hard),
      "true_pair_top1":sum((x["pair_rank"] or 99)<=1 for x in logs)/n,
      "true_pair_top3":sum((x["pair_rank"] or 99)<=3 for x in logs)/n,
      "true_pair_top5":sum((x["pair_rank"] or 99)<=5 for x in logs)/n,
      "second_top1_rate":sum((x["second_rank"] or 99)<=1 for x in logs)/n,
      "second_top2_rate":sum((x["second_rank"] or 99)<=2 for x in logs)/n,
      "second_top3_rate":sum((x["second_rank"] or 99)<=3 for x in logs)/n,
    }

def month_metrics(logs):
    out=[]
    months=sorted(set(x["race_date"][:7] for x in logs))
    for mo in months:
        rr=[x for x in logs if x["race_date"].startswith(mo)]
        out.append({"month":mo,**metrics(rr)})
    return out

def stability_objective(ma,mb,mfull):
    # Selection is driven by stability across two chronological calibration blocks.
    # Hard outside races are diagnostic only; they do not dominate the objective.
    min_sgf=min(ma["second_given_first_capture"],mb["second_given_first_capture"])
    min_complete=min(ma["complete_top2_capture"],mb["complete_top2_capture"])
    min_second=min(ma["second_capture"],mb["second_capture"])
    min_inside=min(ma["inside_capture_given_first"],mb["inside_capture_given_first"])
    min_outside=min(ma["outside_capture_given_first"],mb["outside_capture_given_first"])
    drift=abs(ma["second_given_first_capture"]-mb["second_given_first_capture"])
    cand_pen=max(0.0,mfull["avg_second_candidates"]-2.05)
    return (
      0.42*min_sgf+
      0.24*min_complete+
      0.14*min_second+
      0.10*min_inside+
      0.10*min_outside-
      0.12*drift-
      0.20*cand_pen
    )

def main():
    vrows=v25.load_v19_details()
    v25.fit_v21_selector(vrows)
    v25.mark_v21(vrows)
    _,eb,rb=v25.raw_maps()

    train=v25.make_dataset(vrows,eb,rb,v25.TRAIN_START,v25.TRAIN_END,False)
    cal=v25.make_dataset(vrows,eb,rb,v25.CAL_START,v25.CAL_END,False)
    cal_a=v25.make_dataset(vrows,eb,rb,CAL_A_START,CAL_A_END,False)
    cal_b=v25.make_dataset(vrows,eb,rb,CAL_B_START,CAL_B_END,False)
    test=v25.make_dataset(vrows,eb,rb,v25.TEST_START,v25.TEST_END,False)

    pair_variants=[
      {"name":"blind_d2a","p":{"depth":2,"leaf":18,"l2":1.5,"lr":.045,"iters":190}},
      {"name":"blind_d2b","p":{"depth":2,"leaf":28,"l2":2.5,"lr":.04,"iters":220}},
      {"name":"blind_d3a","p":{"depth":3,"leaf":20,"l2":2.0,"lr":.04,"iters":200}},
      {"name":"blind_d3b","p":{"depth":3,"leaf":30,"l2":3.0,"lr":.035,"iters":240}},
    ]

    models={v["name"]:v26.fit_pair_model(train,v["p"]) for v in pair_variants}
    grid=[]
    for name,pm in models.items():
        for alpha in (.50,.75,1.00,1.25,1.50,2.00):
            for t in (.38,.42,.46,.50,.54,.58,.62):
                for maxk in (2,3):
                    la=evaluate(cal_a,pm,alpha,t,maxk); ma=metrics(la)
                    lb=evaluate(cal_b,pm,alpha,t,maxk); mb=metrics(lb)
                    lf=evaluate(cal,pm,alpha,t,maxk); mf=metrics(lf)
                    if not ma or not mb or not mf:continue
                    if mf["avg_second_candidates"]>2.10:continue
                    obj=stability_objective(ma,mb,mf)
                    grid.append({
                      "variant":name,"alpha":alpha,"threshold":t,"maxk":maxk,
                      "objective":obj,
                      "cal_a":ma,"cal_b":mb,"cal_full":mf
                    })

    grid.sort(key=lambda x:(
      x["objective"],
      min(x["cal_a"]["second_given_first_capture"],x["cal_b"]["second_given_first_capture"]),
      min(x["cal_a"]["complete_top2_capture"],x["cal_b"]["complete_top2_capture"]),
      -x["cal_full"]["avg_second_candidates"]
    ),reverse=True)

    best=grid[0]
    pm=models[best["variant"]]
    test_logs=evaluate(test,pm,best["alpha"],best["threshold"],best["maxk"])
    tm=metrics(test_logs)

    baselines={
      "v23_conditional":{
        "second_capture":0.5342465753424658,
        "complete_top2_capture":0.0,
        "second_given_first_capture":0.5934065934065934,
        "avg_second_candidates":2.0332681017612524,
      },
      "v26_pair_blind_learned_orientation":{
        "second_capture":0.5499021526418787,
        "second_given_first_capture":0.554945054945055,
        "avg_second_candidates":2.0841487279843443,
        "true_pair_top1":0.38551859099804303,
        "true_pair_top3":0.5714285714285714,
        "true_pair_top5":0.7201565557729941,
      },
      "v28_slot_allocation":{
        "second_capture":0.410958904109589,
        "second_given_first_capture":0.4532967032967033,
        "avg_second_candidates":2.0,
      }
    }

    summary={
      "algorithm":"keirin_shogi_v29_top2_pair_partner_orientation",
      "status":"phase_A_candidate",
      "concept":"Preserve v26 unordered Top2 pair posterior; allocate each pair to second-place members only by frozen v21 within-pair first-probability ratio. No learned orientation classifier.",
      "first_core":"v21 adopted and frozen",
      "train_period":[v25.TRAIN_START,v25.TRAIN_END],
      "calibration_period":[v25.CAL_START,v25.CAL_END],
      "calibration_split":[[CAL_A_START,CAL_A_END],[CAL_B_START,CAL_B_END]],
      "test_period":[v25.TEST_START,v25.TEST_END],
      "selected_calibration":best,
      "test_metrics":tm,
      "monthly_test":month_metrics(test_logs),
      "baselines":baselines,
      "delta_vs_v23":{
        "second_capture":tm["second_capture"]-baselines["v23_conditional"]["second_capture"],
        "second_given_first_capture":tm["second_given_first_capture"]-baselines["v23_conditional"]["second_given_first_capture"],
        "avg_second_candidates":tm["avg_second_candidates"]-baselines["v23_conditional"]["avg_second_candidates"],
      },
      "delta_vs_v26":{
        "second_capture":tm["second_capture"]-baselines["v26_pair_blind_learned_orientation"]["second_capture"],
        "second_given_first_capture":tm["second_given_first_capture"]-baselines["v26_pair_blind_learned_orientation"]["second_given_first_capture"],
        "avg_second_candidates":tm["avg_second_candidates"]-baselines["v26_pair_blind_learned_orientation"]["avg_second_candidates"],
      },
      "calibration_top20":grid[:20],
      "validation_guard":{
        "odds_used":False,
        "v21_changed":False,
        "model_selection_uses_2026_h1":False,
        "important":"2026 H1 is a chronological post-2025 test, but it is NOT claimed as a pristine unseen holdout for v29 because prior v25-v28 development already inspected 2026 H1 aggregate results.",
        "true_future_requirement":"Freeze v29 before evaluating newly collected post-2026-06-30 F1 S-class data."
      }
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(test_logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"calibration.json").write_text(json.dumps(grid,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

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

OUT=Path("results/keirin_shogi/v31_top2_membership_second")
OUT.mkdir(parents=True,exist_ok=True)

CAL_A_START,CAL_A_END="2025-10-27","2025-11-30"
CAL_B_START,CAL_B_END="2025-12-01","2025-12-28"

# Core correction:
# The board explicitly allows the same rider on row1 and row2.
# Therefore an unordered Top2 pair does NOT need to be oriented into exact
# first/second roles before it can support row2.
#
# We marginalize pair posterior directly:
#   membership[n] = sum_{pairs containing n} P(pair)
# This is the model's probability mass that rider n belongs to the Top2 set.
# Row2 uses the top membership riders, optionally adding a 3rd rider when
# pair uncertainty justifies it.

def pair_distribution(base,vr,pm):
    nos=sorted(base)
    pairs=[(nos[i],nos[j]) for i in range(len(nos)) for j in range(i+1,len(nos))]
    X=[]
    for a,b in pairs:
        x=v26.pair_features_blind(base,a,b,vr["p1_map"])
        X.append([x[f] for f in pm["features"]])
    raw=np.clip(pm["model"].predict_proba(np.asarray(X,float))[:,1],1e-12,None)
    pp=raw/raw.sum()
    rows=[(a,b,float(p)) for (a,b),p in zip(pairs,pp)]
    rows.sort(key=lambda x:-x[2])
    return rows

def membership_rank(pairs):
    m=defaultdict(float)
    for a,b,p in pairs:
        m[a]+=p;m[b]+=p
    return sorted(m.items(),key=lambda kv:(-kv[1],kv[0]))

def choose(rank,rel3,min3):
    c=[n for n,p in rank[:2]]
    if len(rank)>=3:
        p2=rank[1][1];p3=rank[2][1]
        if p3>=min3 and p3/max(p2,1e-12)>=rel3:
            c.append(rank[2][0])
    return c

def precompute(ds,pm):
    rows=[]
    for rid,dt,base,order,vr in ds:
        if not vr.get("v21_participate"):continue
        pairs=pair_distribution(base,vr,pm)
        rank=membership_rank(pairs)
        f,s=order[:2]
        fc=[n for n in map(v25.ino,vr["candidates"]) if n in base][:2]
        relation="same_line" if base[f]["line_id"]==base[s]["line_id"] else "different_line"
        truepair=frozenset((f,s))
        prank=next((i+1 for i,(a,b,p) in enumerate(pairs) if frozenset((a,b))==truepair),None)
        srank=next((i+1 for i,(n,p) in enumerate(rank) if n==s),None)
        rows.append({
          "race_id":rid,"race_date":dt,
          "actual_first":f,"actual_second":s,
          "first_candidates":fc,"first_hit":int(f in fc),
          "second_in_first_candidates":int(s in fc),
          "relation":relation,"pairs":pairs,"membership":rank,
          "pair_rank":prank,"second_membership_rank":srank
        })
    return rows

def apply(pre,rel3,min3):
    logs=[]
    for r in pre:
        c=choose(r["membership"],rel3,min3)
        s=r["actual_second"]
        logs.append({
          "race_id":r["race_id"],"race_date":r["race_date"],
          "actual_first":r["actual_first"],"actual_second":s,
          "first_candidates":r["first_candidates"],
          "first_hit":r["first_hit"],
          "second_in_first_candidates":r["second_in_first_candidates"],
          "relation":r["relation"],
          "second_candidates":c,
          "second_hit":int(s in c),
          "complete_top2_hit":int(r["first_hit"] and s in c),
          "second_candidate_count":len(c),
          "pair_rank":r["pair_rank"],
          "second_membership_rank":r["second_membership_rank"],
          "membership":[{"no":n,"mass":p} for n,p in r["membership"]],
          "top_pairs":[{"a":a,"b":b,"prob":p} for a,b,p in r["pairs"][:5]]
        })
    return logs

def rate(rows,key):
    return sum(x[key] for x in rows)/len(rows) if rows else 0.0

def metrics(logs):
    n=len(logs)
    if not n:return {}
    first=[x for x in logs if x["first_hit"]]
    inside=[x for x in first if x["second_in_first_candidates"]]
    outside=[x for x in first if not x["second_in_first_candidates"]]
    hard=[x for x in outside if x["relation"]=="different_line"]
    three=[x for x in logs if x["second_candidate_count"]==3]
    return {
      "races":n,
      "second_capture":rate(logs,"second_hit"),
      "complete_top2_capture":rate(logs,"complete_top2_hit"),
      "avg_second_candidates":mean(x["second_candidate_count"] for x in logs),
      "three_candidate_rate":len(three)/n,
      "second_given_first_capture":rate(first,"second_hit"),
      "inside_capture_given_first":rate(inside,"second_hit"),
      "outside_capture_given_first":rate(outside,"second_hit"),
      "hard_diff_line_outside_capture_given_first":rate(hard,"second_hit"),
      "second_membership_top1":sum((x["second_membership_rank"] or 99)<=1 for x in logs)/n,
      "second_membership_top2":sum((x["second_membership_rank"] or 99)<=2 for x in logs)/n,
      "second_membership_top3":sum((x["second_membership_rank"] or 99)<=3 for x in logs)/n,
      "true_pair_top1":sum((x["pair_rank"] or 99)<=1 for x in logs)/n,
      "true_pair_top3":sum((x["pair_rank"] or 99)<=3 for x in logs)/n,
      "true_pair_top5":sum((x["pair_rank"] or 99)<=5 for x in logs)/n,
    }

def objective(a,b,full):
    min_sgf=min(a["second_given_first_capture"],b["second_given_first_capture"])
    min_complete=min(a["complete_top2_capture"],b["complete_top2_capture"])
    min_second=min(a["second_capture"],b["second_capture"])
    min_inside=min(a["inside_capture_given_first"],b["inside_capture_given_first"])
    min_outside=min(a["outside_capture_given_first"],b["outside_capture_given_first"])
    drift=abs(a["complete_top2_capture"]-b["complete_top2_capture"])
    # Candidate count is allowed to adapt, but a permanent third rider should
    # earn its place with measurable capture improvement.
    cell_pen=max(0.0,full["avg_second_candidates"]-2.30)
    always3_pen=max(0.0,full["three_candidate_rate"]-0.60)
    return (
      .34*min_sgf+
      .30*min_complete+
      .16*min_second+
      .10*min_inside+
      .10*min_outside-
      .10*drift-
      .22*cell_pen-
      .10*always3_pen
    )

def monthly(logs):
    out=[]
    for mo in sorted(set(x["race_date"][:7] for x in logs)):
        rr=[x for x in logs if x["race_date"].startswith(mo)]
        out.append({"month":mo,**metrics(rr)})
    return out

def main():
    vrows=v25.load_v19_details();v25.fit_v21_selector(vrows);v25.mark_v21(vrows)
    _,eb,rb=v25.raw_maps()
    train=v25.make_dataset(vrows,eb,rb,v25.TRAIN_START,v25.TRAIN_END,False)
    cal=v25.make_dataset(vrows,eb,rb,v25.CAL_START,v25.CAL_END,False)
    test=v25.make_dataset(vrows,eb,rb,v25.TEST_START,v25.TEST_END,False)

    variants=[
      {"name":"blind_d2a","p":{"depth":2,"leaf":18,"l2":1.5,"lr":.045,"iters":190}},
      {"name":"blind_d2b","p":{"depth":2,"leaf":28,"l2":2.5,"lr":.04,"iters":220}},
      {"name":"blind_d3a","p":{"depth":3,"leaf":20,"l2":2.0,"lr":.04,"iters":200}},
      {"name":"blind_d3b","p":{"depth":3,"leaf":30,"l2":3.0,"lr":.035,"iters":240}},
    ]
    models={v["name"]:v26.fit_pair_model(train,v["p"]) for v in variants}
    grid=[];cache={}
    for name,pm in models.items():
        pre=precompute(cal,pm);cache[name]=pre
        pa=[x for x in pre if CAL_A_START<=x["race_date"]<=CAL_A_END]
        pb=[x for x in pre if CAL_B_START<=x["race_date"]<=CAL_B_END]
        for rel3 in (.55,.65,.75,.85,.95,1.01):
          for min3 in (.00,.15,.20,.25,.30,.35,.40):
            la=apply(pa,rel3,min3);ma=metrics(la)
            lb=apply(pb,rel3,min3);mb=metrics(lb)
            lf=apply(pre,rel3,min3);mf=metrics(lf)
            grid.append({
              "variant":name,"rel3":rel3,"min3":min3,
              "objective":objective(ma,mb,mf),
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
    test_pre=precompute(test,pm)
    logs=apply(test_pre,best["rel3"],best["min3"])
    tm=metrics(logs)

    # strict 2-rider membership baseline for the same selected pair model
    base2=apply(test_pre,1.01,1.0)
    base2m=metrics(base2)

    summary={
      "algorithm":"keirin_shogi_v31_top2_membership_second",
      "status":"phase_A_candidate",
      "concept":"No pair orientation. Convert blind unordered Top2-pair posterior directly into rider Top2-membership mass. Row2 uses top membership riders; third rider is added only under calibrated uncertainty.",
      "first_core":"v21 adopted and frozen",
      "train_period":[v25.TRAIN_START,v25.TRAIN_END],
      "calibration_period":[v25.CAL_START,v25.CAL_END],
      "calibration_split":[[CAL_A_START,CAL_A_END],[CAL_B_START,CAL_B_END]],
      "test_period":[v25.TEST_START,v25.TEST_END],
      "selected_calibration":best,
      "test_metrics":tm,
      "strict_top2_membership_baseline":base2m,
      "monthly_test":monthly(logs),
      "references":{
        "v23":{"second_capture":0.5342465753424658,"second_given_first_capture":0.5934065934065934,"avg_second_candidates":2.0332681017612524},
        "v29":{"second_capture":0.5205479452054794,"second_given_first_capture":0.5796703296703297,"avg_second_candidates":2.0371819960861055},
        "v30":{"second_capture":0.5518590998043053,"second_given_first_capture":0.5714285714285714,"avg_second_candidates":2.0,"complete_top2_capture":0.4070450097847358}
      },
      "validation_guard":{
        "odds_used":False,
        "v21_changed":False,
        "orientation_used":False,
        "model_selection_uses_2026_h1":False,
        "post_2026_06_data_opened":False,
        "note":"2026 H1 is retrospective chronological evaluation only. Future holdout remains sealed."
      },
      "calibration_top20":grid[:20]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"calibration.json").write_text(json.dumps(grid,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

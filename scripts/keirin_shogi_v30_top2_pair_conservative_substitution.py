#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from statistics import mean
import numpy as np

spec=importlib.util.spec_from_file_location("v26","scripts/keirin_shogi_v26_top2_pair_blind_first.py")
v26=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v26)
v25=v26.v25

OUT=Path("results/keirin_shogi/v30_top2_pair_conservative_substitution")
OUT.mkdir(parents=True,exist_ok=True)

CAL_A_START,CAL_A_END="2025-10-27","2025-11-30"
CAL_B_START,CAL_B_END="2025-12-01","2025-12-28"

# v30 principle:
# - default 2nd row = the two adopted v21 first candidates
# - pair model may replace ONE member only when an outside cross-pair
#   provides sufficiently stronger evidence than the inside pair
# - never jump straight to two outsiders
# This treats the v21 pair as the prior and Top2 Pair as a conservative
# falsifier, rather than forcing every race through a fresh 2nd ranking.

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

def pair_prob_map(rows):
    return {frozenset((a,b)):p for a,b,p in rows}

def choose_second(base,vr,pairs,ratio,min_cross,min_adv):
    fc=[n for n in map(v25.ino,vr["candidates"]) if n in base]
    # v21 participants should almost always have two candidates.
    # Fill defensively from frozen p1 ranking without using labels.
    if len(fc)<2:
        extra=sorted(
            ((n,float(vr["p1_map"].get(n,0.0))) for n in base if n not in set(fc)),
            key=lambda kv:(-kv[1],kv[0])
        )
        for n,_ in extra:
            fc.append(n)
            if len(fc)>=2:break
    fc=fc[:2]
    if len(fc)<2:
        return fc,"degenerate",{}

    pm=pair_prob_map(pairs)
    inside_pair=pm.get(frozenset(fc),0.0)
    outside=[n for n in base if n not in set(fc)]

    best=None
    for keep in fc:
        for o in outside:
            p=pm.get(frozenset((keep,o)),0.0)
            cand=(p,keep,o)
            if best is None or cand>best:
                best=cand

    if best is None:
        return fc,"keep_inside",{
          "inside_pair_prob":inside_pair,
          "best_cross_prob":0.0,
          "ratio":0.0,
          "advantage":-inside_pair
        }

    cross,keep,o=best
    rel_ratio=cross/max(inside_pair,1e-12)
    advantage=cross-inside_pair

    do_sub=(cross>=min_cross and rel_ratio>=ratio and advantage>=min_adv)
    if do_sub:
        c=[keep,o]
        mode="substitute_one"
    else:
        c=fc
        mode="keep_inside"

    return c,mode,{
      "inside_pair_prob":inside_pair,
      "best_cross_prob":cross,
      "cross_keep":keep,
      "cross_outsider":o,
      "ratio":rel_ratio,
      "advantage":advantage,
    }

def precompute(ds,pm):
    rows=[]
    for rid,dt,base,order,vr in ds:
        if not vr.get("v21_participate"):continue
        pairs=pair_distribution(base,vr,pm)
        f,s=order[:2]
        fc=[n for n in map(v25.ino,vr["candidates"]) if n in base][:2]
        relation="same_line" if base[f]["line_id"]==base[s]["line_id"] else "different_line"
        truepair=frozenset((f,s))
        prank=next((i+1 for i,(a,b,p) in enumerate(pairs) if frozenset((a,b))==truepair),None)
        rows.append({
          "race_id":rid,"race_date":dt,"base":base,"vr":vr,
          "actual_first":f,"actual_second":s,
          "first_candidates":fc,"first_hit":int(f in fc),
          "second_in_first_candidates":int(s in fc),
          "relation":relation,"pairs":pairs,"pair_rank":prank
        })
    return rows

def apply(pre,ratio,min_cross,min_adv):
    logs=[]
    for r in pre:
        c,mode,diag=choose_second(r["base"],r["vr"],r["pairs"],ratio,min_cross,min_adv)
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
          "mode":mode,"pair_rank":r["pair_rank"],
          "diag":diag,
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
    subs=[x for x in logs if x["mode"]=="substitute_one"]
    kept=[x for x in logs if x["mode"]=="keep_inside"]
    return {
      "races":n,
      "second_capture":rate(logs,"second_hit"),
      "complete_top2_capture":rate(logs,"complete_top2_hit"),
      "avg_second_candidates":mean(len(x["second_candidates"]) for x in logs),
      "second_given_first_capture":rate(first,"second_hit"),
      "inside_capture_given_first":rate(inside,"second_hit"),
      "outside_capture_given_first":rate(outside,"second_hit"),
      "hard_diff_line_outside_capture_given_first":rate(hard,"second_hit"),
      "substitution_rate":len(subs)/n,
      "substitution_capture":rate(subs,"second_hit"),
      "keep_capture":rate(kept,"second_hit"),
      "true_pair_top1":sum((x["pair_rank"] or 99)<=1 for x in logs)/n,
      "true_pair_top3":sum((x["pair_rank"] or 99)<=3 for x in logs)/n,
      "true_pair_top5":sum((x["pair_rank"] or 99)<=5 for x in logs)/n,
    }

def objective(a,b,full):
    # Favor simultaneous row1+row2 correctness and stability.
    # Outside/hard recovery matters, but cannot outweigh destroying inside races.
    min_complete=min(a["complete_top2_capture"],b["complete_top2_capture"])
    min_sgf=min(a["second_given_first_capture"],b["second_given_first_capture"])
    min_second=min(a["second_capture"],b["second_capture"])
    min_inside=min(a["inside_capture_given_first"],b["inside_capture_given_first"])
    min_outside=min(a["outside_capture_given_first"],b["outside_capture_given_first"])
    drift=abs(a["complete_top2_capture"]-b["complete_top2_capture"])
    # Discourage pathological "replace almost every race" behavior.
    sub_pen=max(0.0,full["substitution_rate"]-0.45)
    return (
      .34*min_complete+
      .28*min_sgf+
      .16*min_second+
      .14*min_inside+
      .08*min_outside-
      .10*drift-
      .18*sub_pen
    )

def monthly(logs):
    ret=[]
    for mo in sorted(set(x["race_date"][:7] for x in logs)):
        rr=[x for x in logs if x["race_date"].startswith(mo)]
        ret.append({"month":mo,**metrics(rr)})
    return ret

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
        for ratio in (.80,1.00,1.20,1.40,1.70,2.00):
          for min_cross in (.03,.04,.05,.06,.08,.10):
            for min_adv in (-.02,0.0,.01,.02,.03):
              la=apply(pa,ratio,min_cross,min_adv);ma=metrics(la)
              lb=apply(pb,ratio,min_cross,min_adv);mb=metrics(lb)
              lf=apply(pre,ratio,min_cross,min_adv);mf=metrics(lf)
              grid.append({
                "variant":name,"ratio":ratio,"min_cross":min_cross,"min_adv":min_adv,
                "objective":objective(ma,mb,mf),
                "cal_a":ma,"cal_b":mb,"cal_full":mf
              })

    grid.sort(key=lambda x:(
      x["objective"],
      min(x["cal_a"]["complete_top2_capture"],x["cal_b"]["complete_top2_capture"]),
      min(x["cal_a"]["second_given_first_capture"],x["cal_b"]["second_given_first_capture"]),
      -x["cal_full"]["substitution_rate"]
    ),reverse=True)
    best=grid[0]
    pm=models[best["variant"]]
    test_pre=precompute(test,pm)
    logs=apply(test_pre,best["ratio"],best["min_cross"],best["min_adv"])
    tm=metrics(logs)

    summary={
      "algorithm":"keirin_shogi_v30_top2_pair_conservative_substitution",
      "status":"phase_A_candidate",
      "concept":"Start second row from the two frozen v21 first candidates. Replace at most one member only when blind Top2-pair evidence for a cross pair sufficiently falsifies the inside pair.",
      "first_core":"v21 adopted and frozen",
      "train_period":[v25.TRAIN_START,v25.TRAIN_END],
      "calibration_period":[v25.CAL_START,v25.CAL_END],
      "calibration_split":[[CAL_A_START,CAL_A_END],[CAL_B_START,CAL_B_END]],
      "test_period":[v25.TEST_START,v25.TEST_END],
      "selected_calibration":best,
      "test_metrics":tm,
      "monthly_test":monthly(logs),
      "references":{
        "v23":{"second_capture":0.5342465753424658,"second_given_first_capture":0.5934065934065934,"avg_second_candidates":2.0332681017612524},
        "v29":{"second_capture":0.5205479452054794,"second_given_first_capture":0.5796703296703297,"avg_second_candidates":2.0371819960861055,"complete_top2_capture":0.41291585127201563}
      },
      "validation_guard":{
        "odds_used":False,
        "v21_changed":False,
        "model_selection_uses_2026_h1":False,
        "future_data_opened":False,
        "note":"2026 H1 remains retrospective chronological evaluation only. No post-2026-06-30 F1 data is used for v30 selection."
      },
      "calibration_top20":grid[:20]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"calibration.json").write_text(json.dumps(grid,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

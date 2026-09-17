#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import importlib.util
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
import numpy as np

# This evaluator is intentionally written and committed BEFORE future block data is collected.
# It must not be edited in response to the future result.

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

v19=load_module("v19_future","scripts/keirin_shogi_v19_targeted_participation.py")
v31=load_module("v31_frozen","scripts/keirin_shogi_v31_top2_membership_second.py")

HIST=Path("data/2026_h1/s_class_f1_all")
FUT=Path("data/2026_future_block1/s_class_f1_20260701_20260830")
OUT=Path("results/keirin_shogi/v31_true_future_block1")
OUT.mkdir(parents=True,exist_ok=True)

WARMUP_START=date(2026,6,29)
EVAL_START=date(2026,7,6)
EVAL_END=date(2026,8,30)

FROZEN_REL3=0.65
FROZEN_MIN3=0.30
V21_TARGET_FRACTION=0.25
V21_MAX_FIRST_CANDIDATES=2
V21_LOOKBACK_WEEKS=4

def ino(v):
    try:return int(float(v))
    except:return 0

def load_csv(name):
    out=[]
    for base in (HIST,FUT):
        p=base/name
        if p.exists():
            with p.open("r",encoding="utf-8-sig",newline="") as f:
                out.extend(csv.DictReader(f))
    return out

def ids_between(races,start,end):
    ss=start.isoformat();ee=end.isoformat()
    return [r["race_id"] for r in races
            if ss<=r.get("race_date","")<=ee
            and ino(r.get("entry_count"))==7
            and r.get("meeting_grade")=="F1"
            and "Ｓ級" in r.get("race_type","")]

def result_order(rs):
    vals=[]
    for r in rs:
        p=ino(r.get("finish_position"))
        if p in (1,2,3):vals.append((p,ino(r.get("car_no"))))
    vals.sort()
    return [n for p,n in vals] if [p for p,n in vals]==[1,2,3] else None

def frozen_selector():
    s=json.loads(Path("results/keirin_shogi/v21_quantile_participation/summary.json").read_text(encoding="utf-8"))
    m=s["selector_model"]
    return float(m["intercept"]),np.asarray(m["coefficients"],float)

def selector_score(x,intercept,coef):
    z=float(intercept+np.dot(np.asarray(x,float),coef))
    if z>=0:
        return 1.0/(1.0+math.exp(-z))
    ez=math.exp(z)
    return ez/(1.0+ez)

def threshold_from_history(score_weeks,current_week):
    prior=sorted(w for w in score_weeks if w<current_week)[-V21_LOOKBACK_WEEKS:]
    vals=[r["score"] for w in prior for r in score_weeks[w]
          if r["candidate_count"]<=V21_MAX_FIRST_CANDIDATES]
    if not vals:return 1.0
    n=max(1,int(round(len(vals)*V21_TARGET_FRACTION)))
    return float(sorted(vals)[-n])

def pair_model_frozen():
    # Reconstruct the frozen v31 pair model from the exact historical training block only.
    vrows=v31.v25.load_v19_details()
    v31.v25.fit_v21_selector(vrows)
    v31.v25.mark_v21(vrows)
    _,eb,rb=v31.v25.raw_maps()
    train=v31.v25.make_dataset(vrows,eb,rb,v31.v25.TRAIN_START,v31.v25.TRAIN_END,False)
    params={"depth":2,"leaf":18,"l2":1.5,"lr":0.045,"iters":190}
    return v31.v26.fit_pair_model(train,params)

def metrics(logs):
    if not logs:
        return {
          "races":0,"first_capture":0.0,"second_capture":0.0,
          "complete_top2_capture":0.0,"second_given_first_capture":0.0,
          "avg_first_candidates":0.0,"avg_second_candidates":0.0,
          "three_second_rate":0.0
        }
    first=[x for x in logs if x["first_hit"]]
    return {
      "races":len(logs),
      "first_capture":sum(x["first_hit"] for x in logs)/len(logs),
      "second_capture":sum(x["second_hit"] for x in logs)/len(logs),
      "complete_top2_capture":sum(x["complete_top2_hit"] for x in logs)/len(logs),
      "second_given_first_capture":sum(x["second_hit"] for x in first)/len(first) if first else 0.0,
      "avg_first_candidates":mean(len(x["first_candidates"]) for x in logs),
      "avg_second_candidates":mean(len(x["second_candidates"]) for x in logs),
      "three_second_rate":sum(len(x["second_candidates"])==3 for x in logs)/len(logs),
    }

def main():
    freeze=json.loads(Path("results/keirin_shogi/v31_top2_membership_second/FREEZE.json").read_text(encoding="utf-8"))
    assert freeze["status"]=="FROZEN_BEFORE_TRUE_FUTURE_TEST"
    assert freeze["second_core"]["pair_variant"]=="blind_d2a"
    assert freeze["second_core"]["candidate_rule"]["base_count"]==2

    races=load_csv("races.csv")
    entries=load_csv("entries.csv")
    results=load_csv("results.csv")
    eb=defaultdict(list);rb=defaultdict(list)
    for x in entries:eb[x["race_id"]].append(x)
    for x in results:rb[x["race_id"]].append(x)

    intercept,coef=frozen_selector()
    pm=pair_model_frozen()

    # Seed score-distribution history only from already-existing pre-future v19 OOS rows.
    score_weeks=defaultdict(list)
    old=v31.v25.load_v19_details()
    for r in old:
        if r["week"]>="2026-06-29":continue
        score=selector_score(r["x_selector"],intercept,coef)
        score_weeks[r["week"]].append({
          "score":score,
          "candidate_count":int(r["candidate_count"])
        })

    logs=[];weekly=[]
    cur=WARMUP_START
    while cur<=EVAL_END:
        te=cur+timedelta(days=6)
        h0=cur-timedelta(days=91)
        core_a_end=cur-timedelta(days=36)
        policy_start=cur-timedelta(days=35)
        policy_end=cur-timedelta(days=22)
        final_end=cur-timedelta(days=1)

        core_a=v19.make_races(ids_between(races,h0,core_a_end),eb,rb)
        policy_cal=v19.make_races(ids_between(races,policy_start,policy_end),eb,rb)
        final_train=v19.make_races(ids_between(races,h0,final_end),eb,rb)
        test_ds=v19.make_races(ids_between(races,cur,te),eb,rb)
        if min(len(core_a),len(policy_cal),len(final_train),len(test_ds))==0:
            raise RuntimeError(f"missing walkforward data for {cur}..{te}: "
                               f"{len(core_a)},{len(policy_cal)},{len(final_train)},{len(test_ds)}")

        # Frozen architecture, rolling historical fit only.
        cand_policy=v19.candidate_policy_from_cal(core_a,policy_cal)
        models=v19.v18.fit_ensemble(final_train)

        week=cur.isoformat()
        threshold=threshold_from_history(score_weeks,week)
        generated=[]
        week_eval=[]
        for rid,rr,winner in test_ds:
            pred,tops=v19.v18.predict(rr,models)
            cands=v19.v18.choose(pred,tops,cand_policy)
            x=v19.pred_features(pred,tops,cands)
            score=selector_score(x,intercept,coef)
            participate=(len(cands)<=V21_MAX_FIRST_CANDIDATES and score>=threshold)
            generated.append({"score":score,"candidate_count":len(cands)})

            if cur<EVAL_START:
                continue

            order=result_order(rb.get(rid,[]))
            if not order:
                continue
            base=v31.v25.race_features(eb[rid])
            p1map={ino(x["no"]):float(x["prob"]) for x in pred}
            vr={
              "candidates":cands,
              "p1_map":p1map,
              "v21_participate":participate
            }
            if not participate:
                continue

            pairs=v31.pair_distribution(base,vr,pm)
            rank=v31.membership_rank(pairs)
            second_cands=v31.choose(rank,FROZEN_REL3,FROZEN_MIN3)
            f,s=order[:2]
            rec={
              "race_id":rid,
              "race_date":eb[rid][0].get("race_date",""),
              "week":week,
              "actual_first":f,
              "actual_second":s,
              "first_candidates":[int(n) for n in cands],
              "second_candidates":[int(n) for n in second_cands],
              "first_hit":int(f in cands),
              "second_hit":int(s in second_cands),
              "complete_top2_hit":int(f in cands and s in second_cands),
              "v21_selector_score":score,
              "v21_threshold":threshold,
              "second_membership":[{"no":int(n),"mass":float(p)} for n,p in rank],
              "top_pairs":[{"a":int(a),"b":int(b),"prob":float(p)} for a,b,p in pairs[:5]],
            }
            logs.append(rec);week_eval.append(rec)

        score_weeks[week].extend(generated)
        weekly.append({
          "week":week,
          "period":[cur.isoformat(),te.isoformat()],
          "threshold":threshold,
          "all_7car_f1_s_races":len(test_ds),
          "participants":len(week_eval),
          "metrics":metrics(week_eval) if cur>=EVAL_START else None,
          "candidate_policy":cand_policy,
          "warmup":cur<EVAL_START,
        })
        cur+=timedelta(days=7)

    summary={
      "algorithm":"keirin_shogi_v31_true_future_block1",
      "freeze_commit":"e41ea8debb828c0a4df32db1d9d180f67828bcd0",
      "future_block":["2026-07-06","2026-08-30"],
      "warmup_week":["2026-06-29","2026-07-05"],
      "sealed_remaining_period":"2026-08-31 onward remains unused by this evaluator",
      "policy":{
        "first":"v21 frozen selector coefficients/policy; weekly core fit uses only prior data; threshold uses prior 4 weeks unlabeled score distribution",
        "second":"v31 frozen blind_d2a unordered Top2 membership; rel3=0.65; min3=0.30; no orientation"
      },
      "metrics":metrics(logs),
      "weekly":weekly,
      "guards":{
        "odds_used":False,
        "future_block_used_for_parameter_selection":False,
        "current_week_results_used_for_prediction":False,
        "future_results_used_for_prior_week_training":False,
        "note":"Results are loaded for evaluation and for later-week historical training only. Each week's prediction is built from dates strictly before that week."
      }
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

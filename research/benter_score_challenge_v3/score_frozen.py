#!/usr/bin/env python3
"""Independently score probabilities previously frozen and committed by fit_and_freeze.py."""
from __future__ import annotations
import csv
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research/benter_score_challenge_v3"
RESULTS=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/results.csv"
MODELS=("score_only_logit","v2_line_model","score_tree","no_score_tree","joint_tree")

def read_file(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    manifest=json.loads((OUT/"frozen_manifest.json").read_text(encoding="utf-8"))
    assert manifest["race_result_file_opened_by_this_script"] is False
    assert manifest["holdout_start"]=="2026-07-02"
    forecasts=defaultdict(list)
    for r in read_file(OUT/"frozen_2026h2_probabilities.csv"):
        forecasts[r["race_id"]].append(r)
    outcomes=defaultdict(list)
    for r in read_file(RESULTS):
        if manifest["holdout_start"]<=r.get("race_date","")<=manifest["holdout_end"]:
            outcomes[r["race_id"]].append(r)
    records=[]
    excluded=[]
    for rid,ps in forecasts.items():
        wins=[s["car_no"] for s in outcomes.get(rid,[]) if s.get("finish_position","").strip()=="1"]
        if len(wins)!=1 or wins[0] not in {p["car_no"] for p in ps}:
            excluded.append({"race_id":rid,"winner_count":len(wins)})
            continue
        winner=wins[0]
        pred_ranking={name: {p["car_no"]:float(p[name]) for p in ps} for name in MODELS}
        score_max={p["car_no"]:float(p["race_car_score"]) for p in ps}
        if any(not math.isfinite(z) for z in score_max.values()):
            excluded.append({"race_id":rid,"reason":"Invalid score baseline"})
            continue
        score_argmax=max(score_max,key=score_max.get)
        detail={
            "race_id":rid,"race_date":ps[0]["race_date"],"track":ps[0]["track"],
            "race_no":ps[0]["race_no"],"winner":winner,
            "score_argmax_prediction":score_argmax,
            "score_argmax_hit":int(score_argmax==winner),
            "n_entrants":len(ps),
        }
        for name,v in pred_ranking.items():
            if not abs(sum(v.values())-1)<1e-8:
                raise ValueError(f"Not normalized: {name} {rid}")
            predicted=max(v,key=v.get)
            prob=v[winner]
            detail[f"{name}_prediction"]=predicted
            detail[f"{name}_hit"]=int(predicted==winner)
            detail[f"{name}_actual_probability"]=prob
            detail[f"{name}_log_loss"]=-math.log(max(prob,1e-300))
            detail[f"{name}_brier"]=sum((p-int(c==winner))**2 for c,p in v.items())
        records.append(detail)
    n=len(records)
    if n<600:raise RuntimeError(f"Too few scored holdout races: {n}")
    metrics={}
    for name in ("score_argmax",*MODELS):
        hit_arr=np.array([r[f"{name}_hit"] for r in records])
        met={"races":n,"top1_hits":int(hit_arr.sum()),"top1_accuracy":float(hit_arr.mean())}
        if name!="score_argmax":
            met["log_loss"]=float(np.mean([r[f"{name}_log_loss"] for r in records]))
            met["brier"]=float(np.mean([r[f"{name}_brier"] for r in records]))
        metrics[name]=met

    def clustered_delta(a,b,measurement):
        vals=np.array([r[f"{b}_{measurement}"]-r[f"{a}_{measurement}"]
                      if measurement=="log_loss" else r[f"{a}_{measurement}"]-r[f"{b}_{measurement}"]
                      for r in records],dtype=float)
        groups=defaultdict(list)
        for i,r in enumerate(records):
            groups[r["race_date"]].append(i)
        g=[np.asarray(indices,dtype=int) for indices in groups.values()]
        rng=np.random.default_rng(20261018)
        estimates=[]
        for k in range(3000):
            sample=rng.integers(len(g),size=len(g))
            ids=np.concatenate([g[i] for i in sample])
            estimates.append(float(vals[ids].mean()))
        return {"improvement":float(vals.mean()),"bootstrap_by_day_95pct_interval":
                 [float(x) for x in np.quantile(estimates,[0.025,0.975])],
                "interpretation":"positive means model A beats model B"}
    contrasts={}
    for a in ("joint_tree","no_score_tree","score_tree","v2_line_model"):
        for b in ("score_only_logit","score_argmax"):
            contrasts[f"{a}_vs_{b}_top1"]=clustered_delta(a,b,"hit")
        contrasts[f"{a}_vs_score_only_logit_logloss"]=clustered_delta(a,"score_only_logit","log_loss")
    contrasts["joint_tree_vs_score_tree_logloss"]=clustered_delta("joint_tree","score_tree","log_loss")
    contrasts["joint_tree_vs_score_tree_top1"]=clustered_delta("joint_tree","score_tree","hit")
    contrasts["joint_tree_vs_v2_line_model_logloss"]=clustered_delta("joint_tree","v2_line_model","log_loss")
    daily=[]
    for day in sorted({r["race_date"] for r in records}):
        rr=[r for r in records if r["race_date"]==day]
        d={"date":day,"races":len(rr)}
        for name in ("score_argmax",*MODELS):
            d[f"{name}_hits"]=sum(r[f"{name}_hit"] for r in rr)
        daily.append(d)
    track_scores=defaultdict(lambda:Counter())
    for r in records:
        t=track_scores[r["track"]]
        t["races"]+=1
        for name in ("score_argmax",*MODELS):
            t[f"{name}_hits"]+=r[f"{name}_hit"]
    summary={
        "status":"EXPLORATORY_POST_RACE_ARCHIVED_RACECARDS",
        "scope":"2026-07-02 through 2026-08-30 F1 S-class only; no girls",
        "trained_on":"2023-2025 S-class F1; 2025 for hyperparam selection, 2023-2025 refit",
        "model_predictions_committed_before_outcome_scoring":True,
        "archive_count":manifest["holdout_archive_races"],
        "predicted":manifest["frozen_holdout_races"],
        "scored":n,"unscorable":excluded,
        "metrics":metrics,
        "contrasts":contrasts,
        "track_summary":{k:dict(v) for k,v in track_scores.items()},
        "critical_caveat":"Historical entrant and published predicted-line data were collected Sept 17 after target races; as-of correctness NOT VERIFIED. Do not claim genuine prospective proof or betting edge.",
        "no_betting_returns_evaluated":True
    }
    with (OUT/"individual_results.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with (OUT/"daily_results.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(daily[0].keys()))
        writer.writeheader()
        writer.writerows(daily)
    (OUT/"scorecard.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RESULT",json.dumps({"scored":n,"metrics":metrics,"contrasts":contrasts},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()

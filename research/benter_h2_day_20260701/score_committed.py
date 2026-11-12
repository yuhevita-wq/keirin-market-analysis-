#!/usr/bin/env python3
"""Score *already-committed* predictions without refitting or changing them."""
from __future__ import annotations
import csv, json, math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research/benter_h2_day_20260701"
RESULTS=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830/results.csv"
MANIFEST=OUT/"pre_score_manifest.json"
PREDICTIONS=OUT/"predictions_frozen.csv"
MODELS=("score_only","full_v1","full_plus_role","full_plus_line","no_score_plus_line")


def main():
    assert PREDICTIONS.exists() and MANIFEST.exists(), "Predictions must be committed before scoring"
    frozen=json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert frozen["date"]=="2026-07-01"
    preds=defaultdict(list)
    with PREDICTIONS.open(encoding="utf-8",newline="") as f:
        for rec in csv.DictReader(f):
            assert rec["date"]=="2026-07-01"
            preds[rec["race_id"]].append(rec)
    actual=defaultdict(list)
    with RESULTS.open(encoding="utf-8-sig",newline="") as f:
        for rec in csv.DictReader(f):
            if rec.get("race_date") == "2026-07-01":
                actual[rec["race_id"]].append(rec)
    excluded=[]
    detailed=[]
    win_probs=defaultdict(list)
    top1=Counter()
    briers=defaultdict(list)
    for rid in frozen["eligible_race_ids"]:
        ps=preds[rid]
        winners=[r["car_no"] for r in actual[rid] if r.get("finish_position","").strip()=="1"]
        if len(winners)!=1:
            excluded.append({"race_id":rid,"reason":"Missing or ambiguous first place", "winners":winners})
            continue
        winner=winners[0]
        cars={r["car_no"] for r in ps}
        if winner not in cars:
            excluded.append({"race_id":rid,"reason":"Winner not in forecasts", "winner":winner})
            continue
        row={"race_id":rid,"track":ps[0]["track"],"race_no":int(ps[0]["race_no"]),
             "race_type":ps[0]["race_type"],"actual_winner":winner,
             "car_count":len(ps)}
        for model in MODELS:
            values={r["car_no"]:float(r[model]) for r in ps}
            if abs(sum(values.values())-1)>1e-9:raise AssertionError("Probability sum invalid")
            selected=max(values,key=lambda car:values[car])
            wp=values[winner]
            win_probs[model].append(wp)
            top1[model]+=int(selected==winner)
            briers[model].append(sum((v-(car==winner))**2 for car,v in values.items()))
            row[f"{model}_predicted_car"]=selected
            row[f"{model}_predicted_car_probability"]=values[selected]
            row[f"{model}_actual_winner_probability"]=wp
            row[f"{model}_top1_correct"]=int(selected==winner)
            row[f"{model}_log_loss"]=-math.log(max(wp,1e-300))
        detailed.append(row)
    if not detailed:raise RuntimeError("No results match frozen predictions")
    n=len(detailed)
    baseline=float(np.mean([math.log(row["car_count"]) for row in detailed]))
    metrics={}
    for model in MODELS:
        p=np.array(win_probs[model],dtype=float)
        metrics[model]={
            "races":n,
            "top1_hits":top1[model],
            "top1_accuracy":top1[model]/n,
            "log_loss":float(-np.log(np.maximum(p,1e-300)).mean()),
            "multiclass_brier":float(np.mean(briers[model])),
            "difference_vs_uniform_log_loss":baseline+np.log(np.maximum(p,1e-300)).mean(),
        }
    def delta(a,b):
        lp=np.log(np.maximum(np.array(win_probs[a]),1e-300))-np.log(np.maximum(np.array(win_probs[b]),1e-300))
        rng=np.random.default_rng(20260701)
        sample=rng.integers(n,size=(2000,n))
        boots=lp[sample].mean(axis=1)
        return {"better_if_positive":float(lp.mean()),
                "bootstrap_race_95pct_interval":np.quantile(boots,[.025,.975]).tolist()}
    summary={
        "date":"2026-07-01",
        "scope":"all available F1 S-class races on day, excluding missing/ambiguous outcomes",
        "pre_result_predictions_file":"research/benter_h2_day_20260701/predictions_frozen.csv",
        "training_cutoff":"2026-06-30",
        "entered_races":len(preds),
        "scored_races":n,
        "unscorable":excluded,
        "uniform_log_loss":baseline,
        "metrics":metrics,
        "pairwise_probabilistic_improvement":{
            "full_plus_line_vs_full_v1":delta("full_plus_line","full_v1"),
            "full_plus_line_vs_score_only":delta("full_plus_line","score_only"),
            "full_plus_line_vs_full_plus_role":delta("full_plus_line","full_plus_role"),
            "full_plus_line_vs_no_score_plus_line":delta("full_plus_line","no_score_plus_line")
        },
        "critical_limitation":"2026-07-01 entries and predicted lineup archived 2026-09-17 after the races. They may differ from contemporaneous pre-race factors; NOT a genuine prospective pre-race test.",
        "model_scope":"Only F1 S-class; not other grades or A-class."
    }
    with (OUT/"scored_by_race.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=detailed[0].keys())
        w.writeheader()
        w.writerows(detailed)
    (OUT/"score_report.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()

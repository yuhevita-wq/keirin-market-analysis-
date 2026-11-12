#!/usr/bin/env python3
"""Reproduce top-two exact WINNER inclusion audit from archived frozen probabilities.

No training or changes to original model outputs.
Uses the same 872 single-winner races in the previously committed holdout.
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PRED=ROOT/"frozen_2026h2_probabilities.csv"
ACTUAL=ROOT/"individual_results.csv"
FIELDS=("race_car_score","score_only_logit","v2_line_model","score_tree","no_score_tree","joint_tree")

def load(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    labels={r["race_id"]:r["winner"] for r in load(ACTUAL)}
    by_race=defaultdict(list)
    for row in load(PRED):
        by_race[row["race_id"]].append(row)
    top1={f:0 for f in FIELDS}
    top2={f:0 for f in FIELDS}
    contrast={pair:{"both":0,"model_only":0,"score_only":0,"neither":0}
              for pair in ("v2_line_model","joint_tree")}
    n=0
    for rid,rows in sorted(by_race.items()):
        if rid not in labels: continue
        winner=labels[rid]
        n+=1
        selections={}
        for field in FIELDS:
            ranked=sorted(rows,key=lambda x:(-float(x[field]),int(x["car_no"])))
            selections[field]=[r["car_no"] for r in ranked[:2]]
            top1[field]+=winner==ranked[0]["car_no"]
            top2[field]+=winner in selections[field]
        for field in contrast:
            a=winner in selections[field]
            b=winner in selections["race_car_score"]
            key=("both" if a and b else "model_only" if a else "score_only" if b else "neither")
            contrast[field][key]+=1
    assert n==872,(n,len(labels))
    assert top1["race_car_score"]==292
    assert top1["v2_line_model"]==330
    assert top1["joint_tree"]==312
    assert top2["race_car_score"]==493
    assert top2["v2_line_model"]==523
    assert top2["joint_tree"]==539
    print({"scored_races":n,"top1":top1,"top2":top2,
           "top2_rates":{k:v/n for k,v in top2.items()},"paired_comparisons":contrast})

if __name__=="__main__":
    main()

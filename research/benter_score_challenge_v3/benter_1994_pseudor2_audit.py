#!/usr/bin/env python3
"""Evaluate independent keirin model under Benter (1994) eq. (3).

R^2 = 1 - L(fundamental win probabilities) / L(1/N_race).
This score does NOT establish a cross-sport ranking of model quality.
Reads previously frozen probabilities and scored races, no model refit.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).parent
PRED=HERE/"frozen_2026h2_probabilities.csv"
TRUTH=HERE/"individual_results.csv"

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    actual={r["race_id"]:r for r in read(TRUTH)}
    rows=[r for r in read(PRED) if r["race_id"] in actual]
    by_race=defaultdict(list)
    for r in rows:
        by_race[r["race_id"]].append(r)
    assert len(by_race)==872
    names=("score_only_logit","v2_line_model","joint_tree","no_score_tree")
    stats={name:{"ll":0.0,"winner_count":0} for name in names}
    null=0.0
    dates=defaultdict(list)
    calibrate=[]
    for rid,parts in by_race.items():
        label=actual[rid]["winner"]
        n=len(parts)
        null+=math.log(n)
        dates[parts[0]["race_date"]].append(rid)
        for name in names:
            probs={p["car_no"]:float(p[name]) for p in parts}
            assert abs(sum(probs.values())-1)<1e-9
            stats[name]["ll"]-=math.log(max(probs[label],1e-300))
            stats[name]["winner_count"]+=int(max(probs,key=probs.get)==label)
        for p in parts:
            calibrate.append((p["race_date"],float(p["v2_line_model"]),int(p["car_no"]==label)))
    n=len(by_race)
    print("Benter 1994: fundamental R²=0.1245 over 3,198 HK horse races (32,877 runners)")
    for name,score in stats.items():
        print(name, {"races":n,
                     "mean_logloss":score["ll"]/n,
                     "uniform_logloss":null/n,
                     "benter_style_pseudo_R2":1-score["ll"]/null,
                     "top1_accuracy":score["winner_count"]/n})
    high=[(date,p,y) for date,p,y in calibrate if p>=.4]
    print("V2 40%+ bucket",{
        "riders":len(high),
        "mean_probability":sum(p for _,p,_ in high)/len(high),
        "actual_win_frequency":sum(y for _,_,y in high)/len(high),
        "predicted_wins":sum(p for _,p,_ in high),
        "actual_wins":sum(y for _,_,y in high),
    })
    # Bootstrap by day because racing observations on the same date are dependent.
    dated=defaultdict(list)
    for date,p,y in high:
        dated[date].append((p,y))
    groups=list(dated.values())
    rng=np.random.default_rng(20261018)
    estimates=[]
    for _ in range(4000):
        indices=rng.integers(len(groups),size=len(groups))
        sample=[x for j in indices for x in groups[j]]
        estimates.append(sum(y-p for p,y in sample)/len(sample))
    print("40%+ actual-predicted bootstrap day 95% CI",np.quantile(estimates,[0.025,0.975]).tolist())

if __name__=="__main__":
    main()

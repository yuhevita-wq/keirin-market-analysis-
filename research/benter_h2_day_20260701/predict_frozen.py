#!/usr/bin/env python3
"""One-day frozen-model forecast. Does not read outcome/result files.

Date was fixed ahead of scoring to first available day after 2026-H1: 2026-07-01.
All entry attributes are retrospectively collected, NOT certified historic as-of.
Models are read from the prior V2 models.json and not refitted here.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "research/benter_structural_discovery_v2"))
import discover_v2 as v2

DATA = ROOT / "data/2026_future_block1/s_class_f1_20260701_20260830"
MODELS = ROOT / "research/benter_structural_discovery_v2/output/models.json"
OUT = ROOT / "research/benter_h2_day_20260701"
DATE = "2026-07-01"
COMPARE = ("score_only", "full_v1", "full_plus_role", "full_plus_line", "no_score_plus_line")


def take(day, file):
    with (DATA/file).open(encoding="utf-8-sig",newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("race_date") == day]


def model_probability(x, names, model):
    ix = [v2.BASE_FEATURES.index(f) if f in v2.BASE_FEATURES else
          len(v2.BASE_FEATURES) + v2.STRUCT_FEATURES.index(f) for f in names]
    mat = x[:,ix]
    tr = model["scaler"]
    median=np.array(tr["median"],dtype=float)
    mean=np.array(tr["mean"],dtype=float)
    std=np.array(tr["std"],dtype=float)
    z=np.where(np.isfinite(mat),mat,median)
    xx=np.column_stack(((z-mean)/std, (~np.isfinite(mat)).astype(float)))
    beta=np.array(model["coefficients"],dtype=float)
    assert len(beta) == xx.shape[1]
    logits=xx@beta
    return np.exp(logits-logsumexp(logits))


def main():
    assert DATE >= "2026-07-01"
    models = json.loads(MODELS.read_text(encoding="utf-8"))
    all_races=take(DATE,"races.csv")
    entries=defaultdict(list)
    for r in take(DATE,"entries.csv"):
        entries[r["race_id"]].append(r)
    forecast=[]
    meta=[]
    rejected=[]
    for race in sorted(all_races,key=lambda r:(r["track"],int(r["race_no"]))):
        rid=race["race_id"]
        es=entries.get(rid,[])
        try:
            es=sorted(es,key=lambda x:int(x["car_no"]))
            cars=[int(e["car_no"]) for e in es]
            if len(cars)<5 or len(cars)>9 or len(set(cars)) != len(cars):
                rejected.append({"race_id":rid,"reason":"Invalid entrants"})
                continue
            raw=np.array([[v2.v1.as_number(e.get(k)) for k in v2.BASE_FEATURES] for e in es],dtype=float)
            lineup=v2.parse_line(race.get("predicted_line_formation",""),cars)
            if lineup is None:
                rejected.append({"race_id":rid,"reason":"Incomplete/invalid line"})
                continue
            line=v2.build_interaction({"cars":cars,"x":raw},lineup)
            x=np.column_stack((raw,line))
            if not np.isfinite(raw).any():
                rejected.append({"race_id":rid,"reason":"All base features missing"})
                continue
            probs={name:model_probability(x,models[name]["fields"],models[name]) for name in COMPARE}
            for name,p in probs.items():
                if abs(p.sum()-1)>1e-9:
                    raise ValueError(f"Probability sum not 1: {name} {rid}")
            for idx,car in enumerate(cars):
                record={"date":DATE,"race_id":rid,"track":race["track"],
                       "race_no":race["race_no"],"race_type":race["race_type"],
                       "car_no":car}
                record.update({name:format(p[idx],".14f") for name,p in probs.items()})
                forecast.append(record)
            leader=probs["full_plus_line"].argmax()
            meta.append({"race_id":rid,"track":race["track"],"race_no":int(race["race_no"]),
                         "winner_forecast":cars[int(leader)],"max_probability":float(probs["full_plus_line"][leader]),
                         "input_capture_time":race.get("captured_at_utc"),
                         "lineup":race.get("predicted_line_formation"),
                         "feature_capture_times":sorted({e.get("captured_at_utc","") for e in es})})
        except (ValueError,KeyError,IndexError) as e:
            rejected.append({"race_id":rid,"reason":str(e)})
    if not forecast:raise RuntimeError("No eligible races were predicted")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"predictions_frozen.csv").open("w",encoding="utf-8",newline="") as f:
        fields=["date","race_id","track","race_no","race_type","car_no",*COMPARE]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(forecast)
    (OUT/"pre_score_manifest.json").write_text(json.dumps({
        "date":DATE,"scope":"all S-class F1 races present in archive, no cherry-picking",
        "frozen_models_source":"research/benter_structural_discovery_v2/output/models.json",
        "training_cutoff":"2026-06-30",
        "predicted_races":len(meta),
        "eligible_race_ids":[r["race_id"] for r in meta],
        "models_compared":list(COMPARE),
        "predicted_race_details":meta,
        "excluded_races":rejected,
        "warning":"Inputs were collected after the event (Sept 17, 2026). This is a holdout, not an authentic pre-race forecast."
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"date":DATE,"predicted_races":len(meta),"predicted_rows":len(forecast),
                      "excluded":rejected,"first_prediction":meta[0]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

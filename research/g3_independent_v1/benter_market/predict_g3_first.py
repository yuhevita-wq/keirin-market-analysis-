#!/usr/bin/env python3
"""Calculate fresh G3-only independent first-win q from a genuine pre-race card.

Uses frozen existing G3 2022-2025 V1 weights and transform, and ONLY those.
Uses same 17 fundamental + line-structure + foreign-category feature
extraction employed by original G3 training (no F1 fitted coefficient).
No outcome label, results, odds or payout allowed in card.
"""
from __future__ import annotations
import argparse,json,math,sys,unicodedata
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_g3_first import FIELDS,FEATURE_NAMES,numeric,struct

DEFAULT_MODEL=ROOT/"output/g3_first_win_report.json"
FORBIDDEN={"winner","actual_first","actual_second","actual_third",
           "results","result","finish_position","odds","payout","payment","dividend"}

def norm(s):
    return unicodedata.normalize("NFKC",str(s or "")).upper().strip()

def independent_first(card,model):
    if not isinstance(card,dict):
        raise ValueError("Racecard must be a dict")
    if FORBIDDEN.intersection(card):
        raise ValueError("Do not supply actual finishing results, payouts or market odds as fundamental predictors")
    if norm(card.get("meeting_grade")) not in ("G3","GIII") or not norm(card.get("race_type")).startswith("S級"):
        raise ValueError("Only G3 S級 independent first model")
    if not card.get("race_id") or not card.get("predicted_line_formation"):
        raise ValueError("Original G3 race ID and published predicted formation required")
    if not card.get("source_url") or not card.get("snapshot_captured_at_utc"):
        raise ValueError("Racecard requires verifiable original source and capture time")
    entries=card.get("riders")
    if not isinstance(entries,list) or not 5<=len(entries)<=9:
        raise ValueError("5-9 actual G3 entrants required")
    if any(not isinstance(x,dict) or FORBIDDEN.intersection(x) for x in entries):
        raise ValueError("No actual future results, odds, payouts in riders")
    entries=sorted(entries,key=lambda x:int(x["car_no"]))
    cars=[int(x["car_no"]) for x in entries]
    if len(set(cars))!=len(cars) or any(not 1<=x<=9 for x in cars):
        raise ValueError("Car number duplicate or invalid")
    lines=struct.parse_line(card["predicted_line_formation"],cars)
    if lines is None:
        raise ValueError("Line formation missing/invalid for actual G3 starters")
    if model.get("scope")!="G3 S級 only, no F1 training rows or parameters":
        raise ValueError("G3 model scope mismatch")
    names=tuple(model["selected_features"])
    if any(f not in FEATURE_NAMES for f in names):
        raise ValueError("Unknown learned fundamental feature")
    idx=[FEATURE_NAMES.index(f) for f in names]
    x=np.array([[numeric(e.get(f)) for f in FIELDS] for e in entries],dtype=float)
    foreign=(x[:,FIELDS.index("term")]==999).astype(float)
    x[foreign>0,FIELDS.index("term")]=np.nan
    obj={"id":card["race_id"],"cars":cars,"x":x}
    structure=struct.build_interaction(obj,lines)
    features=np.column_stack((x,structure,foreign))[:,idx]
    tr=model["transform"]
    med=np.asarray(tr["median"],dtype=float)
    mean=np.asarray(tr["mean"],dtype=float)
    std=np.asarray(tr["std"],dtype=float)
    coefficients=np.asarray(model["model_coefficients"],dtype=float)
    n=len(names)
    if len(med)!=n or len(mean)!=n or len(std)!=n or len(coefficients)!=2*n:
        raise ValueError("Frozen G3 model parameters were corrupted")
    bad=~np.isfinite(features)
    data=np.where(bad,med,features)
    design=np.column_stack(((data-mean)/std,bad.astype(float)))
    logits=design@coefficients
    probabilities=np.exp(logits-logsumexp(logits))
    if abs(probabilities.sum()-1)>1e-10 or not np.isfinite(probabilities).all():
        raise ValueError("G3 first probability normalization failed")
    return {
        "race_id":card["race_id"],
        "meeting_grade":"G3",
        "race_type":card["race_type"],
        "probability_model":"G3 independent 1st v1",
        "predicted_line_formation":card["predicted_line_formation"],
        "first_win":[{"car_no":car,"probability":float(p)} for car,p in zip(cars,probabilities)],
        "source_url":card["source_url"],
        "snapshot_captured_at_utc":card["snapshot_captured_at_utc"],
        "source_model":"G3 frozen V1 from model G3 2022-2025 parameters",
        "used_market_odds":False,
        "used_future_result_features":False
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    card=json.loads(args.input.read_text(encoding="utf-8"))
    mod=json.loads(args.model.read_text(encoding="utf-8"))
    prediction=independent_first(card,mod)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(prediction,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_FROZEN_FIRST_READY",prediction["race_id"],len(prediction["first_win"]))

if __name__=="__main__":main()

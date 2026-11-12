#!/usr/bin/env python3
"""Calibrate archived independent fundamental model on JUL/AUG 2026, apply unchanged
to SEP 2026 *without reading Sept outcome files*.

We keep ranking intact (T>=1 softmax temperature; high tail penalty lambda<=1).
High-tail threshold 0.4 is frozen from the earlier Benter calibration audit.
Use July-August for parameter/method selection only. September is true OOT holdout
relative to calibration, not a live prospective pre-race capture.
"""
from __future__ import annotations
import csv
import json
import math
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/benter_structural_discovery_v2"))
import discover_v2 as v2

HERE=Path(__file__).parent
H2=ROOT/"research/benter_score_challenge_v3"
MODEL=ROOT/"research/benter_structural_discovery_v2/output/models.json"
HOLD=HERE/"sep_holdout"
THRESHOLD=.4

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def normalize(weights):
    weights=np.asarray(weights,dtype=float)
    return np.exp(weights-logsumexp(weights))

def calibrate(p,method,theta):
    p=np.asarray(p,dtype=float)
    assert abs(p.sum()-1)<1e-7
    if method=="identity":
        return p
    if method=="temperature":
        assert 1 <= theta <= 3
        return normalize(np.log(np.maximum(p,1e-15))/theta)
    if method=="high_hinge":
        assert 0 <= theta <= 1
        return normalize(np.log(np.maximum(p,1e-15))-theta*np.maximum(0,p-THRESHOLD))
    raise ValueError(method)

def calibration_records():
    actual={r["race_id"]:r["winner"] for r in read(H2/"individual_results.csv")}
    groups=defaultdict(list)
    for r in read(H2/"frozen_2026h2_probabilities.csv"):
        if r["race_id"] in actual:
            groups[r["race_id"]].append(r)
    records=[]
    for rid,rs in groups.items():
        rs.sort(key=lambda r:int(r["car_no"]))
        p=np.array([float(r["v2_line_model"]) for r in rs])
        winner=actual[rid]
        cars=[r["car_no"] for r in rs]
        assert winner in cars
        assert abs(p.sum()-1)<1e-7
        records.append((p,cars.index(winner),rid))
    assert len(records)==872,(len(records))
    return records

def nll(records,method,theta):
    return float(np.mean([-math.log(max(calibrate(p,method,theta)[w],1e-300))
                          for p,w,_ in records]))

def choose_model(records):
    candidates=[{"method":"identity","theta":0.0,"calibration_logloss":nll(records,"identity",0)}]
    for method,limits in [("temperature",(1,3)),("high_hinge",(0,1))]:
        fit=minimize_scalar(lambda t:nll(records,method,t),bounds=limits,method="bounded",
                            options={"xatol":1e-5})
        if not fit.success: raise RuntimeError(f"Calibration not converged: {method}")
        candidates.append({"method":method,"theta":float(fit.x),
                           "calibration_logloss":nll(records,method,float(fit.x))})
    # One scalar learned per method, choose by entire July/August calibration log loss.
    select=min(candidates,key=lambda x:x["calibration_logloss"])
    return select,candidates

def v2_probabilities(r, model):
    names=model["fields"]
    ix=[v2.BASE_FEATURES.index(k) if k in v2.BASE_FEATURES else
        len(v2.BASE_FEATURES)+v2.STRUCT_FEATURES.index(k) for k in names]
    mat=r[:,ix]
    tr=model["scaler"]
    med=np.array(tr["median"],dtype=float)
    mean=np.array(tr["mean"],dtype=float)
    sd=np.array(tr["std"],dtype=float)
    empty=~np.isfinite(mat)
    x=np.column_stack(((np.where(empty,med,mat)-mean)/sd,empty.astype(float)))
    beta=np.array(model["coefficients"],dtype=float)
    assert len(beta)==x.shape[1],(len(beta),x.shape)
    return normalize(x@beta)

def forecast_september(selection):
    models=json.loads(MODEL.read_text(encoding="utf-8"))
    assert "full_plus_line" in models
    model=models["full_plus_line"]
    races=read(HOLD/"races.csv")
    es=defaultdict(list)
    for r in read(HOLD/"entries.csv"):
        es[r["race_id"]].append(r)
    predicted=[]
    rejected=[]
    detail=[]
    for race in sorted(races,key=lambda z:(z["race_date"],z["track"],int(z["race_no"]))):
        rid=race["race_id"]
        try:
            if not ("2026-09-01" <=race["race_date"]<="2026-09-30"):
                raise ValueError("outside locked September window")
            entries=sorted(es.get(rid,[]),key=lambda z:int(z["car_no"]))
            cars=[int(e["car_no"]) for e in entries]
            if not 5<=len(cars)<=9 or len(set(cars))!=len(cars):raise ValueError("invalid cars")
            lines=v2.parse_line(race.get("predicted_line_formation",""),cars)
            if lines is None:raise ValueError("missing or invalid pre-race lineup")
            raw=np.array([[v2.v1.as_number(e.get(k)) for k in v2.BASE_FEATURES] for e in entries],
                         dtype=float)
            sx=v2.build_interaction({"cars":cars,"x":raw},lines)
            x=np.column_stack((raw,sx))
            orig=v2_probabilities(x,model)
            post=calibrate(orig,selection["method"],selection["theta"])
            if abs(orig.sum()-1)>1e-9 or abs(post.sum()-1)>1e-9:
                raise AssertionError("probabilities not normalized")
            # Calibration is strictly rank-preserving.
            if int(np.argmax(orig)) != int(np.argmax(post)):
                raise AssertionError("unexpected rank flip")
            for i,car in enumerate(cars):
                predicted.append({"race_date":race["race_date"],"race_id":rid,
                                  "track":race["track"],"race_no":race["race_no"],
                                  "car_no":car,"original_probability":format(orig[i],".15g"),
                                  "calibrated_probability":format(post[i],".15g")})
            detail.append({"race_id":rid,"race_date":race["race_date"],"n":len(cars)})
        except (ValueError,IndexError,KeyError) as e:
            rejected.append({"race_id":rid,"error":str(e)})
    assert len(detail)>275,(len(detail),len(races))
    with (HERE/"sep_predictions_frozen.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=predicted[0].keys())
        w.writeheader();w.writerows(predicted)
    return detail,rejected,len(races)

def main():
    records=calibration_records()
    sel,candidates=choose_model(records)
    details,rejected,n=forecast_september(sel)
    manifest={
        "model":"Frozen fundamental V2 (independent of all market odds)",
        "base_model":"research/benter_structural_discovery_v2/output/models.json full_plus_line",
        "calibrator_selection":sel,
        "calibrator_candidates":candidates,
        "fixed_high_probability_threshold":THRESHOLD,
        "source_races":"2026-07-02 to 2026-08-30",
        "source_race_count":len(records),
        "frozen_independent_eval_window":"2026-09-01 to 2026-09-30",
        "sept_races_archived":n,
        "sept_races_predicted":len(details),
        "excluded":rejected,
        "forecast_output":"research/benter_calibration_v4/sep_predictions_frozen.csv",
        "model_coefficients_changed":False,
        "target_period_results_opened_before_forecasting":False,
        "historical_racecards":"Race-specific historical entry cards, not current rider profiles",
        "selection_warning":"Calibration method and one scalar selected only on Jul-Aug; Sept outcomes not used."
    }
    (HERE/"calibrator_frozen.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",
                                                encoding="utf-8")
    print("FROZEN",json.dumps({k:v for k,v in manifest.items() if k in (
          "calibrator_selection","calibrator_candidates","sept_races_archived",
          "sept_races_predicted","source_race_count")},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

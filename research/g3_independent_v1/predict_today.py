#!/usr/bin/env python3
"""Apply G3-only model to timestamped pre-start Oct18 Keio-kaku racecards.

Do not read actual results, odds, payout or training validation labels.
"""
from __future__ import annotations
import csv,datetime,hashlib,json,math,sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/g3_independent_v1"))
from train_g3_first import FIELDS,STRUCT,FEATURE_NAMES,numeric,struct
HERE=ROOT/"research/g3_independent_v1/today_20261018"
MODEL=ROOT/"research/g3_independent_v1/output/g3_first_win_report.json"

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)

def main():
    today=json.loads((HERE/"snapshot_audit.json").read_text(encoding="utf-8"))
    if today["date"]!="2026-10-18":
        raise ValueError("Wrong G3 experiment date")
    model=json.loads(MODEL.read_text(encoding="utf-8"))
    if model["scope"]!="G3 S級 only, no F1 training rows or parameters":
        raise ValueError("Wrong model scope")
    features=model["selected_features"]
    ids=[FEATURE_NAMES.index(k) for k in features]
    median=np.array(model["transform"]["median"])
    mean=np.array(model["transform"]["mean"])
    std=np.array(model["transform"]["std"])
    coef=np.array(model["model_coefficients"])
    assert len(coef)==2*len(features)
    entries=defaultdict(list)
    for e in read(HERE/"entries.csv"):
        entries[e["race_id"]].append(e)
    outputs=[]
    audit={"date":"2026-10-18","venue":"京王閣","model":"G3-only frozen 2022-25 fundamental independent win",
           "model_select_2025":"best validation logloss","train_includes_today":False,
           "races":[],"missing":[],"source_cards_sha256":hashlib.sha256(
                (HERE/"races.csv").read_bytes()).hexdigest()}
    tokyo=ZoneInfo("Asia/Tokyo")
    for card in read(HERE/"races.csv"):
        rid=card["race_id"]
        selected=entries.get(rid,[])
        if card.get("meeting_grade")!="G3" or not selected:
            audit["missing"].append({"race":rid,"problem":"bad grade or no entries"});continue
        captured=datetime.datetime.fromisoformat(card["snapshot_captured_at_utc"]).astimezone(tokyo)
        start=card.get("start_time","")
        if start:
            hh,mm=map(int,start.split(":"))
            deadline=datetime.datetime(2026,10,18,hh,mm,tzinfo=tokyo)
            if captured>=deadline:raise RuntimeError("Not pre-race snapshot "+rid)
        selected.sort(key=lambda e:int(e["car_no"]))
        cars=[int(e["car_no"]) for e in selected]
        if len(cars) not in range(5,10) or len(set(cars))!=len(cars):
            audit["missing"].append({"race":rid,"problem":"bad car roster"});continue
        lines=struct.parse_line(card.get("predicted_line_formation",""),cars)
        if lines is None:
            audit["missing"].append({"race":rid,"problem":"no confirmed published line"});continue
        x=np.array([[numeric(e.get(k)) for k in FIELDS] for e in selected],dtype=float)
        foreign=(x[:,FIELDS.index("term")]==999).astype(float)
        x[foreign>0,FIELDS.index("term")]=np.nan
        obj={"cars":cars,"x":x}
        lx=struct.build_interaction(obj,lines)
        arr=np.concatenate((x,lx,foreign[:,None]),axis=1)[:,ids]
        missing=~np.isfinite(arr)
        filled=np.where(missing,median,arr)
        xx=np.column_stack(((filled-mean)/std,missing.astype(float)))
        z=xx@coef
        probs=np.exp(z-logsumexp(z))
        if not np.isfinite(probs).all() or abs(probs.sum()-1)>1e-10:
            raise RuntimeError("Invalid predicted G3 probability")
        for car,p in zip(cars,probs):
            outputs.append({"race_id":rid,"race_no":card["race_no"],"race_date":"2026-10-18",
                            "track":card.get("track"),"car_no":car,
                            "independent_first_probability":format(float(p),".15g"),
                            "captured_at_utc":card["snapshot_captured_at_utc"],
                            "start_time":start})
        audit["races"].append({"race_id":rid,"race_no":card["race_no"],
                               "line":card.get("predicted_line_formation"),
                               "cars":cars,"capture":card["snapshot_captured_at_utc"],
                               "top3":sorted([{"car_no":i,"p":float(v)} for i,v in zip(cars,probs)],
                                             key=lambda x:-x["p"])[:3]})
    output=HERE/"g3_first_independent_probabilities.csv"
    fields=("race_id","race_no","race_date","track","car_no","independent_first_probability","captured_at_utc","start_time")
    with output.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(outputs)
    audit["complete_races"]=len(audit["races"])
    audit["output_sha256"]=hashlib.sha256(output.read_bytes()).hexdigest()
    (HERE/"forecast_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_TODAY_INDEPENDENT_FORECAST",json.dumps(audit,ensure_ascii=False),flush=True)
    if not outputs:raise RuntimeError("No valid G3 race snapshots to forecast")
if __name__=="__main__":main()

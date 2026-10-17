#!/usr/bin/env python3
"""V3: holdout-frozen, odds-independent nonlinear keirin 1st-place model.

Use only 2023–2024 for initial fits and 2025 for candidate selection.
Refit winner-probability ranking on 2023–2025, predict 2026-07-02..08-30
WITHOUT opening those race results. Scores are computed in a separate job
stage after a commit of the frozen predictions.
Retrospectively archived features are not verified as pre-race snapshots.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from collections import Counter,defaultdict
from pathlib import Path
import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/benter_structural_discovery_v2"))
import discover_v2 as v2
v1=v2.v1
OUT=ROOT/"research/benter_score_challenge_v3"
H2=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830"
OLD=ROOT/"research/benter_structural_discovery_v2/output/models.json"
RAW=list(v2.BASE_FEATURES)
LINE=list(v2.STRUCT_FEATURES)
COMPETE=["score","win_rate","top2_rate","top3_rate","b_count","nige_count","makuri_count","mark_count","first_count","outside_count"]
KINDS=("score_tree","no_score_tree","joint_tree")
CONFIGS=[
    {"max_leaf_nodes":15,"max_iter":140,"learning_rate":0.055,"min_samples_leaf":80,"l2_regularization":20.0},
    {"max_leaf_nodes":31,"max_iter":165,"learning_rate":0.045,"min_samples_leaf":90,"l2_regularization":20.0},
    {"max_leaf_nodes":15,"max_iter":220,"learning_rate":0.038,"min_samples_leaf":55,"l2_regularization":12.0},
]
TEMPS=(0.8,1.0,1.2,1.45)
FIELDS=[*RAW,*LINE,*[f"{name}_{suffix}" for name in COMPETE for suffix in ("z","gapmax","rank")],"field_size"]
NOSCORE_FORBIDDEN={"score","teammate_score_support","line_score_imbalance"}|{f"score_{suffix}" for suffix in ("z","gapmax","rank")}
USE={
    "score_tree":("score", "score_z","score_gapmax","score_rank","field_size"),
    "no_score_tree":tuple(f for f in FIELDS if f not in NOSCORE_FORBIDDEN),
    "joint_tree":tuple(FIELDS)
}

def historical_directories():
    return [ROOT/f"data/{y}/s_class_f1_all_parts/{y}_q{q}"
            for y in (2023,2024,2025) for q in (1,2,3,4)
            if (ROOT/f"data/{y}/s_class_f1_all_parts/{y}_q{q}/entries.csv").exists()
            and (ROOT/f"data/{y}/s_class_f1_all_parts/{y}_q{q}/results.csv").exists()]

def read_csv(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def pack_features(cars,raw,lines):
    n=len(cars)
    line=v2.build_interaction({"cars":cars,"x":raw},lines)
    extra=[]
    for feat in COMPETE:
        vals=raw[:,RAW.index(feat)]
        valid=np.isfinite(vals)
        nonnull=vals[valid]
        if len(nonnull)==0:
            extra.extend([np.full(n,np.nan),np.full(n,np.nan),np.full(n,np.nan)])
            continue
        med=float(np.median(nonnull))
        a=np.where(valid,vals,med)
        sd=max(float(a.std()),0.01)
        z=(a-float(a.mean()))/sd
        gapmax=(np.max(a)-a)/sd
        rank=np.array([np.mean(a[valid]>x) if valid[i] else np.nan for i,x in enumerate(a)],dtype=float)
        extra.extend([z,gapmax,rank])
    return np.column_stack((raw,line,*extra,np.full(n,n,dtype=float)))

def rows_for_race(r,lines):
    return pack_features(r["cars"],r["x"],lines)

def collect_past():
    # Avoid reading 2026 (including 2026 H1) outcomes even for validation.
    v1.paths=historical_directories
    races,audit,dirs=v1.read_data()
    line_maps=v2.read_line_maps()
    accepted=[]
    bad=Counter()
    for race in races:
        lines=v2.parse_line(line_maps.get(race["id"],""),race["cars"])
        if not lines:
            bad["incomplete_lineup"]+=1
            continue
        r={**race,"features":rows_for_race(race,lines)}
        accepted.append(r)
    train=[r for r in accepted if r["date"][:4] in ("2023","2024")]
    val=[r for r in accepted if r["date"][:4]=="2025"]
    assert len(train)>7000 and len(val)>4000
    return train,val,{"data_audit":audit,"skipped":dict(bad),"directories":dirs}

def flatten(races,selected):
    indexes=[FIELDS.index(s) for s in selected]
    a=np.concatenate([r["features"][:,indexes] for r in races])
    # Unseen forecast races have no winner label: never inspect outcomes to predict.
    labels=np.concatenate([np.arange(len(r["cars"]))==r["winner"] if "winner" in r else np.zeros(len(r["cars"]),dtype=bool) for r in races]).astype(int)
    boundaries=np.cumsum([0]+[len(r["cars"]) for r in races])
    return a,labels,boundaries

def predictions(races,model,selected,temp):
    x,lab,bound=flatten(races,selected)
    margins=model.decision_function(x)
    probs=[]
    for k in range(len(races)):
        z=margins[bound[k]:bound[k+1]]/temp
        probs.append(np.exp(z-logsumexp(z)))
    return probs

def scores(races,probs):
    n=len(races)
    true=np.array([p[r["winner"]] for r,p in zip(races,probs)])
    hits=sum(np.argmax(p)==r["winner"] for r,p in zip(races,probs))
    return {"races":n,"top1_hits":int(hits),"top1_accuracy":float(hits/n),
            "log_loss":float(-np.log(np.maximum(true,1e-300)).mean()),
            "brier":float(np.mean([np.sum((p-(np.arange(len(p))==r["winner"]))**2)
                   for r,p in zip(races,probs)]))}

def fit(races,selected,cfg):
    x,y,_=flatten(races,selected)
    model=HistGradientBoostingClassifier(**cfg,early_stopping=False,random_state=20261018,
                                         categorical_features=None)
    model.fit(x,y)
    return model

def select(train,val):
    report={}
    choices={}
    for kind in KINDS:
        fields=USE[kind]
        candidates=[]
        for index,cfg in enumerate(CONFIGS):
            model=fit(train,fields,cfg)
            for temp in TEMPS:
                pr=predictions(val,model,fields,temp)
                score=scores(val,pr)
                candidates.append({"index":index,"temperature":temp,**score})
        # First require high top1 on validation; choose best logloss if tied.
        best=min(candidates,key=lambda x:(-x["top1_hits"],x["log_loss"]))
        report[kind]={"selected":best,"candidates":candidates,
                      "feature_count":len(fields)}
        choices[kind]={"configuration":CONFIGS[best["index"]], "temperature":best["temperature"],
                       "fields":fields}
        print("VAL",kind,json.dumps(best),flush=True)
    return report,choices

def collect_h2():
    meta=[r for r in read_csv(H2/"races.csv") if "2026-07-02"<=r["race_date"]<="2026-08-30"
          and "ガールズ" not in r["race_type"]]
    entries=defaultdict(list)
    for e in read_csv(H2/"entries.csv"):
        if "2026-07-02"<=e["race_date"]<="2026-08-30":
            entries[e["race_id"]].append(e)
    races=[]
    rejected=[]
    for m in sorted(meta,key=lambda r:(r["race_date"],r["track"],int(r["race_no"]))):
        rid=m["race_id"]
        es=entries.get(rid,[])
        try:
            es=sorted(es,key=lambda x:int(x["car_no"]))
            cars=[int(e["car_no"]) for e in es]
            if not 5<=len(cars)<=9 or len(set(cars))!=len(cars):
                raise ValueError("Invalid entrants")
            raw=np.array([[v1.as_number(e.get(k)) for k in RAW] for e in es],dtype=float)
            line=v2.parse_line(m.get("predicted_line_formation",""),cars)
            if not line:
                raise ValueError("Invalid or missing predicted lineup")
            if not np.isfinite(raw).any():
                raise ValueError("All rider features missing")
            races.append({"id":rid,"date":m["race_date"],"cars":cars,"track":m["track"],
                          "race_no":m["race_no"],"race_type":m["race_type"],
                          "raw_score":raw[:,RAW.index("score")],
                          "features":pack_features(cars,raw,line),
                          "feature_captured_utc":m.get("captured_at_utc","")})
        except (ValueError,KeyError) as e:
            rejected.append({"race_id":rid,"reason":str(e)})
    return races,rejected,len(meta)

def v2_model_predict(r,models,which):
    z=models[which]
    fields=z["fields"]
    x=r["features"][:,[FIELDS.index(s) for s in fields]]
    tr=z["scaler"]
    mean=np.array(tr["mean"],dtype=float)
    std=np.array(tr["std"],dtype=float)
    med=np.array(tr["median"],dtype=float)
    empty=~np.isfinite(x)
    xx=np.column_stack(((np.where(empty,med,x)-mean)/std,empty.astype(float)))
    beta=np.array(z["coefficients"],dtype=float)
    logits=xx@beta
    return np.exp(logits-logsumexp(logits))

def main():
    train,val,audit=collect_past()
    validate,choices=select(train,val)
    full=train+val
    trees={}
    for kind in KINDS:
        chosen=choices[kind]
        trees[kind]=fit(full,chosen["fields"],chosen["configuration"])
    holdout,rejected,total=collect_h2()
    assert len(holdout)>700,f"Holdout unexpectedly small {len(holdout)}"
    old_models=json.loads(OLD.read_text(encoding="utf-8"))
    trees_pred={kind:predictions(holdout,trees[kind],choices[kind]["fields"],choices[kind]["temperature"]) for kind in KINDS}
    table=[]
    for k,r in enumerate(holdout):
        old_score=v2_model_predict(r,old_models,"score_only")
        old_v2=v2_model_predict(r,old_models,"full_plus_line")
        for i,car in enumerate(r["cars"]):
            table.append({
                "race_id":r["id"],"race_date":r["date"],"track":r["track"],
                "race_no":r["race_no"],"race_type":r["race_type"],"car_no":car,
                "race_car_score":r["raw_score"][i],
                "score_only_logit":old_score[i],
                "v2_line_model":old_v2[i],
                **{kind:trees_pred[kind][k][i] for kind in KINDS}
            })
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"frozen_2026h2_probabilities.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=table[0].keys())
        w.writeheader()
        w.writerows(table)
    joblib.dump({"models":trees,"selection":choices,"features":FIELDS},
                OUT/"frozen_model.joblib",compress=3)
    manifest={
        "research_design":"Train 2023-24, select on 2025 validation, refit on 2023-25, score H2 without retraining",
        "model_must_beat_score_only":"top1 accuracy and log_loss both tested; no post-hoc threshold tuning",
        "training_cutoff":"2025-12-31","holdout_start":"2026-07-02","holdout_end":"2026-08-30",
        "historical_training_races":len(train),"validation_races":len(val),
        "holdout_archive_races":total,"frozen_holdout_races":len(holdout),
        "rejected":rejected,"validation_models":validate,"chosen_configs":choices,
        "audit":audit,
        "models_compared":["score_argmax","score_only_logit","v2_line_model",*KINDS],
        "unverified_as_of":"Entries and line formation for H2 were archived after racing (Sept 17, 2026). Neither historically faithful nor prospectively captured.",
        "race_result_file_opened_by_this_script":False,
    }
    (OUT/"frozen_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FROZEN_HOLDOUT",json.dumps({k:v for k,v in manifest.items() if k in ("historical_training_races","validation_races","frozen_holdout_races","holdout_archive_races")},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

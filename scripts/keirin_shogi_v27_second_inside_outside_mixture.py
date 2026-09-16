#!/usr/bin/env python3
from __future__ import annotations
import json, math, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

spec=importlib.util.spec_from_file_location("v25","scripts/keirin_shogi_v25_top2_pair_second.py")
v25=importlib.util.module_from_spec(spec); spec.loader.exec_module(v25)

OUT=Path("results/keirin_shogi/v27_second_inside_outside_mixture")
OUT.mkdir(parents=True,exist_ok=True)

# Rebuild second place around the observed structure:
# P(2nd=j) = P(inside first-candidate set)*P(j|inside)
#          + P(outside first-candidate set)*P(j|outside)
# No actual-first conditioning anywhere.

RIDER_FEATURES=[
 "z_score","z_win_rate","z_top2_rate","z_top3_rate","z_b_count",
 "z_nige_count","z_makuri_count","z_sashi_count","z_mark_count",
 "z_first_count","z_second_count","z_third_count","z_outside_count",
 "line_pos","line_size","line_mean_score_z","line_max_score_z",
 "line_b_sum_z","line_attack_sum_z"
]

def prepared():
    vrows=v25.load_v19_details();v25.fit_v21_selector(vrows);v25.mark_v21(vrows)
    _,eb,rb=v25.raw_maps()
    return vrows,eb,rb

def datasets(vrows,eb,rb,start,end):
    return v25.make_dataset(vrows,eb,rb,start,end,False)

def gate_features(base,vr):
    fc=list(map(v25.ino,vr["candidates"]))
    if len(fc)<2:
        fc=fc+[fc[0]]
    a,b=fc[:2]
    pa=float(vr["p1_map"].get(a,0));pb=float(vr["p1_map"].get(b,0))
    ca,cb=base[a]["x"],base[b]["x"]
    outsiders=[n for n in base if n not in set(fc)]
    vals_score=[base[n]["x"]["z_score"] for n in outsiders]
    vals_top2=[base[n]["x"]["z_top2_rate"] for n in outsiders]
    vals_mak=[base[n]["x"]["z_makuri_count"] for n in outsiders]
    vals_sashi=[base[n]["x"]["z_sashi_count"] for n in outsiders]
    same=float(base[a]["line_id"]==base[b]["line_id"])
    same_out=sum(1 for n in outsiders if base[n]["line_id"] in {base[a]["line_id"],base[b]["line_id"]})
    # global v21 uncertainty from prior OOS prediction
    probs=[float(x["prob"]) for x in vr["ranking"]]
    ent=-sum(p*math.log(max(p,1e-12)) for p in probs)
    return np.asarray([
      max(pa,pb),min(pa,pb),abs(pa-pb),pa+pb,ent,
      same,
      float(base[a]["line_position"]),float(base[b]["line_position"]),
      max(ca["z_score"],cb["z_score"]),min(ca["z_score"],cb["z_score"]),abs(ca["z_score"]-cb["z_score"]),
      max(ca["z_top2_rate"],cb["z_top2_rate"]),min(ca["z_top2_rate"],cb["z_top2_rate"]),
      max(vals_score),max(vals_top2),max(vals_mak),max(vals_sashi),
      max(vals_score)-min(ca["z_score"],cb["z_score"]),
      max(vals_top2)-min(ca["z_top2_rate"],cb["z_top2_rate"]),
      same_out/5.0,
    ],float)

def relation_summary(base,cand,fc):
    vals=[]
    for f in fc[:2]:
        same=float(base[cand]["line_id"]==base[f]["line_id"])
        posdelta=float(base[cand]["line_position"]-base[f]["line_position"]) if same else 0.0
        vals.append({
          "same":same,"posdelta":posdelta,
          "score_diff":base[cand]["x"]["z_score"]-base[f]["x"]["z_score"],
          "b_diff":base[cand]["x"]["z_b_count"]-base[f]["x"]["z_b_count"],
          "mak_diff":base[cand]["x"]["z_makuri_count"]-base[f]["x"]["z_makuri_count"],
          "sashi_diff":base[cand]["x"]["z_sashi_count"]-base[f]["x"]["z_sashi_count"],
          "mark_diff":base[cand]["x"]["z_mark_count"]-base[f]["x"]["z_mark_count"],
        })
    return vals

def candidate_features(base,cand,fc,vr,mode):
    x=base[cand]["x"]; rel=relation_summary(base,cand,fc)
    p=float(vr["p1_map"].get(cand,0))
    fprobs=[float(vr["p1_map"].get(f,0)) for f in fc[:2]]
    arr=[float(x[f]) for f in RIDER_FEATURES]
    arr += [
      p,p-max(fprobs),p-min(fprobs),
      max(r["same"] for r in rel),sum(r["same"] for r in rel),
      min(r["posdelta"] for r in rel),max(r["posdelta"] for r in rel),
      max(r["score_diff"] for r in rel),min(r["score_diff"] for r in rel),
      max(r["b_diff"] for r in rel),max(r["mak_diff"] for r in rel),
      max(r["sashi_diff"] for r in rel),max(r["mark_diff"] for r in rel),
      1.0 if mode=="inside" else 0.0,
    ]
    if mode=="inside":
        other=next((f for f in fc if f!=cand),cand)
        arr += [
          float(vr["p1_map"].get(cand,0)-vr["p1_map"].get(other,0)),
          float(x["z_score"]-base[other]["x"]["z_score"]),
          float(x["z_top2_rate"]-base[other]["x"]["z_top2_rate"]),
          float(x["z_sashi_count"]-base[other]["x"]["z_sashi_count"]),
          float(base[cand]["line_id"]==base[other]["line_id"])
        ]
    else:
        arr += [0,0,0,0,0]
    return np.asarray(arr,float)

def fit_models(ds,participant_only=True,gateC=.5,depth=2):
    used=[r for r in ds if (r[4].get("v21_participate") or not participant_only)]
    GX=[];Gy=[]; IX=[];Iy=[];IW=[]; OX=[];Oy=[];OW=[]
    gate_n=inside_n=outside_n=0
    for rid,dt,base,order,vr in used:
        f,s=order[:2];fc=list(map(v25.ino,vr["candidates"]))
        if len(fc)==0:continue
        inside=s in fc
        GX.append(gate_features(base,vr));Gy.append(1 if inside else 0);gate_n+=1
        group=fc if inside else [n for n in base if n not in set(fc)]
        mode="inside" if inside else "outside"
        for n in group:
            feat=candidate_features(base,n,fc,vr,mode)
            if inside:
                IX.append(feat);Iy.append(1 if n==s else 0);IW.append(1/len(group))
            else:
                OX.append(feat);Oy.append(1 if n==s else 0);OW.append(1/len(group))
        if inside:inside_n+=1
        else:outside_n+=1

    gate=LogisticRegression(C=gateC,class_weight="balanced",max_iter=800,random_state=27)
    gate.fit(np.asarray(GX,float),np.asarray(Gy,int))
    def hgb(X,y,w,seed):
        m=HistGradientBoostingClassifier(
          learning_rate=.045,max_iter=180,max_depth=depth,min_samples_leaf=18,
          l2_regularization=2.0,random_state=seed)
        m.fit(np.asarray(X,float),np.asarray(y,int),sample_weight=np.asarray(w,float))
        return m
    inside_m=hgb(IX,Iy,IW,271)
    outside_m=hgb(OX,Oy,OW,272)
    return {"gate":gate,"inside":inside_m,"outside":outside_m,
            "train_races":gate_n,"inside_races":inside_n,"outside_races":outside_n}

def normalized_group(model,base,group,fc,vr,mode):
    if not group:return {}
    X=np.vstack([candidate_features(base,n,fc,vr,mode) for n in group])
    raw=np.clip(model.predict_proba(X)[:,1],1e-9,None);raw=raw/raw.sum()
    return {n:float(p) for n,p in zip(group,raw)}

def predict(base,vr,mods):
    fc=list(map(v25.ino,vr["candidates"]))
    inside=[n for n in fc if n in base]
    outside=[n for n in base if n not in set(fc)]
    q=float(mods["gate"].predict_proba(gate_features(base,vr).reshape(1,-1))[0,1])
    pi=normalized_group(mods["inside"],base,inside,fc,vr,"inside")
    po=normalized_group(mods["outside"],base,outside,fc,vr,"outside")
    p={}
    for n,v in pi.items():p[n]=q*v
    for n,v in po.items():p[n]=(1-q)*v
    z=sum(p.values()) or 1
    rank=sorted(((n,v/z) for n,v in p.items()),key=lambda kv:(-kv[1],kv[0]))
    return rank,q

def choose(rank,t):
    return v25.choose_by_cum(rank,t,3)

def evaluate(ds,mods,t):
    logs=[]
    for rid,dt,base,order,vr in ds:
        if not vr.get("v21_participate"):continue
        rank,q=predict(base,vr,mods);c=choose(rank,t)
        f,s=order[:2];fc=list(map(v25.ino,vr["candidates"]))
        rel="same_line" if base[f]["line_id"]==base[s]["line_id"] else "different_line"
        logs.append({
          "race_id":rid,"race_date":dt,"actual_first":f,"actual_second":s,
          "first_candidates":fc,"first_hit":int(f in fc),
          "actual_inside":int(s in fc),"q_inside":q,
          "relation":rel,"second_candidates":c,"second_hit":int(s in c),
          "second_rank":next((i+1 for i,(n,p) in enumerate(rank) if n==s),None),
          "second_candidate_count":len(c)
        })
    return logs

def metrics(logs):
    n=len(logs);first=[x for x in logs if x["first_hit"]]
    outside=[x for x in first if not x["actual_inside"]]
    hard=[x for x in outside if x["relation"]=="different_line"]
    gate_acc=sum((x["q_inside"]>=.5)==bool(x["actual_inside"]) for x in logs)/n
    return {
      "races":n,
      "second_capture":sum(x["second_hit"] for x in logs)/n,
      "avg_second_candidates":mean(x["second_candidate_count"] for x in logs),
      "second_given_first_capture":sum(x["second_hit"] for x in first)/len(first),
      "outside_first_candidates_capture_given_first":sum(x["second_hit"] for x in outside)/len(outside),
      "hard_diff_line_outside_capture_given_first":sum(x["second_hit"] for x in hard)/len(hard),
      "hard_races":len(hard),
      "gate_accuracy":gate_acc,
      "actual_inside_rate":sum(x["actual_inside"] for x in logs)/n,
      "predicted_inside_mean":mean(x["q_inside"] for x in logs),
      "second_top1_rate":sum((x["second_rank"] or 99)<=1 for x in logs)/n,
      "second_top2_rate":sum((x["second_rank"] or 99)<=2 for x in logs)/n,
      "second_top3_rate":sum((x["second_rank"] or 99)<=3 for x in logs)/n,
    }

def main():
    vrows,eb,rb=prepared()
    train=datasets(vrows,eb,rb,v25.TRAIN_START,v25.TRAIN_END)
    cal=datasets(vrows,eb,rb,v25.CAL_START,v25.CAL_END)
    test=datasets(vrows,eb,rb,v25.TEST_START,v25.TEST_END)

    variants=[
      {"name":"participant_d2_c03","participant":True,"depth":2,"C":.3},
      {"name":"participant_d2_c07","participant":True,"depth":2,"C":.7},
      {"name":"participant_d3_c05","participant":True,"depth":3,"C":.5},
      {"name":"all_d2_c05","participant":False,"depth":2,"C":.5},
      {"name":"all_d3_c05","participant":False,"depth":3,"C":.5},
    ]
    grid=[];models={}
    for v in variants:
        mods=fit_models(train,v["participant"],v["C"],v["depth"]);models[v["name"]]=mods
        for t in (.38,.42,.46,.50,.54,.58,.62,.66,.70):
            logs=evaluate(cal,mods,t);m=metrics(logs)
            if m["avg_second_candidates"]>2.10:continue
            obj=(m["second_given_first_capture"]
                 +0.65*m["hard_diff_line_outside_capture_given_first"]
                 +0.30*m["outside_first_candidates_capture_given_first"]
                 +0.12*m["gate_accuracy"]
                 -0.08*max(0,m["avg_second_candidates"]-2.0))
            grid.append({"variant":v["name"],"threshold":t,"objective":obj,**m})
    grid.sort(key=lambda x:(x["objective"],x["second_given_first_capture"],x["hard_diff_line_outside_capture_given_first"],-x["avg_second_candidates"]),reverse=True)
    best=grid[0];mods=models[best["variant"]]
    logs=evaluate(test,mods,best["threshold"]);tm=metrics(logs)

    baseline={"second_capture":0.5342465753424658,"second_given_first_capture":0.5934065934065934,
              "avg_second_candidates":2.0332681017612524,"hard_diff_line_outside_capture_given_first":0.1259259259259259}
    summary={
      "algorithm":"keirin_shogi_v27_second_inside_outside_mixture",
      "concept":"First decide whether 2nd comes from inside the adopted v21 first-candidate set or from outside; then use separate candidate models and mix probabilities. Actual first is never used.",
      "first_core":"v21 frozen",
      "selected_calibration":best,
      "model_train_counts":{k:v for k,v in mods.items() if k.endswith("_races")},
      "test_metrics":tm,
      "v23_baseline":baseline,
      "delta_vs_v23":{k:tm[k]-baseline[k] for k in baseline},
      "calibration_top10":grid[:10],
      "leakage_guard":"Variant and threshold selected only on 2025-10-27..12-28 calibration. 2026 H1 untouched until final test. All first-candidate inputs are historical OOS v21/v19 outputs; odds unused."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"calibration.json").write_text(json.dumps(grid,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

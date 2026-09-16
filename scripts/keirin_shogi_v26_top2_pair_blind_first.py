#!/usr/bin/env python3
from __future__ import annotations
import json, importlib.util
from collections import defaultdict
from pathlib import Path
from statistics import mean
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

spec=importlib.util.spec_from_file_location("v25","scripts/keirin_shogi_v25_top2_pair_second.py")
v25=importlib.util.module_from_spec(spec); spec.loader.exec_module(v25)

OUT=Path("results/keirin_shogi/v26_top2_pair_blind_first")
OUT.mkdir(parents=True,exist_ok=True)

# Pair stage deliberately excludes v21 first-probability features.
# v21 is used only after the unordered top2 pair distribution exists, for orientation.

def pair_features_blind(base,a,b,_p1map):
    xa=base[a]["x"]; xb=base[b]["x"]
    feats=[
        "z_score","z_win_rate","z_top2_rate","z_top3_rate","z_b_count",
        "z_nige_count","z_makuri_count","z_sashi_count","z_mark_count",
        "z_first_count","z_second_count","z_third_count","z_outside_count",
        "line_pos","line_size","line_mean_score_z","line_max_score_z",
        "line_b_sum_z","line_attack_sum_z"
    ]
    x={}
    for f in feats:
        av=float(xa[f]);bv=float(xb[f])
        x[f"{f}_sum"]=av+bv
        x[f"{f}_absdiff"]=abs(av-bv)
        x[f"{f}_max"]=max(av,bv)
        x[f"{f}_min"]=min(av,bv)
    same=float(base[a]["line_id"]==base[b]["line_id"])
    posa=base[a]["line_position"]; posb=base[b]["line_position"]
    x.update({
        "same_line":same,
        "different_line":1.0-same,
        "line_pos_distance":abs(posa-posb) if same else 0.0,
        "adjacent_same_line":1.0 if same and abs(posa-posb)==1 else 0.0,
        "both_line_front":1.0 if posa==1 and posb==1 else 0.0,
        "one_front_one_second":1.0 if sorted([posa,posb])==[1,2] else 0.0,
        "one_front":1.0 if (posa==1) ^ (posb==1) else 0.0,
        "one_second":1.0 if (posa==2) ^ (posb==2) else 0.0,
        "both_top2_line_position":1.0 if posa in (1,2) and posb in (1,2) else 0.0,
        # pair complementarity without v21:
        "score_plus_top2_sum":float(xa["z_score"]+xb["z_score"]+xa["z_top2_rate"]+xb["z_top2_rate"]),
        "attack_plus_mark_sum":float(
            xa["z_nige_count"]+xa["z_makuri_count"]+xa["z_mark_count"]+
            xb["z_nige_count"]+xb["z_makuri_count"]+xb["z_mark_count"]
        ),
        "front_attack_max":max(
            float((xa["z_nige_count"]+xa["z_makuri_count"]) if posa==1 else -9),
            float((xb["z_nige_count"]+xb["z_makuri_count"]) if posb==1 else -9)
        ),
        "chaser_mark_max":max(
            float(xa["z_mark_count"] if posa>=2 else -9),
            float(xb["z_mark_count"] if posb>=2 else -9)
        ),
    })
    return x

def fit_pair_model(ds,params):
    fn=None;X=[];y=[];sw=[]
    for rid,dt,base,order,vr in ds:
        truepair=frozenset(order[:2])
        nos=sorted(base)
        pairs=[(nos[i],nos[j]) for i in range(len(nos)) for j in range(i+1,len(nos))]
        xs=[pair_features_blind(base,a,b,vr["p1_map"]) for a,b in pairs]
        if fn is None:fn=sorted(xs[0])
        for pair,x in zip(pairs,xs):
            X.append([x[f] for f in fn]);y.append(1 if frozenset(pair)==truepair else 0);sw.append(1/len(pairs))
    clf=HistGradientBoostingClassifier(
        learning_rate=params["lr"],max_iter=params["iters"],max_depth=params["depth"],
        min_samples_leaf=params["leaf"],l2_regularization=params["l2"],random_state=26
    )
    clf.fit(np.asarray(X,float),np.asarray(y,int),sample_weight=np.asarray(sw,float))
    return {"model":clf,"features":fn}

def fit_orientation(ds,C):
    return v25.fit_orientation(ds,False,C)

def predict(base,vr,pm,om):
    nos=sorted(base)
    pairs=[(nos[i],nos[j]) for i in range(len(nos)) for j in range(i+1,len(nos))]
    X=[]
    for a,b in pairs:
        x=pair_features_blind(base,a,b,vr["p1_map"])
        X.append([x[f] for f in pm["features"]])
    raw=np.clip(pm["model"].predict_proba(np.asarray(X,float))[:,1],1e-9,None)
    pairp=raw/raw.sum()

    p2=defaultdict(float);prows=[]
    for (a0,b0),pp in zip(pairs,pairp):
        if vr["p1_map"].get(a0,0)>=vr["p1_map"].get(b0,0):a,b=a0,b0
        else:a,b=b0,a0
        ox=v25.orient_features(base,a,b,vr["p1_map"])
        O=np.asarray([[ox[f] for f in om["features"]]],float)
        pa=float(om["model"].predict_proba(O)[0,1])
        p2[b]+=float(pp)*pa
        p2[a]+=float(pp)*(1-pa)
        prows.append((a0,b0,float(pp)))
    z=sum(p2.values()) or 1
    ranking=sorted(((n,p/z) for n,p in p2.items()),key=lambda kv:(-kv[1],kv[0]))
    prows.sort(key=lambda x:-x[2])
    return ranking,prows

def choose(rank,t):
    return v25.choose_by_cum(rank,t,3)

def eval_ds(ds,pm,om,t):
    logs=[]
    for rid,dt,base,order,vr in ds:
        if not vr.get("v21_participate"):continue
        rank,pairs=predict(base,vr,pm,om)
        c=choose(rank,t)
        f,s=order[:2];fc=list(map(v25.ino,vr["candidates"]))
        relation="same_line" if base[f]["line_id"]==base[s]["line_id"] else "different_line"
        truepair=frozenset((f,s))
        prank=next((i+1 for i,(a,b,p) in enumerate(pairs) if frozenset((a,b))==truepair),None)
        logs.append({
            "race_id":rid,"race_date":dt,"actual_first":f,"actual_second":s,
            "first_candidates":fc,"first_hit":int(f in fc),
            "second_in_first_candidates":int(s in fc),"relation":relation,
            "second_candidates":c,"second_hit":int(s in c),
            "second_rank":next((i+1 for i,(n,p) in enumerate(rank) if n==s),None),
            "pair_rank":prank,"second_candidate_count":len(c)
        })
    return logs

def metrics(logs):
    n=len(logs)
    first=[x for x in logs if x["first_hit"]]
    outside=[x for x in first if not x["second_in_first_candidates"]]
    hard=[x for x in outside if x["relation"]=="different_line"]
    return {
        "races":n,
        "second_capture":sum(x["second_hit"] for x in logs)/n,
        "avg_second_candidates":mean(x["second_candidate_count"] for x in logs),
        "second_given_first_capture":sum(x["second_hit"] for x in first)/len(first),
        "outside_first_candidates_capture_given_first":sum(x["second_hit"] for x in outside)/len(outside),
        "hard_diff_line_outside_capture_given_first":sum(x["second_hit"] for x in hard)/len(hard),
        "hard_races":len(hard),
        "true_pair_top1":sum((x["pair_rank"] or 99)<=1 for x in logs)/n,
        "true_pair_top3":sum((x["pair_rank"] or 99)<=3 for x in logs)/n,
        "true_pair_top5":sum((x["pair_rank"] or 99)<=5 for x in logs)/n,
        "hard_true_pair_top5":sum((x["pair_rank"] or 99)<=5 for x in hard)/len(hard),
    }

def main():
    vrows=v25.load_v19_details();v25.fit_v21_selector(vrows);v25.mark_v21(vrows)
    _,eb,rb=v25.raw_maps()
    train=v25.make_dataset(vrows,eb,rb,v25.TRAIN_START,v25.TRAIN_END,False)
    cal=v25.make_dataset(vrows,eb,rb,v25.CAL_START,v25.CAL_END,False)
    test=v25.make_dataset(vrows,eb,rb,v25.TEST_START,v25.TEST_END,False)

    variants=[
      {"name":"blind_d2a","p":{"depth":2,"leaf":18,"l2":1.5,"lr":.045,"iters":190},"C":.45},
      {"name":"blind_d2b","p":{"depth":2,"leaf":28,"l2":2.5,"lr":.04,"iters":220},"C":.45},
      {"name":"blind_d3a","p":{"depth":3,"leaf":20,"l2":2.0,"lr":.04,"iters":200},"C":.55},
      {"name":"blind_d3b","p":{"depth":3,"leaf":30,"l2":3.0,"lr":.035,"iters":240},"C":.55},
    ]
    calgrid=[];mods={}
    for v in variants:
        pm=fit_pair_model(train,v["p"]);om=fit_orientation(train,v["C"]);mods[v["name"]]=(pm,om)
        for t in (.38,.42,.46,.50,.54,.58,.62,.66,.70):
            logs=eval_ds(cal,pm,om,t);m=metrics(logs)
            if m["avg_second_candidates"]>2.10:continue
            objective=(m["second_given_first_capture"]
                       +0.70*m["hard_diff_line_outside_capture_given_first"]
                       +0.25*m["outside_first_candidates_capture_given_first"]
                       +0.15*m["hard_true_pair_top5"]
                       -0.08*max(0,m["avg_second_candidates"]-2.0))
            calgrid.append({"variant":v["name"],"threshold":t,"objective":objective,**m})
    calgrid.sort(key=lambda x:(x["objective"],x["second_given_first_capture"],x["hard_diff_line_outside_capture_given_first"],-x["avg_second_candidates"]),reverse=True)
    best=calgrid[0];pm,om=mods[best["variant"]]
    logs=eval_ds(test,pm,om,best["threshold"]);tm=metrics(logs)

    baseline={"second_capture":0.5342465753424658,"second_given_first_capture":0.5934065934065934,
              "avg_second_candidates":2.0332681017612524,"hard_diff_line_outside_capture_given_first":0.1259259259259259}
    v25m={"second_capture":0.5362035225048923,"second_given_first_capture":0.5521978021978022,
          "avg_second_candidates":2.0273972602739727,"hard_diff_line_outside_capture_given_first":0.06862745098039216}

    summary={
      "algorithm":"keirin_shogi_v26_top2_pair_blind_first",
      "concept":"Unordered top2 pair stage is blind to v21 first probabilities. v21 is used only to orient pair into 1st/2nd after pair likelihood is formed.",
      "first_core":"v21 frozen",
      "selected_calibration":best,
      "test_metrics":tm,
      "v23_baseline":baseline,
      "v25_first_attempt":v25m,
      "delta_vs_v23":{k:tm[k]-baseline[k] for k in baseline},
      "calibration_top10":calgrid[:10],
      "leakage_guard":"Architecture fixed from prior diagnosis; variant/threshold selected on 2025-10-27..12-28 only. 2026 H1 untouched until final test. Odds unused."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"calibration.json").write_text(json.dumps(calgrid,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

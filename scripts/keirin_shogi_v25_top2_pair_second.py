#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# v25: rebuild 2nd place from scratch as:
# 1) predict unordered Top2 pair among all 21 pairs
# 2) orient each pair using OOS v21 first probabilities + pair relations
# 3) marginalize to P(2nd=j)
# Adopted v21 first core remains frozen.

RAW_BASES=[
    Path("data/2025/s_class_f1_all_parts/2025_q3"),
    Path("data/2025/s_class_f1_all_parts/2025_q4"),
    Path("data/2026_h1/s_class_f1_all"),
]
V19=Path("results/keirin_shogi/v19_targeted_participation")
OUT=Path("results/keirin_shogi/v25_top2_pair_second")
OUT.mkdir(parents=True,exist_ok=True)

SELECTOR_TRAIN_START,SELECTOR_TRAIN_END="2025-06-30","2025-10-26"
TRAIN_START,TRAIN_END="2025-07-01","2025-10-26"
CAL_START,CAL_END="2025-10-27","2025-12-28"
TEST_START,TEST_END="2025-12-29","2026-06-28"

V21_TARGET_FRACTION=0.25
V21_MAX_FIRST_CANDIDATES=2
V21_LOOKBACK_WEEKS=4

NUMERIC=[
    "score","win_rate","top2_rate","top3_rate","s_count","b_count",
    "nige_count","makuri_count","sashi_count","mark_count",
    "first_count","second_count","third_count","outside_count"
]

def load_csv_all(name):
    out=[]
    for base in RAW_BASES:
        p=base/name
        if p.exists():
            with p.open("r",encoding="utf-8-sig",newline="") as f:
                out.extend(csv.DictReader(f))
    return out

def num(v):
    try:return float(v)
    except:return 0.0

def ino(v):
    try:return int(float(v))
    except:return 0

def zmap(vals):
    mu=mean(vals); sd=pstdev(vals) or 1.0
    return [(v-mu)/sd for v in vals]

def entropy(pred):
    return -sum(float(x["prob"])*math.log(max(float(x["prob"]),1e-12)) for x in pred)

def selector_feature(d):
    pred=d["ranking"]; tops=d["model_top1s"]; cands=d["candidates"]
    p=[float(x["prob"]) for x in pred]
    top1=ino(pred[0]["no"])
    agree=sum(1 for x in tops if ino(x)==top1)
    mass=sum(float(x["prob"]) for x in pred if ino(x["no"]) in set(map(ino,cands)))
    model_p1=[]
    for x in pred:
        if ino(x["no"])==top1:
            model_p1=[float(v) for v in x.get("model_probs",[])]
            break
    return np.asarray([
        p[0],p[0]-p[1],p[0]+p[1],mass,entropy(pred),agree/3.0,
        len(cands)/3.0,float(np.std(model_p1)) if model_p1 else 0.0
    ],float)

def load_v19_details():
    out=[]
    for d in sorted(V19.iterdir()):
        if not d.is_dir():continue
        p=d/"detail.json"
        if not p.exists():continue
        week=d.name[:10]
        for r in json.loads(p.read_text(encoding="utf-8")):
            rr=dict(r); rr["week"]=week; rr["x_selector"]=selector_feature(r)
            rr["p1_map"]={ino(x["no"]):float(x["prob"]) for x in r["ranking"]}
            out.append(rr)
    return out

def fit_v21_selector(rows):
    tr=[r for r in rows if SELECTOR_TRAIN_START<=r["week"]<=SELECTOR_TRAIN_END]
    X=np.vstack([r["x_selector"] for r in tr])
    y=np.asarray([int(r["candidate_hit"]) for r in tr],int)
    clf=LogisticRegression(C=0.35,class_weight="balanced",max_iter=800,random_state=21)
    clf.fit(X,y)
    for r in rows:
        r["v21_selector_score"]=float(clf.predict_proba(r["x_selector"].reshape(1,-1))[0,1])

def v21_threshold(rows,current_week):
    prior=sorted(set(r["week"] for r in rows if r["week"]<current_week))[-V21_LOOKBACK_WEEKS:]
    vals=[r["v21_selector_score"] for r in rows
          if r["week"] in prior and int(r["candidate_count"])<=V21_MAX_FIRST_CANDIDATES]
    if not vals:return 1.0
    n=max(1,int(round(len(vals)*V21_TARGET_FRACTION)))
    return float(sorted(vals)[-n])

def mark_v21(rows):
    weeks=sorted(set(r["week"] for r in rows))
    for w in weeks:
        thr=v21_threshold(rows,w)
        for r in rows:
            if r["week"]==w:
                r["v21_threshold"]=thr
                r["v21_participate"]=(int(r["candidate_count"])<=2 and r["v21_selector_score"]>=thr)

def raw_maps():
    races=load_csv_all("races.csv")
    entries=load_csv_all("entries.csv")
    results=load_csv_all("results.csv")
    race_by={r["race_id"]:r for r in races}
    eb=defaultdict(list);rb=defaultdict(list)
    for x in entries:eb[x["race_id"]].append(x)
    for x in results:rb[x["race_id"]].append(x)
    return race_by,eb,rb

def result_order(results):
    vals=[]
    for r in results:
        p=ino(r.get("finish_position"))
        if p in (1,2,3):vals.append((p,ino(r["car_no"])))
    vals.sort()
    if [p for p,_ in vals] != [1,2,3]:return None
    return [n for _,n in vals]

def race_features(entries):
    z={}
    for f in NUMERIC:
        vals=[num(r.get(f)) for r in entries]
        zs=zmap(vals)
        for r,v in zip(entries,zs):z[(ino(r["car_no"]),f)]=v

    groups=defaultdict(list)
    for r in entries:
        lid=(r.get("line_id") or "").strip() or f"solo_{r['car_no']}"
        groups[lid].append(r)

    lids=list(groups)
    stats={}
    for lid,rs in groups.items():
        stats[lid]={
            "mean_score":mean(num(x.get("score")) for x in rs),
            "max_score":max(num(x.get("score")) for x in rs),
            "b_sum":sum(num(x.get("b_count")) for x in rs),
            "atk_sum":sum(num(x.get("nige_count"))+num(x.get("makuri_count")) for x in rs),
            "size":len(rs)
        }
    def line_z(key):
        vals=[stats[l][key] for l in lids]
        zs=zmap(vals)
        return {l:v for l,v in zip(lids,zs)}
    lm=line_z("mean_score");lx=line_z("max_score");lb=line_z("b_sum");la=line_z("atk_sum")

    out={}
    for r in entries:
        no=ino(r["car_no"]);lid=(r.get("line_id") or "").strip() or f"solo_{no}"
        pos=ino(r.get("line_position"));size=ino(r.get("line_size")) or stats[lid]["size"]
        x={f"z_{f}":z[(no,f)] for f in NUMERIC}
        x.update({
            "line_pos":float(pos if pos else 4),
            "line_pos1":1.0 if pos==1 else 0.0,
            "line_pos2":1.0 if pos==2 else 0.0,
            "line_pos3p":1.0 if pos>=3 else 0.0,
            "line_size":float(size),
            "line_mean_score_z":lm[lid],
            "line_max_score_z":lx[lid],
            "line_b_sum_z":lb[lid],
            "line_attack_sum_z":la[lid],
        })
        out[no]={"x":x,"line_id":lid,"line_position":pos,"line_size":size}
    return out

PAIR_MEMBER_FEATURES=[
    "z_score","z_win_rate","z_top2_rate","z_top3_rate","z_b_count",
    "z_nige_count","z_makuri_count","z_sashi_count","z_mark_count",
    "line_pos","line_size","line_mean_score_z","line_max_score_z",
    "line_b_sum_z","line_attack_sum_z"
]

def pair_features(base,a,b,p1map):
    xa=base[a]["x"]; xb=base[b]["x"]
    x={}
    for f in PAIR_MEMBER_FEATURES:
        av=float(xa[f]);bv=float(xb[f])
        x[f"{f}_sum"]=av+bv
        x[f"{f}_absdiff"]=abs(av-bv)
        x[f"{f}_max"]=max(av,bv)
        x[f"{f}_min"]=min(av,bv)
    same=float(base[a]["line_id"]==base[b]["line_id"])
    pa=float(p1map.get(a,0.0));pb=float(p1map.get(b,0.0))
    x.update({
        "same_line":same,
        "different_line":1.0-same,
        "line_pos_distance":abs(base[a]["line_position"]-base[b]["line_position"]) if same else 0.0,
        "adjacent_same_line":1.0 if same and abs(base[a]["line_position"]-base[b]["line_position"])==1 else 0.0,
        "p1_sum":pa+pb,
        "p1_absdiff":abs(pa-pb),
        "p1_max":max(pa,pb),
        "p1_min":min(pa,pb),
        "both_p1_top2_like":1.0 if min(pa,pb)>=0.10 else 0.0,
    })
    return x

ORIENT_DIFF_FEATURES=[
    "z_score","z_win_rate","z_top2_rate","z_top3_rate","z_b_count",
    "z_nige_count","z_makuri_count","z_sashi_count","z_mark_count",
    "line_pos","line_mean_score_z","line_max_score_z","line_b_sum_z","line_attack_sum_z"
]

def orient_features(base,a,b,p1map):
    # a/b order is deterministic by higher v21 first probability, not car number.
    xa=base[a]["x"]; xb=base[b]["x"]
    x={}
    for f in ORIENT_DIFF_FEATURES:x[f"{f}_diff"]=float(xa[f])-float(xb[f])
    same=float(base[a]["line_id"]==base[b]["line_id"])
    x.update({
        "p1_prob_diff":float(p1map.get(a,0.0))-float(p1map.get(b,0.0)),
        "same_line":same,
        "a_ahead_same_line":1.0 if same and base[a]["line_position"]<base[b]["line_position"] else 0.0,
        "a_behind_same_line":1.0 if same and base[a]["line_position"]>base[b]["line_position"] else 0.0,
    })
    return x

def make_dataset(vrows,eb,rb,start,end,participants_only=False):
    byrid={r["race_id"]:r for r in vrows}
    ds=[]
    for rid,vr in byrid.items():
        es=eb.get(rid,[])
        if len(es)!=7:continue
        race_date=es[0].get("race_date","")
        if not(start<=race_date<=end):continue
        if participants_only and not vr.get("v21_participate"):continue
        order=result_order(rb.get(rid,[]))
        if not order:continue
        base=race_features(es)
        if any(n not in base for n in order):continue
        ds.append((rid,race_date,base,order,vr))
    return ds

def fit_pair_model(ds,participant_only_train=False,params=None):
    feature_names=None;X=[];y=[];sw=[]
    for rid,dt,base,order,vr in ds:
        if participant_only_train and not vr.get("v21_participate"):continue
        truepair=frozenset(order[:2])
        nos=sorted(base)
        pairs=[(nos[i],nos[j]) for i in range(len(nos)) for j in range(i+1,len(nos))]
        xs=[pair_features(base,a,b,vr["p1_map"]) for a,b in pairs]
        if feature_names is None:feature_names=sorted(xs[0])
        for pair,x in zip(pairs,xs):
            X.append([x[f] for f in feature_names])
            y.append(1 if frozenset(pair)==truepair else 0)
            sw.append(1/len(pairs))
    kw=params or {}
    clf=HistGradientBoostingClassifier(
        learning_rate=kw.get("learning_rate",0.045),
        max_iter=kw.get("max_iter",180),
        max_depth=kw.get("max_depth",3),
        min_samples_leaf=kw.get("min_samples_leaf",20),
        l2_regularization=kw.get("l2",1.5),
        random_state=25
    )
    clf.fit(np.asarray(X,float),np.asarray(y,int),sample_weight=np.asarray(sw,float))
    return {"model":clf,"features":feature_names,"train_rows":len(X)}

def fit_orientation(ds,participant_only_train=False,C=0.6):
    fn=None;X=[];y=[]
    for rid,dt,base,order,vr in ds:
        if participant_only_train and not vr.get("v21_participate"):continue
        f,s=order[:2]
        # order a,b by v21 p1 probability. No result-based ordering.
        pa=vr["p1_map"].get(f,0.0);pb=vr["p1_map"].get(s,0.0)
        if pa>=pb:a,b=f,s
        else:a,b=s,f
        x=orient_features(base,a,b,vr["p1_map"])
        if fn is None:fn=sorted(x)
        X.append([x[k] for k in fn]);y.append(1 if a==f else 0)
    clf=LogisticRegression(C=C,class_weight="balanced",max_iter=800,random_state=251)
    clf.fit(np.asarray(X,float),np.asarray(y,int))
    return {"model":clf,"features":fn,"train_rows":len(X)}

def predict_second(base,vr,pair_model,orient_model):
    nos=sorted(base)
    pairs=[(nos[i],nos[j]) for i in range(len(nos)) for j in range(i+1,len(nos))]
    X=[]
    for a,b in pairs:
        x=pair_features(base,a,b,vr["p1_map"])
        X.append([x[f] for f in pair_model["features"]])
    raw=np.clip(pair_model["model"].predict_proba(np.asarray(X,float))[:,1],1e-9,None)
    pairp=raw/raw.sum()

    p2=defaultdict(float)
    pair_rows=[]
    for (a0,b0),pp in zip(pairs,pairp):
        # orientation input uses higher-v21-prob rider as a
        if vr["p1_map"].get(a0,0.0)>=vr["p1_map"].get(b0,0.0):a,b=a0,b0
        else:a,b=b0,a0
        ox=orient_features(base,a,b,vr["p1_map"])
        O=np.asarray([[ox[f] for f in orient_model["features"]]],float)
        p_a_first=float(orient_model["model"].predict_proba(O)[0,1])
        # if a first, b second; otherwise a second
        p2[b]+=float(pp)*p_a_first
        p2[a]+=float(pp)*(1-p_a_first)
        pair_rows.append({"a":a0,"b":b0,"pair_prob":float(pp),"p_higher_v21_first":p_a_first})
    z=sum(p2.values()) or 1.0
    ranking=sorted(((n,v/z) for n,v in p2.items()),key=lambda kv:(-kv[1],kv[0]))
    pair_rows.sort(key=lambda x:-x["pair_prob"])
    return ranking,pair_rows

def choose_by_cum(rank,t,maxk=3):
    out=[];cum=0
    for n,p in rank[:maxk]:
        out.append(n);cum+=p
        if cum>=t:break
    return out

def eval_rows(ds,pair_model,orient_model,t):
    logs=[]
    for rid,dt,base,order,vr in ds:
        if not vr.get("v21_participate"):continue
        rank,pairs=predict_second(base,vr,pair_model,orient_model)
        c=choose_by_cum(rank,t,3)
        f,s,t3=order
        fc=list(map(ino,vr["candidates"]))
        relation="same_line" if base[f]["line_id"]==base[s]["line_id"] else "different_line"
        logs.append({
            "race_id":rid,"race_date":dt,"actual_first":f,"actual_second":s,
            "first_candidates":fc,"first_hit":int(f in fc),
            "second_in_first_candidates":int(s in fc),
            "relation":relation,
            "second_candidates":c,"second_hit":int(s in c),
            "second_rank":next((i+1 for i,(n,p) in enumerate(rank) if n==s),None),
            "second_candidate_count":len(c),
            "ranking":rank,"top_pairs":pairs[:5]
        })
    return logs

def metrics(logs):
    n=len(logs)
    if not n:return {}
    first=[x for x in logs if x["first_hit"]]
    hard=[x for x in logs if x["first_hit"] and not x["second_in_first_candidates"] and x["relation"]=="different_line"]
    outside=[x for x in logs if x["first_hit"] and not x["second_in_first_candidates"]]
    return {
        "races":n,
        "second_capture":sum(x["second_hit"] for x in logs)/n,
        "avg_second_candidates":mean(x["second_candidate_count"] for x in logs),
        "second_top1_rate":sum((x["second_rank"] or 99)<=1 for x in logs)/n,
        "second_top2_rate":sum((x["second_rank"] or 99)<=2 for x in logs)/n,
        "second_top3_rate":sum((x["second_rank"] or 99)<=3 for x in logs)/n,
        "second_given_first_capture":sum(x["second_hit"] for x in first)/len(first) if first else 0,
        "outside_first_candidates_capture_given_first":sum(x["second_hit"] for x in outside)/len(outside) if outside else 0,
        "hard_diff_line_outside_capture_given_first":sum(x["second_hit"] for x in hard)/len(hard) if hard else 0,
        "hard_races":len(hard),
    }

def main():
    vrows=load_v19_details();fit_v21_selector(vrows);mark_v21(vrows)
    race_by,eb,rb=raw_maps()
    train_ds=make_dataset(vrows,eb,rb,TRAIN_START,TRAIN_END,False)
    cal_ds=make_dataset(vrows,eb,rb,CAL_START,CAL_END,False)
    test_ds=make_dataset(vrows,eb,rb,TEST_START,TEST_END,False)

    variants=[
        {"name":"all_depth2","participant_train":False,"pair_params":{"max_depth":2,"min_samples_leaf":24,"l2":2.0,"learning_rate":0.045,"max_iter":180},"orient_C":0.5},
        {"name":"all_depth3","participant_train":False,"pair_params":{"max_depth":3,"min_samples_leaf":20,"l2":1.5,"learning_rate":0.04,"max_iter":200},"orient_C":0.6},
        {"name":"participant_depth2","participant_train":True,"pair_params":{"max_depth":2,"min_samples_leaf":18,"l2":2.5,"learning_rate":0.045,"max_iter":180},"orient_C":0.5},
        {"name":"participant_depth3","participant_train":True,"pair_params":{"max_depth":3,"min_samples_leaf":16,"l2":2.0,"learning_rate":0.04,"max_iter":200},"orient_C":0.6},
    ]

    calibration=[]
    models={}
    for v in variants:
        pm=fit_pair_model(train_ds,v["participant_train"],v["pair_params"])
        om=fit_orientation(train_ds,v["participant_train"],v["orient_C"])
        models[v["name"]]=(pm,om)
        for threshold in (0.38,0.42,0.46,0.50,0.54,0.58,0.62,0.66,0.70):
            logs=eval_rows(cal_ds,pm,om,threshold)
            m=metrics(logs)
            if not m:continue
            # Hard constraint: do not inflate average candidates.
            if m["avg_second_candidates"]>2.10:continue
            # Reward actual bottleneck directly.
            objective=(m["second_given_first_capture"]
                       +0.55*m["hard_diff_line_outside_capture_given_first"]
                       +0.25*m["outside_first_candidates_capture_given_first"]
                       -0.10*max(0,m["avg_second_candidates"]-2.0))
            calibration.append({"variant":v["name"],"threshold":threshold,"objective":objective,**m})
    calibration.sort(key=lambda x:(x["objective"],x["second_given_first_capture"],x["second_capture"],-x["avg_second_candidates"]),reverse=True)
    best=calibration[0]
    pm,om=models[best["variant"]]

    test_logs=eval_rows(test_ds,pm,om,best["threshold"])
    testm=metrics(test_logs)

    # v23 reference fixed from completed diagnosis
    baseline={
        "second_capture":0.5342465753424658,
        "second_given_first_capture":0.5934065934065934,
        "avg_second_candidates":2.0332681017612524,
        "hard_diff_line_outside_capture_given_first":0.1259259259259259,
    }

    summary={
        "algorithm":"keirin_shogi_v25_top2_pair_second",
        "concept":"Predict unordered top2 pair first, then orient pair with OOS v21 first probabilities and pair relations; marginalize into second-place probabilities.",
        "first_core":"v21 frozen; no changes",
        "train_period":[TRAIN_START,TRAIN_END],
        "calibration_period":[CAL_START,CAL_END],
        "test_period":[TEST_START,TEST_END],
        "selected_calibration":best,
        "test_metrics":testm,
        "v23_baseline":baseline,
        "delta_vs_v23":{
            "second_capture":testm["second_capture"]-baseline["second_capture"],
            "second_given_first_capture":testm["second_given_first_capture"]-baseline["second_given_first_capture"],
            "avg_second_candidates":testm["avg_second_candidates"]-baseline["avg_second_candidates"],
            "hard_diff_line_outside_capture_given_first":testm["hard_diff_line_outside_capture_given_first"]-baseline["hard_diff_line_outside_capture_given_first"],
        },
        "calibration_top10":calibration[:10],
        "leakage_guard":"Model variants and threshold selected only on 2025-10-27..12-28 calibration. 2026 H1 is final untouched test. v21 OOS probabilities/participation are used as historical inputs; odds unused."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(test_logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"calibration.json").write_text(json.dumps(calibration,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

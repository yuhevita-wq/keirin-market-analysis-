#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------------
# v23: adopted v21 first-place core is frozen.
# Build 2nd/3rd conditionally on the first-place candidate set.
# ------------------------------------------------------------------

RAW_BASES=[
    Path("data/2025/s_class_f1_all_parts/2025_q3"),
    Path("data/2025/s_class_f1_all_parts/2025_q4"),
    Path("data/2026_h1/s_class_f1_all"),
]
V19=Path("results/keirin_shogi/v19_targeted_participation")
OUT=Path("results/keirin_shogi/v23_conditional_second_third")
OUT.mkdir(parents=True,exist_ok=True)

SELECTOR_TRAIN_START,SELECTOR_TRAIN_END="2025-06-30","2025-10-26"
RANK_TRAIN_START,RANK_TRAIN_END="2025-07-01","2025-10-26"
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

# ---------- reconstruct adopted v21 participation ------------------

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
        if not d.is_dir(): continue
        p=d/"detail.json"
        if not p.exists(): continue
        week=d.name[:10]
        for r in json.loads(p.read_text(encoding="utf-8")):
            rr=dict(r)
            rr["week"]=week
            rr["x_selector"]=selector_feature(r)
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
    return clf

def v21_threshold(rows,current_week):
    prior=sorted(set(r["week"] for r in rows if r["week"]<current_week))[-V21_LOOKBACK_WEEKS:]
    vals=[r["v21_selector_score"] for r in rows
          if r["week"] in prior and int(r["candidate_count"])<=V21_MAX_FIRST_CANDIDATES]
    if not vals:return 1.0
    n=max(1,int(round(len(vals)*V21_TARGET_FRACTION)))
    return float(sorted(vals)[-n])

def mark_v21_participation(rows):
    weeks=sorted(set(r["week"] for r in rows))
    for w in weeks:
        thr=v21_threshold(rows,w)
        for r in rows:
            if r["week"]==w:
                r["v21_threshold"]=thr
                r["v21_participate"]=(int(r["candidate_count"])<=2 and r["v21_selector_score"]>=thr)

# ---------- race feature engineering -------------------------------

def race_base_features(entries):
    bynum={ino(r["car_no"]):r for r in entries}
    z={}
    for f in NUMERIC:
        vals=[num(r.get(f)) for r in entries]
        zs=zmap(vals)
        for r,v in zip(entries,zs):
            z[(ino(r["car_no"]),f)]=v

    # line aggregates, no hand-made combined strength score
    groups=defaultdict(list)
    for r in entries:
        lid=(r.get("line_id") or "").strip() or f"solo_{r['car_no']}"
        groups[lid].append(r)

    line_stats={}
    mean_scores=[]; max_scores=[]; b_sums=[]; atk_sums=[]
    lids=list(groups)
    for lid in lids:
        rs=groups[lid]
        stat={
            "mean_score":mean(num(x.get("score")) for x in rs),
            "max_score":max(num(x.get("score")) for x in rs),
            "b_sum":sum(num(x.get("b_count")) for x in rs),
            "attack_sum":sum(num(x.get("nige_count"))+num(x.get("makuri_count")) for x in rs),
            "size":len(rs)
        }
        line_stats[lid]=stat
        mean_scores.append(stat["mean_score"]);max_scores.append(stat["max_score"])
        b_sums.append(stat["b_sum"]);atk_sums.append(stat["attack_sum"])
    def dictz(vals):
        zs=zmap(vals); return {lid:v for lid,v in zip(lids,zs)}
    lmz=dictz(mean_scores); lxz=dictz(max_scores); lbz=dictz(b_sums); laz=dictz(atk_sums)

    out={}
    for r in entries:
        no=ino(r["car_no"])
        lid=(r.get("line_id") or "").strip() or f"solo_{no}"
        pos=ino(r.get("line_position")); size=ino(r.get("line_size")) or line_stats[lid]["size"]
        x={}
        for f in NUMERIC:x[f"z_{f}"]=z[(no,f)]
        style=(r.get("style") or "").strip()
        x.update({
            "style_escape":1.0 if style=="逃" else 0.0,
            "style_both":1.0 if style=="両" else 0.0,
            "style_chase":1.0 if style=="追" else 0.0,
            "line_pos1":1.0 if pos==1 else 0.0,
            "line_pos2":1.0 if pos==2 else 0.0,
            "line_pos3p":1.0 if pos>=3 else 0.0,
            "line_size1":1.0 if size==1 else 0.0,
            "line_size2":1.0 if size==2 else 0.0,
            "line_size3p":1.0 if size>=3 else 0.0,
            "line_mean_score_z":lmz[lid],
            "line_max_score_z":lxz[lid],
            "line_b_sum_z":lbz[lid],
            "line_attack_sum_z":laz[lid],
        })
        out[no]={"entry":r,"x":x,"line_id":lid,"line_position":pos,"line_size":size}
    return out

def relation_features(base,cand,first,second=None):
    c=base[cand]; f=base[first]
    x=dict(c["x"])
    samef=float(c["line_id"]==f["line_id"])
    diff_f=float(c["line_id"]!=f["line_id"])
    x.update({
        "same_line_first":samef,
        "diff_line_first":diff_f,
        "pos_delta_first":float(c["line_position"]-f["line_position"]) if samef else 0.0,
        "behind_first_same_line":1.0 if samef and c["line_position"]>f["line_position"] else 0.0,
        "ahead_first_same_line":1.0 if samef and c["line_position"]<f["line_position"] else 0.0,
        "score_diff_first":c["x"]["z_score"]-f["x"]["z_score"],
        "b_diff_first":c["x"]["z_b_count"]-f["x"]["z_b_count"],
        "makuri_diff_first":c["x"]["z_makuri_count"]-f["x"]["z_makuri_count"],
        "sashi_diff_first":c["x"]["z_sashi_count"]-f["x"]["z_sashi_count"],
        "mark_x_same_first":c["x"]["z_mark_count"]*samef,
        "sashi_x_same_first":c["x"]["z_sashi_count"]*samef,
        "makuri_x_diff_first":c["x"]["z_makuri_count"]*diff_f,
        "first_score_z":f["x"]["z_score"],
        "first_b_z":f["x"]["z_b_count"],
        "first_makuri_z":f["x"]["z_makuri_count"],
        "first_sashi_z":f["x"]["z_sashi_count"],
        "first_line_pos1":f["x"]["line_pos1"],
        "first_line_pos2":f["x"]["line_pos2"],
    })
    if second is not None:
        s=base[second]
        sames=float(c["line_id"]==s["line_id"])
        fsame=float(f["line_id"]==s["line_id"])
        x.update({
            "same_line_second":sames,
            "diff_line_second":1.0-sames,
            "first_second_same_line":fsame,
            "pos_delta_second":float(c["line_position"]-s["line_position"]) if sames else 0.0,
            "behind_second_same_line":1.0 if sames and c["line_position"]>s["line_position"] else 0.0,
            "ahead_second_same_line":1.0 if sames and c["line_position"]<s["line_position"] else 0.0,
            "score_diff_second":c["x"]["z_score"]-s["x"]["z_score"],
            "b_diff_second":c["x"]["z_b_count"]-s["x"]["z_b_count"],
            "makuri_diff_second":c["x"]["z_makuri_count"]-s["x"]["z_makuri_count"],
            "sashi_diff_second":c["x"]["z_sashi_count"]-s["x"]["z_sashi_count"],
            "mark_x_same_second":c["x"]["z_mark_count"]*sames,
            "sashi_x_same_second":c["x"]["z_sashi_count"]*sames,
            "second_score_z":s["x"]["z_score"],
            "second_b_z":s["x"]["z_b_count"],
            "second_makuri_z":s["x"]["z_makuri_count"],
            "second_sashi_z":s["x"]["z_sashi_count"],
            "same_line_both":1.0 if samef and sames else 0.0,
        })
    return x

def result_order(results):
    vals=[]
    for r in results:
        p=ino(r.get("finish_position"))
        if p in (1,2,3):vals.append((p,ino(r["car_no"])))
    vals.sort()
    if [p for p,_ in vals] != [1,2,3]: return None
    return [no for _,no in vals]

# ---------- conditional rank models --------------------------------

def make_training_races(race_ids,entries_by,results_by):
    out=[]
    for rid in race_ids:
        es=entries_by.get(rid,[])
        if len(es)!=7:continue
        order=result_order(results_by.get(rid,[]))
        if not order:continue
        base=race_base_features(es)
        if any(no not in base for no in order):continue
        out.append((rid,base,order))
    return out

def fit_conditional_model(ds,stage):
    X=[];y=[];sw=[]
    feature_names=None
    for rid,base,order in ds:
        first,second,third=order
        if stage==2:
            eligible=[n for n in base if n!=first]
            target=second
            xs=[relation_features(base,n,first) for n in eligible]
        else:
            eligible=[n for n in base if n not in (first,second)]
            target=third
            xs=[relation_features(base,n,first,second) for n in eligible]
        if feature_names is None:feature_names=sorted(xs[0])
        for n,x in zip(eligible,xs):
            X.append([x[f] for f in feature_names]); y.append(1 if n==target else 0); sw.append(1/len(eligible))
    clf=HistGradientBoostingClassifier(
        learning_rate=0.045,max_iter=180,max_depth=3,min_samples_leaf=22,
        l2_regularization=1.5,random_state=230+stage
    )
    clf.fit(np.asarray(X,float),np.asarray(y,int),sample_weight=np.asarray(sw,float))
    return {"model":clf,"features":feature_names,"train_races":len(ds)}

def conditional_probs(base,cond_model,first,second=None):
    eligible=[n for n in base if n!=first and (second is None or n!=second)]
    xs=[relation_features(base,n,first,second) for n in eligible]
    X=np.asarray([[x[f] for f in cond_model["features"]] for x in xs],float)
    raw=np.clip(cond_model["model"].predict_proba(X)[:,1],1e-9,None)
    p=raw/raw.sum()
    return {n:float(v) for n,v in zip(eligible,p)}

def first_weights(v19row):
    cands=list(map(ino,v19row["candidates"]))
    pm={ino(x["no"]):float(x["prob"]) for x in v19row["ranking"]}
    vals={n:max(pm.get(n,0.0),1e-9) for n in cands}
    z=sum(vals.values())
    return {n:v/z for n,v in vals.items()}

def predict_second_third(base,v19row,m2,m3):
    fw=first_weights(v19row)
    p2=defaultdict(float)
    branch2={}
    for f,wf in fw.items():
        q=conditional_probs(base,m2,f)
        branch2[f]=q
        for j,p in q.items():p2[j]+=wf*p
    z=sum(p2.values()) or 1.0
    p2={k:v/z for k,v in p2.items()}

    p3=defaultdict(float)
    for f,wf in fw.items():
        q2=branch2[f]
        for s,ps in q2.items():
            q3=conditional_probs(base,m3,f,s)
            for k,pk in q3.items():
                p3[k]+=wf*ps*pk
    z3=sum(p3.values()) or 1.0
    p3={k:v/z3 for k,v in p3.items()}

    r2=sorted(p2.items(),key=lambda kv:(-kv[1],kv[0]))
    r3=sorted(p3.items(),key=lambda kv:(-kv[1],kv[0]))
    return r2,r3

def choose_by_cum(ranking,threshold,maxk=3):
    out=[];cum=0.0
    for n,p in ranking[:maxk]:
        out.append(n);cum+=p
        if cum>=threshold:break
    return out

# ---------- calibration / evaluation -------------------------------

def raw_maps():
    races=load_csv_all("races.csv")
    entries=load_csv_all("entries.csv")
    results=load_csv_all("results.csv")
    race_by={r["race_id"]:r for r in races}
    eb=defaultdict(list);rb=defaultdict(list)
    for x in entries:eb[x["race_id"]].append(x)
    for x in results:rb[x["race_id"]].append(x)
    return race_by,eb,rb

def target_ids(race_by,start,end):
    return [rid for rid,r in race_by.items()
            if start<=r.get("race_date","")<=end
            and ino(r.get("entry_count"))==7
            and r.get("meeting_grade")=="F1"
            and "Ｓ級" in r.get("race_type","")]

def joined_predictions(vrows,base_models,entries_by,results_by,start,end,participants_only=True):
    m2,m3=base_models
    byrid={r["race_id"]:r for r in vrows}
    out=[]
    for rid,r in byrid.items():
        date=r.get("week","")
        # use actual race date from raw entry if possible
        es=entries_by.get(rid,[])
        if len(es)!=7:continue
        race_date=es[0].get("race_date","")
        if not(start<=race_date<=end):continue
        if participants_only and not r.get("v21_participate"):continue
        order=result_order(results_by.get(rid,[]))
        if not order:continue
        base=race_base_features(es)
        try:r2,r3=predict_second_third(base,r,m2,m3)
        except Exception:continue
        out.append({
            "race_id":rid,"race_date":race_date,"order":order,
            "first_candidates":list(map(ino,r["candidates"])),
            "second_ranking":r2,"third_ranking":r3
        })
    return out

def evaluate(rows,t2,t3):
    logs=[]
    for r in rows:
        f,s,t=r["order"]
        fc=r["first_candidates"]
        sc=choose_by_cum(r["second_ranking"],t2,3)
        tc=choose_by_cum(r["third_ranking"],t3,3)
        logs.append({
            "race_id":r["race_id"],"race_date":r["race_date"],
            "actual_1st":f,"actual_2nd":s,"actual_3rd":t,
            "first_candidates":fc,"second_candidates":sc,"third_candidates":tc,
            "first_hit":int(f in fc),"second_hit":int(s in sc),"third_hit":int(t in tc),
            "complete_hit":int(f in fc and s in sc and t in tc),
            "cells":len(fc)+len(sc)+len(tc),
            "second_count":len(sc),"third_count":len(tc)
        })
    n=len(logs)
    first_n=sum(x["first_hit"] for x in logs)
    first_second_n=sum(x["first_hit"] and x["second_hit"] for x in logs)
    return logs,{
        "races":n,
        "first_capture":sum(x["first_hit"] for x in logs)/n if n else 0,
        "second_capture":sum(x["second_hit"] for x in logs)/n if n else 0,
        "third_capture":sum(x["third_hit"] for x in logs)/n if n else 0,
        "complete_capture":sum(x["complete_hit"] for x in logs)/n if n else 0,
        "second_given_first":(sum(x["first_hit"] and x["second_hit"] for x in logs)/first_n) if first_n else 0,
        "third_given_first_second":(sum(x["complete_hit"] for x in logs)/first_second_n) if first_second_n else 0,
        "avg_cells":mean(x["cells"] for x in logs) if logs else 0,
        "avg_second_candidates":mean(x["second_count"] for x in logs) if logs else 0,
        "avg_third_candidates":mean(x["third_count"] for x in logs) if logs else 0,
    }

def calibrate(rows):
    grid=[]
    for t2 in (0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80):
        for t3 in (0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80):
            _,m=evaluate(rows,t2,t3)
            grid.append({"t2":t2,"t3":t3,**m})
    feasible=[g for g in grid if g["avg_cells"]<=7.0]
    pool=feasible if feasible else [g for g in grid if g["avg_cells"]<=8.0]
    pool.sort(key=lambda g:(g["complete_capture"],g["second_given_first"],g["third_given_first_second"],-g["avg_cells"]),reverse=True)
    return pool[0],grid

def main():
    vrows=load_v19_details()
    fit_v21_selector(vrows)
    mark_v21_participation(vrows)

    race_by,eb,rb=raw_maps()
    train_ids=target_ids(race_by,RANK_TRAIN_START,RANK_TRAIN_END)
    train_ds=make_training_races(train_ids,eb,rb)
    m2=fit_conditional_model(train_ds,2)
    m3=fit_conditional_model(train_ds,3)

    cal_rows=joined_predictions(vrows,(m2,m3),eb,rb,CAL_START,CAL_END,True)
    policy,grid=calibrate(cal_rows)

    test_rows=joined_predictions(vrows,(m2,m3),eb,rb,TEST_START,TEST_END,True)
    logs,test_metrics=evaluate(test_rows,policy["t2"],policy["t3"])

    # row-wise capture of old independent v19 first candidates is kept frozen.
    weekly=defaultdict(list)
    for x in logs:weekly[x["race_date"][:7]].append(x)
    monthly=[]
    for month,rs in sorted(weekly.items()):
        n=len(rs)
        monthly.append({
            "month":month,"races":n,
            "complete_capture":sum(x["complete_hit"] for x in rs)/n,
            "first_capture":sum(x["first_hit"] for x in rs)/n,
            "second_capture":sum(x["second_hit"] for x in rs)/n,
            "third_capture":sum(x["third_hit"] for x in rs)/n,
            "avg_cells":mean(x["cells"] for x in rs)
        })

    model_info={
        "stage2_train_races":m2["train_races"],
        "stage3_train_races":m3["train_races"],
        "stage2_feature_count":len(m2["features"]),
        "stage3_feature_count":len(m3["features"]),
    }
    summary={
        "algorithm":"keirin_shogi_v23_conditional_second_third",
        "first_core":"v21 adopted and frozen",
        "train_period":[RANK_TRAIN_START,RANK_TRAIN_END],
        "calibration_period":[CAL_START,CAL_END],
        "test_period":[TEST_START,TEST_END],
        "architecture":{
            "second":"P(2nd=j | first=f, race). Train with actual first; inference mixes over adopted first candidates weighted by v18/v21 first probabilities.",
            "third":"P(3rd=k | first=f, second=s, race). Train with actual first+second; inference marginalizes over first-candidate and second distributions.",
            "relations":"line_id / line_position / line_size plus candidate-vs-first and candidate-vs-second relative features.",
            "candidate_rule":"1st stays v21 max2. 2nd/3rd each adaptive 1-3 by calibrated cumulative probability thresholds."
        },
        "model_info":model_info,
        "calibration_policy":policy,
        "calibration_participant_races":len(cal_rows),
        "test_participant_races":len(test_rows),
        "test_metrics":test_metrics,
        "monthly":monthly,
        "leakage_guard":"2nd/3rd models trained only through 2025-10-26. Candidate thresholds calibrated only 2025-10-27..12-28. 2026 H1 top2/top3 results are untouched until final test. v21 participation is reconstructed from its prior OOS predictions; odds unused."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"calibration_grid.json").write_text(json.dumps(grid,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"README.md").write_text(
        "# v23 conditional second/third\n\n"
        "採用済みv21の1着コアを固定し、2着を1着条件付き、3着を1着+2着条件付きで学習。\n"
        "1着候補を勝手に変更しない。\n",
        encoding="utf-8"
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

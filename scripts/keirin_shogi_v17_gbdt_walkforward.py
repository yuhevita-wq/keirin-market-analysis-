#!/usr/bin/env python3
from __future__ import annotations

import csv, json, math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev

try:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception as e:
    raise SystemExit("scikit-learn/numpy required: "+repr(e))

BASES=[
    Path("data/2024/s_class_f1_all_parts/2024_q1"),
    Path("data/2024/s_class_f1_all_parts/2024_q2"),
    Path("data/2024/s_class_f1_all_parts/2024_q3"),
]
OUT_DIR=Path("results/keirin_shogi/v17_gbdt_walkforward")
OUT_DIR.mkdir(parents=True,exist_ok=True)

START_TEST=date(2024,5,6)
END_TEST=date(2024,9,29)
ACCEPT={
    "top1_min":0.30,
    "candidate_capture_min":0.65,
    "avg_candidates_max":2.20,
    "three_candidate_share_max":0.50,
    "consecutive_weeks":2,
}
BASE_FEATURES=[
    "score","win_rate","b_count","nige_count","makuri_count","sashi_count",
    "mark_count","first_count","second_count","third_count"
]
FEATURE_NAMES=[]

def load_all(name):
    out=[]
    for base in BASES:
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

def target_races(races,start,end):
    ss=start.isoformat(); ee=end.isoformat()
    return {r["race_id"]:r for r in races
            if ss<=r.get("race_date","")<=ee
            and ino(r.get("entry_count"))==7
            and r.get("meeting_grade")=="F1"
            and "Ｓ級" in r.get("race_type","")}

def zvals(vals):
    mu=mean(vals); sd=pstdev(vals) or 1.0
    return [(v-mu)/sd for v in vals]

def race_rows(rows):
    groups=defaultdict(list)
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        groups[lid].append(r)
    lst={}
    for lid,rs in groups.items():
        avg=mean([num(x.get("score")) for x in rs])
        b=sum(num(x.get("b_count")) for x in rs)
        attack=sum(num(x.get("nige_count"))+num(x.get("makuri_count")) for x in rs)
        lst[lid]={
            "size":len(rs),
            "max_score":max(num(x.get("score")) for x in rs),
            "strength":avg+0.8*b+0.8*attack
        }
    raw={f:[num(r.get(f)) for r in rows] for f in BASE_FEATURES}
    z={f:zvals(v) for f,v in raw.items()}
    strengths=[]
    for r in rows:
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        strengths.append(lst[lid]["strength"])
    zls=zvals(strengths)

    out=[]
    for i,r in enumerate(rows):
        lid=(r.get("line_id") or "").strip() or f"solo_{r.get('car_no')}"
        ls=lst[lid]
        others=[v["strength"] for k,v in lst.items() if k!=lid]
        other_best=max(others) if others else ls["strength"]
        pos=ino(r.get("line_position"))
        x=[]
        names=[]
        for f in BASE_FEATURES:
            x.append(z[f][i]); names.append("z_"+f)
        extra={
            "line_pos":float(pos if pos else 4),
            "line_size":float(ls["size"]),
            "z_line_strength":zls[i],
            "gap_to_line_max":(num(r.get("score"))-ls["max_score"])/5.0,
            "line_strength_vs_other":(ls["strength"]-other_best)/10.0,
            "attack":z["nige_count"][i]+z["makuri_count"][i],
            "score_x_pos2":z["score"][i]*(1.0 if pos==2 else 0.0),
            "sashi_x_pos2":z["sashi_count"][i]*(1.0 if pos==2 else 0.0),
            "attack_x_pos1":(z["nige_count"][i]+z["makuri_count"][i])*(1.0 if pos==1 else 0.0),
            "makuri_x_line_adv":z["makuri_count"][i]*((ls["strength"]-other_best)/10.0),
            "score_x_line_adv":z["score"][i]*((ls["strength"]-other_best)/10.0),
        }
        for k,v in extra.items():
            x.append(v); names.append(k)
        global FEATURE_NAMES
        if not FEATURE_NAMES: FEATURE_NAMES=names
        out.append({"no":ino(r["car_no"]),"name":r.get("player_name",""),"x":x})
    return out

def make_races(ids,eb,rb):
    out=[]
    for rid in ids:
        rs=eb.get(rid,[])
        if len(rs)!=7:continue
        wins=[x for x in rb.get(rid,[]) if ino(x.get("finish_position"))==1]
        if len(wins)!=1:continue
        winner=ino(wins[0]["car_no"])
        rr=race_rows(rs)
        if winner in [x["no"] for x in rr]:
            out.append((rid,rr,winner))
    return out

def fit_model(ds):
    X=[];y=[];sw=[]
    for _,rr,winner in ds:
        for a in rr:
            X.append(a["x"]); y.append(1 if a["no"]==winner else 0); sw.append(1/7)
    clf=HistGradientBoostingClassifier(
        learning_rate=0.05,max_iter=140,max_depth=3,min_samples_leaf=18,
        l2_regularization=1.0,random_state=17
    )
    clf.fit(np.asarray(X,float),np.asarray(y,int),sample_weight=np.asarray(sw,float))
    return clf

def predict(rr,model):
    X=np.asarray([a["x"] for a in rr],float)
    raw=model.predict_proba(X)[:,1]
    raw=np.clip(raw,1e-8,None)
    probs=raw/raw.sum()
    out=[{"no":a["no"],"name":a["name"],"prob":float(p)} for a,p in zip(rr,probs)]
    out.sort(key=lambda x:(-x["prob"],x["no"]))
    return out

def choose(pred,pol):
    p1,p2,p3=pred[0]["prob"],pred[1]["prob"],pred[2]["prob"]
    if p1>=pol["p1"] and p1/max(p2,1e-9)>=pol["ratio12"]:
        return [pred[0]["no"]]
    if p1+p2>=pol["cum2"] and p2/max(p3,1e-9)>=pol["ratio23"]:
        return [pred[0]["no"],pred[1]["no"]]
    return [pred[0]["no"],pred[1]["no"],pred[2]["no"]]

def learn_policy(train_ds,val_ds):
    model=fit_model(train_ds)
    cache=[]
    for _,rr,w in val_ds:
        cache.append((predict(rr,model),w))
    grid=[]
    for p1 in (0.30,0.34,0.38,0.42):
      for r12 in (1.35,1.60,1.90):
       for c2 in (0.48,0.54,0.60,0.66):
        for r23 in (1.10,1.25,1.45):
            pol={"p1":p1,"ratio12":r12,"cum2":c2,"ratio23":r23}
            hits=0;counts=[]
            for pred,w in cache:
                c=choose(pred,pol);hits+=int(w in c);counts.append(len(c))
            cap=hits/len(cache);avg=mean(counts);three=sum(k==3 for k in counts)/len(counts)
            obj=cap-0.16*(avg-1)-0.10*three
            grid.append({**pol,"capture":cap,"avg_candidates":avg,"three_share":three,"objective":obj})
    grid.sort(key=lambda x:(x["objective"],x["capture"],-x["avg_candidates"]),reverse=True)
    return grid[0]

def metric(rows):
    n=len(rows)
    return {
        "races":n,
        "top1_hit_rate":sum(r["top1_hit"] for r in rows)/n,
        "candidate_capture_rate":sum(r["candidate_hit"] for r in rows)/n,
        "avg_candidates":mean([r["candidate_count"] for r in rows]),
        "three_candidate_share":sum(r["candidate_count"]==3 for r in rows)/n,
        "candidate_count_distribution":{str(k):sum(r["candidate_count"]==k for r in rows) for k in (1,2,3)}
    }

def pass_week(m):
    return (
        m["top1_hit_rate"]>=ACCEPT["top1_min"] and
        m["candidate_capture_rate"]>=ACCEPT["candidate_capture_min"] and
        m["avg_candidates"]<=ACCEPT["avg_candidates_max"] and
        m["three_candidate_share"]<=ACCEPT["three_candidate_share_max"]
    )

def main():
    races=load_all("races.csv");entries=load_all("entries.csv");results=load_all("results.csv")
    eb=defaultdict(list);rb=defaultdict(list)
    for x in entries:eb[x["race_id"]].append(x)
    for x in results:rb[x["race_id"]].append(x)

    weekly=[];streak=0;checkpoint=None
    cur=START_TEST
    while cur<=END_TEST:
        te=cur+timedelta(days=6)
        tr0=cur-timedelta(days=56); tr1=cur-timedelta(days=1)
        val0=cur-timedelta(days=14)
        inner_train_ids=target_races(races,tr0,val0-timedelta(days=1))
        inner_val_ids=target_races(races,val0,tr1)
        full_ids=target_races(races,tr0,tr1)
        test_ids=target_races(races,cur,te)
        inner_train=make_races(inner_train_ids.keys(),eb,rb)
        inner_val=make_races(inner_val_ids.keys(),eb,rb)
        full=make_races(full_ids.keys(),eb,rb)
        test=make_races(test_ids.keys(),eb,rb)
        if min(len(inner_train),len(inner_val),len(full),len(test))==0:
            cur+=timedelta(days=7);continue
        pol=learn_policy(inner_train,inner_val)
        model=fit_model(full)
        rows=[];detail=[]
        for rid,rr,winner in test:
            pred=predict(rr,model);c=choose(pred,pol)
            rows.append({
                "race_id":rid,"top1_hit":int(pred[0]["no"]==winner),
                "candidate_hit":int(winner in c),"candidate_count":len(c)
            })
            detail.append({"race_id":rid,"actual_1st":winner,"ranking":pred,"candidates":c})
        m=metric(rows);ok=pass_week(m);streak=streak+1 if ok else 0
        rec={
            "test_period":{"start":cur.isoformat(),"end":te.isoformat()},
            "train_period":{"start":tr0.isoformat(),"end":tr1.isoformat()},
            "train_races":len(full),"inner_validation_races":len(inner_val),
            "policy":pol,"metrics":m,"passes_acceptance":ok,"streak":streak
        }
        weekly.append(rec)
        wk=OUT_DIR/f"{cur.isoformat()}_{te.isoformat()}";wk.mkdir(parents=True,exist_ok=True)
        (wk/"summary.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
        (wk/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
        if streak>=ACCEPT["consecutive_weeks"]:
            checkpoint=rec["test_period"]
            break
        cur+=timedelta(days=7)

    final={
        "algorithm":"keirin_shogi_v17_gbdt_walkforward",
        "architecture":"race-relative nonlinear gradient boosting; each race weighted equally, probabilities renormalized within the 7-rider race",
        "acceptance_rule":ACCEPT,
        "weeks_tested":len(weekly),
        "checkpoint":checkpoint,
        "weekly":weekly,
        "method":"毎週、直前8週間で非線形モデルを再学習。候補数1/2/3の判定は直前2週間をinner validationとしてのみ使用し、3人乱用へ罰則。単発の良週では止めず事前条件2週連続を要求。",
        "leakage_guard":"各週より未来の結果は学習・候補ポリシー選択に未使用。オッズ不使用。車番は特徴量不使用。"
    }
    (OUT_DIR/"summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(final,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

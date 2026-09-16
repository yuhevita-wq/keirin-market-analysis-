#!/usr/bin/env python3
from __future__ import annotations

import json, importlib.util
from collections import defaultdict
from datetime import date,timedelta
from pathlib import Path
from statistics import mean
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

spec=importlib.util.spec_from_file_location("v17","scripts/keirin_shogi_v17_gbdt_walkforward.py")
v17=importlib.util.module_from_spec(spec); spec.loader.exec_module(v17)
v17.BASES=[
 Path("data/2024/s_class_f1_all_parts/2024_q1"),
 Path("data/2024/s_class_f1_all_parts/2024_q2"),
 Path("data/2024/s_class_f1_all_parts/2024_q3"),
 Path("data/2024/s_class_f1_all_parts/2024_q4"),
 Path("data/2025/s_class_f1_all_parts/2025_q1"),
 Path("data/2025/s_class_f1_all_parts/2025_q2"),
]

OUT_DIR=Path("results/keirin_shogi/v18_ensemble_walkforward");OUT_DIR.mkdir(parents=True,exist_ok=True)
START_TEST=date(2024,12,30);END_TEST=date(2025,6,29)
ACCEPT={"top1_min":0.30,"candidate_capture_min":0.65,"avg_candidates_max":2.20,"three_candidate_share_max":0.50,"consecutive_weeks":2}

def fit_ensemble(ds):
    X=[];y=[];sw=[]
    for _,rr,winner in ds:
        for a in rr:
            X.append(a["x"]);y.append(1 if a["no"]==winner else 0);sw.append(1/7)
    X=np.asarray(X,float);y=np.asarray(y,int);sw=np.asarray(sw,float)
    specs=[
        dict(learning_rate=.05,max_iter=150,max_depth=3,min_samples_leaf=18,l2_regularization=1.0,random_state=18),
        dict(learning_rate=.04,max_iter=170,max_depth=2,min_samples_leaf=25,l2_regularization=2.0,random_state=181),
        dict(learning_rate=.035,max_iter=200,max_depth=1,min_samples_leaf=28,l2_regularization=2.5,random_state=1818),
    ]
    models=[]
    for kw in specs:
        m=HistGradientBoostingClassifier(**kw);m.fit(X,y,sample_weight=sw);models.append(m)
    return models

def predict(rr,models):
    X=np.asarray([a["x"] for a in rr],float)
    per=[]
    for m in models:
        raw=np.clip(m.predict_proba(X)[:,1],1e-8,None)
        per.append(raw/raw.sum())
    avg=np.mean(np.vstack(per),axis=0)
    rows=[{"no":a["no"],"name":a["name"],"prob":float(p),"model_probs":[float(per[j][i]) for j in range(len(models))]} for i,(a,p) in enumerate(zip(rr,avg))]
    rows.sort(key=lambda x:(-x["prob"],x["no"]))
    model_top1=[]
    for probs in per:
        idx=int(np.argmax(probs));model_top1.append(rr[idx]["no"])
    return rows,model_top1

def choose(pred,tops,pol):
    p1,p2,p3=pred[0]["prob"],pred[1]["prob"],pred[2]["prob"]
    agree=max(tops.count(x) for x in set(tops))
    if agree>=pol["agree_min"] and p1>=pol["p1"] and p1/max(p2,1e-9)>=pol["ratio12"]:
        return [pred[0]["no"]]
    if p1+p2>=pol["cum2"]:
        return [pred[0]["no"],pred[1]["no"]]
    return [pred[0]["no"],pred[1]["no"],pred[2]["no"]]

def learn_policy(train_ds,val_ds):
    models=fit_ensemble(train_ds)
    cache=[]
    for _,rr,w in val_ds:
        p,t=predict(rr,models);cache.append((p,t,w))
    grid=[]
    for agree in (2,3):
      for p1 in (.30,.34,.38,.42):
       for ratio in (1.3,1.5,1.8,2.1):
        for cum2 in (.48,.54,.60,.66):
            pol={"agree_min":agree,"p1":p1,"ratio12":ratio,"cum2":cum2}
            hits=0;counts=[]
            for p,t,w in cache:
                c=choose(p,t,pol);hits+=int(w in c);counts.append(len(c))
            cap=hits/len(cache);avg=mean(counts);three=sum(k==3 for k in counts)/len(counts)
            obj=cap-0.16*(avg-1)-0.10*three
            grid.append({**pol,"capture":cap,"avg_candidates":avg,"three_share":three,"objective":obj})
    grid.sort(key=lambda x:(x["objective"],x["capture"],-x["avg_candidates"]),reverse=True)
    return grid[0]

def metrics(rows):
    n=len(rows)
    return {
      "races":n,
      "top1_hit_rate":sum(x["top1_hit"] for x in rows)/n,
      "candidate_capture_rate":sum(x["candidate_hit"] for x in rows)/n,
      "avg_candidates":mean([x["candidate_count"] for x in rows]),
      "three_candidate_share":sum(x["candidate_count"]==3 for x in rows)/n,
      "candidate_count_distribution":{str(k):sum(x["candidate_count"]==k for x in rows) for k in (1,2,3)}
    }

def passes(m):
    return m["top1_hit_rate"]>=ACCEPT["top1_min"] and m["candidate_capture_rate"]>=ACCEPT["candidate_capture_min"] and m["avg_candidates"]<=ACCEPT["avg_candidates_max"] and m["three_candidate_share"]<=ACCEPT["three_candidate_share_max"]

def main():
    races=v17.load_all("races.csv");entries=v17.load_all("entries.csv");results=v17.load_all("results.csv")
    eb=defaultdict(list);rb=defaultdict(list)
    for x in entries:eb[x["race_id"]].append(x)
    for x in results:rb[x["race_id"]].append(x)

    weekly=[];streak=0;checkpoint=None;cur=START_TEST
    while cur<=END_TEST:
        te=cur+timedelta(days=6);tr0=cur-timedelta(days=56);tr1=cur-timedelta(days=1);val0=cur-timedelta(days=14)
        inner_train=v17.make_races(v17.target_races(races,tr0,val0-timedelta(days=1)).keys(),eb,rb)
        inner_val=v17.make_races(v17.target_races(races,val0,tr1).keys(),eb,rb)
        full=v17.make_races(v17.target_races(races,tr0,tr1).keys(),eb,rb)
        test=v17.make_races(v17.target_races(races,cur,te).keys(),eb,rb)
        if min(len(inner_train),len(inner_val),len(full),len(test))==0:
            cur+=timedelta(days=7);continue
        pol=learn_policy(inner_train,inner_val);models=fit_ensemble(full)
        rows=[];detail=[]
        for rid,rr,winner in test:
            pred,tops=predict(rr,models);c=choose(pred,tops,pol)
            rows.append({"race_id":rid,"top1_hit":int(pred[0]["no"]==winner),"candidate_hit":int(winner in c),"candidate_count":len(c)})
            detail.append({"race_id":rid,"actual_1st":winner,"ranking":pred,"model_top1s":tops,"candidates":c})
        m=metrics(rows);ok=passes(m);streak=streak+1 if ok else 0
        rec={"test_period":{"start":cur.isoformat(),"end":te.isoformat()},"train_period":{"start":tr0.isoformat(),"end":tr1.isoformat()},"train_races":len(full),"inner_validation_races":len(inner_val),"policy":pol,"metrics":m,"passes_acceptance":ok,"streak":streak}
        weekly.append(rec)
        wk=OUT_DIR/f"{cur.isoformat()}_{te.isoformat()}";wk.mkdir(parents=True,exist_ok=True)
        (wk/"summary.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding="utf-8")
        (wk/"detail.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2),encoding="utf-8")
        if streak>=ACCEPT["consecutive_weeks"]:
            checkpoint=rec["test_period"];break
        cur+=timedelta(days=7)

    final={"algorithm":"keirin_shogi_v18_ensemble_walkforward","architecture":"3-model nonlinear ensemble with race-relative features","acceptance_rule":ACCEPT,"weeks_tested":len(weekly),"checkpoint":checkpoint,"weekly":weekly,"method":"3種類の正則化・深さのGBDTをアンサンブルし、モデル合意度とレース内正規化確率から1/2/3人の1着候補を決定。各週直前8週間で再学習、候補ポリシーは直前2週間のinner validationのみで決定。事前基準2週連続まで継続。","leakage_guard":"各週より未来の結果は不使用。オッズ不使用。車番は特徴量不使用。"}
    (OUT_DIR/"summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(final,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

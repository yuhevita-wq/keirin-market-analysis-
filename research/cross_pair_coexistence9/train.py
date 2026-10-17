#!/usr/bin/env python3
"""Observed co-occurrence in historical nine-car G1/G2/G3 races.

Train pair outcomes, NOT a sum/product of rider winning probabilities.
Compare with a rider-only control on untouched 2025 and 2026H1 races.
No race-day results, odds, popularity, or payout may enter features.
The model is *not* a four-ticket model: third-place conditional work remains.
"""
from __future__ import annotations
import importlib.util
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/cross_pair_coexistence9"
SOURCE = ROOT / "scripts/keirin_shogi_ninecar_v2.py"
PAIR_FEATURES = (
    "same_line", "both_front", "front_bante", "both_bante",
    "front_third", "bante_third", "both_third",
    "same_line_adjacent", "different_line_both_front",
    "different_line_both_bante", "different_line_front_bante",
    "both_leaders_high_b", "different_line_leader_b_conflict",
    "leader_b_gap", "leader_nige_collision",
    "rival_front_b_max", "rival_front_b_sum",
    "different_line_both_have_high_b", "both_mark_riders",
    "front_bante_with_strong_front", "different_line_front_bante_with_high_b",
    "both_lines_3car", "one_line_3car", "singleline_rider",
    "line_pair_top3_sum_gap", "line_leader_score_gap",
)
RIDER_FIELDS = ("score", "win_rate", "top3_rate", "b_count", "nige_count", "makuri_count", "sashi_count", "mark_count")
SCALES = (10, 20, 25, 8, 6, 6, 6, 6)
def module():
    spec = importlib.util.spec_from_file_location("ninecar_pair_source", SOURCE)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    return obj

def feature_row(race, a, b, src):
    entries = {src.ino(e["car_no"]): e for e in race.entries}
    # Sort by pre-race top-3 record; never by actual finish or betting rank.
    ea, eb = sorted((entries[a], entries[b]), key=lambda e: (
        -src.fnum(e.get("top3_rate")), -src.fnum(e.get("score")), src.ino(e["car_no"])))
    xs = list(entries.values())
    mean = {k: sum(src.fnum(e.get(k)) for e in xs) / 9 for k in RIDER_FIELDS}
    rider = lambda e: [
        (src.fnum(e.get(k)) - mean[k]) / scale
        for k, scale in zip(RIDER_FIELDS, SCALES)
    ] + [float(src.ino(e.get("line_position")) == p) for p in (1,2,3)] + [
        float(src.ino(e.get("line_size")) == 3)]
    base = rider(ea) + rider(eb)
    pa, pb = src.ino(ea["line_position"]), src.ino(eb["line_position"])
    la, lb = src.ino(ea["line_id"]), src.ino(eb["line_id"])
    same = la == lb
    leaders = {}
    for e in xs:
        li = src.ino(e.get("line_id"))
        if src.ino(e.get("line_position")) == 1:
            leaders[li] = e
    if la not in leaders or lb not in leaders:
        raise ValueError("Missing published line leader")
    fa, fb = leaders[la], leaders[lb]
    ba, bb = src.fnum(fa.get("b_count")), src.fnum(fb.get("b_count"))
    na, nb = src.fnum(fa.get("nige_count")), src.fnum(fb.get("nige_count"))
    other = [src.fnum(e.get("b_count")) for lid,e in leaders.items() if lid not in (la,lb)]
    line_top3 = lambda li: sum(src.fnum(e.get("top3_rate")) for e in xs if src.ino(e["line_id"]) == li)
    mech = [
        float(same),float(pa == 1 and pb == 1),float({pa,pb} == {1,2}),
        float(pa == 2 and pb == 2),float({pa,pb} == {1,3}),
        float({pa,pb} == {2,3}),float(pa == 3 and pb == 3),
        float(same and abs(pa - pb) == 1),float(not same and pa == pb == 1),
        float(not same and pa == pb == 2),float(not same and {pa,pb} == {1,2}),
        float(ba >= 5 and bb >= 5),
        (ba * bb / 100) if not same else 0,
        (abs(ba-bb)/10) if not same else 0,
        (na*nb/100) if not same else 0,
        max(other,default=0)/10, sum(other)/20,
        float(not same and ba >= 5 and bb >= 5),
        float(src.fnum(ea.get("mark_count"))>=2 and src.fnum(eb.get("mark_count"))>=2),
        float({pa,pb} == {1,2} and max(ba,bb) >= 5),
        float(not same and {pa,pb} == {1,2} and ba >= 5 and bb >= 5),
        float(src.ino(ea.get("line_size")) == 3 and src.ino(eb.get("line_size")) == 3),
        float(src.ino(ea.get("line_size")) == 3 or src.ino(eb.get("line_size")) == 3),
        float(src.ino(ea.get("line_size")) == 1 or src.ino(eb.get("line_size")) == 1),
        abs(line_top3(la)-line_top3(lb))/100 if not same else 0,
        abs(src.fnum(fa.get("score"))-src.fnum(fb.get("score")))/10 if not same else 0,
    ]
    assert len(mech) == len(PAIR_FEATURES)
    return base, mech

def make_matrix(races, src):
    xb, xm, top2, top3 = [], [], [], []
    ids = []
    for race in races:
        rows_b, rows_m, labels2, labels3 = [], [], [], []
        winners = set(race.order[:2])
        podium = set(race.order)
        for a,b in combinations(range(1,10),2):
            u,v = feature_row(race,a,b,src)
            rows_b.append(u)
            rows_m.append(v)
            labels2.append(int({a,b} == winners))
            labels3.append(int(a in podium and b in podium))
        if sum(labels2) != 1 or sum(labels3) != 3:
            raise ValueError("Invalid 1-2 / top-3 outcome label")
        xb.append(rows_b); xm.append(rows_m); top2.append(labels2);top3.append(labels3)
        ids.append(race.race_id)
    return np.asarray(xb,dtype=np.float32),np.asarray(xm,dtype=np.float32),np.asarray(top2,dtype=np.uint8),np.asarray(top3,dtype=np.uint8),ids

def fit(x,y):
    n,p,d=x.shape
    scaler=StandardScaler()
    z=scaler.fit_transform(x.reshape(n*p,d))
    clf=LogisticRegression(C=0.5,max_iter=400,solver="lbfgs",tol=1e-6)
    clf.fit(z,y.reshape(-1))
    return scaler,clf

def predict(fitted,x):
    scaler,clf=fitted
    n,p,d=x.shape
    return np.clip(clf.predict_proba(scaler.transform(x.reshape(n*p,d)))[:,1].reshape(n,p),1e-8,1-1e-8)

def metrics(p,y,typ):
    if typ=="top2":
        q=p/p.sum(axis=1,keepdims=True)
        target=np.argmax(y,axis=1)
        selected=np.argmax(q,axis=1)
        nll=float((-np.log(q[np.arange(len(q)),target])).mean())
        return {"race_n":len(y),"race_pair_nll":nll,"top_pair_hit":float(np.mean(selected==target)),
                "top3_pair_capture":float(np.mean([target[i] in np.argsort(-q[i])[:3] for i in range(len(y))]))}
    brier=float(np.mean((p-y)**2))
    nll=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
    selected=np.argmax(p,axis=1)
    return {"race_n":len(y),"pair_brier":brier,"binary_nll":nll,
            "top_pair_both_top3":float(np.mean(y[np.arange(len(y)),selected]))}

def export(fitted):
    sc,clf=fitted
    return {"intercept":float(clf.intercept_[0]),"coef":[float(v) for v in clf.coef_[0]],
            "mean":[float(v) for v in sc.mean_],"scale":[float(v) for v in sc.scale_]}

def compare(test,base,full,name):
    kb="race_pair_nll" if name=="top2" else "binary_nll"
    mb,mf=metrics(base,test,name),metrics(full,test,name)
    return {"baseline":mb,"mechanism":mf,"nll_gain":mb[kb]-mf[kb],"improved":mf[kb]<mb[kb]}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=module()
    raw=src.load_races()
    # Never train on 2025/2026; discard 2026 H2 and future races.
    races=[r for r in raw if "2022-01-01"<=r.race_date<="2026-06-30"
        and r.grade in ("G1","G2","G3")]
    by_year=Counter(r.race_date[:4] for r in races)
    train=[r for r in races if r.race_date<"2025-01-01"]
    tests={"2025":[r for r in races if r.race_date[:4]=="2025"],
           "2026H1":[r for r in races if r.race_date[:4]=="2026"]}
    if len(train)<1000 or min(map(len,tests.values()))<100:
        raise RuntimeError(f"Insufficient historical race outcomes: {dict(by_year)}")
    xb,xm,y2,y3,_=make_matrix(train,src)
    allx=np.concatenate([xb,xm],axis=2)
    models={}
    for name,target in (("top2",y2),("top3",y3)):
        models[name]={"rider_only":fit(xb,target),"mechanism":fit(allx,target)}
    report={"protocol":"Training 2022-2024; untouched 2025 and 2026H1. Labels from REAL finish. Features PRE-RACE only.",
            "source":"data/grade_races g1g2/g3 ZIP via scripts/keirin_shogi_ninecar_v2.py",
            "population":"9-car S-class G1,G2,G3, published line_id and positions, unique 1st/2nd/3rd",
            "years":dict(by_year),"train_races":len(train),
            "expected_labels_per_race":{"top2":1,"top3":3},"holdout":{}}
    for label,data in tests.items():
        xt,mt,t2,t3,_=make_matrix(data,src)
        xx=np.concatenate([xt,mt],axis=2)
        report["holdout"][label]={}
        for name,target in (("top2",t2),("top3",t3)):
            m=models[name]
            b=predict(m["rider_only"],xt)
            f=predict(m["mechanism"],xx)
            report["holdout"][label][name]=compare(target,b,f,name)
        print(label,json.dumps(report["holdout"][label],ensure_ascii=False),flush=True)
    # Both periods AND both outcome definitions must improve on held-out NLL.
    valid=all(report["holdout"][period][label]["improved"]
        for period in tests for label in ("top2","top3"))
    report["validated"]=valid
    report["status"]="HOLDOUT_PASSED" if valid else "HOLDOUT_FAILED_DO_NOT_AUTO_RANK"
    bundle={"version":"observed-pair-compatibility9-v1","trainThrough":"2024-12-31",
            "scope":"9-car S-class G1/G2/G3","feature_names":{"interaction":PAIR_FEATURES,
            "base_per_rider":list(RIDER_FIELDS)+["pos1","pos2","pos3","line3"]},
            "evaluationStatus":report["status"],"canAutoRankCross4":False,
            "models":{label:{name:export(m) for name,m in pair.items()} for label,pair in models.items()}}
    (OUT/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    (OUT/"pair_model.json").write_text(json.dumps(bundle,ensure_ascii=False,separators=(",",":"))+"\n")
    print("RESULT",report["status"],"train",len(train),"holdouts",{k:len(v) for k,v in tests.items()},flush=True)

if __name__=="__main__":
    main()

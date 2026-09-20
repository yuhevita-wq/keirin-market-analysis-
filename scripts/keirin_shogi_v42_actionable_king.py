#!/usr/bin/env python3
from __future__ import annotations
"""
v42 ACTIONABLE KING

Lesson from v41:
- sparse seats were not the main failure: W2 contained 5 KING outcomes ranked within top5.
- gate failed alignment: it skipped all 5 actionable KINGs and bought KING races ranked 53/116/145.

Fix:
- race gate target is no longer "is this a KING race?"
- gate target is "is this an ACTIONABLE KING race for the current exact-seat model?"
  = payout >= 10,000 AND actual combo rank <= seat_k.

Chronology:
- discovery/tuning: 2024-01-01..01-14 (W1+W2; W2 is now learned history)
- untouched holdout: 2024-01-15..01-21 (W3)
Hard constraints:
- exact seats only
- seat_k in {3,4,5,6,8}
- target participation 10%-30%
- no target odds/popularity
- reward capped at 50,000 during selection
"""
import importlib.util, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/keirin_shogi/v42_actionable_king"; OUT.mkdir(parents=True,exist_ok=True)
KING_YEN=10000; REWARD_CAP=50000; UNIT=100
W1=("2024-01-01","2024-01-07"); W2=("2024-01-08","2024-01-14"); W3=("2024-01-15","2024-01-21")
MIN_BUY_RATE=.10; MAX_BUY_RATE=.30

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    assert spec.loader is not None; sys.modules[spec.name]=m; spec.loader.exec_module(m); return m
v41=load_module("v42_base",ROOT/"scripts/keirin_shogi_v41_strict_gate_sparse_seats.py")

def fit_action_gate(rows,k,C):
    X=np.asarray([r["gate_features"] for r in rows],float)
    y=np.asarray([int(r["payout"]>=KING_YEN and r["actual_rank"]<=k) for r in rows],int)
    # If a pathological fold has one class, refuse it cleanly.
    if len(set(y.tolist()))<2:return None,None
    sc=StandardScaler().fit(X)
    clf=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=2500,random_state=42)
    clf.fit(sc.transform(X),y)
    return clf,sc

def apply_gate(rows,clf,sc):
    if clf is None:
        return [{**r,"gate_prob":0.0} for r in rows]
    X=np.asarray([r["gate_features"] for r in rows],float)
    ps=clf.predict_proba(sc.transform(X))[:,1]
    return [{**r,"gate_prob":float(p)} for r,p in zip(rows,ps)]

def thresholds(rows):
    vals=np.asarray([r["gate_prob"] for r in rows],float)
    return sorted({float(np.quantile(vals,q)) for q in np.linspace(.70,.90,17)})

def evaluate(rows,k,th,cap=True):
    logs=[];stake=ret=0
    for r in rows:
        buy=r["gate_prob"]>=th
        hit=bool(buy and r["actual_rank"]<=k)
        st=k*UNIT if buy else 0
        rr=(min(r["payout"],REWARD_CAP) if cap else r["payout"]) if hit else 0
        actionable=bool(r["payout"]>=KING_YEN and r["actual_rank"]<=k)
        stake+=st;ret+=rr
        logs.append({"buy":buy,"hit":hit,"king":r["payout"]>=KING_YEN,"actionable":actionable,"payout":r["payout"],"stake":st,"return":rr})
    return {"stake":stake,"ret":ret,"profit":ret-stake,"buy":sum(x["buy"] for x in logs),"hits":sum(x["hit"] for x in logs),"king_hits":sum(x["hit"] and x["king"] for x in logs),"actionable_total":sum(x["actionable"] for x in logs),"actionable_bought":sum(x["buy"] and x["actionable"] for x in logs),"logs":logs}

def choose_threshold(train_rows,k):
    n=len(train_rows);best=None
    for th in thresholds(train_rows):
        m=evaluate(train_rows,k,th,True);br=m["buy"]/n if n else 0
        if not (MIN_BUY_RATE<=br<=MAX_BUY_RATE):continue
        # Primary: actionable KING recall, then capped profit, then fewer buys.
        recall=m["actionable_bought"]/m["actionable_total"] if m["actionable_total"] else 0
        key=(recall,m["profit"],m["king_hits"],-m["buy"])
        if best is None or key>best[0]:best=(key,th)
    if best:return best[1]
    vals=sorted([r["gate_prob"] for r in train_rows],reverse=True)
    idx=max(0,min(len(vals)-1,int(round(.20*len(vals)))-1))
    return vals[idx] if vals else 1.0

def build_cache(period):
    t,e,r,p=v41.load_range(*period)
    base=v41.build_base(t,e)
    return v41.make_cache(t,e,r,p,base)

def tune_on_w1_w2(c1,c2):
    ids1=sorted(c1);ids2=sorted(c2);grid=[]
    for Cc in (.05,.1,.25,.5,1.0):
        cclf,csc=v41.fit_combo(ids1,c1,Cc)
        for alpha in (.25,.45,.65,.80):
            r1=v41.score_rows(ids1,c1,cclf,csc,alpha);r2=v41.score_rows(ids2,c2,cclf,csc,alpha)
            for k in (3,4,5,6,8):
                for Cg in (.03,.05,.1,.25,.5,1.0):
                    gclf,gsc=fit_action_gate(r1,k,Cg)
                    a1=apply_gate(r1,gclf,gsc);a2=apply_gate(r2,gclf,gsc)
                    th=choose_threshold(a1,k)
                    m=evaluate(a2,k,th,True)
                    br=m["buy"]/len(a2) if a2 else 0
                    exp=m["buy"]*k/210.0;lift=m["hits"]/exp if exp else 0
                    recall=m["actionable_bought"]/m["actionable_total"] if m["actionable_total"] else 0
                    # W2 is now historical validation. Strongly reward actual actionable KING capture.
                    score=m["profit"] + 5000*m["king_hits"] + 3000*m["actionable_bought"] + 500*lift - 250*k
                    grid.append({"combo_C":Cc,"alpha":alpha,"k":k,"gate_C":Cg,"threshold_w1":th,
                      "w2_stake":m["stake"],"w2_return_capped":m["ret"],"w2_profit_capped":m["profit"],"w2_buy":m["buy"],"w2_buy_rate":br,
                      "w2_hits":m["hits"],"w2_king_hits":m["king_hits"],"w2_actionable_total":m["actionable_total"],"w2_actionable_bought":m["actionable_bought"],
                      "w2_actionable_gate_recall":recall,"random_expected_hits":exp,"hit_lift_vs_random":lift,"score":score})
    grid.sort(key=lambda x:(x["score"],x["w2_king_hits"],x["w2_actionable_gate_recall"],x["w2_profit_capped"],-x["k"]),reverse=True)
    return grid

def merge_cache(a,b):
    z=dict(a);z.update(b);return z

def final_train_test(train,test,best):
    ids=sorted(train); k=best["k"]
    cclf,csc=v41.fit_combo(ids,train,best["combo_C"])
    tr=v41.score_rows(ids,train,cclf,csc,best["alpha"])
    gclf,gsc=fit_action_gate(tr,k,best["gate_C"])
    tr=apply_gate(tr,gclf,gsc);th=choose_threshold(tr,k)
    xr=v41.score_rows(sorted(test),test,cclf,csc,best["alpha"]);xr=apply_gate(xr,gclf,gsc)
    logs=[]
    for r in xr:
        buy=r["gate_prob"]>=th;hit=bool(buy and r["actual_rank"]<=k)
        st=k*UNIT if buy else 0;ret=r["payout"] if hit else 0
        verdict="SKIP" if not buy else ("KING" if hit and r["payout"]>=KING_YEN else ("IMPUDENT" if hit else "BEHEADED"))
        actionable=bool(r["payout"]>=KING_YEN and r["actual_rank"]<=k)
        seats=[list(r["combos"][int(i)]) for i in r["order"][:k]]
        logs.append({"race_id":r["rid"],"date":r["date"],"track":r["track"],"race_no":r["race_no"],"actual":list(r["actual"]) if r["actual"] else None,
          "payout_yen":r["payout"],"buy":buy,"verdict":verdict,"hit":hit,"actionable_king":actionable,"actual_rank":r["actual_rank"],"seat_count":k,
          "stake_yen":st,"return_yen":ret,"profit_yen":ret-st,"gate_prob":r["gate_prob"],"gate_threshold":th,"seats":seats})
    bought=[x for x in logs if x["buy"]];hits=[x for x in logs if x["hit"]]
    actionable=[x for x in logs if x["actionable_king"]]
    stake=sum(x["stake_yen"] for x in logs);ret=sum(x["return_yen"] for x in logs);largest=max((x["return_yen"] for x in hits),default=0)
    capped=sum(min(x["return_yen"],REWARD_CAP) for x in hits);exp=len(bought)*k/210.0
    summary={"algorithm":"v42_actionable_king","train_period":["2024-01-01","2024-01-14"],"holdout_period":W3,"holdout_used_for_tuning":False,
      "definition":"gate predicts KING AND exact-seat rank <= k, not KING alone",
      "hard_constraints":{"max_exact_seats":8,"cartesian_betting":False,"target_buy_rate_train_min":MIN_BUY_RATE,"target_buy_rate_train_max":MAX_BUY_RATE},
      "selected":{"combo_C":best["combo_C"],"alpha":best["alpha"],"gate_C":best["gate_C"],"seat_k":k,"gate_threshold":th},
      "races":len(logs),"buy_races":len(bought),"skip_races":len(logs)-len(bought),"buy_rate":len(bought)/len(logs) if logs else 0,
      "verdict_counts":dict(Counter(x["verdict"] for x in logs)),"hits":len(hits),"hit_rate_on_bought":len(hits)/len(bought) if bought else 0,
      "random_expected_hits_at_same_ticket_count":exp,"hit_lift_vs_random":len(hits)/exp if exp else 0,
      "actual_king_races":sum(x["payout_yen"]>=KING_YEN for x in logs),"actionable_king_races":len(actionable),
      "actionable_king_gate_bought":sum(x["buy"] for x in actionable),"actionable_king_gate_recall":sum(x["buy"] for x in actionable)/len(actionable) if actionable else 0,
      "king_hits":sum(x["verdict"]=="KING" for x in logs),"impudent_hits":sum(x["verdict"]=="IMPUDENT" for x in logs),"beheaded":sum(x["verdict"]=="BEHEADED" for x in logs),
      "stake_yen":stake,"payout_yen":ret,"profit_yen":ret-stake,"roi":ret/stake if stake else 0,"capped_payout_yen":capped,"capped_profit_yen":capped-stake,"capped_roi":capped/stake if stake else 0,
      "largest_hit_yen":largest,"payout_without_largest_hit_yen":ret-largest,"profit_without_largest_hit_yen":ret-largest-stake,"roi_without_largest_hit":(ret-largest)/stake if stake else 0,
      "king_details":[x for x in logs if x["verdict"]=="KING"],"actionable_missed":[x for x in actionable if not x["buy"]]}
    return summary,logs

def main():
    c1=build_cache(W1); c2=build_cache(W2); c3=build_cache(W3)
    grid=tune_on_w1_w2(c1,c2);best=grid[0]
    train=merge_cache(c1,c2)
    summary,logs=final_train_test(train,c3,best)
    summary["historical_selection_w1_to_w2"]=best;summary["historical_top10"]=grid[:10]
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"holdout_w3_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

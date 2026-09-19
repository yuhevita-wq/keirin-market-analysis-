#!/usr/bin/env python3
from __future__ import annotations
"""v40 LAST CHANCE KING: exact seats only, max 16, day-CV, capped reward selection."""

import csv, importlib.util, itertools, json, math, re, sys, unicodedata
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/2024/s_class_f1_all_parts/2024_q1"
OUT=ROOT/"results/keirin_shogi/v40_last_chance_king";OUT.mkdir(parents=True,exist_ok=True)
TRAIN=("2024-01-01","2024-01-07");TEST=("2024-01-08","2024-01-14")
KING_YEN=10000;REWARD_CAP=50000;UNIT=100

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    assert spec.loader is not None;sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
runtime=load_module("v40_runtime",ROOT/"scripts/keirin_shogi_v37_auto_place_runtime.py")
v38=load_module("v40_features",ROOT/"scripts/keirin_shogi_v38_king_seat_2024w1.py")
class PassKing:
    @staticmethod
    def predict_king_seat(placed,race):return placed

def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def ino(v):
    try:return int(float(v))
    except:return 0
def norm(v):return unicodedata.normalize("NFKC",str(v or "")).strip()
def triple(v):
    xs=[int(x) for x in re.findall(r"[1-9]",norm(v))]
    return tuple(xs[:3]) if len(xs)>=3 and len(set(xs[:3]))==3 else None

RACES=read(BASE/"races.csv");ENTRIES=read(BASE/"entries.csv");RESULTS=read(BASE/"results.csv");PAYOUTS=read(BASE/"payouts.csv")

def load_range(start,end):
    t={r["race_id"]:r for r in RACES if start<=r.get("race_date","")<=end and ino(r.get("entry_count"))==7 and r.get("meeting_grade")=="F1" and "Ｓ級" in r.get("race_type","")}
    eb=defaultdict(list);rb=defaultdict(list);paid=defaultdict(dict)
    for e in ENTRIES:
        if e.get("race_id") in t:eb[e["race_id"]].append(e)
    for x in RESULTS:
        if x.get("race_id") in t:rb[x["race_id"]].append(x)
    for p in PAYOUTS:
        rid=str(p.get("race_id",""))
        if rid not in t or norm(p.get("ticket_type")) not in {"3連単","三連単"}:continue
        if norm(p.get("status")).lower() not in {"","paid"}:continue
        z=triple(p.get("combination"));y=re.sub(r"[^0-9]","",norm(p.get("payout_yen")))
        if z and y:paid[rid][z]=int(y)
    return t,eb,rb,paid

def actual_order(rows):
    top={}
    for x in rows:
        p=ino(x.get("finish_position"))
        if p in (1,2,3):top[p]=ino(x.get("car_no"))
    return (top[1],top[2],top[3]) if len(top)==3 else None

def build_base(t,eb):
    e=runtime.load_base_engine();n=runtime.load_ninecar_engine();o=runtime.load_sevencar_overlay_engine()
    s=e.build_v21_state();pair=e.build_pair_model();third,fn,freeze=e.build_third_model();out={}
    for rid,r in sorted(t.items(),key=lambda kv:(kv[1].get("race_date",""),kv[1].get("track",""),ino(kv[1].get("race_no")))):
        ers=sorted(eb[rid],key=lambda x:ino(x.get("car_no")))
        live={"race_id":rid,"race_date":r.get("race_date",""),"track":r.get("track",""),"race_no":r.get("race_no",""),"race_type":r.get("race_type",""),"meeting_grade":r.get("meeting_grade",""),"entries":[runtime.normalize_entry(x) for x in ers]}
        p=runtime.runtime_place_one(e,n,o,PassKing(),live,s,pair,third,fn,freeze)
        if p.get("board_generated"):out[rid]=p
    return out

def normal_logs(combos,p):
    fr={int(r["no"]):float(r["probability"]) for r in p["first_ranking"]}
    sr={int(r["no"]):float(r["mass"])/2.0 for r in p["second_membership"]}
    tr={int(r["no"]):float(r["probability"]) for r in p["third_ranking"]}
    return np.asarray([math.log(max(fr[a],1e-12))+math.log(max(sr[b],1e-12))+math.log(max(tr[c],1e-12)) for a,b,c in combos],float)

def make_cache(t,eb,rb,paid,base):
    out={}
    for rid,p in base.items():
        entries=sorted(eb[rid],key=lambda x:ino(x.get("car_no")));cars=sorted(ino(x["car_no"]) for x in entries)
        combos=list(itertools.permutations(cars,3));X=np.asarray([v38.feature_vector(c,p,entries) for c in combos],float)
        actual=actual_order(rb[rid]);idx=combos.index(actual) if actual in combos else -1;payout=paid[rid].get(actual,0) if actual else 0
        out[rid]={"rid":rid,"date":t[rid]["race_date"],"track":t[rid].get("track"),"race_no":ino(t[rid].get("race_no")),"combos":combos,"X":X,"normal_log":normal_logs(combos,p),"actual":actual,"actual_idx":idx,"payout":payout}
    return out

def fit(ids,cache,C):
    X=np.asarray([cache[r]["X"][cache[r]["actual_idx"]] for r in ids if cache[r]["actual_idx"]>=0],float)
    y=np.asarray([int(cache[r]["payout"]>=KING_YEN) for r in ids if cache[r]["actual_idx"]>=0],int)
    w=np.asarray([1.0 if cache[r]["payout"]<KING_YEN else 1.0+min(math.log1p(min(cache[r]["payout"],REWARD_CAP)/KING_YEN),1.5) for r in ids if cache[r]["actual_idx"]>=0],float)
    sc=StandardScaler().fit(X);clf=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=2500,random_state=40)
    clf.fit(sc.transform(X),y,sample_weight=w);return clf,sc

def score_all(ids,cache,clf,sc,alpha):
    rows=[]
    for rid in ids:
        d=cache[rid];kp=clf.predict_proba(sc.transform(d["X"]))[:,1]
        logs=alpha*d["normal_log"]+(1-alpha)*np.log(np.maximum(kp,1e-12))
        order=np.argsort(-logs,kind="stable");m=float(logs[order[0]])
        weights=np.exp(logs-m);gate=float(weights[order[0]]/weights.sum())
        arank=int(np.where(order==d["actual_idx"])[0][0]+1) if d["actual_idx"]>=0 else 999
        rows.append({**d,"order":order,"gate":gate,"actual_rank":arank})
    return rows

def thresholds(rows):
    vals=[r["gate"] for r in rows]
    return sorted({0.0,*[float(np.quantile(vals,q)) for q in (0,.15,.25,.35,.5,.65,.8)]})

def eval_rows(rows,k,th,cap=True):
    stake=ret=buy=hits=kings=0;logs=[]
    for r in rows:
        b=r["gate"]>=th;hit=b and r["actual_rank"]<=k;st=k*UNIT if b else 0
        rr=(min(r["payout"],REWARD_CAP) if cap else r["payout"]) if hit else 0
        stake+=st;ret+=rr;buy+=int(b);hits+=int(hit);kings+=int(hit and r["payout"]>=KING_YEN)
        logs.append((b,hit,r["payout"],st,rr))
    return {"stake":stake,"ret":ret,"profit":ret-stake,"buy":buy,"hits":hits,"kings":kings,"logs":logs}

def choose_gate(rows,k):
    best=None
    for th in thresholds(rows):
        m=eval_rows(rows,k,th,True)
        if m["buy"]<max(4,int(.08*len(rows))):continue
        key=(m["profit"],m["kings"],m["hits"],-m["buy"])
        if best is None or key>best[0]:best=(key,th)
    return best[1] if best else 0.0

def cv_select(cache):
    ids=sorted(cache);days=sorted({cache[r]["date"] for r in ids});grid=[]
    for C in (.05,.1,.25,.5,1.0):
        fold_models={}
        for day in days:
            tr=[r for r in ids if cache[r]["date"]!=day];va=[r for r in ids if cache[r]["date"]==day]
            fold_models[day]=(tr,va,*fit(tr,cache,C))
        for alpha in (.15,.30,.45,.60,.75):
            scored={}
            for day,(tr,va,clf,sc) in fold_models.items():
                scored[day]=(score_all(tr,cache,clf,sc,alpha),score_all(va,cache,clf,sc,alpha))
            for k in (4,6,8,10,12,16):
                all_logs=[]
                for day,(trrows,varows) in scored.items():
                    th=choose_gate(trrows,k);all_logs.extend(eval_rows(varows,k,th,True)["logs"])
                stake=sum(x[3] for x in all_logs);ret=sum(x[4] for x in all_logs);buy=sum(x[0] for x in all_logs);hits=sum(x[1] for x in all_logs);kings=sum(x[1] and x[2]>=KING_YEN for x in all_logs)
                exp=buy*k/210.0;lift=hits/exp if exp else 0
                grid.append({"C":C,"alpha":alpha,"k":k,"cv_stake":stake,"cv_return_capped":ret,"cv_profit_capped":ret-stake,"cv_buy":buy,"cv_hits":hits,"cv_king_hits":kings,"random_expected_hits":exp,"hit_lift_vs_random":lift})
    grid.sort(key=lambda x:(x["cv_profit_capped"],x["cv_king_hits"],x["hit_lift_vs_random"],-x["k"]),reverse=True);return grid

def final(train,test,best):
    ids=sorted(train);clf,sc=fit(ids,train,best["C"]);trrows=score_all(ids,train,clf,sc,best["alpha"]);th=choose_gate(trrows,best["k"])
    rows=score_all(sorted(test),test,clf,sc,best["alpha"]);k=best["k"];logs=[]
    for r in rows:
        buy=r["gate"]>=th;hit=buy and r["actual_rank"]<=k;stake=k*UNIT if buy else 0;ret=r["payout"] if hit else 0
        verdict="SKIP" if not buy else ("KING" if hit and r["payout"]>=KING_YEN else ("IMPUDENT" if hit else "BEHEADED"))
        seats=[list(r["combos"][int(i)]) for i in r["order"][:k]]
        logs.append({"race_id":r["rid"],"date":r["date"],"track":r["track"],"race_no":r["race_no"],"actual":list(r["actual"]) if r["actual"] else None,"payout_yen":r["payout"],"buy":buy,"verdict":verdict,"hit":hit,"actual_rank":r["actual_rank"],"seat_count":k,"stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,"gate_score":r["gate"],"gate_threshold":th,"seats":seats})
    bought=[x for x in logs if x["buy"]];hits=[x for x in logs if x["hit"]];stake=sum(x["stake_yen"] for x in logs);ret=sum(x["return_yen"] for x in logs)
    largest=max((x["return_yen"] for x in hits),default=0);capped=sum(min(x["return_yen"],REWARD_CAP) for x in hits);expected=len(bought)*k/210.0
    summary={"algorithm":"v40_last_chance_king","train_period":TRAIN,"holdout_period":TEST,"holdout_used_for_tuning":False,"exact_seats_only":True,"cartesian_board_betting":False,"max_seats":16,
      "selected":{"C":best["C"],"alpha":best["alpha"],"seat_k":k,"gate_threshold":th},"races":len(logs),"buy_races":len(bought),"skip_races":len(logs)-len(bought),"buy_rate":len(bought)/len(logs) if logs else 0,
      "verdict_counts":dict(Counter(x["verdict"] for x in logs)),"hits":len(hits),"hit_rate_on_bought":len(hits)/len(bought) if bought else 0,
      "random_expected_hits_at_same_ticket_count":expected,"hit_lift_vs_random":len(hits)/expected if expected else 0,"actual_king_races":sum(x["payout_yen"]>=KING_YEN for x in logs),
      "king_hits":sum(x["verdict"]=="KING" for x in logs),"impudent_hits":sum(x["verdict"]=="IMPUDENT" for x in logs),"beheaded":sum(x["verdict"]=="BEHEADED" for x in logs),
      "stake_yen":stake,"payout_yen":ret,"profit_yen":ret-stake,"roi":ret/stake if stake else 0,"reward_cap_yen_for_selection":REWARD_CAP,
      "capped_payout_yen":capped,"capped_profit_yen":capped-stake,"capped_roi":capped/stake if stake else 0,"largest_hit_yen":largest,
      "payout_without_largest_hit_yen":ret-largest,"profit_without_largest_hit_yen":ret-largest-stake,"roi_without_largest_hit":(ret-largest)/stake if stake else 0,
      "king_details":[x for x in logs if x["verdict"]=="KING"],"impudent_details":[x for x in logs if x["verdict"]=="IMPUDENT"]}
    model={"model":"v40_last_chance_king_logit","feature_names":v38.FEATURES,"scaler_mean":[float(x) for x in sc.mean_],"scaler_scale":[float(x) for x in sc.scale_],"coef":[float(x) for x in clf.coef_[0]],"intercept":float(clf.intercept_[0]),"C":best["C"],"alpha":best["alpha"],"seat_k":k,"gate_threshold":th,"king_yen":KING_YEN,"reward_cap":REWARD_CAP,"guards":{"exact_seats_only":True,"max_seats":16,"cartesian_betting":False,"target_odds_used":False,"holdout_used_for_tuning":False}}
    return summary,logs,model

def main():
    tt,te,tr,tp=load_range(*TRAIN);xt,xe,xr,xp=load_range(*TEST)
    tbase=build_base(tt,te);xbase=build_base(xt,xe);train=make_cache(tt,te,tr,tp,tbase);test=make_cache(xt,xe,xr,xp,xbase)
    grid=cv_select(train);best=grid[0];summary,logs,model=final(train,test,best);summary["cv_selected_from_week1"]=best;summary["cv_top10"]=grid[:10]
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");(OUT/"holdout_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8");(OUT/"model.json").write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

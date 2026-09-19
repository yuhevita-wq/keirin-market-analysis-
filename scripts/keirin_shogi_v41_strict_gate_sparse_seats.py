#!/usr/bin/env python3
from __future__ import annotations
"""
v41 KING: strict race gate + sparse exact seats.

Hard fixes:
1) participation first: race gate must target KING-race likelihood and final W1-selected
   threshold is constrained to 10%-30% buy rate.
2) exact seats second: only 3/4/5/6/8 exact trifecta seats are eligible. No Cartesian expansion.

Training/tuning: 2024-01-01..01-07 only, leave-one-day-out CV.
Untouched holdout: 2024-01-08..01-14.
Odds/popularity are never target features. Payout is label/reward only.
"""
import csv, importlib.util, itertools, json, math, re, sys, unicodedata
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/2024/s_class_f1_all_parts/2024_q1"
OUT=ROOT/"results/keirin_shogi/v41_strict_gate_sparse_seats"; OUT.mkdir(parents=True,exist_ok=True)
TRAIN=("2024-01-01","2024-01-07"); TEST=("2024-01-08","2024-01-14")
KING_YEN=10000; REWARD_CAP=50000; UNIT=100
MIN_BUY_RATE=0.10; MAX_BUY_RATE=0.30

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); assert spec.loader is not None
    sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

runtime=load_module("v41_runtime",ROOT/"scripts/keirin_shogi_v37_auto_place_runtime.py")
v38=load_module("v41_features",ROOT/"scripts/keirin_shogi_v38_king_seat_2024w1.py")

class PassKing:
    @staticmethod
    def predict_king_seat(placed,race): return placed

def read(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def ino(v):
    try:return int(float(v))
    except:return 0
def norm(v): return unicodedata.normalize("NFKC",str(v or "")).strip()
def triple(v):
    xs=[int(x) for x in re.findall(r"[1-9]",norm(v))]
    return tuple(xs[:3]) if len(xs)>=3 and len(set(xs[:3]))==3 else None

RACES=read(BASE/"races.csv"); ENTRIES=read(BASE/"entries.csv"); RESULTS=read(BASE/"results.csv"); PAYOUTS=read(BASE/"payouts.csv")

def load_range(start,end):
    t={r["race_id"]:r for r in RACES if start<=r.get("race_date","")<=end and ino(r.get("entry_count"))==7 and r.get("meeting_grade")=="F1" and "Ｓ級" in r.get("race_type","")}
    eb=defaultdict(list);rb=defaultdict(list);paid=defaultdict(dict)
    for e in ENTRIES:
        if e.get("race_id") in t: eb[e["race_id"]].append(e)
    for x in RESULTS:
        if x.get("race_id") in t: rb[x["race_id"]].append(x)
    for p in PAYOUTS:
        rid=str(p.get("race_id",""))
        if rid not in t or norm(p.get("ticket_type")) not in {"3連単","三連単"}: continue
        if norm(p.get("status")).lower() not in {"","paid"}: continue
        z=triple(p.get("combination")); y=re.sub(r"[^0-9]","",norm(p.get("payout_yen")))
        if z and y: paid[rid][z]=int(y)
    return t,eb,rb,paid

def actual_order(rows):
    top={}
    for x in rows:
        p=ino(x.get("finish_position"))
        if p in (1,2,3): top[p]=ino(x.get("car_no"))
    return (top[1],top[2],top[3]) if len(top)==3 else None

def build_base(t,eb):
    e=runtime.load_base_engine(); n=runtime.load_ninecar_engine(); o=runtime.load_sevencar_overlay_engine()
    s=e.build_v21_state(); pair=e.build_pair_model(); third,fn,freeze=e.build_third_model(); out={}
    for rid,r in sorted(t.items(),key=lambda kv:(kv[1].get("race_date",""),kv[1].get("track",""),ino(kv[1].get("race_no")))):
        ers=sorted(eb[rid],key=lambda x:ino(x.get("car_no")))
        live={"race_id":rid,"race_date":r.get("race_date",""),"track":r.get("track",""),"race_no":r.get("race_no",""),"race_type":r.get("race_type",""),"meeting_grade":r.get("meeting_grade",""),"entries":[runtime.normalize_entry(x) for x in ers]}
        p=runtime.runtime_place_one(e,n,o,PassKing(),live,s,pair,third,fn,freeze)
        if p.get("board_generated"): out[rid]=p
    return out

def normal_logs(combos,p):
    fr={int(r["no"]):float(r["probability"]) for r in p["first_ranking"]}
    sr={int(r["no"]):float(r["mass"])/2.0 for r in p["second_membership"]}
    tr={int(r["no"]):float(r["probability"]) for r in p["third_ranking"]}
    return np.asarray([math.log(max(fr[a],1e-12))+math.log(max(sr[b],1e-12))+math.log(max(tr[c],1e-12)) for a,b,c in combos],float)

def make_cache(t,eb,rb,paid,base):
    out={}
    for rid,p in base.items():
        entries=sorted(eb[rid],key=lambda x:ino(x.get("car_no"))); cars=sorted(ino(x["car_no"]) for x in entries)
        combos=list(itertools.permutations(cars,3)); X=np.asarray([v38.feature_vector(c,p,entries) for c in combos],float)
        actual=actual_order(rb[rid]); idx=combos.index(actual) if actual in combos else -1; payout=paid[rid].get(actual,0) if actual else 0
        out[rid]={"rid":rid,"date":t[rid]["race_date"],"track":t[rid].get("track"),"race_no":ino(t[rid].get("race_no")),"combos":combos,"X":X,"normal_log":normal_logs(combos,p),"actual":actual,"actual_idx":idx,"payout":payout}
    return out

def fit_combo(ids,cache,C):
    good=[r for r in ids if cache[r]["actual_idx"]>=0]
    X=np.asarray([cache[r]["X"][cache[r]["actual_idx"]] for r in good],float)
    y=np.asarray([int(cache[r]["payout"]>=KING_YEN) for r in good],int)
    w=np.asarray([1.0 if cache[r]["payout"]<KING_YEN else 1.0+min(math.log1p(min(cache[r]["payout"],REWARD_CAP)/KING_YEN),1.5) for r in good],float)
    sc=StandardScaler().fit(X)
    clf=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=2500,random_state=41)
    clf.fit(sc.transform(X),y,sample_weight=w); return clf,sc

def softmax(x):
    m=float(np.max(x)); z=np.exp(x-m); return z/(z.sum() or 1.0)

def score_rows(ids,cache,clf,sc,alpha):
    rows=[]
    for rid in ids:
        d=cache[rid]
        kp=clf.predict_proba(sc.transform(d["X"]))[:,1]
        combo_log=alpha*d["normal_log"]+(1-alpha)*np.log(np.maximum(kp,1e-12))
        combo_p=softmax(combo_log); order=np.argsort(-combo_p,kind="stable")
        normal_p=softmax(d["normal_log"])
        arank=int(np.where(order==d["actual_idx"])[0][0]+1) if d["actual_idx"]>=0 else 999
        sk=np.sort(kp)[::-1]
        sp=np.sort(combo_p)[::-1]
        sn=np.sort(normal_p)[::-1]
        gate_features=np.asarray([
            sk[0], sk[:3].mean(), sk[:8].mean(), sk.std(), sk[0]-sk[1], sk[0]-sk[7],
            sp[0], sp[:3].sum(), sp[:5].sum(), sp[:8].sum(), sp[0]-sp[1], sp[0]-sp[7],
            sn[0], sn[:3].sum(), sn[:8].sum(),
            -float(np.sum(combo_p*np.log(np.maximum(combo_p,1e-12)))),
            -float(np.sum(normal_p*np.log(np.maximum(normal_p,1e-12)))),
        ],float)
        rows.append({**d,"kp":kp,"combo_p":combo_p,"order":order,"actual_rank":arank,"gate_features":gate_features})
    return rows

def fit_gate(rows,C):
    X=np.asarray([r["gate_features"] for r in rows],float)
    y=np.asarray([int(r["payout"]>=KING_YEN) for r in rows],int)
    sc=StandardScaler().fit(X)
    clf=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=2500,random_state=141)
    clf.fit(sc.transform(X),y); return clf,sc

def apply_gate(rows,clf,sc):
    X=np.asarray([r["gate_features"] for r in rows],float)
    ps=clf.predict_proba(sc.transform(X))[:,1]
    out=[]
    for r,p in zip(rows,ps):
        z=dict(r); z["gate_prob"]=float(p); out.append(z)
    return out

def threshold_candidates(rows):
    vals=np.asarray([r["gate_prob"] for r in rows],float)
    qs=np.linspace(0.70,0.90,9)
    return sorted({float(np.quantile(vals,q)) for q in qs})

def eval_rows(rows,k,th,cap=True):
    logs=[]; stake=ret=0
    for r in rows:
        buy=r["gate_prob"]>=th
        hit=buy and r["actual_rank"]<=k
        st=k*UNIT if buy else 0
        rr=(min(r["payout"],REWARD_CAP) if cap else r["payout"]) if hit else 0
        stake+=st; ret+=rr
        logs.append({"buy":buy,"hit":hit,"payout":r["payout"],"stake":st,"return":rr})
    buy=sum(x["buy"] for x in logs); hits=sum(x["hit"] for x in logs); kings=sum(x["hit"] and x["payout"]>=KING_YEN for x in logs)
    return {"stake":stake,"ret":ret,"profit":ret-stake,"buy":buy,"hits":hits,"kings":kings,"logs":logs}

def choose_threshold(rows,k):
    n=len(rows); best=None
    for th in threshold_candidates(rows):
        m=eval_rows(rows,k,th,True); br=m["buy"]/n if n else 0
        if not (MIN_BUY_RATE<=br<=MAX_BUY_RATE): continue
        # Prefer capped profit, then KING count, then fewer buys.
        key=(m["profit"],m["kings"],m["hits"],-m["buy"])
        if best is None or key>best[0]: best=(key,th,m)
    if best:return best[1]
    # deterministic fallback around 20% buy rate
    vals=sorted([r["gate_prob"] for r in rows],reverse=True)
    idx=max(0,min(len(vals)-1,int(round(.20*len(vals)))-1))
    return vals[idx] if vals else 1.0

def cv_select(cache):
    ids=sorted(cache); days=sorted({cache[r]["date"] for r in ids}); grid=[]
    for Cc in (.05,.1,.25,.5):
      for alpha in (.25,.45,.65):
       for Cg in (.05,.1,.25,.5):
        fold_data={}
        for day in days:
            tr=[r for r in ids if cache[r]["date"]!=day]; va=[r for r in ids if cache[r]["date"]==day]
            cclf,csc=fit_combo(tr,cache,Cc)
            trrows=score_rows(tr,cache,cclf,csc,alpha); varows=score_rows(va,cache,cclf,csc,alpha)
            gclf,gsc=fit_gate(trrows,Cg)
            fold_data[day]=(apply_gate(trrows,gclf,gsc),apply_gate(varows,gclf,gsc))
        for k in (3,4,5,6,8):
            logs=[]
            for day,(trrows,varows) in fold_data.items():
                th=choose_threshold(trrows,k)
                logs.extend(eval_rows(varows,k,th,True)["logs"])
            stake=sum(x["stake"] for x in logs); ret=sum(x["return"] for x in logs)
            buy=sum(x["buy"] for x in logs); hits=sum(x["hit"] for x in logs); kings=sum(x["hit"] and x["payout"]>=KING_YEN for x in logs)
            br=buy/len(ids) if ids else 0; exp=buy*k/210.0; lift=hits/exp if exp else 0
            # Hard preference for sparse gate and sparse seats.
            score=(ret-stake) + 2000*kings + 500*hits - 15000*max(0,br-MAX_BUY_RATE) - 250*k
            grid.append({"combo_C":Cc,"alpha":alpha,"gate_C":Cg,"k":k,"cv_stake":stake,"cv_return_capped":ret,"cv_profit_capped":ret-stake,"cv_buy":buy,"cv_buy_rate":br,"cv_hits":hits,"cv_king_hits":kings,"random_expected_hits":exp,"hit_lift_vs_random":lift,"score":score})
    grid.sort(key=lambda x:(x["score"],x["cv_profit_capped"],x["cv_king_hits"],x["hit_lift_vs_random"],-x["k"]),reverse=True)
    return grid

def final_eval(train,test,best):
    ids=sorted(train)
    cclf,csc=fit_combo(ids,train,best["combo_C"])
    trrows=score_rows(ids,train,cclf,csc,best["alpha"])
    gclf,gsc=fit_gate(trrows,best["gate_C"])
    trrows=apply_gate(trrows,gclf,gsc)
    th=choose_threshold(trrows,best["k"])
    xrows=score_rows(sorted(test),test,cclf,csc,best["alpha"])
    xrows=apply_gate(xrows,gclf,gsc)
    k=best["k"]; logs=[]
    for r in xrows:
        buy=r["gate_prob"]>=th; hit=buy and r["actual_rank"]<=k
        st=k*UNIT if buy else 0; ret=r["payout"] if hit else 0
        verdict="SKIP" if not buy else ("KING" if hit and r["payout"]>=KING_YEN else ("IMPUDENT" if hit else "BEHEADED"))
        seats=[list(r["combos"][int(i)]) for i in r["order"][:k]]
        logs.append({"race_id":r["rid"],"date":r["date"],"track":r["track"],"race_no":r["race_no"],"actual":list(r["actual"]) if r["actual"] else None,"payout_yen":r["payout"],"buy":buy,"verdict":verdict,"hit":hit,"actual_rank":r["actual_rank"],"seat_count":k,"stake_yen":st,"return_yen":ret,"profit_yen":ret-st,"gate_prob":r["gate_prob"],"gate_threshold":th,"seats":seats})
    bought=[x for x in logs if x["buy"]]; hits=[x for x in logs if x["hit"]]
    stake=sum(x["stake_yen"] for x in logs); ret=sum(x["return_yen"] for x in logs); largest=max((x["return_yen"] for x in hits),default=0)
    capped=sum(min(x["return_yen"],REWARD_CAP) for x in hits); exp=len(bought)*k/210.0
    summary={
      "algorithm":"v41_strict_gate_sparse_seats","train_period":TRAIN,"holdout_period":TEST,"holdout_used_for_tuning":False,
      "hard_constraints":{"buy_rate_train_min":MIN_BUY_RATE,"buy_rate_train_max":MAX_BUY_RATE,"max_exact_seats":8,"cartesian_betting":False},
      "selected":{"combo_C":best["combo_C"],"alpha":best["alpha"],"gate_C":best["gate_C"],"seat_k":k,"gate_threshold":th},
      "races":len(logs),"buy_races":len(bought),"skip_races":len(logs)-len(bought),"buy_rate":len(bought)/len(logs) if logs else 0,
      "verdict_counts":dict(Counter(x["verdict"] for x in logs)),"hits":len(hits),"hit_rate_on_bought":len(hits)/len(bought) if bought else 0,
      "random_expected_hits_at_same_ticket_count":exp,"hit_lift_vs_random":len(hits)/exp if exp else 0,
      "actual_king_races":sum(x["payout_yen"]>=KING_YEN for x in logs),"king_hits":sum(x["verdict"]=="KING" for x in logs),"impudent_hits":sum(x["verdict"]=="IMPUDENT" for x in logs),"beheaded":sum(x["verdict"]=="BEHEADED" for x in logs),
      "stake_yen":stake,"payout_yen":ret,"profit_yen":ret-stake,"roi":ret/stake if stake else 0,
      "capped_payout_yen":capped,"capped_profit_yen":capped-stake,"capped_roi":capped/stake if stake else 0,
      "largest_hit_yen":largest,"payout_without_largest_hit_yen":ret-largest,"profit_without_largest_hit_yen":ret-largest-stake,"roi_without_largest_hit":(ret-largest)/stake if stake else 0,
      "king_details":[x for x in logs if x["verdict"]=="KING"],"impudent_details":[x for x in logs if x["verdict"]=="IMPUDENT"]
    }
    model={"model":"v41_strict_gate_sparse_seats","feature_names":v38.FEATURES,"combo_scaler_mean":[float(x) for x in csc.mean_],"combo_scaler_scale":[float(x) for x in csc.scale_],"combo_coef":[float(x) for x in cclf.coef_[0]],"combo_intercept":float(cclf.intercept_[0]),"combo_C":best["combo_C"],"alpha":best["alpha"],"gate_scaler_mean":[float(x) for x in gsc.mean_],"gate_scaler_scale":[float(x) for x in gsc.scale_],"gate_coef":[float(x) for x in gclf.coef_[0]],"gate_intercept":float(gclf.intercept_[0]),"gate_C":best["gate_C"],"seat_k":k,"gate_threshold":th,"guards":{"max_exact_seats":8,"cartesian_betting":False,"target_odds_used":False,"holdout_used_for_tuning":False}}
    return summary,logs,model

def main():
    tt,te,tr,tp=load_range(*TRAIN); xt,xe,xr,xp=load_range(*TEST)
    tbase=build_base(tt,te); xbase=build_base(xt,xe)
    train=make_cache(tt,te,tr,tp,tbase); test=make_cache(xt,xe,xr,xp,xbase)
    grid=cv_select(train); best=grid[0]
    summary,logs,model=final_eval(train,test,best)
    summary["cv_selected_from_week1"]=best; summary["cv_top10"]=grid[:10]
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"holdout_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"model.json").write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

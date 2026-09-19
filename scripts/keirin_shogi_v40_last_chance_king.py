#!/usr/bin/env python3
from __future__ import annotations

"""
v40 LAST CHANCE KING

Design rules fixed before holdout:
- learn KING-shape only from realized outcome triples: KING actual outcomes vs non-KING actual outcomes
- never train the KING classifier on the 209 losing counterfactual triples in each race
- combine structural KING-likeness with frozen normal placement likelihood
- exact trifecta seats only; NEVER buy the Cartesian product of the projected board
- <=16 exact seats/race
- gate is selected only inside 2024-W1 by leave-one-day-out CV
- payout reward is capped at 50,000 yen during model/policy selection so one monster hit cannot crown the model
- final untouched test is 2024-01-08..2024-01-14
- target-race odds/popularity are never features
"""

import csv, importlib.util, itertools, json, math, re, sys, unicodedata
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/2024/s_class_f1_all_parts/2024_q1"
OUT=ROOT/"results/keirin_shogi/v40_last_chance_king"
OUT.mkdir(parents=True,exist_ok=True)
TRAIN=("2024-01-01","2024-01-07")
TEST=("2024-01-08","2024-01-14")
KING_YEN=10000
REWARD_CAP=50000
UNIT=100

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); assert spec.loader is not None
    sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

runtime=load_module("v40_runtime",ROOT/"scripts/keirin_shogi_v37_auto_place_runtime.py")
v38=load_module("v40_features",ROOT/"scripts/keirin_shogi_v38_king_seat_2024w1.py")

class PassKing:
    @staticmethod
    def predict_king_seat(placed,race):
        return placed

def read(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def ino(v):
    try:return int(float(v))
    except:return 0
def norm(v):return unicodedata.normalize("NFKC",str(v or "")).strip()
def triple(v):
    xs=[int(x) for x in re.findall(r"[1-9]",norm(v))]
    if len(xs)>=3 and len(set(xs[:3]))==3:return tuple(xs[:3])
    return None

RACES=read(BASE/"races.csv"); ENTRIES=read(BASE/"entries.csv"); RESULTS=read(BASE/"results.csv"); PAYOUTS=read(BASE/"payouts.csv")

def load_range(start,end):
    targets={r["race_id"]:r for r in RACES if start<=r.get("race_date","")<=end and ino(r.get("entry_count"))==7 and r.get("meeting_grade")=="F1" and "Ｓ級" in r.get("race_type","")}
    eb=defaultdict(list);rb=defaultdict(list);paid=defaultdict(dict)
    for e in ENTRIES:
        if e.get("race_id") in targets:eb[e["race_id"]].append(e)
    for x in RESULTS:
        if x.get("race_id") in targets:rb[x["race_id"]].append(x)
    for p in PAYOUTS:
        rid=str(p.get("race_id",""))
        if rid not in targets or norm(p.get("ticket_type")) not in {"3連単","三連単"}:continue
        if norm(p.get("status")).lower() not in {"","paid"}:continue
        t=triple(p.get("combination")); y=re.sub(r"[^0-9]","",norm(p.get("payout_yen")))
        if t and y:paid[rid][t]=int(y)
    return targets,eb,rb,paid

def actual_order(rows):
    top={}
    for x in rows:
        p=ino(x.get("finish_position"))
        if p in (1,2,3):top[p]=ino(x.get("car_no"))
    return (top.get(1),top.get(2),top.get(3)) if len(top)==3 else None

def build_base(targets,eb):
    engine=runtime.load_base_engine();nine=runtime.load_ninecar_engine();overlay=runtime.load_sevencar_overlay_engine()
    v21=engine.build_v21_state();pair=engine.build_pair_model();third,features,freeze=engine.build_third_model()
    out={}
    for rid,r in sorted(targets.items(),key=lambda kv:(kv[1].get("race_date",""),kv[1].get("track",""),ino(kv[1].get("race_no")))):
        ers=sorted(eb[rid],key=lambda x:ino(x.get("car_no")))
        live={"race_id":rid,"race_date":r.get("race_date",""),"track":r.get("track",""),"race_no":r.get("race_no",""),"race_type":r.get("race_type",""),"meeting_grade":r.get("meeting_grade",""),"entries":[runtime.normalize_entry(e) for e in ers]}
        p=runtime.runtime_place_one(engine,nine,overlay,PassKing(),live,v21,pair,third,features,freeze)
        if p.get("board_generated"):out[rid]=p
    return out

def normal_prob(combo,placed):
    a,b,c=combo
    fr={int(r["no"]):float(r["probability"]) for r in placed["first_ranking"]}
    sr={int(r["no"]):float(r["mass"]) for r in placed["second_membership"]}
    tr={int(r["no"]):float(r["probability"]) for r in placed["third_ranking"]}
    return max(fr[a],1e-9)*max(sr[b]/2.0,1e-9)*max(tr[c],1e-9)

def realized_dataset(ids,targets,eb,rb,paid,base):
    X=[];y=[];weights=[];meta=[]
    for rid in ids:
        actual=actual_order(rb[rid])
        if not actual or rid not in base:continue
        payout=paid[rid].get(actual,0)
        entries=sorted(eb[rid],key=lambda e:ino(e.get("car_no")))
        X.append(v38.feature_vector(actual,base[rid],entries))
        isking=int(payout>=KING_YEN);y.append(isking)
        # modest, capped reward emphasis; no monster payout can dominate
        w=1.0 if not isking else 1.0+min(math.log1p(min(payout,REWARD_CAP)/KING_YEN),1.5)
        weights.append(w);meta.append((rid,actual,payout))
    return np.asarray(X,float),np.asarray(y,int),np.asarray(weights,float),meta

def fit_king(ids,targets,eb,rb,paid,base,C):
    X,y,w,meta=realized_dataset(ids,targets,eb,rb,paid,base)
    sc=StandardScaler().fit(X)
    clf=LogisticRegression(C=C,class_weight="balanced",solver="liblinear",max_iter=2500,random_state=40)
    clf.fit(sc.transform(X),y,sample_weight=w)
    return clf,sc

def score_race(rid,targets,eb,rb,paid,base,clf,sc,alpha):
    placed=base[rid]; entries=sorted(eb[rid],key=lambda e:ino(e.get("car_no")))
    cars=sorted(ino(e["car_no"]) for e in entries)
    combos=list(itertools.permutations(cars,3))
    X=np.asarray([v38.feature_vector(c,placed,entries) for c in combos],float)
    kp=clf.predict_proba(sc.transform(X))[:,1]
    raw=[]
    for combo,kprob in zip(combos,kp):
        npb=normal_prob(combo,placed)
        logscore=alpha*math.log(npb)+(1-alpha)*math.log(max(float(kprob),1e-9))
        raw.append((combo,logscore,float(kprob),npb))
    raw.sort(key=lambda z:(-z[1],z[0]))
    # convert for stable cross-race gate comparison
    m=max(z[1] for z in raw)
    ex=[math.exp(z[1]-m) for z in raw]; denom=sum(ex) or 1.0
    ranked=[]
    for z,e in zip(raw,ex):
        ranked.append({"combo":z[0],"score":e/denom,"king_prob":z[2],"normal_prob":z[3]})
    actual=actual_order(rb[rid]); payout=paid[rid].get(actual,0) if actual else 0
    return {"race_id":rid,"date":targets[rid]["race_date"],"track":targets[rid].get("track"),"race_no":ino(targets[rid].get("race_no")),"actual":actual,"payout":payout,"ranked":ranked}

def threshold_candidates(rows):
    vals=sorted({float(r["ranked"][0]["score"]) for r in rows})
    if not vals:return [0.0]
    qs=[0.0,0.15,0.25,0.35,0.50,0.65,0.80]
    out={0.0}
    for q in qs:
        out.add(float(np.quantile(vals,q)))
    return sorted(out)

def eval_rows(rows,k,threshold,cap=True):
    logs=[];stake=ret=0
    for r in rows:
        buy=float(r["ranked"][0]["score"])>=threshold
        seats=[tuple(x["combo"]) for x in r["ranked"][:k]]
        hit=bool(buy and r["actual"] in seats)
        st=k*UNIT if buy else 0
        rr=(min(r["payout"],REWARD_CAP) if cap else r["payout"]) if hit else 0
        stake+=st;ret+=rr
        logs.append({"buy":buy,"hit":hit,"payout":r["payout"],"stake":st,"return":rr})
    return {"stake":stake,"return":ret,"profit":ret-stake,"buy_count":sum(x["buy"] for x in logs),"hits":sum(x["hit"] for x in logs),"king_hits":sum(x["hit"] and x["payout"]>=KING_YEN for x in logs),"logs":logs}

def choose_gate(train_rows,k):
    best=None
    for th in threshold_candidates(train_rows):
        m=eval_rows(train_rows,k,th,cap=True)
        if m["buy_count"]<max(4,int(.08*len(train_rows))):continue
        key=(m["profit"],m["king_hits"],m["hits"],-m["buy_count"])
        if best is None or key>best[0]:best=(key,th,m)
    return best[1] if best else 0.0

def cv_select(targets,eb,rb,paid,base):
    ids=sorted(base)
    days=sorted({targets[r]["race_date"] for r in ids})
    grid=[]
    for C in (0.05,0.1,0.25,0.5,1.0):
      for alpha in (0.15,0.30,0.45,0.60,0.75):
       for k in (4,6,8,10,12,16):
        fold_logs=[]
        for day in days:
            tr=[rid for rid in ids if targets[rid]["race_date"]!=day]
            va=[rid for rid in ids if targets[rid]["race_date"]==day]
            if not tr or not va:continue
            clf,sc=fit_king(tr,targets,eb,rb,paid,base,C)
            trrows=[score_race(rid,targets,eb,rb,paid,base,clf,sc,alpha) for rid in tr]
            th=choose_gate(trrows,k)
            varows=[score_race(rid,targets,eb,rb,paid,base,clf,sc,alpha) for rid in va]
            m=eval_rows(varows,k,th,cap=True)
            fold_logs.extend(m["logs"])
        stake=sum(x["stake"] for x in fold_logs);ret=sum(x["return"] for x in fold_logs)
        buy=sum(x["buy"] for x in fold_logs);hits=sum(x["hit"] for x in fold_logs)
        kings=sum(x["hit"] and x["payout"]>=KING_YEN for x in fold_logs)
        expected=buy*k/210.0
        lift=hits/expected if expected else 0.0
        # robust selection: capped money first, then KING count, then lift, then fewer seats
        grid.append({"C":C,"alpha":alpha,"k":k,"cv_stake":stake,"cv_return_capped":ret,"cv_profit_capped":ret-stake,"cv_buy":buy,"cv_hits":hits,"cv_king_hits":kings,"random_expected_hits":expected,"hit_lift_vs_random":lift})
    grid.sort(key=lambda x:(x["cv_profit_capped"],x["cv_king_hits"],x["hit_lift_vs_random"],-x["k"]),reverse=True)
    return grid

def final_eval(train_pack,test_pack,params):
    tt,te,tr,tp,tbase=train_pack
    xt,xe,xr,xp,xbase=test_pack
    ids=sorted(tbase)
    clf,sc=fit_king(ids,tt,te,tr,tp,tbase,params["C"])
    train_rows=[score_race(r,tt,te,tr,tp,tbase,clf,sc,params["alpha"]) for r in ids]
    threshold=choose_gate(train_rows,params["k"])
    test_rows=[score_race(r,xt,xe,xr,xp,xbase,clf,sc,params["alpha"]) for r in sorted(xbase)]
    k=params["k"]
    logs=[]
    for r in test_rows:
        buy=float(r["ranked"][0]["score"])>=threshold
        seats=[tuple(x["combo"]) for x in r["ranked"][:k]]
        hit=bool(buy and r["actual"] in seats)
        stake=k*UNIT if buy else 0
        ret=r["payout"] if hit else 0
        if not buy:verdict="SKIP"
        elif hit and r["payout"]>=KING_YEN:verdict="KING"
        elif hit:verdict="IMPUDENT"
        else:verdict="BEHEADED"
        logs.append({"race_id":r["race_id"],"date":r["date"],"track":r["track"],"race_no":r["race_no"],"actual":list(r["actual"]) if r["actual"] else None,"payout_yen":r["payout"],"buy":buy,"verdict":verdict,"hit":hit,"seat_count":k,"stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,"gate_score":float(r["ranked"][0]["score"]),"gate_threshold":threshold,"seats":[list(x) for x in seats]})
    bought=[x for x in logs if x["buy"]]
    hits=[x for x in logs if x["hit"]]
    stake=sum(x["stake_yen"] for x in logs);ret=sum(x["return_yen"] for x in logs)
    top=max((x["return_yen"] for x in hits),default=0)
    ret_without_top=ret-top
    capped_ret=sum(min(x["return_yen"],REWARD_CAP) for x in hits)
    expected=len(bought)*k/210.0
    actual_kings=sum(x["payout_yen"]>=KING_YEN for x in logs)
    summary={
      "algorithm":"v40_last_chance_king",
      "train_period":TRAIN,"holdout_period":TEST,"holdout_used_for_tuning":False,
      "exact_seats_only":True,"cartesian_board_betting":False,"max_seats":16,
      "selected":{"C":params["C"],"alpha":params["alpha"],"seat_k":k,"gate_threshold":threshold},
      "races":len(logs),"buy_races":len(bought),"skip_races":len(logs)-len(bought),"buy_rate":len(bought)/len(logs) if logs else 0,
      "verdict_counts":dict(Counter(x["verdict"] for x in logs)),
      "hits":len(hits),"hit_rate_on_bought":len(hits)/len(bought) if bought else 0,
      "random_expected_hits_at_same_ticket_count":expected,"hit_lift_vs_random":len(hits)/expected if expected else 0,
      "actual_king_races":actual_kings,
      "king_hits":sum(x["verdict"]=="KING" for x in logs),
      "impudent_hits":sum(x["verdict"]=="IMPUDENT" for x in logs),
      "beheaded":sum(x["verdict"]=="BEHEADED" for x in logs),
      "stake_yen":stake,"payout_yen":ret,"profit_yen":ret-stake,"roi":ret/stake if stake else 0,
      "reward_cap_yen_for_selection":REWARD_CAP,
      "capped_payout_yen":capped_ret,"capped_profit_yen":capped_ret-stake,"capped_roi":capped_ret/stake if stake else 0,
      "largest_hit_yen":top,
      "payout_without_largest_hit_yen":ret_without_top,
      "profit_without_largest_hit_yen":ret_without_top-stake,
      "roi_without_largest_hit":ret_without_top/stake if stake else 0,
      "king_details":[x for x in logs if x["verdict"]=="KING"],
      "impudent_details":[x for x in logs if x["verdict"]=="IMPUDENT"],
    }
    return summary,logs,clf,sc,threshold

def main():
    tt,te,tr,tp=load_range(*TRAIN); xt,xe,xr,xp=load_range(*TEST)
    tbase=build_base(tt,te); xbase=build_base(xt,xe)
    train_pack=(tt,te,tr,tp,tbase);test_pack=(xt,xe,xr,xp,xbase)
    grid=cv_select(tt,te,tr,tp,tbase)
    best=grid[0]
    summary,logs,clf,sc,threshold=final_eval(train_pack,test_pack,best)
    summary["cv_selected_from_week1"]={k:v for k,v in best.items()}
    summary["cv_top10"]=grid[:10]
    model={
      "model":"v40_last_chance_king_logit",
      "feature_names":v38.FEATURES,
      "scaler_mean":[float(x) for x in sc.mean_],"scaler_scale":[float(x) for x in sc.scale_],
      "coef":[float(x) for x in clf.coef_[0]],"intercept":float(clf.intercept_[0]),
      "C":best["C"],"alpha":best["alpha"],"seat_k":best["k"],"gate_threshold":threshold,
      "king_yen":KING_YEN,"reward_cap":REWARD_CAP,
      "guards":{"exact_seats_only":True,"max_seats":16,"cartesian_betting":False,"target_odds_used":False,"holdout_used_for_tuning":False},
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"holdout_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"model.json").write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

#!/usr/bin/env python3
from __future__ import annotations

"""
v38 KING SEAT research model.

Intentional research overfit:
- positives are the 29 >=10,000-yen trifecta winners from 2024-01-01..2024-01-07
- all other ordered triples in the same 7-car F1 S-class week are negatives
- target-race odds/popularity are never features
- payouts/results are labels/rewards only

The purpose is NOT a leakage-free profitability claim.
The purpose is to rewrite the board around the structural signature of KING outcomes.
"""

import csv
import importlib.util
import itertools
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/2024/s_class_f1_all_parts/2024_q1"
OUT=ROOT/"results/keirin_shogi/v38_king_seat_2024w1"
OUT.mkdir(parents=True,exist_ok=True)
START,END="2024-01-01","2024-01-07"
KING=10000

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

runtime=load_module("v38_king_runtime",ROOT/"scripts/keirin_shogi_v37_auto_place_runtime.py")

def read_csv(path):
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def ino(v):
    try:return int(float(v))
    except:return 0
def num(v):
    try:return float(v)
    except:return 0.0
def norm(v):
    return unicodedata.normalize("NFKC",str(v or "")).strip()
def triple(v):
    xs=[int(x) for x in re.findall(r"[1-9]",norm(v))]
    if len(xs)>=3 and len(set(xs[:3]))==3:return tuple(xs[:3])
    return None
def sigmoid(z):
    if z>=0:return 1.0/(1.0+math.exp(-z))
    ez=math.exp(z);return ez/(1.0+ez)

FEATURES=[
 "f_rank","s_rank","t_rank",
 "f_prob","s_mass","t_prob",
 "rank_sum","rank_max","rank_min","rank_spread",
 "outsider_count","f_outside","s_outside","t_outside",
 "f_in_base","s_in_base","t_in_base",
 "strong_middle","weak_ends","middle_vs_ends",
 "f_score_z","s_score_z","t_score_z",
 "f_win_z","s_win_z","t_win_z",
 "f_top2_z","s_top2_z","t_top2_z",
 "f_top3_z","s_top3_z","t_top3_z",
 "f_b_z","s_b_z","t_b_z",
 "f_attack_z","s_attack_z","t_attack_z",
 "f_sashi_z","s_sashi_z","t_sashi_z",
 "f_mark_z","s_mark_z","t_mark_z",
 "f_line_pos","s_line_pos","t_line_pos",
 "same_line_fs","same_line_ft","same_line_st","all_same_line",
 "different_line_count",
]

RAW_Z=[
 ("score","score_z"),("win_rate","win_z"),("top2_rate","top2_z"),("top3_rate","top3_z"),
 ("b_count","b_z"),("sashi_count","sashi_z"),("mark_count","mark_z")
]

def zmap(entries,key):
    vals=np.asarray([num(e.get(key)) for e in entries],float)
    mu=float(vals.mean()); sd=float(vals.std()) or 1.0
    return {ino(e["car_no"]):(num(e.get(key))-mu)/sd for e in entries}

def entry_maps(entries):
    out={ino(e["car_no"]):dict(e) for e in entries}
    z={}
    for raw,name in RAW_Z:
        z[name]=zmap(entries,raw)
    attacks=[num(e.get("nige_count"))+num(e.get("makuri_count")) for e in entries]
    mu=float(np.mean(attacks));sd=float(np.std(attacks)) or 1.0
    z["attack_z"]={ino(e["car_no"]):(num(e.get("nige_count"))+num(e.get("makuri_count"))-mu)/sd for e in entries}
    return out,z

def ranking_maps(placed):
    fr={int(r["no"]):(i+1,float(r["probability"])) for i,r in enumerate(placed["first_ranking"])}
    sr={int(r["no"]):(i+1,float(r["mass"])) for i,r in enumerate(placed["second_membership"])}
    tr={int(r["no"]):(i+1,float(r["probability"])) for i,r in enumerate(placed["third_ranking"])}
    return fr,sr,tr

def feature_vector(combo,placed,entries):
    a,b,c=combo
    fr,sr,tr=ranking_maps(placed)
    em,z=entry_maps(entries)
    ra,pa=fr[a]; rb,pb=sr[b]; rc,pc=tr[c]
    ranks=[ra,rb,rc]
    base1=set(map(int,placed["first_candidates"]));base2=set(map(int,placed["second_candidates"]));base3=set(map(int,placed["third_candidates"]))
    la=str(em[a].get("line_id","")).strip();lb=str(em[b].get("line_id","")).strip();lc=str(em[c].get("line_id","")).strip()
    same_fs=float(bool(la) and la==lb);same_ft=float(bool(la) and la==lc);same_st=float(bool(lb) and lb==lc)
    allsame=float(same_fs and same_ft)
    vals={
      "f_rank":ra/7.0,"s_rank":rb/7.0,"t_rank":rc/7.0,
      "f_prob":pa,"s_mass":pb,"t_prob":pc,
      "rank_sum":sum(ranks)/21.0,"rank_max":max(ranks)/7.0,"rank_min":min(ranks)/7.0,
      "rank_spread":(max(ranks)-min(ranks))/6.0,
      "outsider_count":sum(r>=4 for r in ranks)/3.0,
      "f_outside":float(ra>=4),"s_outside":float(rb>=4),"t_outside":float(rc>=4),
      "f_in_base":float(a in base1),"s_in_base":float(b in base2),"t_in_base":float(c in base3),
      "strong_middle":float(rb<=2 and ra>=4 and rc>=4),
      "weak_ends":float(ra>=4 and rc>=4),
      "middle_vs_ends":((ra+rc)/2.0-rb)/6.0,
      "f_line_pos":min(num(em[a].get("line_position")),4)/4.0,
      "s_line_pos":min(num(em[b].get("line_position")),4)/4.0,
      "t_line_pos":min(num(em[c].get("line_position")),4)/4.0,
      "same_line_fs":same_fs,"same_line_ft":same_ft,"same_line_st":same_st,"all_same_line":allsame,
      "different_line_count":len({x for x in (la,lb,lc) if x})/3.0,
    }
    for prefix,no in (("f",a),("s",b),("t",c)):
        vals[f"{prefix}_score_z"]=z["score_z"][no]
        vals[f"{prefix}_win_z"]=z["win_z"][no]
        vals[f"{prefix}_top2_z"]=z["top2_z"][no]
        vals[f"{prefix}_top3_z"]=z["top3_z"][no]
        vals[f"{prefix}_b_z"]=z["b_z"][no]
        vals[f"{prefix}_attack_z"]=z["attack_z"][no]
        vals[f"{prefix}_sashi_z"]=z["sashi_z"][no]
        vals[f"{prefix}_mark_z"]=z["mark_z"][no]
    return [float(vals[n]) for n in FEATURES]

def load_week():
    races=read_csv(BASE/"races.csv");entries=read_csv(BASE/"entries.csv");results=read_csv(BASE/"results.csv");payouts=read_csv(BASE/"payouts.csv")
    targets={r["race_id"]:r for r in races if START<=r.get("race_date","")<=END and ino(r.get("entry_count"))==7 and r.get("meeting_grade")=="F1" and "Ｓ級" in r.get("race_type","")}
    eb=defaultdict(list); rb=defaultdict(list)
    for e in entries:
        if e.get("race_id") in targets:eb[e["race_id"]].append(e)
    for x in results:
        if x.get("race_id") in targets:rb[x["race_id"]].append(x)
    paid={}
    for p in payouts:
        rid=str(p.get("race_id",""))
        if rid not in targets or norm(p.get("ticket_type")) not in {"3連単","三連単"}:continue
        if norm(p.get("status")).lower() not in {"","paid"}:continue
        t=triple(p.get("combination")); y=re.sub(r"[^0-9]","",norm(p.get("payout_yen")))
        if t and y: paid.setdefault(rid,{})[t]=int(y)
    return targets,eb,rb,paid

def build_base_predictions(targets,eb):
    engine=runtime.load_base_engine();nine=runtime.load_ninecar_engine();overlay=runtime.load_sevencar_overlay_engine()
    v21=engine.build_v21_state();pair=engine.build_pair_model();third,features,freeze=engine.build_third_model()
    out={}
    for rid,r in sorted(targets.items(),key=lambda kv:(kv[1].get("race_date",""),kv[1].get("track",""),ino(kv[1].get("race_no")))):
        ers=sorted(eb[rid],key=lambda x:ino(x.get("car_no")))
        if len(ers)!=7:continue
        live={"race_id":rid,"race_date":r.get("race_date",""),"track":r.get("track",""),"race_no":r.get("race_no",""),"race_type":r.get("race_type",""),"meeting_grade":r.get("meeting_grade",""),"entries":[runtime.normalize_entry(e) for e in ers]}
        p=runtime.runtime_place_one(engine,nine,overlay,live,v21,pair,third,features,freeze)
        if p.get("board_generated"):out[rid]=p
    return out

def actual_order(rows):
    top={}
    for x in rows:
        p=ino(x.get("finish_position"))
        if p in (1,2,3):top[p]=ino(x.get("car_no"))
    return (top.get(1),top.get(2),top.get(3)) if len(top)==3 else None

def train_model(targets,eb,rb,paid,basepred):
    X=[];y=[];meta=[]
    for rid,p in basepred.items():
        actual=actual_order(rb[rid])
        payout=paid.get(rid,{}).get(actual,0) if actual else 0
        king=bool(payout>=KING)
        entries=sorted(eb[rid],key=lambda e:ino(e.get("car_no")))
        cars=sorted(ino(e["car_no"]) for e in entries)
        for combo in itertools.permutations(cars,3):
            X.append(feature_vector(combo,p,entries))
            y.append(1 if king and combo==actual else 0)
            meta.append((rid,combo))
    X=np.asarray(X,float);y=np.asarray(y,int)
    scaler=StandardScaler().fit(X)
    Xs=scaler.transform(X)
    model=LogisticRegression(C=0.35,class_weight="balanced",max_iter=2500,solver="liblinear",random_state=38)
    model.fit(Xs,y)
    return model,scaler,meta,X,y

def score_races(model,scaler,targets,eb,rb,paid,basepred):
    rows=[]
    for rid,p in basepred.items():
        entries=sorted(eb[rid],key=lambda e:ino(e.get("car_no")))
        cars=sorted(ino(e["car_no"]) for e in entries)
        combos=list(itertools.permutations(cars,3))
        X=np.asarray([feature_vector(c,p,entries) for c in combos],float)
        probs=model.predict_proba(scaler.transform(X))[:,1]
        ranked=sorted(zip(combos,probs),key=lambda x:(-float(x[1]),x[0]))
        actual=actual_order(rb[rid]); payout=paid.get(rid,{}).get(actual,0) if actual else 0
        actual_rank=next((i+1 for i,(c,_) in enumerate(ranked) if c==actual),None)
        rows.append({
          "race_id":rid,"date":targets[rid].get("race_date"),"track":targets[rid].get("track"),"race_no":ino(targets[rid].get("race_no")),
          "actual":list(actual) if actual else None,"actual_payout":payout,"actual_label":"KING" if payout>=KING else ("SLAVE" if payout>0 else "DEATH"),
          "actual_king_rank":actual_rank,"max_king_score":float(ranked[0][1]),
          "ranked":[{"combo":list(c),"score":float(s)} for c,s in ranked],
        })
    return sorted(rows,key=lambda r:(r["date"],r["track"],r["race_no"]))

def eval_policy(rows,seat_k,threshold):
    log=[]; stake=payout=0;kings_total=kings_captured=0;king_payout_total=king_payout_captured=0
    for r in rows:
        isking=r["actual_payout"]>=KING
        if isking:
            kings_total+=1;king_payout_total+=r["actual_payout"]
        buy=r["max_king_score"]>=threshold
        selected=[tuple(x["combo"]) for x in r["ranked"][:seat_k]]
        actual=tuple(r["actual"]) if r["actual"] else None
        hit=buy and actual in selected
        rp=r["actual_payout"] if hit else 0
        st=seat_k*100 if buy else 0
        if isking and hit:
            kings_captured+=1;king_payout_captured+=r["actual_payout"]
        stake+=st;payout+=rp
        log.append({"race_id":r["race_id"],"buy":buy,"hit":hit,"actual_label":r["actual_label"],"actual_payout":r["actual_payout"],"stake":st,"return":rp,"profit":rp-st})
    buy_count=sum(x["buy"] for x in log)
    king_capture=kings_captured/kings_total if kings_total else 0
    king_payout_share=king_payout_captured/king_payout_total if king_payout_total else 0
    precision=kings_captured/buy_count if buy_count else 0
    buy_rate=buy_count/len(rows) if rows else 0
    roi=payout/stake if stake else 0
    objective=3.0*king_capture+1.5*king_payout_share+0.8*precision+0.35*min(roi,3.0)-0.50*buy_rate-0.010*seat_k
    return {
      "seat_k":seat_k,"threshold":threshold,"buy_count":buy_count,"buy_rate":buy_rate,
      "king_total":kings_total,"king_captured":kings_captured,"king_capture":king_capture,
      "king_payout_total":king_payout_total,"king_payout_captured":king_payout_captured,"king_payout_share":king_payout_share,
      "king_precision_per_buy":precision,"stake":stake,"payout":payout,"profit":payout-stake,"roi":roi,
      "objective":objective,"log":log
    }

def search_policy(rows):
    maxscores=sorted(set(round(float(r["max_king_score"]),12) for r in rows),reverse=True)
    # candidate thresholds based on race-rank cutoffs plus all-buy floor
    cuts={0.0}
    for n in (8,12,16,20,24,29,36,48,64,80,100):
        if n<=len(maxscores): cuts.add(maxscores[n-1])
    grid=[]
    for k in (4,6,8,10,12,16,20,24):
        for t in sorted(cuts):
            grid.append(eval_policy(rows,k,t))
    grid.sort(key=lambda x:(x["objective"],x["king_captured"],x["king_payout_share"],x["profit"]),reverse=True)
    return grid

def project_board(top):
    return {
      "first":sorted(set(int(x["combo"][0]) for x in top)),
      "second":sorted(set(int(x["combo"][1]) for x in top)),
      "third":sorted(set(int(x["combo"][2]) for x in top)),
    }

def main():
    targets,eb,rb,paid=load_week()
    basepred=build_base_predictions(targets,eb)
    model,scaler,meta,X,y=train_model(targets,eb,rb,paid,basepred)
    rows=score_races(model,scaler,targets,eb,rb,paid,basepred)
    grid=search_policy(rows)
    best=grid[0]
    seat_k=int(best["seat_k"]);threshold=float(best["threshold"])
    diagnostics=[]
    for r in rows:
        top=r["ranked"][:seat_k]
        board=project_board(top)
        actual=tuple(r["actual"]) if r["actual"] else None
        board_hit=bool(actual and actual[0] in board["first"] and actual[1] in board["second"] and actual[2] in board["third"])
        diagnostics.append({
          "race_id":r["race_id"],"date":r["date"],"track":r["track"],"race_no":r["race_no"],
          "actual":r["actual"],"actual_payout":r["actual_payout"],"actual_label":r["actual_label"],
          "actual_king_rank":r["actual_king_rank"],"max_king_score":r["max_king_score"],
          "participate":r["max_king_score"]>=threshold,"top_king_tickets":top,
          "king_board":board,"board_capture":board_hit,
        })
    kings=[x for x in diagnostics if x["actual_label"]=="KING"]
    model_json={
      "model":"v38_king_seat_2024w1_logit",
      "research_status":"INTENTIONAL_OVERFIT_KING_DISCOVERY",
      "training_period":[START,END],
      "king_definition_yen":KING,
      "feature_names":FEATURES,
      "scaler_mean":[float(x) for x in scaler.mean_],
      "scaler_scale":[float(x) for x in scaler.scale_],
      "coef":[float(x) for x in model.coef_[0]],
      "intercept":float(model.intercept_[0]),
      "policy":{"seat_k":seat_k,"threshold":threshold},
      "guards":{"target_odds_used":False,"target_popularity_used":False,"labels_use_results_and_payouts":True},
    }
    summary={
      "algorithm":"keirin_shogi_v38_king_seat_2024w1",
      "purpose":"29 KING outcomes are primary teachers; create KING seats and a KING-suitability buy gate.",
      "warning":"In-sample intentional overfit. Not evidence of future profitability.",
      "races":len(rows),"king_races":len(kings),"positive_combo_labels":int(y.sum()),"all_combo_examples":len(y),
      "selected_policy":{k:v for k,v in best.items() if k!="log"},
      "king_board_capture":sum(x["board_capture"] for x in kings)/len(kings) if kings else 0,
      "king_board_capture_count":sum(x["board_capture"] for x in kings),
      "king_actual_rank_distribution":{str(k):sum(x["actual_king_rank"]==k for x in kings) for k in range(1,25)},
      "top_coefficients":sorted(
        [{"feature":n,"coef":float(c)} for n,c in zip(FEATURES,model.coef_[0])],
        key=lambda x:abs(x["coef"]),reverse=True
      )[:20],
      "policy_top20":[{k:v for k,v in g.items() if k!="log"} for g in grid[:20]],
    }
    (OUT/"model.json").write_text(json.dumps(model_json,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"diagnostics.json").write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

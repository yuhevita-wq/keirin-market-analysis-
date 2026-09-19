#!/usr/bin/env python3
from __future__ import annotations
"""
v39 KING GATE: race-level buy/skip model learned from the same 2024-W1 29 KING races.

This intentionally overfits the discovery week. It exists to give KING its own
chair AND its own entrance gate. It must be tested on a later untouched week
before replacing production participation.
"""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT=Path(__file__).resolve().parents[1]
V38=ROOT/"scripts/keirin_shogi_v38_king_seat_2024w1.py"
OUT=ROOT/"results/keirin_shogi/v39_king_gate_2024w1";OUT.mkdir(parents=True,exist_ok=True)

spec=importlib.util.spec_from_file_location("v38kg",V38);v38=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v38;spec.loader.exec_module(v38)

def race_features(r):
    scores=np.asarray([float(x["score"]) for x in r["ranked"]],float)
    # KING-shape density, not odds: how many triples the v38 structural model
    # regards as unusually KING-like, plus concentration/dispersion.
    top=np.sort(scores)[::-1]
    return [
      float(top[0]),float(top[1]),float(top[2]),float(top[5]),float(top[9]),float(top[19]),float(top[23]),
      float(top[:4].mean()),float(top[:8].mean()),float(top[:16].mean()),float(top[:24].mean()),
      float(top.std()),float(top[0]-top[5]),float(top[0]-top[23]),
      float((scores>=.99).sum()/len(scores)),float((scores>=.95).sum()/len(scores)),
      float((scores>=.90).sum()/len(scores)),float((scores>=.75).sum()/len(scores)),
    ]
FEATURES=["top1","top2","top3","top6","top10","top20","top24","mean4","mean8","mean16","mean24","std","gap1_6","gap1_24","share99","share95","share90","share75"]

def main():
    targets,eb,rb,paid=v38.load_week();base=v38.build_base_predictions(targets,eb)
    combo,scaler,_,_,_=v38.train_model(targets,eb,rb,paid,base)
    rows=v38.score_races(combo,scaler,targets,eb,rb,paid,base)
    X=np.asarray([race_features(r) for r in rows],float)
    y=np.asarray([int(r["actual_payout"]>=v38.KING) for r in rows],int)
    # Discovery model is deliberately allowed to fit the 29 KING signatures hard.
    gate=RandomForestClassifier(n_estimators=600,max_depth=None,min_samples_leaf=1,max_features=None,class_weight="balanced",random_state=39)
    gate.fit(X,y)
    prob=gate.predict_proba(X)[:,1]
    # Select the narrowest threshold that still keeps every discovery KING.
    king_probs=prob[y==1]
    threshold=float(king_probs.min()-1e-12)
    pred=prob>=threshold
    tp=int(((pred)&(y==1)).sum());fp=int(((pred)&(y==0)).sum());fn=int(((~pred)&(y==1)).sum())
    # KING SEAT itself: keep top 24 exact triples from v38. Projection is shown
    # separately, but purchase seats remain exact triples to avoid cartesian explosion.
    seat_k=24
    king_hits=0;king_total=int(y.sum());stake=payout=0
    logs=[]
    for r,p,buy,label in zip(rows,prob,pred,y):
        seats=[tuple(x["combo"]) for x in r["ranked"][:seat_k]]
        actual=tuple(r["actual"])
        hit=bool(buy and actual in seats)
        if label and hit:king_hits+=1
        st=seat_k*100 if buy else 0
        ret=r["actual_payout"] if hit else 0
        stake+=st;payout+=ret
        logs.append({"race_id":r["race_id"],"date":r["date"],"track":r["track"],"race_no":r["race_no"],"king_gate_score":float(p),"buy":bool(buy),"actual_label":r["actual_label"],"actual_payout":r["actual_payout"],"king_seats":[list(x) for x in seats],"hit":hit})
    artifact={
      "algorithm":"keirin_shogi_v39_king_gate_2024w1",
      "status":"DISCOVERY_OVERFIT_NOT_YET_PRODUCTION",
      "training_period":[v38.START,v38.END],
      "king_races":king_total,
      "buy_races":int(pred.sum()),"true_king_buys":tp,"false_buys":fp,"missed_kings":fn,
      "king_gate_recall":tp/king_total,
      "seat_k":seat_k,"king_seat_exact_capture":king_hits/king_total,
      "stake_yen":stake,"payout_yen":payout,"profit_yen":payout-stake,"roi":payout/stake if stake else 0,
      "threshold":threshold,
      "feature_names":FEATURES,
      "feature_importance":sorted([{"feature":n,"importance":float(v)} for n,v in zip(FEATURES,gate.feature_importances_)],key=lambda x:-x["importance"]),
      "guard":"29 KING discovery week only. No odds/popularity features. Do not claim future performance until next-week holdout.",
      "logs":logs
    }
    (OUT/"summary.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in artifact.items() if k!="logs"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()

#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/keirin_shogi/v38_king_seat_2024w1"
DIAG=OUT/"diagnostics.json"
MODEL=OUT/"model.json"
SUMMARY=OUT/"summary.json"

GATE_FEATURES=[
 "max_score","top3_mean","top6_mean","top12_mean","top24_mean",
 "top24_min","top24_std","gap_1_2","gap_1_6","gap_1_24",
 "first_count","second_count","third_count","board_ticket_count",
 "score_mass_top3","score_mass_top6",
]

def gate_features(row):
    ss=[float(x["score"]) for x in row["top_king_tickets"]]
    def avg(n):return float(np.mean(ss[:min(n,len(ss))])) if ss else 0.0
    def ticket_count(board):
        return sum(1 for a in board["first"] for b in board["second"] for c in board["third"] if len({a,b,c})==3)
    total=sum(ss) or 1.0
    vals={
      "max_score":ss[0] if ss else 0.0,
      "top3_mean":avg(3),"top6_mean":avg(6),"top12_mean":avg(12),"top24_mean":avg(24),
      "top24_min":ss[-1] if ss else 0.0,
      "top24_std":float(np.std(ss)) if ss else 0.0,
      "gap_1_2":(ss[0]-ss[1]) if len(ss)>1 else 0.0,
      "gap_1_6":(ss[0]-ss[5]) if len(ss)>5 else 0.0,
      "gap_1_24":(ss[0]-ss[-1]) if ss else 0.0,
      "first_count":len(row["king_board"]["first"])/7.0,
      "second_count":len(row["king_board"]["second"])/7.0,
      "third_count":len(row["king_board"]["third"])/7.0,
      "board_ticket_count":ticket_count(row["king_board"])/210.0,
      "score_mass_top3":sum(ss[:3])/total,
      "score_mass_top6":sum(ss[:6])/total,
    }
    return [float(vals[n]) for n in GATE_FEATURES]

def board_ticket_count(board):
    return sum(1 for a in board["first"] for b in board["second"] for c in board["third"] if len({a,b,c})==3)

def eval_gate(rows,probs,threshold):
    buy=[float(p)>=threshold for p in probs]
    total_king=sum(r["actual_label"]=="KING" for r in rows)
    monetizable=sum(r["actual_label"]=="KING" and r["board_capture"] for r in rows)
    captured=0;allking_buy=0;stake=payout=0;king_payout_total=sum(r["actual_payout"] for r in rows if r["actual_label"]=="KING")
    king_payout_captured=0
    log=[]
    for r,b,p in zip(rows,buy,probs):
        st=board_ticket_count(r["king_board"])*100 if b else 0
        hit=bool(b and r["board_capture"])
        ret=int(r["actual_payout"]) if hit else 0
        if r["actual_label"]=="KING" and b:allking_buy+=1
        if r["actual_label"]=="KING" and hit:
            captured+=1;king_payout_captured+=int(r["actual_payout"])
        stake+=st;payout+=ret
        log.append({"race_id":r["race_id"],"prob":float(p),"buy":bool(b),"board_tickets":board_ticket_count(r["king_board"]),"hit":hit,"label":r["actual_label"],"payout":r["actual_payout"],"profit":ret-st})
    bc=sum(buy);br=bc/len(rows)
    cap=captured/total_king if total_king else 0
    possible_cap=captured/monetizable if monetizable else 0
    precision=captured/bc if bc else 0
    share=king_payout_captured/king_payout_total if king_payout_total else 0
    roi=payout/stake if stake else 0
    target_rate=total_king/len(rows)
    objective=(3.2*cap+1.2*possible_cap+1.4*precision+1.2*share+0.35*min(roi/3.0,1.0)
               -1.1*abs(br-target_rate)-1.5*max(0.0,br-0.50))
    return {
      "threshold":threshold,"buy_count":bc,"buy_rate":br,"total_king":total_king,"monetizable_king":monetizable,
      "king_captured":captured,"king_capture_all":cap,"king_capture_possible":possible_cap,
      "king_precision_per_buy":precision,"king_payout_share":share,
      "stake":stake,"payout":payout,"profit":payout-stake,"roi":roi,"objective":objective,"log":log
    }

def serialize_tree(clf):
    t=clf.tree_
    return {
      "children_left":[int(x) for x in t.children_left],
      "children_right":[int(x) for x in t.children_right],
      "feature":[int(x) for x in t.feature],
      "threshold":[float(x) for x in t.threshold],
      "value":[[float(v) for v in row[0]] for row in t.value],
    }

def main():
    rows=json.loads(DIAG.read_text(encoding="utf-8"))
    X=np.asarray([gate_features(r) for r in rows],float)
    # Learn a gate for a race where the KING board actually has a throne ready.
    y=np.asarray([1 if r["actual_label"]=="KING" and r["board_capture"] else 0 for r in rows],int)
    grid=[]
    for depth in (2,3,4,5,6):
      for leaf in (2,3,4,5,6,8):
        clf=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight="balanced",random_state=3801)
        clf.fit(X,y)
        probs=clf.predict_proba(X)[:,1]
        thresholds=sorted(set([0.25,0.35,0.45,0.5,0.55,0.65,0.75]+[float(x) for x in probs]))
        for th in thresholds:
            m=eval_gate(rows,probs,th)
            if 8<=m["buy_count"]<=65:
                grid.append((m["objective"],m,clf))
    grid.sort(key=lambda x:(x[0],x[1]["king_captured"],x[1]["king_precision_per_buy"],x[1]["profit"]),reverse=True)
    _,best,clf=grid[0]
    model=json.loads(MODEL.read_text(encoding="utf-8"))
    model["gate"]={
      "model":"decision_tree_king_board_ready",
      "feature_names":GATE_FEATURES,
      "tree":serialize_tree(clf),
      "threshold":float(best["threshold"]),
      "training_positive":"KING and king_board captures actual trifecta",
      "training_positive_count":int(y.sum()),
    }
    MODEL.write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding="utf-8")
    summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary["king_gate"]={k:v for k,v in best.items() if k!="log"}
    summary["king_gate_top20"]=[{k:v for k,v in m.items() if k!="log"} for _,m,_ in grid[:20]]
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"gate_log.json").write_text(json.dumps(best["log"],ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary["king_gate"],ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

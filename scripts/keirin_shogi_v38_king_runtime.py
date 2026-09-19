#!/usr/bin/env python3
from __future__ import annotations

"""v38 KING SEAT runtime projector.

Consumes the frozen JSON artifact produced by keirin_shogi_v38_king_seat_2024w1.py.
No target-race odds, popularity, results or payouts are read here.
"""

import itertools
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"results/keirin_shogi/v38_king_seat_2024w1/model.json"

RAW_Z=[
 ("score","score_z"),("win_rate","win_z"),("top2_rate","top2_z"),("top3_rate","top3_z"),
 ("b_count","b_z"),("sashi_count","sashi_z"),("mark_count","mark_z")
]

def ino(v):
    try:return int(float(v))
    except:return 0
def num(v):
    try:return float(v)
    except:return 0.0
def sigmoid(z):
    if z>=0:return 1.0/(1.0+math.exp(-z))
    ez=math.exp(z);return ez/(1.0+ez)

def zmap(entries,key):
    vals=[num(e.get(key)) for e in entries]
    mu=mean(vals);sd=pstdev(vals) or 1.0
    return {ino(e["car_no"]):(num(e.get(key))-mu)/sd for e in entries}

def entry_maps(entries):
    out={ino(e["car_no"]):dict(e) for e in entries}
    z={}
    for raw,name in RAW_Z:z[name]=zmap(entries,raw)
    attacks=[num(e.get("nige_count"))+num(e.get("makuri_count")) for e in entries]
    mu=mean(attacks);sd=pstdev(attacks) or 1.0
    z["attack_z"]={ino(e["car_no"]):(num(e.get("nige_count"))+num(e.get("makuri_count"))-mu)/sd for e in entries}
    return out,z

def ranking_maps(placed):
    fr={int(r["no"]):(i+1,float(r["probability"])) for i,r in enumerate(placed["first_ranking"])}
    sr={int(r["no"]):(i+1,float(r["mass"])) for i,r in enumerate(placed["second_membership"])}
    tr={int(r["no"]):(i+1,float(r["probability"])) for i,r in enumerate(placed["third_ranking"])}
    return fr,sr,tr

def feature_values(combo,placed,entries):
    a,b,c=combo
    fr,sr,tr=ranking_maps(placed)
    em,z=entry_maps(entries)
    ra,pa=fr[a];rb,pb=sr[b];rc,pc=tr[c]
    ranks=[ra,rb,rc]
    base1=set(map(int,placed["first_candidates"]));base2=set(map(int,placed["second_candidates"]));base3=set(map(int,placed["third_candidates"]))
    la=str(em[a].get("line_id","")).strip();lb=str(em[b].get("line_id","")).strip();lc=str(em[c].get("line_id","")).strip()
    same_fs=float(bool(la) and la==lb);same_ft=float(bool(la) and la==lc);same_st=float(bool(lb) and lb==lc)
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
      "same_line_fs":same_fs,"same_line_ft":same_ft,"same_line_st":same_st,
      "all_same_line":float(same_fs and same_ft),
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
    return vals

def load_model(path=MODEL):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def combo_score(combo,placed,entries,model):
    vals=feature_values(combo,placed,entries)
    names=model["feature_names"];means=model["scaler_mean"];scales=model["scaler_scale"];coef=model["coef"]
    z=float(model["intercept"])
    for name,mu,sd,w in zip(names,means,scales,coef):
        x=float(vals[name]);den=float(sd) or 1.0
        z+=float(w)*((x-float(mu))/den)
    return sigmoid(z)

def predict_king_seat(placed,race,model_path=MODEL):
    if len(race.get("entries",[]))!=7:
        return placed
    model=load_model(model_path)
    entries=race["entries"]
    cars=sorted(ino(e["car_no"]) for e in entries)
    ranked=[]
    for combo in itertools.permutations(cars,3):
        ranked.append((combo,combo_score(combo,placed,entries,model)))
    ranked.sort(key=lambda x:(-x[1],x[0]))
    seat_k=int(model["policy"]["seat_k"]);threshold=float(model["policy"]["threshold"])
    top=ranked[:seat_k]
    first=sorted({c[0] for c,_ in top});second=sorted({c[1] for c,_ in top});third=sorted({c[2] for c,_ in top})
    score=float(top[0][1]) if top else 0.0
    result=dict(placed)
    result.update({
      "board_policy":"v38_king_seat_2024w1",
      "participate":bool(score>=threshold),
      "first_candidates":first,"second_candidates":second,"third_candidates":third,
      "king_score":score,"king_threshold":threshold,"king_seat_k":seat_k,
      "king_top_tickets":[{"first":c[0],"second":c[1],"third":c[2],"score":float(s)} for c,s in top],
      "king_base_board":{
        "first":list(map(int,placed.get("first_candidates",[]))),
        "second":list(map(int,placed.get("second_candidates",[]))),
        "third":list(map(int,placed.get("third_candidates",[]))),
        "v21_participate":bool(placed.get("participate")),
      },
      "versions":{
        **dict(placed.get("versions",{})),
        "seven_car_primary":"v38_king_seat_2024w1",
      },
      "adoption_note":"29 KING outcomes learned as primary seats; research-overfit v38",
      "research_warning":"2024-01-01..01-07 intentional in-sample KING fit; future validation not yet evidence.",
    })
    if not result["participate"]:
        result["skip_reason"]="v38 KING適性が閾値未満"
    else:
        result.pop("skip_reason",None)
    return result

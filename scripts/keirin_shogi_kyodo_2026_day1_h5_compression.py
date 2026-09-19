#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
from collections import defaultdict
import joblib, numpy as np

ROOT=Path(__file__).resolve().parents[1]
V32_PATH=ROOT/"scripts/keirin_shogi_ninecar_v32.py"
SNAP=ROOT/"data/research/kyodo_2026_day1_snapshot.json"
MODEL=ROOT/"results/keirin_shogi/ninecar_v32/model.joblib"
OUT=ROOT/"results/keirin_shogi/kyodo_2026_day1_h5_compression.json"
STAKE=100

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

v32=load("v32_2026_day1_h5",V32_PATH)
v2=v32.v2

def pair_key(a,b): return tuple(sorted((int(a),int(b))))

def union_pairs(board):
    cars=sorted(set(board[0])|set(board[1])|set(board[2]))
    return {pair_key(a,b) for i,a in enumerate(cars) for b in cars[i+1:]}

def pair_probs(joint):
    out=defaultdict(float)
    for a,b,c,p in joint:
        out[pair_key(a,b)]+=p; out[pair_key(a,c)]+=p; out[pair_key(b,c)]+=p
    return dict(out)

def top3_inclusion(joint):
    p1,p2,p3=v2.marginals(joint)
    return {n:p1[n]+p2[n]+p3[n] for n in p1}

def paid_map(x):
    out={}
    for k,v in x["actual"]["wide"].items():
        a,b=map(int,k.split("-")); out[pair_key(a,b)]=int(v)
    return out

def eval_tickets(tickets,paid):
    hits=sorted(set(tickets)&set(paid))
    return {
        "tickets":len(tickets),
        "payout":sum(paid[p] for p in hits),
        "hit_count":len(hits),
        "hits":[list(x) for x in hits],
    }

def summarize(rows,key):
    bets=[r for r in rows if r[key]["tickets"]>0]
    tickets=sum(r[key]["tickets"] for r in bets)
    payout=sum(r[key]["payout"] for r in bets)
    hit_races=sum(r[key]["payout"]>0 for r in bets)
    return {
        "races":len(bets),"tickets":tickets,"stake_yen":tickets*STAKE,
        "payout_yen":payout,"profit_yen":payout-tickets*STAKE,
        "roi_pct":100*payout/(tickets*STAKE) if tickets else None,
        "hit_races":hit_races,
        "hit_rate_pct":100*hit_races/len(bets) if bets else None,
        "multi_hit_races":sum(r[key]["hit_count"]>=2 for r in bets),
        "triple_hit_races":sum(r[key]["hit_count"]>=3 for r in bets),
        "avg_tickets":tickets/len(bets) if bets else None,
        "max_race_payout_yen":max((r[key]["payout"] for r in bets),default=0),
    }

def predict_row(payload,bundle):
    race=v32.live_race_from_payload(payload)
    base=bundle["base_bundle"]
    feats,strong,p1,p2,joint=v32.state_features(
        race,base["models"],float(base["alpha"]),float(base["beta"])
    )
    X=np.asarray([[feats[f] for f in bundle["state_feature_names"]]],dtype=float)
    pp=bundle["state_model"].predict_proba(X)[0]
    scored={
        "race":race,"features":feats,"strong":strong,"p1":p1,"p2":p2,"joint":joint,
        "state_probs":{v32.STATE_NAMES[i]:float(pp[i]) for i in range(3)}
    }
    board,mass,participate,dominant,action=v32.apply_overlay(scored,bundle["rules"])
    return scored,board,mass,participate,dominant,action

def main():
    data=json.loads(SNAP.read_text(encoding="utf-8"))
    bundle=joblib.load(MODEL)
    rows=[]
    skipped=[]
    for x in data["races"]:
        r,board,mass,participate,dominant,action=predict_row(x,bundle)
        if not participate:
            skipped.append({"race_no":x["race_no"],"race_id":x["race_id"],"reason":"v32 participation skip"})
            continue

        base=union_pairs(board)
        pp=pair_probs(r["joint"])
        inc=top3_inclusion(r["joint"])
        ranked=sorted(inc,key=lambda n:(-inc[n],n))
        strong_pair=pair_key(ranked[0],ranked[1])

        line_of={int(e["car_no"]):int(float(e.get("line_id") or 0)) for e in r["race"].entries}
        same_line=[p for p in base if line_of.get(p[0],0)>0 and line_of.get(p[0])==line_of.get(p[1])]
        strong_line_pair=max(same_line,key=lambda p:(pp.get(p,0.0),-p[0],-p[1]),default=None)

        h1=set(base); h1.discard(strong_pair)
        h2=set(h1)
        if strong_line_pair is not None: h2.discard(strong_line_pair)

        h3_pair=max(h2,key=lambda p:(pp.get(p,0.0),-p[0],-p[1]),default=None)
        h3=set(h2)
        if h3_pair is not None: h3.discard(h3_pair)

        h4_pair=max(h3,key=lambda p:(pp.get(p,0.0),-p[0],-p[1]),default=None)
        h4=set(h3)
        if h4_pair is not None: h4.discard(h4_pair)

        centers=[]
        if strong_pair in base: centers.append(("H1",strong_pair))
        if strong_line_pair is not None and strong_line_pair in h1: centers.append(("H2",strong_line_pair))
        if h3_pair is not None and h3_pair in h2: centers.append(("H3",h3_pair))
        if h4_pair is not None and h4_pair in h3: centers.append(("H4",h4_pair))
        touch={n:0 for n in inc}
        for _,p in centers:
            touch[p[0]]+=1; touch[p[1]]+=1

        rel={p:touch.get(p[0],0)+touch.get(p[1],0) for p in h4}
        order=sorted(h4,key=lambda p:(-rel[p],-pp.get(p,0.0),p[0],p[1]))

        states={"base":set(base),"h1":h1,"h2":h2,"h3":h3,"h4":h4}
        for target in (6,5,4,3):
            t=set(h4)
            for p in order:
                if len(t)<=target: break
                t.discard(p)
            states[f"h5_{target}"]=t

        paid=paid_map(x)
        rec={
            "race_no":x["race_no"],"race_id":x["race_id"],
            "actual_order":x["actual"]["order"],
            "board":[list(z) for z in board],
            "board_unique_riders":len(set(board[0])|set(board[1])|set(board[2])),
            "board_mass":mass,"overlay_action":action,
            "h1_pair":list(strong_pair),
            "h2_pair":list(strong_line_pair) if strong_line_pair else None,
            "h3_pair":list(h3_pair) if h3_pair else None,
            "h4_pair":list(h4_pair) if h4_pair else None,
            "history_touch":touch,
            "h5_cut_order":[list(p) for p in order],
        }
        for key,t in states.items():
            rec[key]=eval_tickets(t,paid)
            rec[key]["selected_pairs"]=[list(p) for p in sorted(t)]
        rows.append(rec)

    report={
        "study":"2026 Kyodo day1 pre-race snapshot: H1-H4 + iterative H5 to 6/5/4/3",
        "snapshot_commit":data["source_snapshot_commit"],
        "snapshot_generated_at_jst":data["snapshot_generated_at_jst"],
        "prediction_model":bundle["model_name"],
        "inference_integrity":{
            "snapshot_before_first_race":True,
            "snapshot_time_jst":data["snapshot_generated_at_jst"],
            "official_grade_corrected_to":"G2",
            "official_race_type_corrected_to":"S級一次予選",
            "odds_used_for_selection":False,
            "results_used_for_selection":False,
            "payouts_used_for_selection":False,
        },
        "skipped":skipped,
        "base":summarize(rows,"base"),"h1":summarize(rows,"h1"),"h2":summarize(rows,"h2"),
        "h3":summarize(rows,"h3"),"h4":summarize(rows,"h4"),
        "h5_6":summarize(rows,"h5_6"),"h5_5":summarize(rows,"h5_5"),
        "h5_4":summarize(rows,"h5_4"),"h5_3":summarize(rows,"h5_3"),
        "races":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

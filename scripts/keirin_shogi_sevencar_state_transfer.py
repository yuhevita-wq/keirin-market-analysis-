#!/usr/bin/env python3
from __future__ import annotations

"""Transfer nine-car strong-state lessons into the 7-car F1 v21/v31/v37 stack.

Study:
- v21: extra COLLAPSE filter on current participants.
- v31: SOFT_FAIL keeps strong rider in row1 and duplicates into row2 if needed.
- v37: test whether also forcing the strong rider into row3 helps; keep only if forward-positive.
- dominant rider: explicit first-row protection, but current v21 top rider is already row1.

No odds/popularity/payout used.
"""

import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/keirin_shogi/sevencar_state_transfer"
OUT.mkdir(parents=True,exist_ok=True)

def load_mod(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

v37=load_mod("v37_transfer","scripts/keirin_shogi_v37_shrunk_board_third.py")
v36=v37.v36
v35=v37.v35
v32=v37.v32
v31=v32.v31

FREEZE=json.loads((ROOT/"results/keirin_shogi/v37_shrunk_board_third/FREEZE.json").read_text(encoding="utf-8"))
STATES=("WIN","SOFT_FAIL","COLLAPSE")
CAL_A=("2025-10-27","2025-11-30")
CAL_B=("2025-12-01","2025-12-28")
H1_EARLY=("2025-12-29","2026-03-31")
H1_LATE=("2026-04-01","2026-06-28")
FUTURE=("2026-07-06","2026-08-30")

def ino(v):
    try:return int(float(v))
    except:return 0

def load_json(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))

def load_future_bases():
    base=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830"
    def read(name):
        with (base/name).open("r",encoding="utf-8-sig",newline="") as h:
            return list(csv.DictReader(h))
    races=read("races.csv"); entries=read("entries.csv"); results=read("results.csv")
    eb=defaultdict(list); rb=defaultdict(list)
    for x in entries: eb[x["race_id"]].append(x)
    for x in results: rb[x["race_id"]].append(x)
    out={}
    for race in races:
        rid=race["race_id"]
        if race.get("meeting_grade")!="F1" or "Ｓ級" not in race.get("race_type","") or ino(race.get("entry_count"))!=7:
            continue
        if len(eb[rid])!=7: continue
        order=v32.result_order(rb[rid])
        if order is None: continue
        out[str(rid)]={"race_id":rid,"race_date":race["race_date"],"race_type":race.get("race_type",""),
                       "base":v32.enrich_base(eb[rid]),"order":order}
    return out

def calibration_logs_and_races():
    races=v32.load_races()
    rmap={str(r["race_id"]):r for r in races}
    pair_model,contexts,_=v35.exact_v31_contexts()
    train=v32.period(races,*v35.FINAL_TRAIN)
    x,y,names=v35.build_training_matrix(train,None)
    third=v35.fit_third_model(x,y,FREEZE["selected_variant"])
    cal=[r for r in v32.period(races,*v35.FINAL_CALIBRATION) if str(r["race_id"]) in contexts]
    prepared=v35.prepare_prediction(cal,pair_model,names,contexts)
    cond=v36.raw_conditionals(prepared,third)
    rows=v36.aggregate(prepared,cond,FREEZE["fixed_score"])
    logs=v35.apply(rows,FREEZE["selected_policy"],True)
    return logs,rmap

def rank_desc(vals):
    order=sorted(vals,key=lambda n:(-vals[n],n))
    return {n:i+1 for i,n in enumerate(order)}

def state_of(race,strong):
    if int(race["order"][0])==strong:return 0
    if strong in set(map(int,race["order"][1:3])):return 1
    return 2

def features(log,race):
    base=race["base"]
    first=list(map(int,log["first_candidates"]))
    second=list(map(int,log["second_candidates"]))
    third=list(map(int,log["third_candidates"]))
    strong=first[0]
    mem_rows=log.get("top2_membership") or log.get("membership") or log.get("second_membership") or []
    mem={int(x["no"]):float(x.get("mass",0.0)) for x in mem_rows}
    mem_order=sorted(mem,key=lambda n:(-mem[n],n))
    mr={n:i+1 for i,n in enumerate(mem_order)}
    top_pairs=log.get("top_pairs",[])
    xstrong=base[strong]["x"]
    zkeys=("z_score","z_win_rate","z_top2_rate","z_top3_rate","z_b_count","z_first_count","z_second_count","z_third_count","z_outside_count")
    ranks={}
    for k in zkeys:
        vals={n:float(base[n]["x"].get(k,0.0)) for n in base}
        ranks[k]=rank_desc(vals)

    strong_line=str(base[strong]["line_id"])
    line_members=defaultdict(list)
    for n in base: line_members[str(base[n]["line_id"])].append(n)
    def lmean(lid,k):
        ns=line_members[lid]
        return float(np.mean([float(base[n]["x"].get(k,0.0)) for n in ns]))
    strong_line_score=lmean(strong_line,"z_score")
    rival_scores=[lmean(lid,"z_score") for lid in line_members if lid!=strong_line]
    rival_best=max(rival_scores) if rival_scores else 0.0
    top2_same=0.0
    if len(first)>=2:
        top2_same=float(str(base[first[0]]["line_id"])==str(base[first[1]]["line_id"]))
    vals=[mem[n] for n in mem_order]
    top_pair=float(top_pairs[0].get("prob",top_pairs[0].get("probability",0.0))) if top_pairs else 0.0
    top3pair=sum(float(x.get("prob",x.get("probability",0.0))) for x in top_pairs[:3])
    feats={
        "strong_membership":float(mem.get(strong,0.0)),
        "strong_membership_rank":float(mr.get(strong,7)),
        "membership_top1":float(vals[0] if vals else 0.0),
        "membership_gap12":float(vals[0]-vals[1] if len(vals)>1 else 0.0),
        "membership_top2_mass":float(sum(vals[:2])),
        "top_pair_prob":top_pair,
        "top3_pair_mass":float(top3pair),
        "first_count":float(len(first)),
        "second_count":float(len(second)),
        "third_count":float(len(third)),
        "strong_in_second":float(strong in second),
        "strong_in_third":float(strong in third),
        "strong_line_pos":float(base[strong]["line_position"]),
        "strong_line_size":float(base[strong]["x"].get("line_size",1.0)),
        "strong_line_score":strong_line_score,
        "best_rival_line_score":rival_best,
        "strong_vs_rival_line_score":strong_line_score-rival_best,
        "top2_first_same_line":top2_same,
        "num_lines":float(len(line_members)),
    }
    for k in zkeys:
        feats[f"strong_{k}"]=float(xstrong.get(k,0.0))
        feats[f"strong_{k}_rank"]=float(ranks[k][strong])
    rt=str(race["race_type"])
    for lab in ("予選","一般","準決勝","特選","選抜","決勝"):
        feats[f"race_{lab}"]=float(lab in rt)
    return feats,strong

def rows_from(logs,rmap):
    out=[]
    for log in logs:
        rid=str(log["race_id"])
        race=rmap.get(rid)
        if not race: continue
        feats,strong=features(log,race)
        out.append({"log":log,"race":race,"features":feats,"strong":strong,"state":state_of(race,strong)})
    return out

def fit_state(rows):
    names=list(rows[0]["features"])
    X=np.asarray([[r["features"][k] for k in names] for r in rows],float)
    y=np.asarray([r["state"] for r in rows],int)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,class_weight="balanced",random_state=7037))
    model.fit(X,y)
    return model,names

def score(rows,model,names):
    X=np.asarray([[r["features"][k] for k in names] for r in rows],float)
    pp=model.predict_proba(X)
    out=[]
    for r,p in zip(rows,pp):
        q=dict(r);q["probs"]={STATES[i]:float(p[i]) for i in range(3)};out.append(q)
    return out

def replace_lowest(cands,strong,ranking,key):
    c=list(map(int,cands))
    if strong in c:return c
    scoremap={int(x["no"]):float(x[key]) for x in ranking}
    victim=min(c,key=lambda n:(scoremap.get(n,-1e9),-n))
    c.remove(victim);c.append(strong)
    return sorted(c)

def apply_variant(row,rules,variant):
    log=row["log"]; strong=row["strong"]; p=row["probs"]
    first=list(map(int,log["first_candidates"])); second=list(map(int,log["second_candidates"])); third=list(map(int,log["third_candidates"]))
    participate=p["COLLAPSE"] < rules["collapse_threshold"]
    action="BASE"
    if participate and p["SOFT_FAIL"]>=rules["soft_threshold"] and p["COLLAPSE"]<rules["soft_collapse_max"]:
        if variant in ("SECOND","SECOND_THIRD"):
            second=replace_lowest(second,strong,log.get("top2_membership") or log.get("membership") or log.get("second_membership") or [],"mass")
            action="SOFT_SECOND"
        if variant=="SECOND_THIRD":
            third=replace_lowest(third,strong,log["third_ranking"],"probability")
            action="SOFT_SECOND_THIRD"
    a1,a2,a3=map(int,row["race"]["order"][:3])
    return {
        "participate":participate,"action":action,"first":first,"second":second,"third":third,
        "first_hit":int(a1 in first),"second_hit":int(a2 in second),"third_hit":int(a3 in third),
        "full":int(a1 in first and a2 in second and a3 in third),"state":row["state"],"strong":strong
    }

def choose_rules(scored):
    # Tune on a held-out calibration block.
    col=np.asarray([r["probs"]["COLLAPSE"] for r in scored])
    soft=np.asarray([r["probs"]["SOFT_FAIL"] for r in scored])
    best=None
    for qc in (0.70,0.80,0.90):
        ct=float(np.quantile(col,qc))
        for qs in (0.65,0.75,0.85,0.90):
            st=float(np.quantile(soft,qs))
            for scm_q in (0.50,0.60,0.70):
                scm=float(np.quantile(col,scm_q))
                rules={"collapse_threshold":ct,"soft_threshold":st,"soft_collapse_max":scm}
                vals=[apply_variant(r,rules,"SECOND") for r in scored]
                keep=[x for x in vals if x["participate"]]
                if len(keep)<int(0.65*len(vals)):continue
                full=np.mean([x["full"] for x in keep])
                second=np.mean([x["second_hit"] for x in keep])
                soft_sel=[x for x in vals if x["action"]=="SOFT_SECOND"]
                if len(soft_sel)<8:continue
                key=(full,second,-len(vals)+len(keep),np.mean([x["second_hit"] for x in soft_sel]))
                rec={"rules":rules,"keep_rate":len(keep)/len(vals),"keep_full":float(full),"keep_second":float(second),
                     "soft_n":len(soft_sel),"soft_second":float(np.mean([x["second_hit"] for x in soft_sel]))}
                if best is None or key>best[0]:best=(key,rec)
    if best:return best[1]
    return {"rules":{"collapse_threshold":float(np.quantile(col,.8)),"soft_threshold":float(np.quantile(soft,.85)),
                     "soft_collapse_max":float(np.quantile(col,.6))}}

def eval_block(scored,rules):
    out={}
    for variant in ("BASE","SECOND","SECOND_THIRD"):
        vals=[apply_variant(r,rules,variant) for r in scored]
        keep=[x for x in vals if x["participate"]]
        out[variant]={
            "races":len(vals),"participation_rate":len(keep)/len(vals),
            "first_capture":float(np.mean([x["first_hit"] for x in vals])),
            "second_capture":float(np.mean([x["second_hit"] for x in vals])),
            "third_capture":float(np.mean([x["third_hit"] for x in vals])),
            "full_capture":float(np.mean([x["full"] for x in vals])),
            "participant_full_capture":float(np.mean([x["full"] for x in keep])) if keep else None,
            "participant_first_capture":float(np.mean([x["first_hit"] for x in keep])) if keep else None,
            "participant_second_capture":float(np.mean([x["second_hit"] for x in keep])) if keep else None,
            "participant_third_capture":float(np.mean([x["third_hit"] for x in keep])) if keep else None,
            "soft_action_n":sum(x["action"]!="BASE" for x in vals),
        }
    # Dominant diagnostic only: high WIN score + strongest raw-score rank1 + strong line advantage.
    wins=np.asarray([r["probs"]["WIN"] for r in scored])
    wt=float(np.quantile(wins,.9))
    dom=[r for r in scored if r["probs"]["WIN"]>=wt and r["features"]["strong_z_score_rank"]<=1 and r["features"]["strong_vs_rival_line_score"]>0]
    out["DOMINANT_DIAGNOSTIC"]={
        "n":len(dom),
        "actual_win_rate":float(np.mean([r["state"]==0 for r in dom])) if dom else None,
        "actual_top3_rate":float(np.mean([r["state"]!=2 for r in dom])) if dom else None,
    }
    return out

def main():
    cal_logs,hist_map=calibration_logs_and_races()
    h1_logs=load_json("results/keirin_shogi/v37_shrunk_board_third/race_log.json")
    future_logs=load_json("results/keirin_shogi/v37_true_future_block1/race_log.json")
    future_map=load_future_bases()

    cal_rows=rows_from(cal_logs,hist_map)
    h1_rows=rows_from(h1_logs,hist_map)
    fut_rows=rows_from(future_logs,future_map)
    calA=[r for r in cal_rows if CAL_A[0]<=r["race"]["race_date"]<=CAL_A[1]]
    calB=[r for r in cal_rows if CAL_B[0]<=r["race"]["race_date"]<=CAL_B[1]]

    m1,n1=fit_state(calA)
    sb=score(calB,m1,n1)
    rule1=choose_rules(sb)["rules"]
    sh1=score(h1_rows,m1,n1)
    h1_eval=eval_block(sh1,rule1)

    # True-future setup: learn state model using only pre-April data, tune rules Apr-Jun.
    pre=[*cal_rows,*[r for r in h1_rows if H1_EARLY[0]<=r["race"]["race_date"]<=H1_EARLY[1]]]
    tune=[r for r in h1_rows if H1_LATE[0]<=r["race"]["race_date"]<=H1_LATE[1]]
    m2,n2=fit_state(pre)
    st=score(tune,m2,n2)
    rule2=choose_rules(st)["rules"]
    sf=score(fut_rows,m2,n2)
    fut_eval=eval_block(sf,rule2)

    # Production candidate after Aug: all historical participant labels are now prior.
    prod_train=[*cal_rows,*h1_rows,*fut_rows]
    mp,npnames=fit_state(prod_train)
    sp=score(fut_rows,mp,npnames)
    prod_rules=choose_rules(sp)["rules"]

    bundle={"model":mp,"feature_names":npnames,"rules":prod_rules,
            "model_name":"sevencar_v21_v31_v37_state_transfer",
            "trained_through":"2026-08-30",
            "policy":{"collapse":"extra v21 participant filter","soft":"keep row1 and duplicate strong into row2","third":"only adopt if true-future SECOND_THIRD beats SECOND"}}
    import joblib
    joblib.dump(bundle,OUT/"model.joblib",compress=3)

    report={
      "study":"sevencar_transfer_from_ninecar",
      "data":{"calibration_rows":len(cal_rows),"h1_rows":len(h1_rows),"true_future_rows":len(fut_rows)},
      "rules":{"h1_forward":rule1,"true_future":rule2,"production_after_2026_08_30":prod_rules},
      "evaluation":{"2026_h1_forward":h1_eval,"2026_07_08_true_future":fut_eval},
      "references":{
        "v37_h1":{"full":0.32093933463796476,"first":0.7123287671232876,"second":0.5909980430528375,"third":0.5342465753424658},
        "v37_true_future":{"full":0.3202614379084967,"first":0.673202614379085,"second":0.6078431372549019,"third":0.5228758169934641}
      },
      "guards":{"odds_used":False,"popularity_used":False,"payout_used":False,"true_future_used_for_pre_future_rules":False}
    }
    (OUT/"evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

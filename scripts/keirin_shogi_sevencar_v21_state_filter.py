#!/usr/bin/env python3
from __future__ import annotations

"""Seven-car v21 state-aware participation study on the full OOS race pool.

Train collapse risk from all v19/v21 OOS races, then use it only as an extra
filter after the existing v21 participation gate. True-future 2026 Jul-Aug is
used only for evaluation after pre-future rules are frozen.
"""

import csv, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/keirin_shogi/sevencar_v21_state_filter"
OUT.mkdir(parents=True,exist_ok=True)

def load_mod(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec); assert spec.loader is not None
    sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

v25=load_mod("v25_v21_state","scripts/keirin_shogi_v25_top2_pair_second.py")
STATES=("WIN","SOFT_FAIL","COLLAPSE")
RAW_KEYS=("z_score","z_win_rate","z_top2_rate","z_top3_rate","z_b_count","z_first_count","z_second_count","z_third_count","z_outside_count",
          "line_pos","line_size","line_mean_score_z","line_max_score_z","line_b_sum_z","line_attack_sum_z")

def ino(v):
    try:return int(float(v))
    except:return 0

def future_maps():
    base=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830"
    def read(n):
        with (base/n).open("r",encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
    races=read("races.csv"); entries=read("entries.csv"); results=read("results.csv")
    eb=defaultdict(list);rb=defaultdict(list)
    for x in entries:eb[x["race_id"]].append(x)
    for x in results:rb[x["race_id"]].append(x)
    return {r["race_id"]:r for r in races},eb,rb

def rank_desc(base,key):
    order=sorted(base,key=lambda n:(-float(base[n]["x"].get(key,0.0)),n))
    return {n:i+1 for i,n in enumerate(order)}

def make_features(base,strong,selector_score,threshold,candidate_count):
    ranks={k:rank_desc(base,k) for k in RAW_KEYS if k.startswith("z_")}
    x=base[strong]["x"]
    line=str(base[strong]["line_id"])
    line_members=defaultdict(list)
    for n in base:line_members[str(base[n]["line_id"])].append(n)
    def lm(lid,k):return float(np.mean([float(base[n]["x"].get(k,0.0)) for n in line_members[lid]]))
    strong_line=lm(line,"z_score")
    rivals=[lm(lid,"z_score") for lid in line_members if lid!=line]
    rival=max(rivals) if rivals else 0.0
    f={
      "selector_score":float(selector_score),"selector_margin":float(selector_score-threshold),
      "candidate_count":float(candidate_count),"strong_line_score":strong_line,
      "best_rival_line_score":rival,"strong_vs_rival_line":strong_line-rival,
      "strong_line_position":float(base[strong]["line_position"]),"strong_line_size":float(base[strong]["line_size"]),
      "num_lines":float(len(line_members))
    }
    for k in RAW_KEYS:
        f[f"strong_{k}"]=float(x.get(k,0.0))
        if k in ranks:f[f"strong_{k}_rank"]=float(ranks[k][strong])
    return f

def state(order,strong):
    if order[0]==strong:return 0
    if strong in order[1:3]:return 1
    return 2

def historical_rows():
    rows=v25.load_v19_details();v25.fit_v21_selector(rows);v25.mark_v21(rows)
    _,eb,rb=v25.raw_maps()
    out=[]
    for r in rows:
        rid=str(r["race_id"])
        order=v25.result_order(rb.get(rid,[]))
        if not order or len(eb.get(rid,[]))!=7:continue
        base=v25.race_features(eb[rid])
        strong=ino(r["ranking"][0]["no"])
        feats=make_features(base,strong,r["v21_selector_score"],r["v21_threshold"],r["candidate_count"])
        out.append({
          "race_id":rid,"date":r["week"],"features":feats,"state":state(order,strong),
          "participate":bool(r["v21_participate"]),"candidate_hit":int(order[0] in set(map(ino,r["candidates"]))),
          "top1_hit":int(order[0]==strong)
        })
    return out

def future_rows():
    logs=json.loads((ROOT/"results/keirin_shogi/v31_true_future_block1/race_log.json").read_text(encoding="utf-8"))
    _,eb,rb=future_maps()
    out=[]
    for log in logs:
        rid=str(log["race_id"]); order=v25.result_order(rb.get(rid,[]))
        if not order or len(eb.get(rid,[]))!=7:continue
        base=v25.race_features(eb[rid])
        strong=int(log["first_candidates"][0])
        feats=make_features(base,strong,float(log["v21_selector_score"]),float(log["v21_threshold"]),len(log["first_candidates"]))
        out.append({
          "race_id":rid,"date":log["race_date"],"features":feats,"state":state(order,strong),
          "participate":True,"candidate_hit":int(order[0] in set(map(int,log["first_candidates"]))),
          "top1_hit":int(order[0]==strong)
        })
    return out

def fit(rows):
    names=list(rows[0]["features"]);X=np.asarray([[r["features"][n] for n in names] for r in rows],float);y=np.asarray([r["state"] for r in rows],int)
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,class_weight="balanced",random_state=2137));m.fit(X,y)
    return m,names

def score(rows,m,names):
    X=np.asarray([[r["features"][n] for n in names] for r in rows],float);pp=m.predict_proba(X)
    out=[]
    for r,p in zip(rows,pp):
        q=dict(r);q["collapse"]=float(p[2]);q["soft"]=float(p[1]);q["win"]=float(p[0]);out.append(q)
    return out

def choose(scored):
    part=[r for r in scored if r["participate"]]
    vals=np.asarray([r["collapse"] for r in part])
    best=None
    for q in (0.70,0.75,0.80,0.85,0.90,0.95):
        th=float(np.quantile(vals,q));keep=[r for r in part if r["collapse"]<th]
        if len(keep)<0.65*len(part):continue
        cap=float(np.mean([r["candidate_hit"] for r in keep]));top=float(np.mean([r["top1_hit"] for r in keep]))
        collapse=float(np.mean([r["state"]==2 for r in keep]))
        key=(cap,top,-collapse,len(keep))
        rec={"threshold":th,"keep_rate":len(keep)/len(part),"candidate_capture":cap,"top1_hit":top,"collapse_rate":collapse}
        if best is None or key>best[0]:best=(key,rec)
    return best[1]

def evaluate(scored,rule):
    part=[r for r in scored if r["participate"]]
    keep=[r for r in part if r["collapse"]<rule["threshold"]]
    cut=[r for r in part if r["collapse"]>=rule["threshold"]]
    def m(x):
        return {"n":len(x),"candidate_capture":float(np.mean([r["candidate_hit"] for r in x])) if x else None,
                "top1_hit":float(np.mean([r["top1_hit"] for r in x])) if x else None,
                "collapse_rate":float(np.mean([r["state"]==2 for r in x])) if x else None}
    return {"base":m(part),"keep":m(keep),"cut":m(cut),"keep_fraction_of_v21":len(keep)/len(part) if part else None}

def main():
    hist=historical_rows();fut=future_rows()
    train=[r for r in hist if "2025-06-30"<=r["date"]<="2025-10-26"]
    cal=[r for r in hist if "2025-10-27"<=r["date"]<="2025-12-28"]
    h1=[r for r in hist if "2025-12-29"<=r["date"]<="2026-06-28"]
    m1,n1=fit(train);sc=score(cal,m1,n1);rule1=choose(sc);h1s=score(h1,m1,n1)

    # true future: learn through Mar, tune Apr-Jun, then freeze.
    pre=[r for r in hist if "2025-06-30"<=r["date"]<="2026-03-31"]
    tune=[r for r in hist if "2026-04-01"<=r["date"]<="2026-06-28"]
    m2,n2=fit(pre);st=score(tune,m2,n2);rule2=choose(st);sf=score(fut,m2,n2)

    # production after Aug.
    prod=[*hist,*fut];mp,npn=fit(prod);spr=score(fut,mp,npn);rulep=choose(spr)
    import joblib
    joblib.dump({"model":mp,"feature_names":npn,"rule":rulep,"model_name":"sevencar_v21_state_filter","trained_through":"2026-08-30"},OUT/"model.joblib",compress=3)

    report={"study":"sevencar_v21_state_filter_full_oos","counts":{"train":len(train),"cal":len(cal),"h1":len(h1),"future_participants":len(fut)},
      "rules":{"h1_forward":rule1,"true_future":rule2,"production":rulep},
      "evaluation":{"2026_h1":evaluate(h1s,rule1),"2026_07_08_true_future":evaluate(sf,rule2)},
      "guards":{"odds_used":False,"popularity_used":False,"payout_used":False,"future_used_for_prefuture_rule":False}}
    (OUT/"evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

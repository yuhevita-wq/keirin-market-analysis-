#!/usr/bin/env python3
from __future__ import annotations

"""Identify which conventionally lower-ranked rider causes a reversal.

Train only on 2024 out-of-sample baseline predictions, evaluate on 2025 and
2026 H1. Candidate pools are riders ranked 4-9 by the rider-only baseline.
"""

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "scripts/keirin_shogi_ninecar_v3.py"
OUT = ROOT / "results/keirin_shogi/ninecar_v32_reversal_actor_study.json"


def load_v3():
    spec = importlib.util.spec_from_file_location("ninecar_v3_actor_base", V3_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


v3 = load_v3()
v2 = v3.v2
METRICS = ("score", "win_rate", "top2_rate", "top3_rate", "b_count", "first_count", "outside_count")
STAGES = tuple(v2.STAGE_PATTERNS)


def rank_map(vals, reverse=True):
    order = sorted(vals, key=lambda n: ((-vals[n]) if reverse else vals[n], n))
    return {n:i+1 for i,n in enumerate(order)}


def baseline(race, models):
    base, _ = v2.base_map(race)
    x = np.asarray([base[n] for n in range(1,10)])
    p1a = v2.normalize(models["first"].predict_proba(x)[:,1])
    p2a = v2.normalize(models["second"].predict_proba(x)[:,1])
    p1 = {n:float(p1a[n-1]) for n in range(1,10)}
    p2 = {n:float(p2a[n-1]) for n in range(1,10)}
    return p1,p2


def race_candidate_rows(race, models, target):
    em = {v2.ino(e.get("car_no")):e for e in race.entries}
    p1,p2 = baseline(race, models)
    pmain = p1 if target=="first" else p2
    ranks = rank_map(pmain)
    p1r = rank_map(p1); p2r = rank_map(p2)
    top = min(p1, key=lambda n:(-p1[n],n))
    top2_strength = sorted(p1,key=lambda n:(-p1[n],n))[1]

    metric_vals = {m:{n:v2.fnum(em[n].get(m)) for n in range(1,10)} for m in METRICS}
    metric_ranks = {
        m:rank_map(vals, reverse=(m!="outside_count"))
        for m,vals in metric_vals.items()
    }

    line_mass=defaultdict(float)
    line_members=defaultdict(list)
    for n in range(1,10):
        lid=v2.ino(em[n].get("line_id"))
        line_mass[lid]+=p1[n]
        line_members[lid].append(n)
    top_line=v2.ino(em[top].get("line_id"))
    top_line_mass=line_mass[top_line]
    strongest_rival=max((m for lid,m in line_mass.items() if lid!=top_line), default=0.0)

    rows=[]
    for n in range(1,10):
        if ranks[n] < 4:
            continue
        e=em[n]
        lid=v2.ino(e.get("line_id"))
        members=line_members[lid]
        within=sorted(members,key=lambda x:(-p1[x],x))
        within_rank=within.index(n)+1
        cand_rank_spread=float(np.std([metric_ranks[m][n] for m in METRICS]))
        top_rank_spread=float(np.std([metric_ranks[m][top] for m in METRICS]))
        feats={
            "candidate_main_prob":pmain[n],
            "candidate_main_rank":float(ranks[n]),
            "candidate_p1":p1[n],
            "candidate_p1_rank":float(p1r[n]),
            "candidate_p2":p2[n],
            "candidate_p2_rank":float(p2r[n]),
            "candidate_line_position":float(v2.ino(e.get("line_position"))),
            "candidate_line_size":float(v2.ino(e.get("line_size"))),
            "candidate_same_line_as_top":float(lid==top_line),
            "candidate_line_mass":float(line_mass[lid]),
            "candidate_line_vs_top_mass":float(line_mass[lid]-top_line_mass),
            "candidate_line_vs_strongest_rival":float(line_mass[lid]-strongest_rival),
            "candidate_within_line_p1_rank":float(within_rank),
            "candidate_line_top3_count":float(sum(p1r[x]<=3 for x in members)),
            "candidate_line_top5_count":float(sum(p1r[x]<=5 for x in members)),
            "candidate_metric_rank_spread":cand_rank_spread,
            "top_p1":p1[top],
            "top_p1_gap":p1[top]-p1[top2_strength],
            "top_line_mass":float(top_line_mass),
            "strongest_rival_line_mass":float(strongest_rival),
            "top_metric_rank_spread":top_rank_spread,
            "num_lines":float(len(line_mass)),
        }
        for m in METRICS:
            feats[f"candidate_{m}_rank"]=float(metric_ranks[m][n])
            feats[f"candidate_minus_top_{m}"]=float(metric_vals[m][n]-metric_vals[m][top])
        for g in ("G1","G2","G3"):
            feats[f"grade_{g}"]=float(race.grade==g)
        for s in STAGES:
            feats[f"stage_{s}"]=float(s in race.race_type)
        actual = race.order[0] if target=="first" else race.order[1]
        rows.append({
            "race_id":race.race_id,
            "features":feats,
            "label":int(n==actual),
            "candidate":n,
            "baseline_rank":int(ranks[n]),
            "actual":actual,
            "is_reversal":bool(ranks[actual]>=4),
        })
    return rows


def build_year(year,races,target):
    models=v3.fit_models([r for r in races if r.year<year])
    out=[]
    for race in [r for r in races if r.year==year]:
        out.extend(race_candidate_rows(race,models,target))
    return out


def fit_actor(train_rows):
    feature_names=list(train_rows[0]["features"].keys())
    X=np.asarray([[r["features"][f] for f in feature_names] for r in train_rows],dtype=float)
    y=np.asarray([r["label"] for r in train_rows],dtype=int)
    model=make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000,class_weight="balanced",random_state=20260918)
    )
    model.fit(X,y)
    coefs=model.named_steps["logisticregression"].coef_[0]
    imp=sorted(
        [{"feature":f,"coef":float(c)} for f,c in zip(feature_names,coefs)],
        key=lambda x:abs(x["coef"]),reverse=True
    )
    return model,feature_names,imp


def evaluate_actor(model,feature_names,rows):
    by=defaultdict(list)
    for r in rows: by[r["race_id"]].append(r)
    reversal=[]
    for rid,group in by.items():
        if not group[0]["is_reversal"]:
            continue
        X=np.asarray([[r["features"][f] for f in feature_names] for r in group],dtype=float)
        scores=model.predict_proba(X)[:,1]
        ranked=[r for _,r in sorted(zip(scores,group),key=lambda x:(-x[0],x[1]["candidate"]))]
        actual=group[0]["actual"]
        pos=next(i+1 for i,r in enumerate(ranked) if r["candidate"]==actual)
        naive_ranked=sorted(group,key=lambda r:(r["baseline_rank"],r["candidate"]))
        naive_pos=next(i+1 for i,r in enumerate(naive_ranked) if r["candidate"]==actual)
        reversal.append({
            "race_id":rid,
            "actual":actual,
            "actor_rank":pos,
            "naive_rank":naive_pos,
        })
    n=len(reversal)
    return {
        "reversal_races":n,
        "actor_top1_capture":float(np.mean([x["actor_rank"]<=1 for x in reversal])) if n else None,
        "actor_top2_capture":float(np.mean([x["actor_rank"]<=2 for x in reversal])) if n else None,
        "actor_top3_capture":float(np.mean([x["actor_rank"]<=3 for x in reversal])) if n else None,
        "naive_top1_capture":float(np.mean([x["naive_rank"]<=1 for x in reversal])) if n else None,
        "naive_top2_capture":float(np.mean([x["naive_rank"]<=2 for x in reversal])) if n else None,
        "naive_top3_capture":float(np.mean([x["naive_rank"]<=3 for x in reversal])) if n else None,
        "actor_minus_naive_top1":float(np.mean([x["actor_rank"]<=1 for x in reversal])-np.mean([x["naive_rank"]<=1 for x in reversal])) if n else None,
        "actor_minus_naive_top2":float(np.mean([x["actor_rank"]<=2 for x in reversal])-np.mean([x["naive_rank"]<=2 for x in reversal])) if n else None,
        "actor_minus_naive_top3":float(np.mean([x["actor_rank"]<=3 for x in reversal])-np.mean([x["naive_rank"]<=3 for x in reversal])) if n else None,
        "mean_actor_rank":float(np.mean([x["actor_rank"] for x in reversal])) if n else None,
        "mean_naive_rank":float(np.mean([x["naive_rank"] for x in reversal])) if n else None,
        "rank_distribution":{str(k):int(sum(x["actor_rank"]==k for x in reversal)) for k in range(1,7)},
    }


def main():
    races=v2.load_races()
    report={"study":"ninecar_v32_reversal_actor"}
    for target in ("first","second"):
        rows24=build_year(2024,races,target)
        rows25=build_year(2025,races,target)
        rows26=build_year(2026,races,target)
        model,names,imp=fit_actor(rows24)
        report[target]={
            "candidate_definition":"baseline rank 4-9 only",
            "training":"2024 OOS candidates",
            "importance_top20":imp[:20],
            "2025_forward":evaluate_actor(model,names,rows25),
            "2026_h1_forward":evaluate_actor(model,names,rows26),
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

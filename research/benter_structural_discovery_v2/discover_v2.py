#!/usr/bin/env python3
"""Pre-registered fundamental prediction comparisons, independently of market odds.

Candidate discovery: lineup-role and teammate/rival interaction information.
Out-of-time holdout 2026 H1 is evaluated once after selecting regularization
on 2025 validation, never used to select candidate features or tune a model.
VERY IMPORTANT: historical racecards are retrospectively captured and
their pre-race as-of fidelity is NOT certified. Hence exploratory only.
"""
from __future__ import annotations
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/benter_fundamental_v1"))
import train_v1 as v1

# V1's matrix builder has a hard-coded 17-feature dimension; replace only
# its padding with an input-dimension-aware version for the ablation study.
def matrix_variable_width(races, scaler):
    n = len(races)
    p = len(scaler.median) * 2
    x = np.zeros((n, 9, p), dtype=float)
    mask = np.zeros((n, 9), dtype=bool)
    winners = np.zeros(n, dtype=int)
    for i, race in enumerate(races):
        k = len(race["cars"])
        x[i, :k] = scaler.apply(race["x"])
        mask[i, :k] = True
        winners[i] = race["winner"]
    return x, mask, winners

v1.matrix = matrix_variable_width

OUT = ROOT / "research/benter_structural_discovery_v2/output"
BASE_FEATURES = list(v1.FEATURES)
STRUCT_FEATURES = [
    "leader", "second", "third_or_later", "solo", "line_size",
    "leader_b", "follower_mark", "lead_follow_synergy",
    "line_leader_first", "line_follow_first",
    "line_leader_pressure_share", "rival_front_b",
    "lead_attack_versatility", "line_mate_count",
    "line_head_vs_follower_win_gap", "own_role_front_b",
    "solo_front_pressure", "teammate_score_support",
    "line_score_imbalance"
]
ROLE_ONLY = ["leader", "second", "third_or_later", "solo", "line_size", "line_mate_count"]
CANDIDATES = {
    "score_only": (["score"], []),
    "score_plus_role": (["score"], ROLE_ONLY),
    "no_score_plus_role": ([f for f in BASE_FEATURES if f != "score"], ROLE_ONLY),
    "full_plus_role": (BASE_FEATURES, ROLE_ONLY),
    "raw_without_score": ([f for f in BASE_FEATURES if f != "score"], []),
    "score_plus_line": (["score"], [f for f in STRUCT_FEATURES if f not in ("teammate_score_support", "line_score_imbalance")]),
    "no_score_plus_line": ([f for f in BASE_FEATURES if f != "score"], [f for f in STRUCT_FEATURES if f not in ("teammate_score_support", "line_score_imbalance")]),
    "full_v1": (BASE_FEATURES, []),
    "full_plus_line": (BASE_FEATURES, STRUCT_FEATURES),
}
ALPHAS = (0.0001, 0.003, 0.03)
FIX_RE = re.compile(r"^[1-9](?:-[1-9])*$")


def parse_line(s, cars):
    if not s:
        return None
    bits = [b.strip() for b in s.split("/")]
    if not bits or any(not FIX_RE.match(b) for b in bits):
        return None
    parsed = [[int(x) for x in b.split("-")] for b in bits]
    flat = [x for line in parsed for x in line]
    if len(flat) != len(set(flat)) or set(flat) != set(cars):
        return None
    return parsed


def numeric(v):
    x = v1.as_number(v)
    return float(x) if np.isfinite(x) else 0.0


def build_interaction(r, lines):
    n = len(r["cars"])
    result = np.zeros((n, len(STRUCT_FEATURES)), dtype=float)
    if lines is None:
        return result
    cars = r["cars"]
    lookup = {car:idx for idx, car in enumerate(cars)}
    def val(car, field):
        return numeric(r["x"][lookup[car], BASE_FEATURES.index(field)])
    def l1(car, field):
        return math.log1p(max(val(car, field), 0.0))
    heads = [b[0] for b in lines]
    for lineup in lines:
        leader = lineup[0]
        front_b = l1(leader, "b_count")
        mark = l1(lineup[1], "mark_count") if len(lineup) >= 2 else 0.0
        first = l1(leader, "first_count")
        follower_first = l1(lineup[1], "first_count") if len(lineup) >= 2 else 0.0
        competitors = [h for h in heads if h != leader]
        pressure = sum(max(val(h, "b_count"),0) for h in competitors)
        pressure_max = max((max(val(h,"b_count"),0) for h in competitors),default=0)
        all_front = sum(max(val(h,"b_count"),0) for h in heads)
        front_share = max(val(leader,"b_count"),0) / (1.0+all_front)
        leader_score = val(leader,"score")
        followers_score = [val(c,"score") for c in lineup[1:]]
        teammate_score = np.mean(followers_score) if followers_score else 0.0
        head_vs_fol = first - follower_first
        for pos,car in enumerate(lineup):
            # Compare leader's and line-mates' interactive roles, rather than ranking solo strengths.
            pos_indicator = [float(pos==0),float(pos==1),float(pos>=2),float(len(lineup)==1)]
            lead_b = front_b if pos>=1 else 0.
            support_mark = mark if pos==0 else 0.
            synergy = math.sqrt(max(0,front_b * mark)) if pos<=1 else 0.
            own_front = math.log1p(max(val(car,"b_count"),0)) if pos==0 else 0.
            teammate_strength = teammate_score if pos==0 else leader_score
            m = [
                *pos_indicator,
                float(len(lineup)),lead_b,support_mark,synergy,
                first if pos>=1 else 0.,follower_first if pos==0 else 0.,
                front_share if pos==0 else 0.,
                math.log1p(pressure_max) if pos==0 else 0.,
                math.log1p(max(val(leader,"nige_count"),0)+max(val(leader,"makuri_count"),0)) if pos==0 else 0.,
                float(len(lineup)-1),
                head_vs_fol if pos<=1 else 0.,
                own_front,
                float(len(lineup)==1)*math.log1p(pressure),
                teammate_strength,
                abs(teammate_strength - val(car,"score")),
            ]
            result[lookup[car]] = np.array(m)
    assert result.shape[1] == len(STRUCT_FEATURES)
    return result


def read_line_maps():
    maps = {}
    by_dir = Counter()
    for d in v1.paths():
        p = d / "races.csv"
        if not p.exists():
            by_dir["missing_race_metadata"] += 1
            continue
        with p.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                maps[row["race_id"]] = row.get("predicted_line_formation","")
    return maps


def make_views(all_races, raw_fields, struct_fields):
    base_ix = [BASE_FEATURES.index(k) for k in raw_fields]
    line_ix = [STRUCT_FEATURES.index(k) for k in struct_fields]
    return [
        {**r, "x":np.concatenate((r["x"][:,base_ix], r["line_x"][:,line_ix]),axis=1)}
        for r in all_races
    ]


def pred_and_metrics(races, scaler, beta):
    x, mask, winners = v1.matrix(races, scaler)
    p = v1.probabilities(x, mask, beta)
    idx = np.arange(len(races))
    wp = np.maximum(p[idx,winners], 1e-300)
    score = {
        "n_races":len(races),
        "log_loss":float(-np.log(wp).mean()),
        "mean_uniform_log_loss":float(np.mean([math.log(len(r["cars"])) for r in races])),
        "top1_accuracy":float(np.mean(np.argmax(p,axis=1)==winners)),
        "brier": float(np.mean([
            np.square(p[i,:len(r["cars"])] - (np.arange(len(r["cars"])) == r["winner"])).sum()
            for i,r in enumerate(races)
        ])),
    }
    return score, wp, p


def bootstrap_delta(challenger, baseline, seed=20261017, reps=2000):
    # Positive means challenger beats baseline in loss. Resample races, not horses.
    gains = np.log(np.maximum(challenger,1e-300)) - np.log(np.maximum(baseline,1e-300))
    rng = np.random.default_rng(seed)
    vals = [float(gains[rng.integers(len(gains),size=len(gains))].mean()) for _ in range(reps)]
    return {
        "n_races": len(gains),
        "mean_improvement": float(gains.mean()),
        "bootstrap_95pct_interval": [float(v) for v in np.quantile(vals,[.025,.975])],
        "interpretation": "positive=lower log loss than baseline; interval crossing zero is inconclusive"
    }


def main():
    races, audit, by_dir = v1.read_data()
    maps = read_line_maps()
    count = Counter()
    for r in races:
        lines = parse_line(maps.get(r["id"], ""), r["cars"])
        if lines is None:
            count["no_valid_line"] += 1
        else:
            count["valid_pre_race_published_line_unverified_snapshot"] += 1
            count["with_two_or_more_lines"] += int(len(lines)>=2)
        r["line_x"] = build_interaction(r,lines)
    results = {}
    winning_p = {}
    model_info = {}
    candidate_factors = {}
    validation_counts = {}
    for name,(base_fields,line_fields) in CANDIDATES.items():
        view = make_views(races,base_fields,line_fields)
        train,val,test = v1.partitions(view)
        scaler = v1.Transform(train)
        validations = []
        for alpha in ALPHAS:
            beta = v1.fit(train,scaler,alpha)
            scores,wp,p = pred_and_metrics(val,scaler,beta)
            validations.append({"alpha":alpha,"validation_log_loss":scores["log_loss"]})
        pick = min(validations,key=lambda z:z["validation_log_loss"])["alpha"]
        final_transform = v1.Transform(train+val)
        fit_coefficients = v1.fit(train+val,final_transform,pick)
        testscore, wp, predictions = pred_and_metrics(test,final_transform,fit_coefficients)
        validations.sort(key=lambda z:z["validation_log_loss"])
        results[name] = {"selected_alpha":pick,"validation_candidates":validations,**testscore}
        winning_p[name] = wp
        candidate_factors[name] = {"raw":base_fields,"structure":line_fields}
        model_info[name] = {
            "fields":base_fields+line_fields, "scaler":final_transform.export(),
            "coefficients":fit_coefficients.tolist(), "alpha":pick,
        }
        if name in ("score_only","raw_without_score","no_score_plus_line","full_plus_line"):
            OUT.mkdir(parents=True,exist_ok=True)
            with (OUT / f"test_{name}_win_probability.csv").open("w",encoding="utf-8",newline="") as file:
                w = csv.writer(file)
                w.writerow(("race_id","date","car_no","estimated_win_probability","actual_winner"))
                for i,r in enumerate(test):
                    for j,car in enumerate(r["cars"]):
                        w.writerow((r["id"],r["date"],car,f"{predictions[i,j]:.10f}",int(j==r["winner"])))
        print(name,json.dumps(testscore),flush=True)
    comparisons = {}
    for name in CANDIDATES:
        if name=="score_only":continue
        comparisons[f"{name}_vs_score_only"]=bootstrap_delta(winning_p[name],winning_p["score_only"])
    for a,b in [("full_plus_line","full_v1"),("no_score_plus_line","raw_without_score"),
                ("full_plus_line","full_plus_role"),("score_plus_line","score_plus_role"),
                ("no_score_plus_line","no_score_plus_role")]:
        comparisons[f"{a}_vs_{b}"]=bootstrap_delta(winning_p[a],winning_p[b])
    report = {
        "design":"Pre-declared score-only, no-score and line-teammate models; 2026 holdout evaluated once.",
        "scope":"F1 S-class stored racecards; other race classes NOT evaluated",
        "status":"EXPLORATORY_NOT_AS_OF_CERTIFIED",
        "serious_data_limitation":"Historical entries and line predictions gathered in 2026; pre-race as-of values unverified. No live predictive or profit claim.",
        "audit":audit,"dir_audit":by_dir,"line_audit":dict(count),
        "splits":{"training":"2023-2024","validation":"2025","test":"2026-01 to 2026-06"},
        "models":results,"comparisons":comparisons,
        "discoveries_are_associations_not_proof_of_profitable_edges":True,
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"models.json").write_text(json.dumps(model_info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("comparison:",json.dumps(comparisons,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

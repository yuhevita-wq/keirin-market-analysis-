#!/usr/bin/env python3
"""Past-only F1 S-class independent V2 win forecasts, special foreign term=999 encoded as a category.

Target results are deliberately never opened during the forecast stage.
Retrospective racecards are NOT equivalent to a verified live as-of snapshot.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/benter_fundamental_v1"))
sys.path.insert(0, str(ROOT / "research/benter_structural_discovery_v2"))
import train_v1 as v1
import discover_v2 as v2

OUT = Path(__file__).resolve().parent / "frozen_term_unscaled"
ALPHA = .0001  # fixed V2 full_plus_line baseline; no tuning on target outcomes
PERIODS = {
    **{f"{year}q{q}": (f"{year}-{3*q-2:02d}-01",
                       f"{year+int(q==4)}-{3*(q%4)+1:02d}-01")
       for year in (2024, 2025) for q in range(1,5)},
    "2026q1": ("2026-01-01","2026-04-01"),
    "2026q2": ("2026-04-01","2026-07-01"),
    "2026julaug": ("2026-07-01","2026-09-01"),
    "2026sep": ("2026-09-01","2026-10-01"),
}

def sources():
    dirs = []
    for year in (2023,2024,2025):
        for q in range(1,5):
            d = ROOT / f"data/{year}/s_class_f1_all_parts/{year}_q{q}"
            if (d/"entries.csv").exists() and (d/"races.csv").exists():
                dirs.append(d)
    for p in ("data/2026_h1/s_class_f1_all",
              "data/2026_future_block1/s_class_f1_20260701_20260830",
              "research/benter_calibration_v4/sep_holdout"):
        d = ROOT/p
        if (d/"entries.csv").exists() and (d/"races.csv").exists():
            dirs.append(d)
    return dirs

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:
        yield from csv.DictReader(handle)

def is_f1_s(r):
    grade=unicodedata.normalize("NFKC",(r.get("meeting_grade") or "").strip()).upper()
    group=unicodedata.normalize("NFKC",(r.get("race_type") or "").strip()).upper()
    return grade=="F1" and group.startswith("S級")

def ids_sha(ids):
    return hashlib.sha256(("\n".join(sorted(ids))+"\n").encode()).hexdigest()

def load(start,end):
    training=[]; target=[]; audit=Counter(); seen=set()
    for d in sources():
        cards={r["race_id"]:r for r in rows(d/"races.csv") if r.get("race_id")}
        entrants=defaultdict(list)
        for r in rows(d/"entries.csv"):
            if (r.get("race_date") or "")<end:
                entrants[r["race_id"]].append(r)
        past={rid for rid,ee in entrants.items() if ee[0].get("race_date","")<start}
        winners=defaultdict(list)
        if past and (d/"results.csv").exists():
            # Only earlier winners may enter the fit. This does not use target labels.
            for r in rows(d/"results.csv"):
                if r.get("race_id") in past and str(r.get("finish_position","")).strip()=="1":
                    winners[r["race_id"]].append(r.get("car_no"))
        for rid,ee in entrants.items():
            if rid in seen:
                audit["duplicate_race_id"]+=1
                continue
            seen.add(rid)
            card=cards.get(rid)
            if card is None or not is_f1_s(card):
                audit["missing_card_or_wrong_scope"]+=1
                continue
            date=ee[0].get("race_date","")
            if not date or any(e.get("race_date")!=date for e in ee):
                audit["invalid_date"]+=1
                continue
            try:
                ee.sort(key=lambda e:int(e["car_no"]))
                cars=[int(e["car_no"]) for e in ee]
            except (ValueError,KeyError):
                audit["invalid_car_id"]+=1
                continue
            if not 5<=len(cars)<=9 or len(set(cars))!=len(cars):
                audit["irregular_field"]+=1
                continue
            lines=v2.parse_line(card.get("predicted_line_formation",""),cars)
            if lines is None:
                audit["unusable_line"]+=1
                continue
            x=np.asarray([[v1.as_number(e.get(key)) for key in v1.FEATURES] for e in ee],dtype=float)
            if not np.isfinite(x).any():
                audit["all_features_missing"]+=1
                continue
            obj={"id":rid,"date":date,"cars":cars,"x":x}
            # KDreams uses "999" as a special FOREIGN rider category, NOT a
            # 999th training graduation class. Never scale it as ordinal 999.
            # Preserve the categorical information and keep those riders IN scope.
            foreign=(x[:,v1.FEATURES.index("term")]==999.0).astype(float)
            x[foreign>0,v1.FEATURES.index("term")]=np.nan
            obj["x"]=x
            line_x=v2.build_interaction(obj,lines)
            obj["x"]=np.concatenate((x,line_x,foreign.reshape(-1,1)),axis=1)
            if date<start:
                ww=winners.get(rid,[])
                if len(ww)!=1 or not ww[0] or int(ww[0]) not in cars:
                    audit["historical_winner_missing_or_ambiguous"]+=1
                    continue
                obj["winner"]=cars.index(int(ww[0]))
                training.append(obj)
            elif date<end:
                target.append(obj)
    training.sort(key=lambda r:(r["date"],r["id"]))
    target.sort(key=lambda r:(r["date"],r["id"]))
    if len(training)<2000 or not target:
        raise RuntimeError(f"Training/target race count insufficient: {len(training)}/{len(target)}")
    if training[-1]["date"]>=target[0]["date"]:
        raise RuntimeError("Past-only cutoff violated")
    return training,target,audit

def forecast(period):
    start,end=PERIODS[period]
    train,target,audit=load(start,end)
    transform=v1.Transform(train)
    # Rare binary categories must NOT be divided by their tiny empirical
    # standard deviation. Keep the feature at 0/1, under the existing L2
    # penalty. Ordinary continuous term remains standardized as usual.
    category_idx=len(v1.FEATURES)+len(v2.STRUCT_FEATURES)
    assert category_idx==len(train[0]["x"][0])-1
    transform.median[category_idx]=0.
    transform.mean[category_idx]=0.
    transform.std[category_idx]=1.
    # Sparse 0/1 foreign feature needs more L-BFGS iterations for convergence.
    # Numerical optimization only; objective, penalty, and features unchanged.
    base_minimize=v1.minimize
    def extend_iterations(*args,**kwargs):
        kwargs["options"]={**kwargs.get("options",{}),"maxiter":2400}
        return base_minimize(*args,**kwargs)
    v1.minimize=extend_iterations
    beta=v1.fit(train,transform,ALPHA)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f"independent_win_{period}.csv"
    with path.open("w",encoding="utf-8",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(("race_id","race_date","car_no","independent_win_probability"))
        for race in target:
            z=transform.apply(race["x"])@beta
            p=np.exp(z-logsumexp(z))
            if not np.isfinite(p).all() or np.min(p)<=0 or abs(p.sum()-1)>1e-10:
                raise RuntimeError(f"Invalid probability: {race['id']}")
            for car,v in zip(race["cars"],p):
                w.writerow((race["id"],race["date"],car,format(float(v),".16g")))
    manifest={
        "status":"FROZEN_PRIOR_TRAINED_FORECAST_NO_TARGET_LABELS_READ",
        "scope":"F1 S級 only",
        "period":period,
        "period_start":start,"period_end_exclusive":end,
        "train_max_date":train[-1]["date"],
        "training_races":len(train),
        "training_ids_sha256":ids_sha([r["id"] for r in train]),
        "target_races":len(target),
        "target_ids_sha256":ids_sha([r["id"] for r in target]),
        "target_rider_rows":sum(len(r["cars"]) for r in target),
        "field_sizes":dict(Counter(len(r["cars"]) for r in target)),
        "V2_model_family":"full_plus_line logistic + foreign_term999_unscaled_binary_category",
        "alpha_fixed":ALPHA,
        "term999_encoding":"numeric term treated missing; foreign indicator 0/1 NEVER standardized by rarity; preserve all F1 S-class riders",
        "feature_names":v2.CANDIDATES["full_plus_line"][0]+v2.CANDIDATES["full_plus_line"][1]+["term_999_foreign_category"],
        "market_odds_used":False,
        "target_results_read":False,
        "V4_calibration_applied":False,
        "V4_note":"V4 was trained on 2026 July-Aug and must not be back-applied to earlier periods as a past-only transform",
        "pre_race_asof_snapshot_verified":False,
        "existing_V7_params_unchanged":True,
        "prediction_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "audit":dict(audit),
        "model":{"coefficients":beta.tolist(),"transform":transform.export()},
    }
    (OUT/f"manifest_{period}.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("model","feature_names")},
                     ensure_ascii=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--period",required=True,choices=tuple(PERIODS))
    forecast(parser.parse_args().period)

"""Accuracy-only learner for a keirin outcome-model selection branch.

Primary metric (the paper's winner-accuracy analogue): ONE 1st-place rider
per race, accuracy on 2023 October development races.
Secondary: ONE exact ordered 3連単 among all permutations, not a box.
For conditional 2nd/3rd probabilities, use a strictly frozen 2023 Jan-Aug
model (the original 2023 autonomous generation-01). Candidate algorithms
can learn the 1st-place model; they DO NOT optimize market odds or ROI.
2023 Nov-Dec are tested at most ONCE after the development goal is met.
Do not advertise this as an untouched published-paper replication.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_lines_2023 import load, matrix, feature
from chapter1_autonomous import keep_columns, SEED
from chapter1_blind_exam import load_frozen_champion
from backtest_line_rf_2024_q1 import infer_new_chunk

OUT = Path("docs/paper-only-ai-2023/chapter1_accuracy_ai")
STATE = OUT / "state.json"
LATEST = OUT / "latest.json"
MAX_GENERATIONS = 24
GENERATIONS_PER_RUN = 4
MIN_OCT_RACES = 1200
# Research target, not a threshold from Walsh & Joshi's paper:
WIN_TARGET = 0.45
TRIFECTA_TOP1_TARGET = 0.115
WIN_BASELINE_REQUIRED_GAIN = 0.01
TRIFECTA_BASELINE_REQUIRED_GAIN = 0.005
CONFIRM_WIN_TARGET = 0.44
CONFIRM_TRI_TARGET = 0.10
TEMPERATURES = (0.75, 1.0, 1.25, 1.5)
EPS = 1e-12


def create_state():
    return {
        "schema": "chapter1-accuracy-learner-v1",
        "generation": 0,
        "goal": {"winner_top1": WIN_TARGET, "exact_trifecta_top1": TRIFECTA_TOP1_TARGET,
                 "minimum_gain_vs_october_frozen_baseline_winner": WIN_BASELINE_REQUIRED_GAIN,
                 "minimum_gain_vs_october_frozen_baseline_trifecta": TRIFECTA_BASELINE_REQUIRED_GAIN,
                 "min_october_races": MIN_OCT_RACES,
                 "confirm_nov_dec_winner": CONFIRM_WIN_TARGET,
                 "confirm_nov_dec_trifecta": CONFIRM_TRI_TARGET},
        "champion": None, "attempts": [], "confirmation_used": False,
        "confirmation": None, "status": "SEARCHING",
        "max_generations_on_same_october_data": MAX_GENERATIONS,
    }


def validate_state(state):
    if state.get("schema") != "chapter1-accuracy-learner-v1":
        raise RuntimeError("Research state schema mismatch")
    if int(state["generation"]) != len(state["attempts"]):
        raise RuntimeError("Generation research history mismatch")
    if state["goal"] != create_state()["goal"]:
        raise RuntimeError("Unapproved after-the-fact accuracy target change")
    if state["generation"] > MAX_GENERATIONS:
        raise RuntimeError("Unbounded fixed-development data exploration")


def hypothesis_eligible(score, targets):
    return (score["races"] >= MIN_OCT_RACES
            and score["winner_top1"] > targets["winner"]
            and score["trifecta_top1"] > targets["trifecta"])


def generate_config(generation, index, champion):
    rng = np.random.default_rng(SEED + 1019 * generation + index)
    if generation == 1:
        baselines = [
            {"family": "extra_trees", "n_estimators": 160,
             "max_depth": 12, "min_samples_leaf": 8,
             "max_features": 0.8, "mask": "all"},
            {"family": "hgb", "max_iter": 180, "max_leaf_nodes": 20,
             "learning_rate": 0.045, "min_samples_leaf": 60,
             "l2_regularization": 12.0, "mask": "all"},
            {"family": "rf", "n_estimators": 150, "max_depth": 12,
             "min_samples_leaf": 10, "max_features": 0.75, "mask": "all"},
            {"family": "lr", "C": 0.75, "mask": "all"},
        ]
        if index <= len(baselines):
            return baselines[index-1]
    family = str(rng.choice(["hgb", "rf", "extra_trees", "lr"],
                            p=[0.35, 0.2, 0.35, 0.10]))
    masks = ("all", "drop_age", "drop_term", "drop_gear",
             "drop_rates", "drop_style")
    mask = str(rng.choice(masks, p=[0.55, 0.1, 0.05, 0.05, 0.1, 0.15]))
    if champion and rng.random() < 0.4:
        parent = champion["config"]
        family = parent["family"]
        if rng.random() < 0.6:
            mask = parent["mask"]
    if family in ("rf", "extra_trees"):
        return {"family": family,
                "n_estimators": int(rng.choice([100, 140, 180, 250])),
                "max_depth": int(rng.choice([8, 10, 12, 15, 20, 26])),
                "min_samples_leaf": int(rng.choice([3, 6, 10, 16, 26])),
                "max_features": float(rng.choice([0.5, 0.7, 0.9, 1.0])),
                "mask": mask}
    if family == "hgb":
        return {"family": family,
                "max_iter": int(rng.choice([90, 140, 180, 250])),
                "max_leaf_nodes": int(rng.choice([12, 20, 30])),
                "learning_rate": float(rng.choice([0.025, 0.045, 0.08, 0.12])),
                "min_samples_leaf": int(rng.choice([25, 50, 80, 120])),
                "l2_regularization": float(rng.choice([2, 8, 20, 50])),
                "mask": mask}
    return {"family": "lr",
            "C": float(rng.choice([0.1, 0.5, 1, 2.0, 5.0])),
            "mask": mask}


def make_estimator(config, g, i):
    seed = SEED + 100 * g + i
    if config["family"] == "extra_trees":
        return ExtraTreesClassifier(n_estimators=config["n_estimators"],
                                    max_depth=config["max_depth"],
                                    min_samples_leaf=config["min_samples_leaf"],
                                    max_features=config["max_features"],
                                    n_jobs=2, random_state=seed)
    if config["family"] == "rf":
        return RandomForestClassifier(n_estimators=config["n_estimators"],
                                      max_depth=config["max_depth"],
                                      min_samples_leaf=config["min_samples_leaf"],
                                      max_features=config["max_features"],
                                      n_jobs=2, random_state=seed)
    if config["family"] == "hgb":
        return HistGradientBoostingClassifier(max_iter=config["max_iter"],
                                              max_leaf_nodes=config["max_leaf_nodes"],
                                              max_depth=8,
                                              learning_rate=config["learning_rate"],
                                              min_samples_leaf=config["min_samples_leaf"],
                                              l2_regularization=config["l2_regularization"],
                                              early_stopping=False, random_state=seed)
    if config["family"] == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=config["C"], solver="liblinear", max_iter=400, random_state=seed))
    raise ValueError(config)


def load_original_conditional_models():
    """Read the immutable Sep-calibrated Jan-Aug 2023 three-stage foundation."""
    state_path = Path("docs/paper-only-ai-2023/autonomous_chapter1/state.json")
    state = json.loads(state_path.read_text("utf-8"))
    source = state["attempts"][0]["candidates"][0]
    if source["id"] != "g01-c01":
        raise RuntimeError("The inherited original model identity changed")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="accuracy_foundation_") as td:
        models, sha = load_frozen_champion(source, td)
    return models, sha


def conditional_cache(races, frozen):
    """Use only published pre-race cards and conditional probabilities.

    For every possible winning rider, hold the highest P(second,third|first)
    and its exact ordered ticket. This avoids leakage from actual finish order.
    """
    output = {}
    items = list(races.items())
    for start in range(0, len(items), 24):
        chunk = items[start:start+24]
        prediction = infer_new_chunk(chunk, frozen)
        for rid, race in chunk:
            cars = sorted(race["cars"], key=int)
            raw = frozen[1].predict_proba(
                np.asarray([feature(race, a, use_line=True) for a in cars],
                           dtype=np.float32))[:, 1]
            p_first = raw / np.sum(raw)
            distro = prediction[rid]["3連単"]
            # For each possible first place, maximize the full conditional
            # distribution; neither ground truth nor later odds enter here.
            best = {}
            for a, p1 in zip(cars, p_first):
                candidates = [(float(p / p1), ticket) for ticket, p in distro.items()
                              if ticket[0] == a]
                prob, ticket = max(candidates, key=lambda x: (x[0], x[1]))
                best[a] = (prob, ticket)
            baseline_first = cars[int(np.argmax(p_first))]
            baseline_order = max(
                cars, key=lambda a: p_first[cars.index(a)] * best[a][0])
            output[rid] = {
                "cars": cars, "conditional": best,
                "real_winner": race["order"][0],
                "real_order": tuple(race["order"]),
                "baseline_win": baseline_first,
                "baseline_trifecta": best[baseline_order][1],
            }
        print("CONDITIONAL_CACHE_PROGRESS", start+len(chunk), len(items), flush=True)
    return output


REL_INPUTS = (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17)
REL_EXTRA_DIMENSIONS = len(REL_INPUTS)*2 + 10


def race_relative_features(races):
    """Only information known from pre-race cards; never uses finishing positions.

    Relative athlete standing, line strength, rank and neighboring competitors
    often carry more signal than one athlete's absolute stats. The exact same
    method is applied to train, development and held-out races.
    """
    all_rows = []
    for race in races.values():
        cars = sorted(race["cars"],key=int)
        raw = np.stack([race["cars"][k] for k in cars]).astype(float)
        selected = raw[:,REL_INPUTS]
        avg = selected.mean(axis=0)
        sd = selected.std(axis=0)+1.0
        centered = np.clip((selected-avg)/sd,-4.0,4.0)
        percentiles = np.empty_like(selected)
        for j in range(selected.shape[1]):
            vals = selected[:,j]
            for i in range(len(cars)):
                percentiles[i,j] = (np.sum(vals<vals[i])+
                                    0.5*np.sum(vals==vals[i]))/len(vals)
        scores=raw[:,4]
        total=float(np.maximum(scores.sum(),1.0))
        maximum=float(np.max(scores))
        race_avg=float(np.mean(scores))
        positions=race["positions"]
        members={}
        for k in cars:
            group,rank,size = positions[k]
            members.setdefault(group,[]).append(k)
        idx={car:i for i,car in enumerate(cars)}
        line_best=[]
        for group, names in members.items():
            line_best.append((group,max(float(scores[idx[k]]) for k in names)))
        line_best.sort(key=lambda x:x[1],reverse=True)
        best_line_rank={group:rank for rank,(group,_) in enumerate(line_best)}
        extra=np.empty((len(cars),10),dtype=float)
        for i,k in enumerate(cars):
            group,rank,size=positions[k]
            teammates=members[group]
            line_scores=np.asarray([float(scores[idx[t]]) for t in teammates])
            best_tm=float(line_scores.max())
            leader=next((t for t in teammates if positions[t][1]==1),teammates[0])
            lead_score=float(scores[idx[leader]])
            others=np.asarray([float(scores[idx[t]]) for t in teammates if t!=k])
            other_best=float(others.max()) if len(others) else race_avg
            extra[i,:]=[
                len(teammates)/len(cars),
                (float(line_scores.mean())-race_avg)/15.0,
                (lead_score-maximum)/15.0,
                float(line_scores.sum())/total,
                (other_best-maximum)/15.0,
                (best_tm-maximum)/15.0,
                rank/max(len(teammates),1),
                float(rank==1),
                best_line_rank[group]/max(len(members)-1,1),
                len(cars)/9.0
            ]
        all_rows.append(np.column_stack((centered,percentiles,extra)))
    answer=np.asarray(np.concatenate(all_rows,axis=0),dtype=np.float32)
    if not np.isfinite(answer).all() or answer.shape[1]!=REL_EXTRA_DIMENSIONS:
        raise RuntimeError("Invalid relative-feature matrix")
    return answer


def selected_columns(config):
    cols=keep_columns(1,config["mask"])
    mode=config.get("engineering","raw")
    if mode=="relative_v1":
        cols=np.concatenate((cols,np.arange(22,22+REL_EXTRA_DIMENSIONS)))
    elif mode!="raw":
        raise ValueError("Unsupported accuracy feature architecture")
    return cols


def candidate_inputs(raw,relative,config):
    if config.get("engineering","raw")=="relative_v1":
        if len(raw)!=len(relative):
            raise ValueError("Relative/rider rows do not align")
        return np.column_stack((raw,relative))
    return raw


def winner_matrix(races):
    X, y, gid = matrix(races, 1, use_line=True)
    expected = sum(len(r["cars"]) for r in races.values())
    if len(X) != expected:
        raise RuntimeError("Unexpected winner feature matrix length")
    return X, y, gid


def evaluate(raw_scores, races, cache, temperature):
    """One exact winner and one exact 3連単 per race; all participants counted."""
    raw_scores = np.clip(np.asarray(raw_scores, dtype=float), EPS, 1.0)
    hits_first = hits_tri = 0
    predictions = {}
    cursor = 0
    for rid, race in races.items():
        info = cache[rid]
        cars = info["cars"]
        n = len(cars)
        sl = raw_scores[cursor:cursor+n]
        cursor += n
        z = np.power(sl, 1.0/temperature)
        p1 = z / z.sum()
        who = cars[int(np.argmax(p1))]
        tri_first = max(cars, key=lambda a: float(p1[cars.index(a)])*
                        info["conditional"][a][0])
        ticket = info["conditional"][tri_first][1]
        hits_first += int(who == info["real_winner"])
        hits_tri += int(ticket == info["real_order"])
        predictions[rid] = (who, ticket)
    if cursor != len(raw_scores):
        raise RuntimeError("Mismatched output/race alignment")
    races_n = len(races)
    return {
        "races": races_n, "winner_hits": hits_first,
        "trifecta_hits": hits_tri,
        "winner_top1": hits_first/races_n,
        "trifecta_top1": hits_tri/races_n,
        "one_pick_per_race_each_metric": True,
    }


def baseline_metrics(races, cache):
    n = len(races)
    hits_win = sum(int(v["baseline_win"] == v["real_winner"])
                   for v in cache.values())
    hits_tri = sum(int(v["baseline_trifecta"] == v["real_order"])
                   for v in cache.values())
    return {"races": n, "winner_hits": hits_win, "trifecta_hits": hits_tri,
            "winner_top1": hits_win/n, "trifecta_top1": hits_tri/n}


def objectives(october_baseline):
    return {
        "winner": max(WIN_TARGET,
                      october_baseline["winner_top1"]+WIN_BASELINE_REQUIRED_GAIN),
        "trifecta": max(TRIFECTA_TOP1_TARGET,
                       october_baseline["trifecta_top1"]+TRIFECTA_BASELINE_REQUIRED_GAIN),
    }


def rank_candidate(score):
    return (score["winner_top1"], score["trifecta_top1"])


def save_champion(model, rec):
    OUT.mkdir(parents=True, exist_ok=True)
    asset = OUT / ("accuracy_champion_" + rec["id"] + ".joblib")
    joblib.dump({"model": model, "columns": selected_columns(rec["config"]),
                 "temperature": rec["temperature"],
                 "config": rec["config"],
                 "selection": rec["october"]}, asset, compress=3)
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    return {"file": str(asset), "sha256": digest,
            "generation": rec["generation"], "candidate_id": rec["id"]}


def candidate_scores(config, iteration, index, fit, dev, fit_relative, dev_relative, dev_races, dev_cache):
    Xtrain, ytrain, _ = fit
    Xdev, _, _ = dev
    cols = selected_columns(config)
    model = make_estimator(config, iteration, index)
    Xtrain = candidate_inputs(Xtrain,fit_relative,config)
    Xdev = candidate_inputs(Xdev,dev_relative,config)
    model.fit(Xtrain[:,cols],ytrain)
    raw = model.predict_proba(Xdev[:,cols])[:,1]
    trials = [evaluate(raw, dev_races, dev_cache, t) | {"temperature":t}
              for t in TEMPERATURES]
    best = max(trials, key=lambda r: (r["winner_top1"],
                                      r["trifecta_top1"],
                                      -abs(r["temperature"]-1)))
    return model,best,trials


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    state = json.loads(STATE.read_text("utf-8")) if STATE.exists() else create_state()
    validate_state(state)
    if state["status"] in ("CONFIRMED_HISTORICAL_GOAL", "MAX_DEVELOPMENT_BUDGET_EXHAUSTED"):
        print("ACCURACY_GOAL_ALREADY_TERMINAL", state["status"], flush=True)
        return
    if state["generation"] >= MAX_GENERATIONS:
        print("ACCURACY_MAX_GENERATIONS_REACHED__FRESH_DATA_NEEDED",flush=True)
        return
    foundation, digest = load_original_conditional_models()
    initial, heldout, audit = load()
    train = {rid:r for rid,r in initial.items() if r["date"] <= "2023-08-31"}
    dev = {rid:r for rid,r in initial.items()
           if "2023-10-01" <= r["date"] <= "2023-10-31"}
    if len(train) < 10000 or len(dev) < MIN_OCT_RACES:
        raise RuntimeError("Training or development samples unavailable")
    print("ACC_DATA_BOUNDARIES",len(train),len(dev),flush=True)
    frozen_oct = conditional_cache(dev, foundation)
    prior = baseline_metrics(dev,frozen_oct)
    goal = objectives(prior)
    print("FIXED_ACCURACY_TARGETS", json.dumps({
        "october_baseline":prior,"winner_target":goal["winner"],
        "trifecta_target":goal["trifecta"]}), flush=True)
    fit = winner_matrix(train)
    develop = winner_matrix(dev)
    rel_fit = race_relative_features(train)
    rel_dev = race_relative_features(dev)
    new_assets = []
    pending_holdout = None
    for g in range(state["generation"]+1,
                   min(MAX_GENERATIONS,state["generation"]+GENERATIONS_PER_RUN)+1):
        n_candidates = 4 if g == 1 else 3
        generation_results = []
        for i in range(1,n_candidates+1):
            config = generate_config(g,i,state["champion"])
            # New feature-engineering architecture introduced at generation 5;
            # the formerly frozen goal and historical confirmation stay intact.
            config["engineering"]="relative_v1" if g >= 5 else "raw"
            model,score,alternatives = candidate_scores(
                config,g,i,fit,develop,rel_fit,rel_dev,dev,frozen_oct)
            rec={"id":f"acc-g{g:02d}-c{i:02d}", "generation":g,
                 "config":config,"temperature":score["temperature"],
                 "october":score,"alternative_temperatures":alternatives}
            generation_results.append(rec)
            if (state["champion"] is None or
                    rank_candidate(score)>rank_candidate(state["champion"]["october"])):
                state["champion"] = rec
                item = save_champion(model,rec)
                rec["model_asset"] = item
                new_assets.append(item)
            print("ACC_CANDIDATE",rec["id"],json.dumps({
                "first":score["winner_top1"],
                "one_trifecta":score["trifecta_top1"],
                "goal_hit":hypothesis_eligible(score,goal)}),flush=True)
            if hypothesis_eligible(score,goal) and not state["confirmation_used"]:
                # This part runs once, only after the Oct-only choice is fixed.
                # The old 2023 Nov/Dec records were consulted in earlier work,
                # so this is a development-era historic confirmation, not a
                # truly never-seen exam.
                pending_holdout=(rec,model)
                break
        state["attempts"].append({"generation":g,"results":generation_results})
        state["generation"]=g
        if pending_holdout:
            break
    if pending_holdout:
        rec, model = pending_holdout
        nov = {rid:r for rid,r in heldout.items()
               if "2023-11-01" <= r["date"] <= "2023-11-30"}
        dec = {rid:r for rid,r in heldout.items()
               if "2023-12-01" <= r["date"] <= "2023-12-31"}
        checks = {}
        for month, subset in (("2023-11",nov),("2023-12",dec)):
            if len(subset)<400:
                raise RuntimeError("Incomplete confirmation cohort")
            cache = conditional_cache(subset, foundation)
            X,_,_=winner_matrix(subset)
            cols=selected_columns(rec["config"])
            X=candidate_inputs(X,race_relative_features(subset),rec["config"])
            raw=model.predict_proba(X[:,cols])[:,1]
            check=evaluate(raw,subset,cache,rec["temperature"])
            check["baseline"]=baseline_metrics(subset,cache)
            check["passes_predeclared_goal"]=(check["winner_top1"]>=CONFIRM_WIN_TARGET
                and check["trifecta_top1"]>=CONFIRM_TRI_TARGET
                and check["winner_top1"]>check["baseline"]["winner_top1"])
            checks[month]=check
        state["confirmation_used"]=True
        success=all(v["passes_predeclared_goal"] for v in checks.values())
        state["confirmation"]={"candidate_id":rec["id"],
             "already_consulted_historical_months":True,"monthly_scores":checks,
             "both_months_passed":success,
             "no_second_try_on_same_2023_nov_dec":True}
        state["status"]=("CONFIRMED_HISTORICAL_GOAL" if success else
                         "SEARCHING_AFTER_FAILED_HISTORICAL_CONFIRMATION")
        print("ACCURACY_HISTORICAL_CONFIRMATION",json.dumps(
            state["confirmation"],ensure_ascii=False),flush=True)
    if state["generation"]>=MAX_GENERATIONS and state["status"]!="CONFIRMED_HISTORICAL_GOAL":
        state["status"]="MAX_DEVELOPMENT_BUDGET_EXHAUSTED"
    validate_state(state)
    report={
        "schema":state["schema"],"generated_at":datetime.now(timezone.utc).isoformat(),
        "status":state["status"],"generation":state["generation"],
        "training_races":len(train),"october_development_races":len(dev),
        "october_frozen_baseline":prior,"thresholds_strictly_greater_than":goal,
        "current_best":state["champion"],
        "confirmation_used":state["confirmation_used"],
        "confirmation":state["confirmation"],
        "model_promotions_this_run":new_assets,
        "conditional_2023_foundation_immutable_release_sha":digest,
        "market_odds_used":False,"roi_used_for_optimization":False,
        "2024plus_results_used":False,
        "paper_accuracy_branch_equivalence":"winner top1 accuracy is the closest keirin analogue; exact trifecta top1 is an additional metric",
        "limitations":[
            "Only first-place rider model is trainable; 2nd/3rd conditional models remain frozen from the earlier 2023 AI, so it is not a fully end-to-end independent architecture.",
            "Generations 5+ add pre-race race-relative/rider-rank/line-strength features, fitted without seeing future outcomes.",
            "October 2023 is reused across candidates and generations; only development targets can be claimed from repeated October scoring.",
            "Nov-Dec 2023 are scored once per state; these months were consulted in earlier project research and are NOT a pristine holdout.",
            "Never guarantee reaching 45 percent by repeating the same fixed historical data; stop after 24 generations and demand a newly designed chronology.",
            "One exact 1st rider and one exact ordered trifecta, not a box or expanded formation.",
            "Prediction accuracy is not equivalent to positive expected value or profitable betting.",
        ]}
    LATEST.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ACCURACY_AUTONOMY_REPORT",json.dumps({
        "generation":state["generation"],"status":state["status"],
        "oct_baseline_winner":prior["winner_top1"],
        "oct_baseline_trifecta":prior["trifecta_top1"],
        "best":state["champion"]["october"] if state["champion"] else None,
        "goal":goal},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

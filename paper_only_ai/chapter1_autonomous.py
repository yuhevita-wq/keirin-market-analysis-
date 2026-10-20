"""Autonomous bounded scientific model search toward Walsh & Joshi's calibration hypothesis.

The *software*, not ChatGPT, searches and compares model candidates. 2024 Q1/Q2
outcomes (known), all 2024+ labels and all odds remain inaccessible to its
development selection. The full-year 2023 LINE RF is retained, not overwritten.
This is an initial autonomous research engine, NOT a paper replication certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_lines_2023 import load, matrix, normalise
from train_2023 import FEATURES

OUT = Path("docs/paper-only-ai-2023/autonomous_chapter1")
STATE_PATH = OUT / "state.json"
SEED = 230306021
MAX_GENERATIONS = 24
TRAIN_END = "2023-08-31"
SEPT = ("2023-09-01", "2023-09-30")
OCT = ("2023-10-01", "2023-10-31")
EPS = 1e-12


def initial_state():
    return {
        "schema": "chapter1-autonomous-research-v1", "generation": 0,
        "max_generations_before_fresh_data_review": MAX_GENERATIONS,
        "champions": {"accuracy": None, "calibration": None},
        "attempts": [], "locked_external_test_executed": False,
        "locked_external_test_policy": (
            "2024 Q1 and Q2 are already observed. Never optimize on their odds, "
            "results, payouts or ROI. Run a genuinely unseen future test only "
            "ONCE after model/selection policy is frozen; never tune after seeing "
            "that holdout."
        ),
    }


def check_state(state):
    if state["schema"] != "chapter1-autonomous-research-v1":
        raise ValueError("Unsupported autonomous generation ledger")
    if int(state["generation"]) != len(state["attempts"]):
        raise ValueError("Ledger and generation inconsistent")
    if [x["generation"] for x in state["attempts"]] != list(
        range(1, len(state["attempts"]) + 1)
    ):
        raise ValueError("Generations are not an append-only sequence")


def read_inherited_model():
    """Require the actual 2023 RF model archive and verify every immutable weight."""
    path = Path(os.environ.get("FROZEN_LINE_FULL_DIR", "frozen_line_full"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["training_date_range"] != ["2023-01-01", "2023-12-31"]:
        raise RuntimeError("Unexpected inherited training dates")
    if int(manifest["trained_races"]) != 20349:
        raise RuntimeError("Unexpected inherited model cohort")
    digests = {}
    for i in (1, 2, 3):
        data = (path / f"stage_{i}.joblib").read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest["stages"][str(i)]["sha256"]:
            raise RuntimeError("Inherited 2023 RF weights corrupted")
        digests[str(i)] = digest
    return {"trained_races": manifest["trained_races"],
            "training_date_range": manifest["training_date_range"],
            "immutable_stage_sha256": digests,
            "original_model_unchanged": True}


def configs_for_generation(g, state):
    """Deterministic mutation of previous development champion plus exploration."""
    rng = np.random.default_rng(SEED + g)
    configs = []
    if g == 1:
        configs += [
            {"algorithm": "RF", "n_estimators": 90, "max_depth": 10,
             "min_samples_leaf": 12, "max_features": 0.7,
             "mask": "all", "origin": "inherited-RF-architecture"},
            {"algorithm": "LR", "C": 1.0, "mask": "all",
             "origin": "paper-comparison-candidate"},
        ]
    parent = state.get("champions", {}).get("calibration")
    for i in range(1 if g == 1 else 2):
        algorithm = "RF" if rng.random() < 0.75 else "LR"
        mask = str(rng.choice(["all", "drop_age", "drop_term", "drop_gear",
                               "drop_rates", "drop_style"]))
        if parent is not None and i == 0 and rng.random() < 0.5:
            algorithm = parent["config"]["algorithm"]
            if rng.random() < 0.5:
                mask = parent["config"]["mask"]
        if algorithm == "RF":
            configs.append({
                "algorithm": "RF",
                "n_estimators": int(rng.choice([60, 90, 120])),
                "max_depth": int(rng.choice([6, 8, 10, 12, 15])),
                "min_samples_leaf": int(rng.choice([6, 10, 16, 24, 36])),
                "max_features": float(rng.choice([0.5, 0.7, 1.0])),
                "mask": mask, "origin": "autonomous-mutation",
            })
        else:
            configs.append({
                "algorithm": "LR", "C": float(rng.choice(
                    [0.05, 0.2, 0.5, 1.0, 2.0, 5.0])),
                "mask": mask, "origin": "autonomous-mutation",
            })
    unique, seen = [], set()
    for c in configs:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def keep_columns(stage, mask):
    """Vary rider features while keeping genuine published LINE features."""
    dims = {1: 22, 2: 65, 3: 110}
    blocks = {1: 1, 2: 3, 3: 5}
    excluded = {
        "all": (), "drop_age": (1,), "drop_term": (2,),
        "drop_gear": (3,), "drop_rates": (15, 16, 17),
        "drop_style": (5, 6, 7, 8, 9, 10),
    }[mask]
    if len(FEATURES) != 18:
        raise RuntimeError("Unexpected original rider feature schema")
    bad = {block * 18 + j for block in range(blocks[stage])
           for j in excluded}
    cols = np.array([i for i in range(dims[stage]) if i not in bad],
                    dtype=np.int32)
    assert len(cols) > 0
    assert set(range(blocks[stage] * 18, dims[stage])).issubset(set(cols))
    return cols


def model_for(config, stage):
    if config["algorithm"] == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=float(config["C"]), solver="liblinear", max_iter=300,
            random_state=SEED + stage))
    if config["algorithm"] == "RF":
        return RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=float(config["max_features"]),
            random_state=SEED + stage, n_jobs=2)
    raise ValueError("Unknown candidate")


def conditional_prob(log_scores, group, temp):
    if not 0.4 <= temp <= 3.0:
        raise ValueError("Out of declared temperature search range")
    values = np.exp((log_scores - float(np.max(log_scores))) / temp)
    p = normalise(values, group)
    if not np.isfinite(p).all():
        raise RuntimeError("Nonfinite probability")
    return p


def score_stage(p, y, group):
    n = int(group.max()) + 1
    if int(y.sum()) != n or np.any(np.bincount(group, weights=y) != 1):
        raise ValueError("Conditional outcome must have one truth per race")
    nll = -np.log(np.maximum(p[y == 1], EPS)).mean()
    _, starts = np.unique(group, return_index=True)
    ends = list(starts[1:]) + [len(group)]
    chosen = np.zeros(len(p), dtype=bool)
    for start, end in zip(starts, ends):
        chosen[start + int(np.argmax(p[start:end]))] = True
    accuracy = float(y[chosen].sum() / n)
    bins = np.minimum(np.floor(p * 20).astype(int), 19)
    ece, occupied = 0.0, 0
    for b in range(20):
        use = bins == b
        if use.any():
            occupied += 1
            ece += float(use.mean() * abs(y[use].mean() - p[use].mean()))
    return {
        "conditional_nll": float(nll),
        "conditional_top1_accuracy": accuracy,
        "conditional_positive_ece20": ece,
        "occupied_bins_20": occupied,
        "paper_80pct_bin_occupancy_proxy_passed": occupied >= 16,
    }


def matrices_for(races):
    built = {}
    for s in (1, 2, 3):
        X, y, groups = matrix(races, s, use_line=True)
        if len(X) < 1000:
            raise ValueError("Training/selection race cohort too small")
        built[s] = (X, y, groups)
    return built


def evaluate_candidate(config, train, fit, dev, generation, index):
    measurements, model_records = [], []
    for s in (1, 2, 3):
        cols = keep_columns(s, config["mask"])
        Xt, yt, _ = train[s]
        Xf, yf, gf = fit[s]
        Xv, yv, gv = dev[s]
        model = model_for(config, s)
        model.fit(Xt[:, cols], yt)
        log_fit = np.log(np.clip(model.predict_proba(Xf[:, cols])[:, 1], EPS, 1))
        log_dev = np.log(np.clip(model.predict_proba(Xv[:, cols])[:, 1], EPS, 1))
        def fit_nll(t):
            return float(-np.log(np.maximum(
                conditional_prob(log_fit, gf, float(t))[yf == 1], EPS)).mean())
        opt = minimize_scalar(fit_nll, method="bounded",
                              bounds=(0.5, 2.5), options={"xatol": 1e-3})
        if not opt.success:
            raise RuntimeError("Temperature calibration failed")
        temperature = float(opt.x)
        before = score_stage(conditional_prob(log_dev, gv, 1.0), yv, gv)
        after = score_stage(conditional_prob(log_dev, gv, temperature), yv, gv)
        measurements.append({
            "stage": s, "temperature_fitted_on_september_only": temperature,
            "september_calibration_nll": fit_nll(temperature),
            "october_before": before, "october_after": after,
        })
        model_records.append({"stage": s, "estimator": model,
                              "columns": cols, "temperature": temperature})
    metrics = [d["october_after"] for d in measurements]
    summary = {
        "stagewise_mean_accuracy_proxy": float(np.mean([
            d["conditional_top1_accuracy"] for d in metrics])),
        "ordered_finish_joint_nll": float(sum([
            d["conditional_nll"] for d in metrics])),
        "mean_conditional_positive_ece20": float(np.mean([
            d["conditional_positive_ece20"] for d in metrics])),
        "all_three_80pct_occupancy_proxies_passed": all([
            d["paper_80pct_bin_occupancy_proxy_passed"] for d in metrics]),
        "stage_evaluations": measurements,
    }
    return {
        "id": f"g{generation:02d}-c{index:02d}",
        "config": config, "development": summary,
    }, model_records


def is_improvement(candidate, champion, criterion):
    if champion is None:
        return True
    new = candidate["development"]
    old = champion["development"]
    if criterion == "accuracy":
        return (new["stagewise_mean_accuracy_proxy"] >
                old["stagewise_mean_accuracy_proxy"] + 1e-10)
    if criterion == "calibration":
        n_pass = new["all_three_80pct_occupancy_proxies_passed"]
        o_pass = old["all_three_80pct_occupancy_proxies_passed"]
        if n_pass != o_pass:
            return bool(n_pass)
        return (new["mean_conditional_positive_ece20"] <
                old["mean_conditional_positive_ece20"] - 1e-10)
    raise ValueError(criterion)


def save_promoted_weights(records, promotions):
    assets = []
    for rec, models in records:
        roles = [role for role, candidate in promotions.items()
                 if candidate == rec["id"]]
        if not roles:
            continue
        root = OUT / "candidates" / rec["id"]
        root.mkdir(parents=True, exist_ok=True)
        for m in models:
            joblib.dump(m, root / f"stage_{m['stage']}.joblib", compress=3)
        (root / "manifest.json").write_text(json.dumps(
            {"candidate_id": rec["id"], "roles": roles, "config": rec["config"],
             "development": rec["development"], "trained_through": TRAIN_END,
             "temperature_fit": SEPT, "candidate_check": OCT},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        archive = OUT / f"{rec['id']}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for p in sorted(root.iterdir()):
                bundle.add(p, arcname=p.name)
        assets.append({"candidate_id": rec["id"], "roles": roles,
                       "archive": str(archive),
                       "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    return assets


def main():
    global OUT, STATE_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    OUT = Path(args.out)
    STATE_PATH = OUT / "state.json"
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_PATH.read_text("utf-8")) if STATE_PATH.exists() else initial_state()
    check_state(state)
    inherited = read_inherited_model()
    if state["generation"] >= MAX_GENERATIONS:
        print("NEEDS_FRESH_UNSEEN_DATA_OR_NEW_EXPERIMENT_DESIGN", flush=True)
        return
    g = state["generation"] + 1
    train_2023, _, source_audit = load()
    learning = {rid: r for rid, r in train_2023.items()
                if r["date"] <= TRAIN_END}
    calibration = {rid: r for rid, r in train_2023.items()
                   if SEPT[0] <= r["date"] <= SEPT[1]}
    development = {rid: r for rid, r in train_2023.items()
                   if OCT[0] <= r["date"] <= OCT[1]}
    if min(len(learning), len(calibration), len(development)) < 200:
        raise RuntimeError("Incomplete forward chronological development sets")
    mats = (matrices_for(learning), matrices_for(calibration),
            matrices_for(development))
    records, promotions = [], {}
    for idx, config in enumerate(configs_for_generation(g, state), 1):
        rec, weights = evaluate_candidate(config, *mats, g, idx)
        records.append((rec, weights))
        for objective in ("accuracy", "calibration"):
            if is_improvement(rec, state["champions"][objective], objective):
                state["champions"][objective] = rec
                promotions[objective] = rec["id"]
        print("AI_CANDIDATE", rec["id"], "accuracy",
              rec["development"]["stagewise_mean_accuracy_proxy"],
              "calibration_ece", rec["development"]["mean_conditional_positive_ece20"],
              flush=True)
    assets = save_promoted_weights(records, promotions)
    state["attempts"].append({
        "generation": g,
        "candidates": [r for r, _ in records],
        "promotions": promotions,
        "trained_max_date": TRAIN_END, "calibration_only": SEPT,
        "selection_only": OCT,
    })
    state["generation"] = g
    check_state(state)
    report = {
        "state": "AUTONOMOUS_GENERATION_COMPLETE",
        "generation": g,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "inherited_frozen_2023_model": inherited,
        "train_races": len(learning),
        "temperature_calibration_races": len(calibration),
        "development_races": len(development),
        "training_end": TRAIN_END,
        "temperature_calibration_dates": SEPT,
        "model_selection_dates": OCT,
        "both_model_selection_branches": state["champions"],
        "new_candidate_records": [r for r, _ in records],
        "newly_promoted": promotions,
        "permanently_save_these_weights": assets,
        "externally_certified": False,
        "chapter1_goal_reached": False,
        "external_test_locked": True,
        "limitations": [
            "This is bounded hyperparameter, model-family and feature-mask search, not a free-form scientist or identical NBA paper replication.",
            "These are stagewise conditional accuracy/ECE proxy metrics, NOT complete 3連単/3連複 classwise-ECE or ROI.",
            "80% bin-occupancy proxy is enforced/recorded; most multinomial keirin models may structurally fail.",
            "Reusing October 2023 over many generations can overfit; the 24-generation budget then requires new independent evidence.",
            "2023 Nov-Dec heldout observations and all 2024+ information are excluded from new optimization.",
            "The separately frozen full-year 2023 RF is never modified.",
            "No 2024+ labels, market odds, expected values, stake selection or bets were accessed in this run.",
        ],
    }
    (OUT / "latest.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    STATE_PATH.write_text(json.dumps(
        state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("AUTONOMOUS_GENERATION_FINISHED", json.dumps({
        "generation": g, "promotions": promotions,
        "accuracy": state["champions"]["accuracy"]["development"]["stagewise_mean_accuracy_proxy"],
        "calibration_ece": state["champions"]["calibration"]["development"]["mean_conditional_positive_ece20"],
        "all_bin_gates_passed": state["champions"]["calibration"]["development"]["all_three_80pct_occupancy_proxies_passed"],
    }), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v32 = load_module("v32_for_v34", "scripts/keirin_shogi_v32_top3_membership.py")

OUT = Path("results/keirin_shogi/v34_direct_third_residual")
SCRIPT = Path(__file__)

MODEL_FOLDS = [
    {
        "name": "2024_h2",
        "train": ["2024-01-01", "2024-06-30"],
        "validation": ["2024-07-01", "2024-12-31"],
    },
    {
        "name": "2025_h1",
        "train": ["2024-01-01", "2024-12-31"],
        "validation": ["2025-01-01", "2025-06-30"],
    },
    {
        "name": "2025_pre_calibration",
        "train": ["2024-01-01", "2025-06-30"],
        "validation": ["2025-07-01", "2025-10-26"],
    },
]

POLICY_STAGES = [
    {
        "name": "policy_2024q4",
        "train": ["2024-01-01", "2024-06-30"],
        "calibration": ["2024-07-01", "2024-09-30"],
        "test": ["2024-10-01", "2024-12-31"],
    },
    {
        "name": "policy_2025q2",
        "train": ["2024-01-01", "2024-12-31"],
        "calibration": ["2025-01-01", "2025-03-31"],
        "test": ["2025-04-01", "2025-06-30"],
    },
    {
        "name": "policy_2025q4",
        "train": ["2024-01-01", "2025-06-30"],
        "calibration": ["2025-07-01", "2025-10-26"],
        "test": ["2025-10-27", "2025-12-28"],
    },
]

DIAGNOSTIC = {
    "name": "reopened_v32_holdout_diagnostic",
    "train": ["2024-01-01", "2025-10-26"],
    "calibration": ["2025-10-27", "2025-12-28"],
    "test": ["2025-12-29", "2026-06-28"],
}

FUTURE_POLICY_TRAIN = ["2024-01-01", "2026-03-31"]
FUTURE_POLICY_CALIBRATION = ["2026-04-01", "2026-06-28"]
FUTURE_CAL_A = ["2026-04-01", "2026-05-15"]
FUTURE_CAL_B = ["2026-05-16", "2026-06-28"]
FUTURE_REFIT = ["2024-01-01", "2026-06-28"]
TRUE_FUTURE = ["2026-07-06", "2026-08-30"]

MODEL_VARIANTS = [
    {
        "name": "direct_d2_uniform",
        "depth": 2,
        "leaf": 28,
        "l2": 2.0,
        "lr": 0.045,
        "iters": 190,
        "positive_mass": 1.0 / 7.0,
    },
    {
        "name": "direct_d2_mild",
        "depth": 2,
        "leaf": 32,
        "l2": 2.5,
        "lr": 0.045,
        "iters": 220,
        "positive_mass": 0.25,
    },
    {
        "name": "direct_d2_balanced",
        "depth": 2,
        "leaf": 40,
        "l2": 3.0,
        "lr": 0.040,
        "iters": 240,
        "positive_mass": 0.50,
    },
    {
        "name": "direct_d3_uniform",
        "depth": 3,
        "leaf": 30,
        "l2": 2.5,
        "lr": 0.040,
        "iters": 210,
        "positive_mass": 1.0 / 7.0,
    },
    {
        "name": "direct_d3_mild",
        "depth": 3,
        "leaf": 36,
        "l2": 3.0,
        "lr": 0.038,
        "iters": 230,
        "positive_mass": 0.25,
    },
    {
        "name": "direct_d3_balanced",
        "depth": 3,
        "leaf": 48,
        "l2": 4.0,
        "lr": 0.035,
        "iters": 250,
        "positive_mass": 0.50,
    },
]

RANK_FEATURES = [
    "z_score",
    "z_win_rate",
    "z_top2_rate",
    "z_top3_rate",
    "z_s_count",
    "z_b_count",
    "z_nige_count",
    "z_makuri_count",
    "z_sashi_count",
    "z_mark_count",
    "z_first_count",
    "z_second_count",
    "z_third_count",
    "z_outside_count",
    "line_mean_score_z",
    "line_max_score_z",
    "line_b_sum_z",
    "line_attack_sum_z",
]

BASE_FEATURES = [
    *RANK_FEATURES,
    "line_pos",
    "line_size",
    "line_pos1",
    "line_pos2",
    "line_pos3p",
    "style_escape",
    "style_both",
    "style_chase",
]

GAP3_GRID = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.00, 1.25)
GAP4_GRID = (None, 0.0, 0.10, 0.20, 0.30, 0.40)
BETA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)


def race_type_features(race_type: str) -> dict[str, float]:
    labels = ("予選", "一般", "準決勝", "特選", "選抜", "決勝")
    return {f"race_type_{label}": float(label in race_type) for label in labels}


def rider_features(race: dict[str, object], no: int) -> dict[str, float]:
    base = race["base"]
    x = base[no]["x"]
    output = {name: float(x[name]) for name in BASE_FEATURES}
    for name in RANK_FEATURES:
        values = sorted(
            ((candidate, float(base[candidate]["x"][name])) for candidate in base),
            key=lambda item: (-item[1], item[0]),
        )
        rank = next(index + 1 for index, (candidate, _) in enumerate(values) if candidate == no)
        current = float(x[name])
        output[f"{name}_rank01"] = (rank - 1) / 6.0
        output[f"{name}_gap_to_max"] = max(value for _, value in values) - current
        output[f"{name}_gap_to_min"] = current - min(value for _, value in values)

    output.update(
        {
            "third_minus_first_z": float(x["z_third_count"] - x["z_first_count"]),
            "third_minus_second_z": float(x["z_third_count"] - x["z_second_count"]),
            "top3_minus_top2_rate_z": float(x["z_top3_rate"] - x["z_top2_rate"]),
            "mark_plus_sashi_z": float(x["z_mark_count"] + x["z_sashi_count"]),
            "attack_z": float(x["z_nige_count"] + x["z_makuri_count"]),
        }
    )
    output.update(race_type_features(str(race["race_type"])))
    return output


def build_matrix(
    races: list[dict[str, object]], feature_names: list[str] | None = None
):
    vectors = []
    labels = []
    groups = []
    local_names = feature_names
    offset = 0
    for race in races:
        riders = sorted(race["base"])
        features = [rider_features(race, no) for no in riders]
        if local_names is None:
            local_names = sorted(features[0])
        actual_third = int(race["order"][2])
        for no, values in zip(riders, features):
            vectors.append([values[name] for name in local_names])
            labels.append(int(no == actual_third))
        groups.append(
            {"race": race, "riders": riders, "start": offset, "end": offset + len(riders)}
        )
        offset += len(riders)
    if local_names is None:
        raise ValueError("No races for direct-third matrix")
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        local_names,
        groups,
    )


def fit_model(x: np.ndarray, y: np.ndarray, variant: dict[str, object]):
    positive_mass = float(variant["positive_mass"])
    sample_weight = np.where(y == 1, positive_mass, (1.0 - positive_mass) / 6.0)
    model = HistGradientBoostingClassifier(
        learning_rate=float(variant["lr"]),
        max_iter=int(variant["iters"]),
        max_depth=int(variant["depth"]),
        min_samples_leaf=int(variant["leaf"]),
        l2_regularization=float(variant["l2"]),
        random_state=34,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def normalize(raw: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(raw, dtype=float), 1e-12, None)
    return values / values.sum()


def evaluate_ranking(model, x, y, groups):
    raw = model.predict_proba(x)[:, 1]
    ranks = []
    losses = []
    for group in groups:
        start, end = int(group["start"]), int(group["end"])
        probabilities = normalize(raw[start:end])
        true_index = int(np.argmax(y[start:end]))
        order = np.argsort(-probabilities, kind="stable")
        ranks.append(int(np.flatnonzero(order == true_index)[0]) + 1)
        losses.append(-math.log(max(float(probabilities[true_index]), 1e-12)))
    count = len(groups)
    return {
        "races": count,
        "actual_third_top1": sum(rank <= 1 for rank in ranks) / count,
        "actual_third_top2": sum(rank <= 2 for rank in ranks) / count,
        "actual_third_top3": sum(rank <= 3 for rank in ranks) / count,
        "actual_third_top4": sum(rank <= 4 for rank in ranks) / count,
        "mean_nll": mean(losses),
    }


def ranking_score(metrics: dict[str, float | int]) -> float:
    return (
        0.20 * float(metrics["actual_third_top1"])
        + 0.30 * float(metrics["actual_third_top2"])
        + 0.50 * float(metrics["actual_third_top3"])
        - 0.015 * float(metrics["mean_nll"])
    )


def select_variant(races: list[dict[str, object]]):
    records = {str(v["name"]): {"variant": v, "folds": []} for v in MODEL_VARIANTS}
    feature_names = None
    for fold in MODEL_FOLDS:
        train = v32.period(races, *fold["train"])
        validation = v32.period(races, *fold["validation"])
        x_train, y_train, feature_names, _ = build_matrix(train, feature_names)
        x_validation, y_validation, _, groups = build_matrix(validation, feature_names)
        for variant in MODEL_VARIANTS:
            model = fit_model(x_train, y_train, variant)
            metrics = evaluate_ranking(model, x_validation, y_validation, groups)
            score = ranking_score(metrics)
            records[str(variant["name"])]["folds"].append(
                {"fold": fold, "score": score, "metrics": metrics}
            )
            print(
                f"[v34] {fold['name']} {variant['name']} score={score:.5f} "
                f"third@3={metrics['actual_third_top3']:.4f}",
                flush=True,
            )
        del x_train, y_train, x_validation, y_validation

    selection = []
    for record in records.values():
        scores = [float(row["score"]) for row in record["folds"]]
        selection.append(
            {
                "variant": record["variant"],
                "objective": 0.55 * min(scores) + 0.45 * mean(scores),
                "minimum_fold_score": min(scores),
                "mean_fold_score": mean(scores),
                "folds": record["folds"],
            }
        )
    selection.sort(
        key=lambda row: (float(row["objective"]), float(row["minimum_fold_score"])),
        reverse=True,
    )
    return selection[0]["variant"], selection, feature_names


def predict_rows(races, model, feature_names):
    x, _, _, groups = build_matrix(races, feature_names)
    raw = model.predict_proba(x)[:, 1]
    rows = []
    for group in groups:
        start, end = int(group["start"]), int(group["end"])
        probabilities = normalize(raw[start:end])
        race = group["race"]
        riders = group["riders"]
        rows.append(
            {
                "race_id": race["race_id"],
                "race_date": race["race_date"],
                "race_type": race["race_type"],
                "order": race["order"],
                "direct_probabilities": {
                    no: float(probability) for no, probability in zip(riders, probabilities)
                },
            }
        )
    return rows


def zscores(values: dict[int, float]) -> dict[int, float]:
    vector = np.asarray(list(values.values()), dtype=float)
    standard_deviation = float(vector.std()) or 1.0
    average = float(vector.mean())
    return {no: (float(value) - average) / standard_deviation for no, value in values.items()}


def combined_scores(
    direct: dict[int, float], top2_membership: dict[int, float] | None, beta: float
):
    direct_z = zscores(direct)
    if top2_membership is None or beta == 0.0:
        return direct_z
    top2_z = zscores(top2_membership)
    return {no: direct_z[no] - beta * top2_z.get(no, 0.0) for no in direct_z}


def choose(scores: dict[int, float], gap3: float, gap4: float | None):
    ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    candidates = [ranking[0][0], ranking[1][0]]
    gap23 = float(ranking[1][1] - ranking[2][1])
    gap34 = float(ranking[2][1] - ranking[3][1])
    if gap23 <= gap3:
        candidates.append(ranking[2][0])
        if gap4 is not None and gap34 <= gap4:
            candidates.append(ranking[3][0])
    return candidates, ranking, gap23, gap34


def apply_simple(rows, policy):
    logs = []
    for row in rows:
        scores = combined_scores(row["direct_probabilities"], None, 0.0)
        candidates, ranking, gap23, gap34 = choose(
            scores, float(policy["gap3"]), policy["gap4"]
        )
        third = int(row["order"][2])
        logs.append(
            {
                "race_id": row["race_id"],
                "race_date": row["race_date"],
                "actual_third": third,
                "third_candidates": candidates,
                "third_hit": int(third in candidates),
                "third_candidate_count": len(candidates),
                "gap_rank2_rank3": gap23,
                "gap_rank3_rank4": gap34,
                "ranking": ranking,
            }
        )
    return logs


def simple_metrics(logs):
    count = len(logs)
    return {
        "races": count,
        "third_capture": sum(row["third_hit"] for row in logs) / count,
        "avg_third_candidates": mean(row["third_candidate_count"] for row in logs),
        "two_candidate_rate": sum(row["third_candidate_count"] == 2 for row in logs) / count,
        "three_candidate_rate": sum(row["third_candidate_count"] == 3 for row in logs) / count,
        "four_candidate_rate": sum(row["third_candidate_count"] == 4 for row in logs) / count,
    }


def split_halves(rows):
    midpoint = max(1, len(rows) // 2)
    return rows[:midpoint], rows[midpoint:]


def simple_objective(a, b, full):
    minimum = min(float(a["third_capture"]), float(b["third_capture"]))
    drift = abs(float(a["third_capture"]) - float(b["third_capture"]))
    return (
        0.72 * minimum
        + 0.28 * float(full["third_capture"])
        - 0.10 * max(0.0, float(full["avg_third_candidates"]) - 2.0)
        - 0.20 * float(full["four_candidate_rate"])
        - 0.10 * drift
    )


def calibrate_simple(rows):
    rows_a, rows_b = split_halves(rows)
    grid = []
    for gap3 in GAP3_GRID:
        for gap4 in GAP4_GRID:
            policy = {"gap3": gap3, "gap4": gap4}
            a = simple_metrics(apply_simple(rows_a, policy))
            b = simple_metrics(apply_simple(rows_b, policy))
            full = simple_metrics(apply_simple(rows, policy))
            if float(full["avg_third_candidates"]) > 2.85:
                continue
            grid.append(
                {**policy, "objective": simple_objective(a, b, full), "cal_a": a, "cal_b": b, "cal_full": full}
            )
    grid.sort(
        key=lambda row: (
            float(row["objective"]),
            min(float(row["cal_a"]["third_capture"]), float(row["cal_b"]["third_capture"])),
            -float(row["cal_full"]["avg_third_candidates"]),
        ),
        reverse=True,
    )
    return grid


def run_policy_stage(races, stage, variant, feature_names):
    train = v32.period(races, *stage["train"])
    calibration = v32.period(races, *stage["calibration"])
    test = v32.period(races, *stage["test"])
    x, y, feature_names, _ = build_matrix(train, feature_names)
    model = fit_model(x, y, variant)
    del x, y
    calibration_rows = predict_rows(calibration, model, feature_names)
    grid = calibrate_simple(calibration_rows)
    policy = {key: grid[0][key] for key in ("gap3", "gap4")}
    test_logs = apply_simple(predict_rows(test, model, feature_names), policy)
    metrics = simple_metrics(test_logs)
    print(
        f"[v34] {stage['name']} policy={policy} third={metrics['third_capture']:.4f} "
        f"avg={metrics['avg_third_candidates']:.3f}",
        flush=True,
    )
    return {
        "stage": stage,
        "train_races": len(train),
        "selected_policy": policy,
        "calibration_metrics": grid[0]["cal_full"],
        "test_metrics": metrics,
    }


def attach_contexts(rows, contexts):
    output = []
    for row in rows:
        context = contexts.get(str(row["race_id"]))
        if context is None:
            continue
        output.append({**row, **context})
    output.sort(key=lambda row: (row["race_date"], row["race_id"]))
    return output


def apply_board(rows, policy):
    logs = []
    for row in rows:
        scores = combined_scores(
            row["direct_probabilities"], row["top2_membership"], float(policy["beta"])
        )
        candidates, ranking, gap23, gap34 = choose(
            scores, float(policy["gap3"]), policy["gap4"]
        )
        first, second, third = map(int, row["order"][:3])
        first_hit = int(first in row["first_candidates"])
        second_hit = int(second in row["second_candidates"])
        third_hit = int(third in candidates)
        logs.append(
            {
                "race_id": row["race_id"],
                "race_date": row["race_date"],
                "race_type": row["race_type"],
                "actual_first": first,
                "actual_second": second,
                "actual_third": third,
                "first_candidates": row["first_candidates"],
                "second_candidates": row["second_candidates"],
                "third_candidates": candidates,
                "first_hit": first_hit,
                "second_hit": second_hit,
                "third_hit": third_hit,
                "first_second_hit": int(first_hit and second_hit),
                "complete_board_hit": int(first_hit and second_hit and third_hit),
                "third_candidate_count": len(candidates),
                "board_cell_count": len(row["first_candidates"])
                + len(row["second_candidates"])
                + len(candidates),
                "gap_rank2_rank3": gap23,
                "gap_rank3_rank4": gap34,
                "third_score_ranking": [
                    {"no": no, "score": float(score)} for no, score in ranking
                ],
                "direct_probabilities": [
                    {"no": no, "probability": float(probability)}
                    for no, probability in sorted(
                        row["direct_probabilities"].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
                "top2_membership": row["top2_ranking"],
            }
        )
    return logs


def board_objective(a, b, full):
    min_third = min(float(a["third_capture"]), float(b["third_capture"]))
    min_complete = min(
        float(a["complete_board_capture"]), float(b["complete_board_capture"])
    )
    min_conditional = min(
        float(a["third_given_first_second"]), float(b["third_given_first_second"])
    )
    drift = abs(float(a["third_capture"]) - float(b["third_capture"]))
    return (
        0.44 * min_third
        + 0.30 * min_complete
        + 0.16 * min_conditional
        + 0.10 * float(full["third_capture"])
        - 0.10 * max(0.0, float(full["avg_third_candidates"]) - 2.0)
        - 0.20 * float(full["four_candidate_rate"])
        - 0.10 * drift
    )


def calibrate_board(rows):
    rows_a = [row for row in rows if FUTURE_CAL_A[0] <= row["race_date"] <= FUTURE_CAL_A[1]]
    rows_b = [row for row in rows if FUTURE_CAL_B[0] <= row["race_date"] <= FUTURE_CAL_B[1]]
    grid = []
    for beta in BETA_GRID:
        for gap3 in GAP3_GRID:
            for gap4 in GAP4_GRID:
                policy = {"beta": beta, "gap3": gap3, "gap4": gap4}
                a = v32.board_metrics(apply_board(rows_a, policy))
                b = v32.board_metrics(apply_board(rows_b, policy))
                full = v32.board_metrics(apply_board(rows, policy))
                if float(full["avg_third_candidates"]) > 2.85:
                    continue
                grid.append(
                    {**policy, "objective": board_objective(a, b, full), "cal_a": a, "cal_b": b, "cal_full": full}
                )
    grid.sort(
        key=lambda row: (
            float(row["objective"]),
            min(float(row["cal_a"]["third_capture"]), float(row["cal_b"]["third_capture"])),
            min(
                float(row["cal_a"]["complete_board_capture"]),
                float(row["cal_b"]["complete_board_capture"]),
            ),
            -float(row["cal_full"]["avg_third_candidates"]),
        ),
        reverse=True,
    )
    return grid


def script_hash():
    return hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    races = v32.load_races()
    print(f"[v34] loaded {len(races)} historical races; future block not loaded", flush=True)
    selected_variant, model_selection, feature_names = select_variant(races)
    print(f"[v34] selected model {selected_variant['name']}", flush=True)

    stages = [
        run_policy_stage(races, stage, selected_variant, feature_names)
        for stage in POLICY_STAGES
    ]
    diagnostic = run_policy_stage(races, DIAGNOSTIC, selected_variant, feature_names)
    diagnostic["validity_note"] = (
        "Development diagnostic only: v34 was proposed after v32 exposed this period."
    )

    policy_train = v32.period(races, *FUTURE_POLICY_TRAIN)
    x, y, feature_names, _ = build_matrix(policy_train, feature_names)
    policy_model = fit_model(x, y, selected_variant)
    del x, y
    calibration_races = v32.period(races, *FUTURE_POLICY_CALIBRATION)
    direct_rows = predict_rows(calibration_races, policy_model, feature_names)
    contexts, v31_audit = v32.build_v31_contexts()
    board_rows = attach_contexts(direct_rows, contexts)
    board_grid = calibrate_board(board_rows)
    selected_policy = {
        key: board_grid[0][key] for key in ("beta", "gap3", "gap4")
    }
    print(
        f"[v34] future policy={selected_policy} "
        f"third={board_grid[0]['cal_full']['third_capture']:.4f} "
        f"complete={board_grid[0]['cal_full']['complete_board_capture']:.4f}",
        flush=True,
    )

    stage_captures = [float(stage["test_metrics"]["third_capture"]) for stage in stages]
    stage_counts = [float(stage["test_metrics"]["avg_third_candidates"]) for stage in stages]
    diagnostic_capture = float(diagnostic["test_metrics"]["third_capture"])
    ready_for_future = (
        mean(stage_captures) > v32.V23_REFERENCE["third_capture"]
        and min(stage_captures) >= 0.50
        and max(stage_counts) <= v32.V23_REFERENCE["avg_third_candidates"]
        and diagnostic_capture > v32.V23_REFERENCE["third_capture"]
    )

    summary = {
        "algorithm": "keirin_shogi_v34_direct_third_residual",
        "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST" if ready_for_future else "development_failed_not_frozen",
        "concept": (
            "Predict exact third directly across all seven riders without conditioning on a "
            "predicted first or second. v31 Top2 membership is a calibrated residual penalty, "
            "never an eligibility constraint."
        ),
        "model_selection": model_selection,
        "selected_variant": selected_variant,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "historical_policy_walk_forward": stages,
        "reopened_v32_holdout_diagnostic": diagnostic,
        "future_lock": {
            "policy_model_training_period": FUTURE_POLICY_TRAIN,
            "policy_model_training_races": len(policy_train),
            "calibration_period": FUTURE_POLICY_CALIBRATION,
            "calibration_split": [FUTURE_CAL_A, FUTURE_CAL_B],
            "calibration_participant_races": len(board_rows),
            "selected_policy": selected_policy,
            "selected_calibration_metrics": board_grid[0]["cal_full"],
            "future_refit_period": FUTURE_REFIT,
            "true_future_period": TRUE_FUTURE,
            "v31_audit": v31_audit,
        },
        "ready_for_true_future": ready_for_future,
        "acceptance_rule": {
            "mean_historical_stage_capture_above_v23": v32.V23_REFERENCE["third_capture"],
            "minimum_historical_stage_capture": 0.50,
            "maximum_average_candidates": v32.V23_REFERENCE["avg_third_candidates"],
            "reopened_diagnostic_capture_above_v23": v32.V23_REFERENCE["third_capture"],
        },
        "guards": {
            "future_block_loaded": False,
            "odds_or_popularity_used": False,
            "random_shuffle": False,
            "v21_changed": False,
            "v31_changed": False,
            "top2_membership_used_as_constraint": False,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "calibration.json").write_text(
        json.dumps(board_grid, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = "# v34 Direct Third Residual\n\n"
    readme += "Top3集合だけでは実3着の順序を分離できなかったため、7人を直接比較する独立3着モデルへ変更。\n"
    readme += "v31は候補除外に使わず、校正された重複ペナルティとしてだけ使う。\n\n"
    readme += "| block | third capture | avg candidates |\n|---|---:|---:|\n"
    for stage in stages:
        readme += (
            f"| {stage['stage']['test'][0]}〜{stage['stage']['test'][1]} "
            f"| {float(stage['test_metrics']['third_capture']):.2%} "
            f"| {float(stage['test_metrics']['avg_third_candidates']):.3f} |\n"
        )
    readme += f"\n判定: **{'完全未来試験へ凍結' if ready_for_future else '開発不合格'}**\n"
    readme += "\n```bash\npython scripts/keirin_shogi_v34_direct_third_residual.py\n```\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    freeze_path = OUT / "FREEZE.json"
    if ready_for_future:
        freeze = {
            "model": "keirin_shogi_v34_direct_third_residual",
            "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST",
            "source_script_sha256": script_hash(),
            "selected_variant": selected_variant,
            "selected_policy": selected_policy,
            "future_refit_period": FUTURE_REFIT,
            "calibration_period": FUTURE_POLICY_CALIBRATION,
            "true_future_period": TRUE_FUTURE,
            "future_tuning_allowed": False,
        }
        freeze_path.write_text(
            json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif freeze_path.exists():
        freeze_path.unlink()

    print(
        json.dumps(
            {
                "ready_for_true_future": ready_for_future,
                "selected_variant": selected_variant,
                "stage_metrics": [stage["test_metrics"] for stage in stages],
                "diagnostic_metrics": diagnostic["test_metrics"],
                "future_policy": selected_policy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

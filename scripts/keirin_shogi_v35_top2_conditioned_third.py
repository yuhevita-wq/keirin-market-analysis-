#!/usr/bin/env python3
from __future__ import annotations

"""v35: exact-third completion of the frozen unordered Top2 posterior.

The third model is trained as P(third=c | unordered actual Top2 pair, race).
At prediction time it is evaluated for every one of the 21 pair hypotheses and
marginalized with the (unchanged) v31 pair posterior.  No rider is filtered by
v21 or v31; a low-probability pair branch can therefore still rescue a miss.
"""

import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
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


v32 = load_module("v32_for_v35", "scripts/keirin_shogi_v32_top3_membership.py")
v31 = v32.v31
v26 = v31.v26
v25 = v31.v25

OUT = Path("results/keirin_shogi/v35_top2_conditioned_third")
SCRIPT = Path(__file__)

PAIR_PARAMS = {"depth": 2, "leaf": 18, "l2": 1.5, "lr": 0.045, "iters": 190}

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

FINAL_TRAIN = ["2024-01-01", "2025-10-26"]
FINAL_CALIBRATION = ["2025-10-27", "2025-12-28"]
FINAL_CAL_A = ["2025-10-27", "2025-11-30"]
FINAL_CAL_B = ["2025-12-01", "2025-12-28"]
REOPENED_DIAGNOSTIC = ["2025-12-29", "2026-06-28"]
TRUE_FUTURE = ["2026-07-06", "2026-08-30"]

MODEL_VARIANTS = [
    {"name": "cond_d2a", "depth": 2, "leaf": 24, "l2": 1.5, "lr": 0.045, "iters": 190},
    {"name": "cond_d2b", "depth": 2, "leaf": 36, "l2": 2.5, "lr": 0.040, "iters": 230},
    {"name": "cond_d3a", "depth": 3, "leaf": 28, "l2": 2.0, "lr": 0.040, "iters": 210},
    {"name": "cond_d3b", "depth": 3, "leaf": 42, "l2": 3.0, "lr": 0.035, "iters": 250},
]

RIDER_FEATURES = [
    "z_score", "z_win_rate", "z_top2_rate", "z_top3_rate", "z_s_count",
    "z_b_count", "z_nige_count", "z_makuri_count", "z_sashi_count",
    "z_mark_count", "z_first_count", "z_second_count", "z_third_count",
    "z_outside_count", "line_pos", "line_size", "line_mean_score_z",
    "line_max_score_z", "line_b_sum_z", "line_attack_sum_z", "line_pos1",
    "line_pos2", "line_pos3p", "style_escape", "style_both", "style_chase",
]


def race_tuples(races: list[dict[str, object]]):
    return [
        (race["race_id"], race["race_date"], race["base"], race["order"], {"p1_map": {}})
        for race in races
    ]


def fit_pair_model(races: list[dict[str, object]]):
    return v26.fit_pair_model(race_tuples(races), PAIR_PARAMS)


def relation_features(
    race: dict[str, object], pair: tuple[int, int], candidate: int
) -> dict[str, float]:
    base = race["base"]
    a, b = pair
    xc = base[candidate]["x"]
    xa = base[a]["x"]
    xb = base[b]["x"]
    output: dict[str, float] = {}

    for feature in RIDER_FEATURES:
        cv = float(xc[feature])
        av = float(xa[feature])
        bv = float(xb[feature])
        output[f"candidate_{feature}"] = cv
        output[f"candidate_minus_pair_mean_{feature}"] = cv - (av + bv) / 2.0
        output[f"candidate_minus_pair_max_{feature}"] = cv - max(av, bv)

    pair_values = v26.pair_features_blind(base, a, b, {})
    for name, value in pair_values.items():
        output[f"pair_{name}"] = float(value)

    # The triple aggregates carry the original unordered-Top3 signal.  They do
    # not impose eligibility; they are explanatory inputs to the completion.
    triple_values = v32.triple_features(base, tuple(sorted((a, b, candidate))))
    for name, value in triple_values.items():
        output[f"triple_{name}"] = float(value)

    line_c = str(base[candidate]["line_id"])
    pos_c = int(base[candidate]["line_position"])
    pair_lines = [str(base[a]["line_id"]), str(base[b]["line_id"])]
    pair_positions = [int(base[a]["line_position"]), int(base[b]["line_position"])]
    same = [line_c == line for line in pair_lines]
    output.update(
        {
            "candidate_same_line_count": float(sum(same)),
            "candidate_different_from_both": float(not any(same)),
            "candidate_behind_pair_count": float(
                sum(match and pos_c > pos for match, pos in zip(same, pair_positions))
            ),
            "candidate_ahead_pair_count": float(
                sum(match and pos_c < pos for match, pos in zip(same, pair_positions))
            ),
            "candidate_adjacent_pair_count": float(
                sum(match and abs(pos_c - pos) == 1 for match, pos in zip(same, pair_positions))
            ),
            "candidate_immediately_behind_count": float(
                sum(match and pos_c == pos + 1 for match, pos in zip(same, pair_positions))
            ),
            "candidate_immediately_ahead_count": float(
                sum(match and pos_c + 1 == pos for match, pos in zip(same, pair_positions))
            ),
            "pair_same_line_candidate_same": float(pair_lines[0] == pair_lines[1] == line_c),
            "pair_same_line_candidate_other": float(pair_lines[0] == pair_lines[1] != line_c),
            "triple_distinct_lines": float(len(set(pair_lines + [line_c]))),
            "candidate_line_position": float(pos_c),
        }
    )
    race_type = str(race["race_type"])
    for label in ("予選", "一般", "準決勝", "特選", "選抜", "決勝"):
        output[f"race_type_{label}"] = float(label in race_type)
    return output


def build_training_matrix(
    races: list[dict[str, object]], feature_names: list[str] | None = None
):
    vectors: list[list[float]] = []
    labels: list[int] = []
    local_names = feature_names
    for race in races:
        first, second, third = map(int, race["order"][:3])
        pair = tuple(sorted((first, second)))
        candidates = [no for no in sorted(race["base"]) if no not in pair]
        rows = [relation_features(race, pair, no) for no in candidates]
        if local_names is None:
            local_names = sorted(rows[0])
        for no, values in zip(candidates, rows):
            vectors.append([values[name] for name in local_names])
            labels.append(int(no == third))
    if local_names is None:
        raise ValueError("No races available for v35 training")
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        local_names,
    )


def fit_third_model(x: np.ndarray, y: np.ndarray, variant: dict[str, object]):
    # One positive and four negatives exist in every conditional branch.
    weights = np.where(y == 1, 0.20, 0.20)
    model = HistGradientBoostingClassifier(
        learning_rate=float(variant["lr"]),
        max_iter=int(variant["iters"]),
        max_depth=int(variant["depth"]),
        min_samples_leaf=int(variant["leaf"]),
        l2_regularization=float(variant["l2"]),
        random_state=35,
    )
    model.fit(x, y, sample_weight=weights)
    return model


def normalized(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    return clipped / clipped.sum()


def pair_distribution(race: dict[str, object], pair_model):
    return v31.pair_distribution(race["base"], {"p1_map": {}}, pair_model)


def predict_one(
    race: dict[str, object], pair_model, third_model, feature_names: list[str],
    supplied_pairs: list[tuple[int, int, float]] | None = None,
):
    pairs = supplied_pairs if supplied_pairs is not None else pair_distribution(race, pair_model)
    scores: defaultdict[int, float] = defaultdict(float)
    branch_rows = []
    vectors = []
    for a, b, probability in pairs:
        candidates = [no for no in sorted(race["base"]) if no not in (a, b)]
        for no in candidates:
            values = relation_features(race, (a, b), no)
            vectors.append([values[name] for name in feature_names])
        branch_rows.append((a, b, float(probability), candidates))

    raw = third_model.predict_proba(np.asarray(vectors, dtype=np.float32))[:, 1]
    offset = 0
    for a, b, pair_probability, candidates in branch_rows:
        conditional = normalized(raw[offset : offset + len(candidates)])
        offset += len(candidates)
        for no, probability in zip(candidates, conditional):
            scores[no] += pair_probability * float(probability)
    total = sum(scores.values()) or 1.0
    probabilities = {no: value / total for no, value in scores.items()}
    ranking = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    return probabilities, ranking, pairs


def predict_rows(races, pair_model, third_model, feature_names, supplied_contexts=None):
    prepared = prepare_prediction(races, pair_model, feature_names, supplied_contexts)
    return predict_prepared(prepared, third_model)


def prepare_prediction(races, pair_model, feature_names, supplied_contexts=None):
    vectors = []
    groups = []
    offset = 0
    for race in races:
        context = supplied_contexts.get(str(race["race_id"])) if supplied_contexts else None
        if supplied_contexts is not None and context is None:
            continue
        pairs = context["pairs"] if context is not None else pair_distribution(race, pair_model)
        branches = []
        for a, b, probability in pairs:
            candidates = [no for no in sorted(race["base"]) if no not in (a, b)]
            start = offset
            for no in candidates:
                values = relation_features(race, (a, b), no)
                vectors.append([values[name] for name in feature_names])
                offset += 1
            branches.append(
                {
                    "probability": float(probability),
                    "candidates": candidates,
                    "start": start,
                    "end": offset,
                }
            )
        groups.append(
            {
                "race": race,
                "context": context,
                "pairs": pairs,
                "branches": branches,
            }
        )
    return {
        "x": np.asarray(vectors, dtype=np.float32),
        "groups": groups,
    }


def predict_prepared(prepared, third_model):
    raw = third_model.predict_proba(prepared["x"])[:, 1]
    rows = []
    for group in prepared["groups"]:
        race = group["race"]
        context = group["context"]
        scores: defaultdict[int, float] = defaultdict(float)
        for branch in group["branches"]:
            conditional = normalized(raw[branch["start"] : branch["end"]])
            for no, probability in zip(branch["candidates"], conditional):
                scores[no] += float(branch["probability"]) * float(probability)
        total = sum(scores.values()) or 1.0
        probabilities = {no: value / total for no, value in scores.items()}
        ranking = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        rows.append(
            {
                "race_id": race["race_id"],
                "race_date": race["race_date"],
                "race_type": race["race_type"],
                "order": race["order"],
                "probabilities": probabilities,
                "ranking": ranking,
                "pairs": group["pairs"],
                **({k: v for k, v in context.items() if k != "pairs"} if context else {}),
            }
        )
    rows.sort(key=lambda row: (row["race_date"], row["race_id"]))
    return rows


def ranking_metrics(rows):
    ranks = []
    losses = []
    for row in rows:
        third = int(row["order"][2])
        rank = next(i + 1 for i, (no, _) in enumerate(row["ranking"]) if no == third)
        ranks.append(rank)
        losses.append(-math.log(max(float(row["probabilities"][third]), 1e-12)))
    count = len(rows)
    return {
        "races": count,
        "actual_third_top1": sum(rank <= 1 for rank in ranks) / count,
        "actual_third_top2": sum(rank <= 2 for rank in ranks) / count,
        "actual_third_top3": sum(rank <= 3 for rank in ranks) / count,
        "actual_third_top4": sum(rank <= 4 for rank in ranks) / count,
        "mean_nll": mean(losses),
    }


def ranking_objective(metrics):
    return (
        0.18 * float(metrics["actual_third_top1"])
        + 0.30 * float(metrics["actual_third_top2"])
        + 0.52 * float(metrics["actual_third_top3"])
        - 0.015 * float(metrics["mean_nll"])
    )


def select_variant(races):
    records = {str(v["name"]): {"variant": v, "folds": []} for v in MODEL_VARIANTS}
    feature_names = None
    for fold in MODEL_FOLDS:
        train = v32.period(races, *fold["train"])
        validation = v32.period(races, *fold["validation"])
        pair_model = fit_pair_model(train)
        x, y, feature_names = build_training_matrix(train, feature_names)
        prepared = prepare_prediction(validation, pair_model, feature_names)
        for variant in MODEL_VARIANTS:
            model = fit_third_model(x, y, variant)
            rows = predict_prepared(prepared, model)
            metrics = ranking_metrics(rows)
            score = ranking_objective(metrics)
            records[str(variant["name"])]["folds"].append(
                {"fold": fold, "score": score, "metrics": metrics}
            )
            print(
                f"[v35] {fold['name']} {variant['name']} score={score:.5f} "
                f"third@3={metrics['actual_third_top3']:.4f}",
                flush=True,
            )
    selection = []
    for record in records.values():
        scores = [float(row["score"]) for row in record["folds"]]
        selection.append(
            {
                "variant": record["variant"],
                "objective": 0.60 * min(scores) + 0.40 * mean(scores),
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


def policy_specs():
    policies = []
    for threshold in (0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.66, 0.70):
        policies.append({"kind": "cumulative", "threshold": threshold, "max_count": 3})
        policies.append({"kind": "cumulative", "threshold": threshold, "max_count": 4})
    for rel3 in (0.50, 0.60, 0.70, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90):
        policies.append({"kind": "ratio", "rel3": rel3, "rel4": None})
        for rel4 in (0.55, 0.70, 0.85):
            policies.append({"kind": "ratio", "rel3": rel3, "rel4": rel4})
    return policies


def choose(ranking: list[tuple[int, float]], policy: dict[str, object]):
    candidates = [ranking[0][0], ranking[1][0]]
    if policy["kind"] == "cumulative":
        total = ranking[0][1] + ranking[1][1]
        while total < float(policy["threshold"]) and len(candidates) < int(policy["max_count"]):
            candidates.append(ranking[len(candidates)][0])
            total += ranking[len(candidates) - 1][1]
    else:
        if ranking[2][1] / max(ranking[1][1], 1e-12) >= float(policy["rel3"]):
            candidates.append(ranking[2][0])
            rel4 = policy["rel4"]
            if rel4 is not None and ranking[3][1] / max(ranking[2][1], 1e-12) >= float(rel4):
                candidates.append(ranking[3][0])
    return candidates


def apply(rows, policy, board=False):
    logs = []
    for row in rows:
        candidates = choose(row["ranking"], policy)
        first, second, third = map(int, row["order"][:3])
        first_candidates = row.get("first_candidates", [])
        second_candidates = row.get("second_candidates", [])
        first_hit = int(first in first_candidates) if board else 0
        second_hit = int(second in second_candidates) if board else 0
        third_hit = int(third in candidates)
        logs.append(
            {
                "race_id": row["race_id"],
                "race_date": row["race_date"],
                "race_type": row["race_type"],
                "actual_first": first,
                "actual_second": second,
                "actual_third": third,
                "first_candidates": first_candidates,
                "second_candidates": second_candidates,
                "third_candidates": candidates,
                "first_hit": first_hit,
                "second_hit": second_hit,
                "third_hit": third_hit,
                "first_second_hit": int(board and first_hit and second_hit),
                "complete_board_hit": int(board and first_hit and second_hit and third_hit),
                "third_candidate_count": len(candidates),
                "board_cell_count": len(first_candidates) + len(second_candidates) + len(candidates),
                "third_ranking": [
                    {"no": int(no), "probability": float(probability)}
                    for no, probability in row["ranking"]
                ],
                "top2_membership": row.get("top2_ranking", []),
                "top_pairs": [
                    {"a": int(a), "b": int(b), "probability": float(probability)}
                    for a, b, probability in row["pairs"][:5]
                ],
            }
        )
    return logs


def metrics(logs, board=False):
    count = len(logs)
    output = {
        "races": count,
        "third_capture": sum(row["third_hit"] for row in logs) / count,
        "avg_third_candidates": mean(row["third_candidate_count"] for row in logs),
        "two_candidate_rate": sum(row["third_candidate_count"] == 2 for row in logs) / count,
        "three_candidate_rate": sum(row["third_candidate_count"] == 3 for row in logs) / count,
        "four_candidate_rate": sum(row["third_candidate_count"] == 4 for row in logs) / count,
    }
    if board:
        first_second = [row for row in logs if row["first_second_hit"]]
        output.update(
            {
                "first_capture": sum(row["first_hit"] for row in logs) / count,
                "second_capture": sum(row["second_hit"] for row in logs) / count,
                "complete_board_capture": sum(row["complete_board_hit"] for row in logs) / count,
                "third_given_first_second": (
                    sum(row["third_hit"] for row in first_second) / len(first_second)
                    if first_second else 0.0
                ),
                "avg_first_candidates": mean(len(row["first_candidates"]) for row in logs),
                "avg_second_candidates": mean(len(row["second_candidates"]) for row in logs),
                "avg_board_cells": mean(row["board_cell_count"] for row in logs),
            }
        )
    return output


def split_rows(rows, bounds_a=None, bounds_b=None):
    if bounds_a and bounds_b:
        return (
            [row for row in rows if bounds_a[0] <= row["race_date"] <= bounds_a[1]],
            [row for row in rows if bounds_b[0] <= row["race_date"] <= bounds_b[1]],
        )
    midpoint = max(1, len(rows) // 2)
    return rows[:midpoint], rows[midpoint:]


def policy_objective(a, b, full, board=False):
    min_third = min(float(a["third_capture"]), float(b["third_capture"]))
    drift = abs(float(a["third_capture"]) - float(b["third_capture"]))
    value = 0.48 * min_third + 0.52 * float(full["third_capture"])
    if board:
        value += 0.22 * min(float(a["complete_board_capture"]), float(b["complete_board_capture"]))
        value += 0.10 * min(float(a["third_given_first_second"]), float(b["third_given_first_second"]))
    # Candidate count is already hard-capped at the v23 reference (2.85).
    # Inside that feasible set, charge a small marginal cost instead of
    # allowing the cost term to dominate the requested capture comparison.
    value -= 0.005 * max(0.0, float(full["avg_third_candidates"]) - 2.0)
    value -= 0.24 * float(full["four_candidate_rate"])
    value -= 0.05 * drift
    return value


def calibrate(rows, board=False, bounds_a=None, bounds_b=None):
    rows_a, rows_b = split_rows(rows, bounds_a, bounds_b)
    grid = []
    for policy in policy_specs():
        a = metrics(apply(rows_a, policy, board), board)
        b = metrics(apply(rows_b, policy, board), board)
        full = metrics(apply(rows, policy, board), board)
        if float(full["avg_third_candidates"]) > 2.85:
            continue
        grid.append(
            {
                "policy": policy,
                "objective": policy_objective(a, b, full, board),
                "cal_a": a,
                "cal_b": b,
                "cal_full": full,
            }
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
    pair_model = fit_pair_model(train)
    x, y, feature_names = build_training_matrix(train, feature_names)
    third_model = fit_third_model(x, y, variant)
    calibration_rows = predict_rows(calibration, pair_model, third_model, feature_names)
    grid = calibrate(calibration_rows)
    policy = grid[0]["policy"]
    test_rows = predict_rows(test, pair_model, third_model, feature_names)
    test_metrics = metrics(apply(test_rows, policy))
    print(
        f"[v35] {stage['name']} policy={policy} "
        f"third={test_metrics['third_capture']:.4f} avg={test_metrics['avg_third_candidates']:.3f}",
        flush=True,
    )
    return {
        "stage": stage,
        "train_races": len(train),
        "selected_policy": policy,
        "calibration_metrics": grid[0]["cal_full"],
        "test_metrics": test_metrics,
    }


def exact_v31_contexts():
    vrows = v25.load_v19_details()
    v25.fit_v21_selector(vrows)
    v25.mark_v21(vrows)
    _, entries_by, results_by = v25.raw_maps()
    train = v25.make_dataset(vrows, entries_by, results_by, v25.TRAIN_START, v25.TRAIN_END, False)
    pair_model = v26.fit_pair_model(train, PAIR_PARAMS)
    contexts = {}
    counts = {}
    for name, bounds in (("calibration", FINAL_CALIBRATION), ("diagnostic", REOPENED_DIAGNOSTIC)):
        dataset = v25.make_dataset(vrows, entries_by, results_by, bounds[0], bounds[1], False)
        selected = 0
        for race_id, race_date, base, order, vrow in dataset:
            if not vrow.get("v21_participate"):
                continue
            pairs = v31.pair_distribution(base, vrow, pair_model)
            membership = v31.membership_rank(pairs)
            first_candidates = [no for no in map(v25.ino, vrow["candidates"]) if no in base][:2]
            second_candidates = v31.choose(membership, 0.65, 0.30)
            contexts[str(race_id)] = {
                "pairs": pairs,
                "first_candidates": first_candidates,
                "first_probabilities": {
                    int(no): float(probability)
                    for no, probability in vrow["p1_map"].items()
                },
                "second_candidates": second_candidates,
                "top2_ranking": [
                    {"no": int(no), "mass": float(mass)} for no, mass in membership
                ],
            }
            selected += 1
        counts[name] = selected
    return pair_model, contexts, {
        "v31_training_period": [v25.TRAIN_START, v25.TRAIN_END],
        "v31_training_races": len(train),
        "participant_context_counts": counts,
        "pair_variant": "blind_d2a",
        "pair_params": PAIR_PARAMS,
        "second_rule": {"base": 2, "rel3": 0.65, "min3": 0.30},
    }


def monthly(logs):
    output = []
    for month in sorted({row["race_date"][:7] for row in logs}):
        month_rows = [row for row in logs if row["race_date"].startswith(month)]
        output.append({"month": month, **metrics(month_rows, True)})
    return output


def script_hash():
    return hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    races = v32.load_races()
    races_by_id = {str(race["race_id"]): race for race in races}
    print(f"[v35] loaded {len(races)} historical races; future block not loaded", flush=True)

    selected_variant, selection, feature_names = select_variant(races)
    print(f"[v35] selected model {selected_variant['name']}", flush=True)
    stages = [
        run_policy_stage(races, stage, selected_variant, feature_names)
        for stage in POLICY_STAGES
    ]

    pair_model, contexts, v31_audit = exact_v31_contexts()
    train = v32.period(races, *FINAL_TRAIN)
    x, y, feature_names = build_training_matrix(train, feature_names)
    third_model = fit_third_model(x, y, selected_variant)
    calibration_races = [
        race for race in v32.period(races, *FINAL_CALIBRATION)
        if str(race["race_id"]) in contexts
    ]
    diagnostic_races = [
        race for race in v32.period(races, *REOPENED_DIAGNOSTIC)
        if str(race["race_id"]) in contexts
    ]
    calibration_rows = predict_rows(
        calibration_races, pair_model, third_model, feature_names, contexts
    )
    board_grid = calibrate(
        calibration_rows, True, FINAL_CAL_A, FINAL_CAL_B
    )
    selected_policy = board_grid[0]["policy"]
    diagnostic_rows = predict_rows(
        diagnostic_races, pair_model, third_model, feature_names, contexts
    )
    diagnostic_logs = apply(diagnostic_rows, selected_policy, True)
    diagnostic_metrics = metrics(diagnostic_logs, True)
    print(
        f"[v35] final policy={selected_policy} cal_third={board_grid[0]['cal_full']['third_capture']:.4f} "
        f"diagnostic_third={diagnostic_metrics['third_capture']:.4f} "
        f"complete={diagnostic_metrics['complete_board_capture']:.4f} "
        f"avg={diagnostic_metrics['avg_third_candidates']:.3f}",
        flush=True,
    )

    stage_capture = [float(row["test_metrics"]["third_capture"]) for row in stages]
    stage_counts = [float(row["test_metrics"]["avg_third_candidates"]) for row in stages]
    historical_pass = (
        mean(stage_capture) > v32.V23_REFERENCE["third_capture"]
        and min(stage_capture) >= 0.50
        and max(stage_counts) <= v32.V23_REFERENCE["avg_third_candidates"]
    )
    diagnostic_pass = (
        float(diagnostic_metrics["third_capture"]) > v32.V23_REFERENCE["third_capture"]
        and float(diagnostic_metrics["avg_third_candidates"]) <= v32.V23_REFERENCE["avg_third_candidates"]
        and float(diagnostic_metrics["complete_board_capture"]) > v32.V23_REFERENCE["complete_capture"]
    )
    ready = bool(historical_pass and diagnostic_pass)

    summary = {
        "algorithm": "keirin_shogi_v35_top2_conditioned_third",
        "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST" if ready else "development_failed_not_frozen",
        "concept": (
            "For every unordered Top2 pair, predict the remaining rider that completes third; "
            "marginalize all 21 conditional branches with the unchanged v31 pair posterior."
        ),
        "data_inventory": {
            "historical_complete_races": len(races),
            "by_year": dict(Counter(str(race["race_date"])[:4] for race in races)),
            "future_block_loaded": False,
        },
        "model_selection": selection,
        "selected_variant": selected_variant,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "historical_policy_walk_forward": stages,
        "final_calibration": {
            "period": FINAL_CALIBRATION,
            "split": [FINAL_CAL_A, FINAL_CAL_B],
            "participant_races": len(calibration_rows),
            "selected_policy": selected_policy,
            "selected_metrics": board_grid[0]["cal_full"],
        },
        "reopened_2026_h1_diagnostic": {
            "period": REOPENED_DIAGNOSTIC,
            "note": "Not a fresh future test: v32-v34 aggregate results were already observed before v35.",
            "metrics": diagnostic_metrics,
            "monthly": monthly(diagnostic_logs),
        },
        "references": {"v23": v32.V23_REFERENCE},
        "acceptance": {
            "historical_pass": historical_pass,
            "diagnostic_pass": diagnostic_pass,
            "ready_for_true_future": ready,
        },
        "future_lock": {
            "third_training_period": FINAL_TRAIN,
            "v31": v31_audit,
            "true_future_period": TRUE_FUTURE,
        },
        "guards": {
            "odds_or_popularity_used": False,
            "random_shuffle": False,
            "v21_changed": False,
            "v31_parameters_changed": False,
            "v21_or_v31_used_as_candidate_filter": False,
            "future_block_loaded": False,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "race_log.json").write_text(
        json.dumps(diagnostic_logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "calibration.json").write_text(
        json.dumps(board_grid, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "block_evaluation.json").write_text(
        json.dumps(stages, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = "# v35 Top2-conditioned third\n\n"
    readme += "v31の21個のTop2組確率を変更せず、各組に対する残り5人の3着条件付き確率を周辺化する。\n"
    readme += "v21/v31は候補除外に使わない。2026 H1は再開封診断であり、完全未来試験ではない。\n\n"
    readme += "| test block | third capture | avg third candidates |\n|---|---:|---:|\n"
    for stage in stages:
        result = stage["test_metrics"]
        readme += (
            f"| {stage['stage']['test'][0]}〜{stage['stage']['test'][1]} "
            f"| {float(result['third_capture']):.2%} "
            f"| {float(result['avg_third_candidates']):.3f} |\n"
        )
    readme += (
        f"| reopened {REOPENED_DIAGNOSTIC[0]}〜{REOPENED_DIAGNOSTIC[1]} "
        f"| {float(diagnostic_metrics['third_capture']):.2%} "
        f"| {float(diagnostic_metrics['avg_third_candidates']):.3f} |\n\n"
    )
    readme += f"判定: **{'完全未来試験へ凍結' if ready else '開発不合格'}**\n\n"
    readme += "```bash\npython scripts/keirin_shogi_v35_top2_conditioned_third.py\n```\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    freeze_path = OUT / "FREEZE.json"
    if ready:
        freeze = {
            "model": "keirin_shogi_v35_top2_conditioned_third",
            "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST",
            "source_script_sha256": script_hash(),
            "selected_variant": selected_variant,
            "selected_policy": selected_policy,
            "third_training_period": FINAL_TRAIN,
            "calibration_period": FINAL_CALIBRATION,
            "v31_pair_variant": "blind_d2a",
            "v31_pair_params": PAIR_PARAMS,
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
                "ready_for_true_future": ready,
                "selected_variant": selected_variant,
                "stage_metrics": [row["test_metrics"] for row in stages],
                "selected_policy": selected_policy,
                "diagnostic_metrics": diagnostic_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

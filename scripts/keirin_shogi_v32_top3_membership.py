#!/usr/bin/env python3
from __future__ import annotations

import csv
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


# v32 changes only row 3.  The adopted v21 row-1 policy and frozen v31 row-2
# policy are reconstructed through their existing implementations.
spec = importlib.util.spec_from_file_location(
    "v31", "scripts/keirin_shogi_v31_top2_membership_second.py"
)
v31 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v31)
v26 = v31.v26
v25 = v31.v25


OUT = Path("results/keirin_shogi/v32_top3_membership")
SCRIPT = Path(__file__)

RAW_BASES = [
    *sorted(Path("data/2024/s_class_f1_all_parts").glob("2024_q*")),
    *sorted(Path("data/2025/s_class_f1_all_parts").glob("2025_q*")),
    Path("data/2026_h1/s_class_f1_all"),
]

DEV_FOLDS = [
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

FINAL_TRAIN = ["2024-01-01", "2025-10-26"]
CALIBRATION = ["2025-10-27", "2025-12-28"]
CAL_A = ["2025-10-27", "2025-11-30"]
CAL_B = ["2025-12-01", "2025-12-28"]
TEST = ["2025-12-29", "2026-06-28"]

V23_REFERENCE = {
    "third_capture": 0.5303326810176126,
    "complete_capture": 0.2974559686888454,
    "avg_third_candidates": 2.847358121330724,
    "third_given_first_second": 0.7037037037037037,
}

# The triple classifier is deliberately blind to v21/v31 outputs.  Those
# outputs enter after the unordered Top3 posterior exists.  This keeps v32
# capable of rescuing a v21/v31 miss instead of making either an eligibility
# constraint, and avoids manufacturing in-sample v31 scores for 2024.
RIDER_FEATURES = [
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
    "line_pos",
    "line_size",
    "line_mean_score_z",
    "line_max_score_z",
    "line_b_sum_z",
    "line_attack_sum_z",
    "line_pos1",
    "line_pos2",
    "line_pos3p",
    "style_escape",
    "style_both",
    "style_chase",
]

MODEL_VARIANTS = [
    {
        "name": "d2_mild",
        "depth": 2,
        "leaf": 34,
        "l2": 2.0,
        "lr": 0.045,
        "iters": 210,
        "positive_mass": 0.16,
    },
    {
        "name": "d2_balanced",
        "depth": 2,
        "leaf": 42,
        "l2": 3.0,
        "lr": 0.040,
        "iters": 240,
        "positive_mass": 0.50,
    },
    {
        "name": "d3_mild",
        "depth": 3,
        "leaf": 36,
        "l2": 2.5,
        "lr": 0.040,
        "iters": 220,
        "positive_mass": 0.16,
    },
    {
        "name": "d3_balanced",
        "depth": 3,
        "leaf": 48,
        "l2": 3.5,
        "lr": 0.035,
        "iters": 250,
        "positive_mass": 0.50,
    },
]


def num(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ino(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def result_order(rows: list[dict[str, str]]) -> list[int] | None:
    values = []
    for row in rows:
        position = ino(row.get("finish_position"))
        if position in (1, 2, 3):
            values.append((position, ino(row.get("car_no"))))
    values.sort()
    if [p for p, _ in values] != [1, 2, 3]:
        return None
    return [no for _, no in values]


def enrich_base(
    entries: list[dict[str, str]],
) -> dict[int, dict[str, object]]:
    base = v25.race_features(entries)
    by_no = {ino(row.get("car_no")): row for row in entries}
    for no, rider in base.items():
        row = by_no[no]
        position = ino(row.get("line_position"))
        style = (row.get("style") or "").strip()
        x = rider["x"]
        x.update(
            {
                "line_pos1": 1.0 if position == 1 else 0.0,
                "line_pos2": 1.0 if position == 2 else 0.0,
                "line_pos3p": 1.0 if position >= 3 else 0.0,
                "style_escape": 1.0 if style == "逃" else 0.0,
                "style_both": 1.0 if style == "両" else 0.0,
                "style_chase": 1.0 if style == "追" else 0.0,
            }
        )
    return base


def load_races() -> list[dict[str, object]]:
    race_map: dict[str, dict[str, str]] = {}
    entries_by: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    results_by: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for base in RAW_BASES:
        if not (base / "races.csv").exists():
            continue
        for row in load_csv(base / "races.csv"):
            race_map[row["race_id"]] = row
        for row in load_csv(base / "entries.csv"):
            entries_by[row["race_id"]].append(row)
        for row in load_csv(base / "results.csv"):
            results_by[row["race_id"]].append(row)

    races = []
    for race_id, race in race_map.items():
        entries = entries_by.get(race_id, [])
        if (
            race.get("meeting_grade") != "F1"
            or "Ｓ級" not in race.get("race_type", "")
            or ino(race.get("entry_count")) != 7
            or len(entries) != 7
        ):
            continue
        order = result_order(results_by.get(race_id, []))
        if order is None:
            continue
        base = enrich_base(entries)
        if set(order) - set(base):
            continue
        races.append(
            {
                "race_id": race_id,
                "race_date": race["race_date"],
                "race_type": race.get("race_type", ""),
                "base": base,
                "order": order,
            }
        )
    races.sort(key=lambda row: (row["race_date"], row["race_id"]))
    return races


def period(
    races: list[dict[str, object]], start: str, end: str
) -> list[dict[str, object]]:
    return [row for row in races if start <= row["race_date"] <= end]


def triple_features(
    base: dict[int, dict[str, object]], triple: tuple[int, int, int]
) -> dict[str, float]:
    xs = [base[no]["x"] for no in triple]
    out: dict[str, float] = {}
    for feature in RIDER_FEATURES:
        values = np.asarray([float(x[feature]) for x in xs], dtype=float)
        out[f"{feature}_mean"] = float(values.mean())
        out[f"{feature}_min"] = float(values.min())
        out[f"{feature}_max"] = float(values.max())
        out[f"{feature}_std"] = float(values.std())

    line_ids = [str(base[no]["line_id"]) for no in triple]
    line_positions = [int(base[no]["line_position"]) for no in triple]
    counts = Counter(line_ids)
    same_line_pairs = sum(
        line_ids[i] == line_ids[j] for i, j in combinations(range(3), 2)
    )
    adjacent_pairs = sum(
        line_ids[i] == line_ids[j]
        and abs(line_positions[i] - line_positions[j]) == 1
        for i, j in combinations(range(3), 2)
    )
    contains_line_12 = 0
    for line_id in counts:
        positions = {
            line_positions[i] for i, candidate in enumerate(line_ids) if candidate == line_id
        }
        contains_line_12 += int(1 in positions and 2 in positions)

    distinct_lines = len(counts)
    max_same_line = max(counts.values())
    out.update(
        {
            "distinct_line_count": float(distinct_lines),
            "same_line_pair_count": float(same_line_pairs),
            "adjacent_same_line_pair_count": float(adjacent_pairs),
            "max_same_line_members": float(max_same_line),
            "all_different_lines": float(distinct_lines == 3),
            "exactly_two_same_line": float(sorted(counts.values()) == [1, 2]),
            "all_same_line": float(max_same_line == 3),
            "contains_same_line_positions_1_2": float(contains_line_12 > 0),
            "line_front_count": float(sum(p == 1 for p in line_positions)),
            "line_second_count": float(sum(p == 2 for p in line_positions)),
            "line_third_or_later_count": float(sum(p >= 3 for p in line_positions)),
        }
    )
    return out


def build_matrix(
    races: list[dict[str, object]], feature_names: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, object]]]:
    feature_names_local = feature_names
    vectors: list[list[float]] = []
    labels: list[int] = []
    groups: list[dict[str, object]] = []
    offset = 0
    for race in races:
        base = race["base"]
        triples = list(combinations(sorted(base), 3))
        true_set = frozenset(race["order"][:3])
        features = [triple_features(base, triple) for triple in triples]
        if feature_names_local is None:
            feature_names_local = sorted(features[0])
        for triple, values in zip(triples, features):
            vectors.append([values[name] for name in feature_names_local])
            labels.append(int(frozenset(triple) == true_set))
        groups.append(
            {
                "race": race,
                "triples": triples,
                "start": offset,
                "end": offset + len(triples),
            }
        )
        offset += len(triples)
    if feature_names_local is None:
        raise ValueError("No races available for feature construction")
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        feature_names_local,
        groups,
    )


def fit_top3(
    x: np.ndarray, y: np.ndarray, variant: dict[str, object]
) -> HistGradientBoostingClassifier:
    positive_mass = float(variant["positive_mass"])
    negative_count = 34.0
    sample_weight = np.where(
        y == 1, positive_mass, (1.0 - positive_mass) / negative_count
    )
    model = HistGradientBoostingClassifier(
        learning_rate=float(variant["lr"]),
        max_iter=int(variant["iters"]),
        max_depth=int(variant["depth"]),
        min_samples_leaf=int(variant["leaf"]),
        l2_regularization=float(variant["l2"]),
        random_state=32,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def normalized_probabilities(raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw, dtype=float), 1e-12, None)
    return clipped / clipped.sum()


def membership_from_triples(
    triples: list[tuple[int, int, int]], probabilities: np.ndarray
) -> dict[int, float]:
    membership: defaultdict[int, float] = defaultdict(float)
    for triple, probability in zip(triples, probabilities):
        for no in triple:
            membership[no] += float(probability)
    return dict(membership)


def rank_of(ranking: list[tuple[int, float]], target: int) -> int:
    return next(i + 1 for i, (no, _) in enumerate(ranking) if no == target)


def evaluate_top3_model(
    model: HistGradientBoostingClassifier,
    x: np.ndarray,
    y: np.ndarray,
    groups: list[dict[str, object]],
) -> dict[str, float | int]:
    raw = model.predict_proba(x)[:, 1]
    triple_ranks: list[int] = []
    third_ranks: list[int] = []
    losses: list[float] = []
    for group in groups:
        start, end = int(group["start"]), int(group["end"])
        probabilities = normalized_probabilities(raw[start:end])
        triples = group["triples"]
        true_index = int(np.argmax(y[start:end]))
        order = np.argsort(-probabilities, kind="stable")
        triple_rank = int(np.flatnonzero(order == true_index)[0]) + 1
        membership = membership_from_triples(triples, probabilities)
        ranking = sorted(membership.items(), key=lambda item: (-item[1], item[0]))
        actual_third = int(group["race"]["order"][2])
        triple_ranks.append(triple_rank)
        third_ranks.append(rank_of(ranking, actual_third))
        losses.append(-math.log(max(float(probabilities[true_index]), 1e-12)))

    count = len(groups)
    return {
        "races": count,
        "true_triple_top1": sum(rank <= 1 for rank in triple_ranks) / count,
        "true_triple_top3": sum(rank <= 3 for rank in triple_ranks) / count,
        "true_triple_top5": sum(rank <= 5 for rank in triple_ranks) / count,
        "true_triple_top10": sum(rank <= 10 for rank in triple_ranks) / count,
        "actual_third_membership_top1": sum(rank <= 1 for rank in third_ranks) / count,
        "actual_third_membership_top2": sum(rank <= 2 for rank in third_ranks) / count,
        "actual_third_membership_top3": sum(rank <= 3 for rank in third_ranks) / count,
        "actual_third_membership_top4": sum(rank <= 4 for rank in third_ranks) / count,
        "mean_true_triple_nll": mean(losses),
    }


def fold_score(metrics: dict[str, float | int]) -> float:
    return (
        0.24 * float(metrics["true_triple_top1"])
        + 0.20 * float(metrics["true_triple_top3"])
        + 0.16 * float(metrics["true_triple_top5"])
        + 0.18 * float(metrics["actual_third_membership_top2"])
        + 0.22 * float(metrics["actual_third_membership_top3"])
        - 0.012 * float(metrics["mean_true_triple_nll"])
    )


def select_variant(
    races: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[str]]:
    records: dict[str, dict[str, object]] = {
        str(variant["name"]): {"variant": variant, "folds": []}
        for variant in MODEL_VARIANTS
    }
    feature_names: list[str] | None = None
    for fold in DEV_FOLDS:
        train = period(races, *fold["train"])
        validation = period(races, *fold["validation"])
        print(
            f"[v32] build {fold['name']}: train={len(train)} validation={len(validation)}",
            flush=True,
        )
        x_train, y_train, feature_names, _ = build_matrix(train, feature_names)
        x_validation, y_validation, _, groups = build_matrix(
            validation, feature_names
        )
        for variant in MODEL_VARIANTS:
            model = fit_top3(x_train, y_train, variant)
            metrics = evaluate_top3_model(model, x_validation, y_validation, groups)
            score = fold_score(metrics)
            records[str(variant["name"])]["folds"].append(
                {"fold": fold, "score": score, "metrics": metrics}
            )
            print(
                f"[v32] {fold['name']} {variant['name']} score={score:.5f} "
                f"third@3={metrics['actual_third_membership_top3']:.4f}",
                flush=True,
            )
        del x_train, y_train, x_validation, y_validation

    selection: list[dict[str, object]] = []
    for record in records.values():
        scores = [float(row["score"]) for row in record["folds"]]
        objective = 0.55 * min(scores) + 0.45 * mean(scores)
        selection.append(
            {
                "variant": record["variant"],
                "selection_objective": objective,
                "minimum_fold_score": min(scores),
                "mean_fold_score": mean(scores),
                "folds": record["folds"],
            }
        )
    selection.sort(
        key=lambda row: (
            float(row["selection_objective"]),
            float(row["minimum_fold_score"]),
        ),
        reverse=True,
    )
    return selection[0]["variant"], selection, feature_names or []


def predict_top3(
    model: HistGradientBoostingClassifier,
    feature_names: list[str],
    base: dict[int, dict[str, object]],
) -> tuple[dict[int, float], list[dict[str, object]]]:
    triples = list(combinations(sorted(base), 3))
    features = [triple_features(base, triple) for triple in triples]
    x = np.asarray(
        [[values[name] for name in feature_names] for values in features],
        dtype=np.float32,
    )
    probabilities = normalized_probabilities(model.predict_proba(x)[:, 1])
    membership = membership_from_triples(triples, probabilities)
    triple_rows = [
        {"members": list(triple), "probability": float(probability)}
        for triple, probability in zip(triples, probabilities)
    ]
    triple_rows.sort(key=lambda row: -float(row["probability"]))
    return membership, triple_rows


def build_v31_contexts() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    vrows = v25.load_v19_details()
    v25.fit_v21_selector(vrows)
    v25.mark_v21(vrows)
    _, entries_by, results_by = v25.raw_maps()
    train = v25.make_dataset(
        vrows, entries_by, results_by, v25.TRAIN_START, v25.TRAIN_END, False
    )
    pair_params = {"depth": 2, "leaf": 18, "l2": 1.5, "lr": 0.045, "iters": 190}
    pair_model = v26.fit_pair_model(train, pair_params)

    contexts: dict[str, dict[str, object]] = {}
    counts: dict[str, int] = {}
    for name, bounds in (("calibration", CALIBRATION), ("test", TEST)):
        dataset = v25.make_dataset(
            vrows, entries_by, results_by, bounds[0], bounds[1], False
        )
        selected = 0
        for race_id, race_date, base, order, vrow in dataset:
            if not vrow.get("v21_participate"):
                continue
            pairs = v31.pair_distribution(base, vrow, pair_model)
            membership_ranking = v31.membership_rank(pairs)
            top2_membership = {no: float(mass) for no, mass in membership_ranking}
            first_candidates = [
                no for no in map(v25.ino, vrow["candidates"]) if no in base
            ][:2]
            second_candidates = v31.choose(membership_ranking, 0.65, 0.30)
            contexts[race_id] = {
                "race_id": race_id,
                "race_date": race_date,
                "order": order,
                "first_candidates": first_candidates,
                "second_candidates": second_candidates,
                "top2_membership": top2_membership,
                "top2_ranking": [
                    {"no": no, "mass": float(mass)}
                    for no, mass in membership_ranking
                ],
            }
            selected += 1
        counts[name] = selected
    audit = {
        "v31_train_races": len(train),
        "participant_context_counts": counts,
        "pair_variant": "blind_d2a",
        "pair_params": pair_params,
        "second_candidate_rule": {
            "base_count": 2,
            "add_third_membership_min": 0.30,
            "add_third_ratio_to_rank2_min": 0.65,
        },
    }
    return contexts, audit


def zscores(values: dict[int, float]) -> dict[int, float]:
    numbers = np.asarray(list(values.values()), dtype=float)
    standard_deviation = float(numbers.std()) or 1.0
    average = float(numbers.mean())
    return {no: (float(value) - average) / standard_deviation for no, value in values.items()}


def score_method(
    top3_membership: dict[int, float],
    top2_membership: dict[int, float],
    beta: float,
) -> dict[int, float]:
    return {
        no: float(top3_membership[no]) - beta * float(top2_membership.get(no, 0.0))
        for no in top3_membership
    }


def choose_third(
    scores: dict[int, float], gap3: float, gap4: float | None
) -> tuple[list[int], list[tuple[int, float]], dict[str, float]]:
    standardized = zscores(scores)
    ranking = sorted(standardized.items(), key=lambda item: (-item[1], item[0]))
    candidates = [ranking[0][0], ranking[1][0]]
    actual_gap3 = float(ranking[1][1] - ranking[2][1])
    actual_gap4 = float(ranking[2][1] - ranking[3][1])
    if actual_gap3 <= gap3:
        candidates.append(ranking[2][0])
        if gap4 is not None and actual_gap4 <= gap4:
            candidates.append(ranking[3][0])
    return candidates, ranking, {"gap_rank2_rank3": actual_gap3, "gap_rank3_rank4": actual_gap4}


def prediction_rows(
    races_by_id: dict[str, dict[str, object]],
    contexts: dict[str, dict[str, object]],
    model: HistGradientBoostingClassifier,
    feature_names: list[str],
    bounds: list[str],
) -> list[dict[str, object]]:
    rows = []
    for race_id, context in contexts.items():
        if not (bounds[0] <= context["race_date"] <= bounds[1]):
            continue
        race = races_by_id.get(race_id)
        if race is None:
            continue
        top3_membership, triples = predict_top3(model, feature_names, race["base"])
        rows.append(
            {
                **context,
                "race_type": race["race_type"],
                "top3_membership": top3_membership,
                "top3_ranking": [
                    {"no": no, "mass": float(mass)}
                    for no, mass in sorted(
                        top3_membership.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
                "top_triples": triples[:5],
            }
        )
    rows.sort(key=lambda row: (row["race_date"], row["race_id"]))
    return rows


def apply_policy(
    rows: list[dict[str, object]], policy: dict[str, object]
) -> list[dict[str, object]]:
    logs = []
    beta = float(policy["beta"])
    for row in rows:
        raw_scores = score_method(
            row["top3_membership"], row["top2_membership"], beta
        )
        candidates, ranking, gaps = choose_third(
            raw_scores, float(policy["gap3"]), policy["gap4"]
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
                "complete_board_hit": int(first_hit and second_hit and third_hit),
                "first_second_hit": int(first_hit and second_hit),
                "third_candidate_count": len(candidates),
                "board_cell_count": len(row["first_candidates"])
                + len(row["second_candidates"])
                + len(candidates),
                "score_method": policy["method"],
                "beta": beta,
                "score_ranking": [
                    {"no": no, "zscore": float(score), "raw_score": raw_scores[no]}
                    for no, score in ranking
                ],
                **gaps,
                "top3_membership": row["top3_ranking"],
                "top2_membership": row["top2_ranking"],
                "top_triples": row["top_triples"],
            }
        )
    return logs


def safe_rate(rows: list[dict[str, object]], key: str) -> float:
    return sum(int(row[key]) for row in rows) / len(rows) if rows else 0.0


def board_metrics(logs: list[dict[str, object]]) -> dict[str, float | int]:
    count = len(logs)
    if not count:
        return {"races": 0}
    first_second = [row for row in logs if row["first_second_hit"]]
    return {
        "races": count,
        "first_capture": safe_rate(logs, "first_hit"),
        "second_capture": safe_rate(logs, "second_hit"),
        "third_capture": safe_rate(logs, "third_hit"),
        "complete_board_capture": safe_rate(logs, "complete_board_hit"),
        "third_given_first_second": safe_rate(first_second, "third_hit"),
        "avg_first_candidates": mean(len(row["first_candidates"]) for row in logs),
        "avg_second_candidates": mean(len(row["second_candidates"]) for row in logs),
        "avg_third_candidates": mean(int(row["third_candidate_count"]) for row in logs),
        "avg_board_cells": mean(int(row["board_cell_count"]) for row in logs),
        "two_candidate_rate": sum(row["third_candidate_count"] == 2 for row in logs) / count,
        "three_candidate_rate": sum(row["third_candidate_count"] == 3 for row in logs) / count,
        "four_candidate_rate": sum(row["third_candidate_count"] == 4 for row in logs) / count,
    }


def calibration_objective(
    metrics_a: dict[str, float | int],
    metrics_b: dict[str, float | int],
    full: dict[str, float | int],
) -> float:
    min_third = min(float(metrics_a["third_capture"]), float(metrics_b["third_capture"]))
    min_complete = min(
        float(metrics_a["complete_board_capture"]),
        float(metrics_b["complete_board_capture"]),
    )
    min_conditional = min(
        float(metrics_a["third_given_first_second"]),
        float(metrics_b["third_given_first_second"]),
    )
    drift = abs(float(metrics_a["third_capture"]) - float(metrics_b["third_capture"]))
    average_count = float(full["avg_third_candidates"])
    four_rate = float(full["four_candidate_rate"])
    return (
        0.45 * min_third
        + 0.30 * min_complete
        + 0.15 * min_conditional
        + 0.10 * float(full["third_capture"])
        - 0.10 * max(0.0, average_count - 2.0)
        - 0.18 * four_rate
        - 0.10 * drift
    )


def method_specs() -> list[dict[str, object]]:
    return [
        {"method": "A_top3_membership", "beta": 0.0},
        {"method": "B_top3_minus_top2", "beta": 1.0},
        {"method": "C_blend_beta_0.25", "beta": 0.25},
        {"method": "C_blend_beta_0.50", "beta": 0.50},
        {"method": "C_blend_beta_0.75", "beta": 0.75},
        {"method": "C_blend_beta_1.25", "beta": 1.25},
        {"method": "C_blend_beta_1.50", "beta": 1.50},
    ]


def calibrate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grid = []
    rows_a = [row for row in rows if CAL_A[0] <= row["race_date"] <= CAL_A[1]]
    rows_b = [row for row in rows if CAL_B[0] <= row["race_date"] <= CAL_B[1]]
    for method in method_specs():
        for gap3 in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.00, 1.25):
            for gap4 in (None, 0.0, 0.10, 0.20, 0.30, 0.40):
                policy = {**method, "gap3": gap3, "gap4": gap4}
                metrics_a = board_metrics(apply_policy(rows_a, policy))
                metrics_b = board_metrics(apply_policy(rows_b, policy))
                full = board_metrics(apply_policy(rows, policy))
                if float(full["avg_third_candidates"]) > 2.85:
                    continue
                grid.append(
                    {
                        **policy,
                        "objective": calibration_objective(metrics_a, metrics_b, full),
                        "cal_a": metrics_a,
                        "cal_b": metrics_b,
                        "cal_full": full,
                    }
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


def best_by_method(grid: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in grid:
        best.setdefault(str(row["method"]), row)
    return list(best.values())


def fixed_count_diagnostics(
    rows: list[dict[str, object]], beta: float, method: str
) -> list[dict[str, object]]:
    output = []
    for count in (2, 3, 4):
        logs = []
        for row in rows:
            scores = score_method(row["top3_membership"], row["top2_membership"], beta)
            ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            candidates = [no for no, _ in ranking[:count]]
            first, second, third = map(int, row["order"][:3])
            first_hit = int(first in row["first_candidates"])
            second_hit = int(second in row["second_candidates"])
            third_hit = int(third in candidates)
            logs.append(
                {
                    "first_candidates": row["first_candidates"],
                    "second_candidates": row["second_candidates"],
                    "third_hit": third_hit,
                    "first_hit": first_hit,
                    "second_hit": second_hit,
                    "first_second_hit": int(first_hit and second_hit),
                    "complete_board_hit": int(first_hit and second_hit and third_hit),
                    "third_candidate_count": count,
                    "board_cell_count": len(row["first_candidates"])
                    + len(row["second_candidates"])
                    + count,
                }
            )
        output.append({"method": method, "beta": beta, "fixed_count": count, **board_metrics(logs)})
    return output


def grouped_metrics(logs: list[dict[str, object]], key_name: str, key_fn) -> list[dict[str, object]]:
    output = []
    for key in sorted({key_fn(row) for row in logs}):
        group = [row for row in logs if key_fn(row) == key]
        output.append({key_name: key, **board_metrics(group)})
    return output


def method_holdout_comparison(
    calibration_grid: list[dict[str, object]], test_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    output = []
    for calibration in best_by_method(calibration_grid):
        policy = {
            key: calibration[key] for key in ("method", "beta", "gap3", "gap4")
        }
        output.append(
            {
                "policy_selected_on_calibration": policy,
                "calibration_metrics": calibration["cal_full"],
                "holdout_metrics": board_metrics(apply_policy(test_rows, policy)),
            }
        )
    return output


def script_hash() -> str:
    return hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    races = load_races()
    races_by_id = {str(row["race_id"]): row for row in races}
    year_counts = Counter(str(row["race_date"])[:4] for row in races)
    print(f"[v32] valid races by year: {dict(year_counts)}", flush=True)

    selected_variant, walk_forward, feature_names = select_variant(races)
    print(f"[v32] selected model: {selected_variant['name']}", flush=True)

    final_train = period(races, *FINAL_TRAIN)
    x_train, y_train, feature_names, _ = build_matrix(final_train, feature_names)
    final_model = fit_top3(x_train, y_train, selected_variant)
    print(f"[v32] final model fitted on {len(final_train)} races", flush=True)
    del x_train, y_train

    contexts, v31_audit = build_v31_contexts()
    calibration_rows = prediction_rows(
        races_by_id, contexts, final_model, feature_names, CALIBRATION
    )
    calibration_grid = calibrate(calibration_rows)
    if not calibration_grid:
        raise RuntimeError("No feasible third-candidate calibration remained")
    selected_policy = {
        key: calibration_grid[0][key]
        for key in ("method", "beta", "gap3", "gap4")
    }
    print(
        f"[v32] selected policy: {selected_policy} "
        f"cal_third={calibration_grid[0]['cal_full']['third_capture']:.4f}",
        flush=True,
    )

    # The holdout is touched only after model, blend, and candidate rule are fixed.
    test_rows = prediction_rows(races_by_id, contexts, final_model, feature_names, TEST)
    logs = apply_policy(test_rows, selected_policy)
    test_metrics = board_metrics(logs)
    print(
        f"[v32] holdout races={len(logs)} third={test_metrics['third_capture']:.4f} "
        f"complete={test_metrics['complete_board_capture']:.4f} "
        f"avg3={test_metrics['avg_third_candidates']:.3f}",
        flush=True,
    )

    accepted = (
        float(test_metrics["third_capture"]) > V23_REFERENCE["third_capture"]
        and float(test_metrics["avg_third_candidates"])
        <= V23_REFERENCE["avg_third_candidates"]
    )

    fixed_calibration = []
    fixed_test = []
    for method in method_specs():
        fixed_calibration.extend(
            fixed_count_diagnostics(
                calibration_rows, float(method["beta"]), str(method["method"])
            )
        )
        fixed_test.extend(
            fixed_count_diagnostics(test_rows, float(method["beta"]), str(method["method"])
            )
        )

    monthly = grouped_metrics(logs, "month", lambda row: str(row["race_date"])[:7])
    by_race_type = grouped_metrics(logs, "race_type", lambda row: str(row["race_type"]))
    method_comparison = method_holdout_comparison(calibration_grid, test_rows)

    summary = {
        "algorithm": "keirin_shogi_v32_top3_membership",
        "status": "accepted" if accepted else "rejected_after_untouched_holdout",
        "concept": (
            "Predict one unordered Top3 set among 35 triples, normalize the set posterior, "
            "and marginalize it into rider Top3-membership. Compare A=Top3 membership, "
            "B=Top3-Top2 boundary, and C=calibration-only learned Top3-beta*Top2."
        ),
        "data_audit": {
            "raw_bases": [str(path) for path in RAW_BASES],
            "valid_f1_s_class_7_races_by_year": dict(year_counts),
            "odds_files_loaded": False,
        },
        "frozen_dependencies": {
            "first_core": "v21 quantile participation; unchanged",
            "second_core": "v31 blind_d2a Top2 membership; unchanged",
            "v31_reconstruction": v31_audit,
        },
        "top3_model": {
            "training_period": FINAL_TRAIN,
            "training_races": len(final_train),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "selected_variant": selected_variant,
            "selection_rule": "0.55 * worst development-fold score + 0.45 * mean development-fold score",
            "walk_forward_development": walk_forward,
            "dependency_guard": (
                "The Top3 classifier receives neither v21 nor v31 outputs. v31 enters only "
                "after Top3 inference in B/C, so v32 can rescue a v21/v31 miss and 2024 "
                "training cannot receive later v31 labels."
            ),
        },
        "calibration_period": CALIBRATION,
        "calibration_split": [CAL_A, CAL_B],
        "calibration_participant_races": len(calibration_rows),
        "selected_policy": selected_policy,
        "selected_calibration_metrics": calibration_grid[0]["cal_full"],
        "best_calibration_by_method": best_by_method(calibration_grid),
        "fixed_count_calibration_diagnostics": fixed_calibration,
        "holdout_period": TEST,
        "holdout_participant_races": len(logs),
        "holdout_metrics": test_metrics,
        "holdout_method_comparison_without_reselection": method_comparison,
        "fixed_count_holdout_diagnostics_without_reselection": fixed_test,
        "monthly_holdout": monthly,
        "race_type_holdout": by_race_type,
        "v23_reference": V23_REFERENCE,
        "delta_vs_v23": {
            "third_capture": float(test_metrics["third_capture"])
            - V23_REFERENCE["third_capture"],
            "complete_capture": float(test_metrics["complete_board_capture"])
            - V23_REFERENCE["complete_capture"],
            "avg_third_candidates": float(test_metrics["avg_third_candidates"])
            - V23_REFERENCE["avg_third_candidates"],
            "third_given_first_second": float(test_metrics["third_given_first_second"])
            - V23_REFERENCE["third_given_first_second"],
        },
        "acceptance_rule": {
            "third_capture_strictly_above_v23": V23_REFERENCE["third_capture"],
            "avg_third_candidates_at_most_v23": V23_REFERENCE["avg_third_candidates"],
        },
        "accepted": accepted,
        "leakage_guard": {
            "random_shuffle": False,
            "odds_or_popularity_used": False,
            "future_labels_used_for_thresholds": False,
            "development_order": [fold["name"] for fold in DEV_FOLDS],
            "model_locked_before_calibration": True,
            "model_and_candidate_policy_locked_before_2026_h1": True,
            "holdout_alternatives_reported_but_not_used_for_reselection": True,
        },
    }

    block_metrics = {
        "monthly": monthly,
        "race_type": by_race_type,
        "candidate_count": grouped_metrics(
            logs, "third_candidate_count", lambda row: int(row["third_candidate_count"])
        ),
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "race_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "calibration.json").write_text(
        json.dumps(calibration_grid, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "block_metrics.json").write_text(
        json.dumps(block_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# v32 Top3 Membership

## 結論

- 判定: **{'採用' if accepted else '不採用'}**
- 2026年前半ホールドアウト: {len(logs)}R
- 3着捕捉率: {float(test_metrics['third_capture']):.2%}
- 平均3着候補数: {float(test_metrics['avg_third_candidates']):.3f}人
- 完全盤面捕捉率: {float(test_metrics['complete_board_capture']):.2%}
- 選択方式: {selected_policy['method']} (beta={selected_policy['beta']}, gap3={selected_policy['gap3']}, gap4={selected_policy['gap4']})

## 時系列

モデル候補は2024年後半、2025年前半、2025年7月〜10月26日の順で検証した。
最終モデルは2024年1月1日〜2025年10月26日だけで再学習した。
A/B/Cの方式と2〜4人の可変候補ルールは2025年10月27日〜12月28日だけで校正した。
2026年1月以降はすべて固定後に一度だけ評価した。

## A/B/C

- A: `Top3Membership`
- B: `Top3Membership - Top2Membership`
- C: `Top3Membership - beta * Top2Membership`（betaは校正期間だけで選択）

3着候補は常に2人を置き、2位と3位の標準化スコア差が小さい時だけ3人目を追加する。
4人目は3位と4位も接近した場合だけ許可し、校正目的関数で強く減点する。

## 再現

```bash
python -m pip install scikit-learn
python scripts/keirin_shogi_v32_top3_membership.py
```

オッズ、人気、支持率は読み込まない。v21とv31のルールは変更していない。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    freeze_path = OUT / "FREEZE.json"
    if accepted:
        freeze = {
            "model": "keirin_shogi_v32_top3_membership",
            "status": "FROZEN_AFTER_UNTOUCHED_2026_H1_HOLDOUT",
            "source_script_sha256": script_hash(),
            "first_core": "v21 unchanged",
            "second_core": "v31 unchanged",
            "selected_top3_variant": selected_variant,
            "selected_policy": selected_policy,
            "training_period": FINAL_TRAIN,
            "calibration_period": CALIBRATION,
            "holdout_period": TEST,
            "holdout_metrics": test_metrics,
            "future_tuning_allowed": False,
            "future_rule": (
                "Any change after seeing this holdout must use a new version and a later, "
                "previously unopened future block."
            ),
        }
        freeze_path.write_text(
            json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif freeze_path.exists():
        freeze_path.unlink()

    print(json.dumps({"accepted": accepted, "holdout_metrics": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

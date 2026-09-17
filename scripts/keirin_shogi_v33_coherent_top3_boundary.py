#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v32 = load_module("v32", "scripts/keirin_shogi_v32_top3_membership.py")
v31 = v32.v31
v26 = v32.v26
v25 = v32.v25

OUT = Path("results/keirin_shogi/v33_coherent_top3_boundary")
SCRIPT = Path(__file__)

TOP3_VARIANT = {
    "name": "d2_mild",
    "depth": 2,
    "leaf": 34,
    "l2": 2.0,
    "lr": 0.045,
    "iters": 210,
    "positive_mass": 0.16,
}
PAIR_PARAMS = {"depth": 2, "leaf": 18, "l2": 1.5, "lr": 0.045, "iters": 190}

ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)
CUM2_THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
CUM3_THRESHOLDS = (None, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

STAGES = [
    {
        "name": "stage1_2024q4",
        "train": ["2024-01-01", "2024-06-30"],
        "calibration": ["2024-07-01", "2024-09-30"],
        "test": ["2024-10-01", "2024-12-31"],
    },
    {
        "name": "stage2_2025q2",
        "train": ["2024-01-01", "2024-12-31"],
        "calibration": ["2025-01-01", "2025-03-31"],
        "test": ["2025-04-01", "2025-06-30"],
    },
    {
        "name": "stage3_2025q4",
        "train": ["2024-01-01", "2025-06-30"],
        "calibration": ["2025-07-01", "2025-10-26"],
        "test": ["2025-10-27", "2025-12-28"],
    },
]

REOPENED_DIAGNOSTIC = {
    "name": "reopened_v32_holdout_diagnostic",
    "train": ["2024-01-01", "2025-10-26"],
    "calibration": ["2025-10-27", "2025-12-28"],
    "test": ["2025-12-29", "2026-06-28"],
}

FUTURE_MODEL_TRAIN = ["2024-01-01", "2026-03-31"]
FUTURE_POLICY_CALIBRATION = ["2026-04-01", "2026-06-28"]
FUTURE_POLICY_SPLIT_A = ["2026-04-01", "2026-05-15"]
FUTURE_POLICY_SPLIT_B = ["2026-05-16", "2026-06-28"]
FUTURE_REFIT = ["2024-01-01", "2026-06-28"]
TRUE_FUTURE = ["2026-07-06", "2026-08-30"]

V23_REFERENCE = v32.V23_REFERENCE


def raw_dataset(races: list[dict[str, object]]) -> list[tuple]:
    return [
        (
            str(race["race_id"]),
            str(race["race_date"]),
            race["base"],
            race["order"],
            {"p1_map": {}},
        )
        for race in races
    ]


def fit_top3_model(
    races: list[dict[str, object]], feature_names: list[str] | None = None
):
    x, y, feature_names, _ = v32.build_matrix(races, feature_names)
    model = v32.fit_top3(x, y, TOP3_VARIANT)
    del x, y
    return model, feature_names


def fit_pair_model(races: list[dict[str, object]]):
    return v26.fit_pair_model(raw_dataset(races), PAIR_PARAMS)


def exact_v31_pair_model():
    vrows = v25.load_v19_details()
    v25.fit_v21_selector(vrows)
    v25.mark_v21(vrows)
    _, entries_by, results_by = v25.raw_maps()
    train = v25.make_dataset(
        vrows, entries_by, results_by, v25.TRAIN_START, v25.TRAIN_END, False
    )
    return v26.fit_pair_model(train, PAIR_PARAMS), len(train)


def pair_probabilities(base: dict[int, dict[str, object]], pair_model) -> dict[frozenset, float]:
    rows = v31.pair_distribution(base, {"p1_map": {}}, pair_model)
    return {frozenset((a, b)): float(probability) for a, b, probability in rows}


def coherent_boundary(
    triple_rows: list[dict[str, object]],
    pair_probs: dict[frozenset, float],
    alpha: float,
) -> dict[int, float]:
    scores: defaultdict[int, float] = defaultdict(float)
    for triple_row in triple_rows:
        members = tuple(map(int, triple_row["members"]))
        triple_probability = float(triple_row["probability"])
        complement_weights = {}
        for third in members:
            complement = frozenset(no for no in members if no != third)
            pair_probability = max(pair_probs.get(complement, 0.0), 1e-12)
            complement_weights[third] = pair_probability**alpha
        total = sum(complement_weights.values()) or 1.0
        for third, weight in complement_weights.items():
            scores[third] += triple_probability * weight / total
    normalization = sum(scores.values()) or 1.0
    return {no: value / normalization for no, value in scores.items()}


def predict_rows(
    races: list[dict[str, object]],
    top3_model,
    feature_names: list[str],
    pair_model,
    label: str,
) -> list[dict[str, object]]:
    rows = []
    for index, race in enumerate(races, 1):
        _, triple_rows = v32.predict_top3(top3_model, feature_names, race["base"])
        pair_probs = pair_probabilities(race["base"], pair_model)
        scores_by_alpha = {
            str(alpha): coherent_boundary(triple_rows, pair_probs, alpha)
            for alpha in ALPHAS
        }
        rows.append(
            {
                "race_id": race["race_id"],
                "race_date": race["race_date"],
                "race_type": race["race_type"],
                "order": race["order"],
                "scores_by_alpha": scores_by_alpha,
                "top_triples": triple_rows[:5],
            }
        )
        if index % 500 == 0:
            print(f"[v33] {label}: predicted {index}/{len(races)}", flush=True)
    return rows


def choose(scores: dict[int, float], cum2: float, cum3: float | None):
    ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    candidates = [ranking[0][0], ranking[1][0]]
    top2_mass = float(ranking[0][1] + ranking[1][1])
    top3_mass = top2_mass + float(ranking[2][1])
    if top2_mass < cum2:
        candidates.append(ranking[2][0])
        if cum3 is not None and top3_mass < cum3:
            candidates.append(ranking[3][0])
    return candidates, ranking, top2_mass, top3_mass


def apply_simple(rows: list[dict[str, object]], policy: dict[str, object]):
    logs = []
    alpha = str(float(policy["alpha"]))
    for row in rows:
        scores = row["scores_by_alpha"][alpha]
        candidates, ranking, top2_mass, top3_mass = choose(
            scores, float(policy["cum2"]), policy["cum3"]
        )
        actual_third = int(row["order"][2])
        logs.append(
            {
                "race_id": row["race_id"],
                "race_date": row["race_date"],
                "race_type": row["race_type"],
                "actual_third": actual_third,
                "third_candidates": candidates,
                "third_hit": int(actual_third in candidates),
                "third_candidate_count": len(candidates),
                "top2_probability_mass": top2_mass,
                "top3_probability_mass": top3_mass,
                "third_probability_ranking": [
                    {"no": no, "probability": float(probability)}
                    for no, probability in ranking
                ],
            }
        )
    return logs


def simple_metrics(logs: list[dict[str, object]]) -> dict[str, float | int]:
    count = len(logs)
    if not count:
        return {"races": 0}
    return {
        "races": count,
        "third_capture": sum(row["third_hit"] for row in logs) / count,
        "avg_third_candidates": mean(row["third_candidate_count"] for row in logs),
        "two_candidate_rate": sum(row["third_candidate_count"] == 2 for row in logs) / count,
        "three_candidate_rate": sum(row["third_candidate_count"] == 3 for row in logs) / count,
        "four_candidate_rate": sum(row["third_candidate_count"] == 4 for row in logs) / count,
    }


def simple_objective(a, b, full) -> float:
    min_capture = min(float(a["third_capture"]), float(b["third_capture"]))
    drift = abs(float(a["third_capture"]) - float(b["third_capture"]))
    return (
        0.72 * min_capture
        + 0.28 * float(full["third_capture"])
        - 0.10 * max(0.0, float(full["avg_third_candidates"]) - 2.0)
        - 0.20 * float(full["four_candidate_rate"])
        - 0.10 * drift
    )


def split_halves(rows: list[dict[str, object]]):
    midpoint = max(1, len(rows) // 2)
    return rows[:midpoint], rows[midpoint:]


def calibrate_simple(rows: list[dict[str, object]]):
    rows_a, rows_b = split_halves(rows)
    grid = []
    for alpha in ALPHAS:
        for cum2 in CUM2_THRESHOLDS:
            for cum3 in CUM3_THRESHOLDS:
                policy = {"alpha": alpha, "cum2": cum2, "cum3": cum3}
                metrics_a = simple_metrics(apply_simple(rows_a, policy))
                metrics_b = simple_metrics(apply_simple(rows_b, policy))
                full = simple_metrics(apply_simple(rows, policy))
                if float(full["avg_third_candidates"]) > 2.85:
                    continue
                grid.append(
                    {
                        **policy,
                        "objective": simple_objective(metrics_a, metrics_b, full),
                        "cal_a": metrics_a,
                        "cal_b": metrics_b,
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


def run_stage(
    all_races: list[dict[str, object]],
    stage: dict[str, object],
    feature_names: list[str] | None,
    exact_pair: bool = False,
):
    train = v32.period(all_races, *stage["train"])
    calibration = v32.period(all_races, *stage["calibration"])
    test = v32.period(all_races, *stage["test"])
    print(
        f"[v33] {stage['name']}: train={len(train)} calibration={len(calibration)} test={len(test)}",
        flush=True,
    )
    top3_model, feature_names = fit_top3_model(train, feature_names)
    if exact_pair:
        pair_model, pair_train_races = exact_v31_pair_model()
    else:
        pair_model = fit_pair_model(train)
        pair_train_races = len(train)
    calibration_rows = predict_rows(
        calibration, top3_model, feature_names, pair_model, f"{stage['name']} calibration"
    )
    grid = calibrate_simple(calibration_rows)
    policy = {key: grid[0][key] for key in ("alpha", "cum2", "cum3")}
    test_rows = predict_rows(test, top3_model, feature_names, pair_model, f"{stage['name']} test")
    test_logs = apply_simple(test_rows, policy)
    test_metrics = simple_metrics(test_logs)
    print(
        f"[v33] {stage['name']} policy={policy} third={test_metrics['third_capture']:.4f} "
        f"avg={test_metrics['avg_third_candidates']:.3f}",
        flush=True,
    )
    return (
        {
            "stage": stage,
            "train_races": len(train),
            "pair_train_races": pair_train_races,
            "selected_policy": policy,
            "calibration_metrics": grid[0]["cal_full"],
            "test_metrics": test_metrics,
            "calibration_top20": grid[:20],
        },
        feature_names,
    )


def add_pair_probabilities_to_contexts(
    contexts: dict[str, dict[str, object]],
    races_by_id: dict[str, dict[str, object]],
    pair_model,
):
    output = {}
    for race_id, context in contexts.items():
        race = races_by_id.get(race_id)
        if race is None:
            continue
        output[race_id] = {
            **context,
            "race_type": race["race_type"],
            "pair_probabilities": pair_probabilities(race["base"], pair_model),
        }
    return output


def board_prediction_rows(
    races: list[dict[str, object]],
    contexts: dict[str, dict[str, object]],
    top3_model,
    feature_names: list[str],
):
    rows = []
    for index, race in enumerate(races, 1):
        context = contexts.get(str(race["race_id"]))
        if context is None:
            continue
        _, triple_rows = v32.predict_top3(top3_model, feature_names, race["base"])
        scores_by_alpha = {
            str(alpha): coherent_boundary(
                triple_rows, context["pair_probabilities"], alpha
            )
            for alpha in ALPHAS
        }
        rows.append(
            {
                **context,
                "top_triples": triple_rows[:5],
                "scores_by_alpha": scores_by_alpha,
            }
        )
        if index % 200 == 0:
            print(f"[v33] future calibration predicted {index}/{len(races)}", flush=True)
    rows.sort(key=lambda row: (row["race_date"], row["race_id"]))
    return rows


def apply_board(rows: list[dict[str, object]], policy: dict[str, object]):
    logs = []
    alpha = str(float(policy["alpha"]))
    for row in rows:
        candidates, ranking, top2_mass, top3_mass = choose(
            row["scores_by_alpha"][alpha],
            float(policy["cum2"]),
            policy["cum3"],
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
                "top2_probability_mass": top2_mass,
                "top3_probability_mass": top3_mass,
                "third_probability_ranking": [
                    {"no": no, "probability": float(probability)}
                    for no, probability in ranking
                ],
                "top_triples": row["top_triples"],
            }
        )
    return logs


def board_objective(a, b, full) -> float:
    min_third = min(float(a["third_capture"]), float(b["third_capture"]))
    min_complete = min(
        float(a["complete_board_capture"]), float(b["complete_board_capture"])
    )
    min_conditional = min(
        float(a["third_given_first_second"]),
        float(b["third_given_first_second"]),
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


def calibrate_board(rows: list[dict[str, object]]):
    rows_a = [
        row
        for row in rows
        if FUTURE_POLICY_SPLIT_A[0] <= row["race_date"] <= FUTURE_POLICY_SPLIT_A[1]
    ]
    rows_b = [
        row
        for row in rows
        if FUTURE_POLICY_SPLIT_B[0] <= row["race_date"] <= FUTURE_POLICY_SPLIT_B[1]
    ]
    grid = []
    for alpha in ALPHAS:
        for cum2 in CUM2_THRESHOLDS:
            for cum3 in CUM3_THRESHOLDS:
                policy = {"alpha": alpha, "cum2": cum2, "cum3": cum3}
                metrics_a = v32.board_metrics(apply_board(rows_a, policy))
                metrics_b = v32.board_metrics(apply_board(rows_b, policy))
                full = v32.board_metrics(apply_board(rows, policy))
                if float(full["avg_third_candidates"]) > 2.85:
                    continue
                grid.append(
                    {
                        **policy,
                        "objective": board_objective(metrics_a, metrics_b, full),
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


def script_hash() -> str:
    return hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_races = v32.load_races()
    races_by_id = {str(race["race_id"]): race for race in all_races}
    print(f"[v33] loaded {len(all_races)} unique complete races", flush=True)

    feature_names = None
    stages = []
    for stage in STAGES:
        result, feature_names = run_stage(all_races, stage, feature_names)
        stages.append(result)

    reopened, feature_names = run_stage(
        all_races, REOPENED_DIAGNOSTIC, feature_names, exact_pair=True
    )
    reopened["validity_note"] = (
        "Development diagnostic only. v33 architecture was proposed after v32 opened this "
        "period, so these metrics are not an adoption holdout."
    )

    # Lock the policy for the still-unseen Jul-Aug block.  The model used to
    # calibrate the policy ends before Apr-Jun; the future evaluator will refit
    # the same fixed architecture through Jun 28 without changing the policy.
    future_train = v32.period(all_races, *FUTURE_MODEL_TRAIN)
    top3_model, feature_names = fit_top3_model(future_train, feature_names)
    exact_pair_model, pair_train_races = exact_v31_pair_model()
    contexts, v31_audit = v32.build_v31_contexts()
    contexts = add_pair_probabilities_to_contexts(contexts, races_by_id, exact_pair_model)
    future_calibration_races = v32.period(all_races, *FUTURE_POLICY_CALIBRATION)
    future_calibration_rows = board_prediction_rows(
        future_calibration_races, contexts, top3_model, feature_names
    )
    future_grid = calibrate_board(future_calibration_rows)
    selected_future_policy = {
        key: future_grid[0][key] for key in ("alpha", "cum2", "cum3")
    }
    print(
        f"[v33] future policy={selected_future_policy} "
        f"cal_third={future_grid[0]['cal_full']['third_capture']:.4f} "
        f"cal_complete={future_grid[0]['cal_full']['complete_board_capture']:.4f}",
        flush=True,
    )

    stage_capture = [float(row["test_metrics"]["third_capture"]) for row in stages]
    stage_counts = [float(row["test_metrics"]["avg_third_candidates"]) for row in stages]
    reproducible = (
        min(stage_capture) >= V23_REFERENCE["third_capture"]
        and max(stage_counts) <= V23_REFERENCE["avg_third_candidates"]
    )

    summary = {
        "algorithm": "keirin_shogi_v33_coherent_top3_boundary",
        "status": (
            "FROZEN_BEFORE_TRUE_FUTURE_TEST"
            if reproducible
            else "development_failed_not_frozen"
        ),
        "concept": (
            "For each unordered Top3 set T, normalize the frozen-v31 probabilities of "
            "T's three internal Top2 pairs. The member excluded from an internal pair "
            "receives that pair's conditional third-boundary mass. Marginalize over T."
        ),
        "formula": (
            "P3(i)=sum_{T contains i} PTop3(T) * "
            "PTop2(T\\{i})^alpha / sum_{j in T} PTop2(T\\{j})^alpha"
        ),
        "top3_variant_fixed_from_v32_pre_holdout_development": TOP3_VARIANT,
        "pair_core": {"variant": "v31 blind_d2a", "params": PAIR_PARAMS},
        "historical_walk_forward": stages,
        "historical_reproducibility_rule": {
            "minimum_stage_third_capture": V23_REFERENCE["third_capture"],
            "maximum_stage_avg_candidates": V23_REFERENCE["avg_third_candidates"],
        },
        "historical_reproducible": reproducible,
        "reopened_v32_holdout_diagnostic": reopened,
        "future_lock": {
            "top3_policy_training_period": FUTURE_MODEL_TRAIN,
            "top3_policy_training_races": len(future_train),
            "v31_pair_training_races": pair_train_races,
            "calibration_period": FUTURE_POLICY_CALIBRATION,
            "calibration_split": [FUTURE_POLICY_SPLIT_A, FUTURE_POLICY_SPLIT_B],
            "calibration_participant_races": len(future_calibration_rows),
            "selected_policy": selected_future_policy,
            "selected_calibration_metrics": future_grid[0]["cal_full"],
            "future_refit_period": FUTURE_REFIT,
            "true_future_period": TRUE_FUTURE,
            "v31_audit": v31_audit,
        },
        "guards": {
            "odds_or_popularity_used": False,
            "random_shuffle": False,
            "v21_changed": False,
            "v31_changed": False,
            "v32_2026_h1_reused_as_adoption_holdout": False,
            "true_future_data_loaded": False,
            "fourth_candidate_penalized": True,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "calibration.json").write_text(
        json.dumps(future_grid, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# v33 Coherent Top3 Boundary

v32の単純な `Top3Membership - Top2Membership` は、別々に正規化した分布の差であり、
3着確率として整合しなかった。v33は各Top3集合の内部だけでTop2ペアを再正規化し、
そのペアから外れる1人へ3着境界確率を与える。

## 開発結果

| future block | 3着捕捉 | 平均候補 |
|---|---:|---:|
"""
    for row in stages:
        readme += (
            f"| {row['stage']['test'][0]}〜{row['stage']['test'][1]} "
            f"| {float(row['test_metrics']['third_capture']):.2%} "
            f"| {float(row['test_metrics']['avg_third_candidates']):.3f} |\n"
        )
    readme += f"""

判定: **{'未来試験へ凍結' if reproducible else '開発不合格'}**

2026年前半はv32で既に開封済みのため、v33の採用判定には使わない。
次の採用判定は固定後の2026-07-06〜2026-08-30だけで行う。

```bash
python scripts/keirin_shogi_v33_coherent_top3_boundary.py
```
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    freeze_path = OUT / "FREEZE.json"
    if reproducible:
        freeze = {
            "model": "keirin_shogi_v33_coherent_top3_boundary",
            "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST",
            "source_script_sha256": script_hash(),
            "top3_variant": TOP3_VARIANT,
            "v31_pair_params": PAIR_PARAMS,
            "selected_policy": selected_future_policy,
            "top3_future_refit_period": FUTURE_REFIT,
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
                "historical_reproducible": reproducible,
                "stage_test_metrics": [row["test_metrics"] for row in stages],
                "reopened_diagnostic": reopened["test_metrics"],
                "future_policy": selected_future_policy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

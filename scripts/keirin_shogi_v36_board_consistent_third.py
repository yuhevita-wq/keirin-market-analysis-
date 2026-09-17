#!/usr/bin/env python3
from __future__ import annotations

"""v36: softly align v35 Top2 branches with the frozen board context.

All 21 v31 pair branches remain alive.  v21 first probabilities and v31
membership are used only to reweight (never remove) branches before the v35
conditional third probabilities are marginalized.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v35 = load_module("v35_for_v36", "scripts/keirin_shogi_v35_top2_conditioned_third.py")
v32 = v35.v32

OUT = Path("results/keirin_shogi/v36_board_consistent_third")
SCRIPT = Path(__file__)

SELECTED_VARIANT = {
    "name": "cond_d2a",
    "depth": 2,
    "leaf": 24,
    "l2": 1.5,
    "lr": 0.045,
    "iters": 190,
}

SCORE_SPECS = [
    {"mode": "global", "gamma": 0.0},
    *[
        {"mode": mode, "gamma": gamma}
        for mode in ("v21_pair", "soft_cross", "candidate_cross")
        for gamma in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
    ],
]

FAIR_V23_ON_V31 = {
    "races": 511,
    "third_capture": 0.5303326810176126,
    "avg_third_candidates": 2.847358121330724,
    "complete_board_capture": 0.3131115459882583,
    "third_given_first_second": 0.730593607305936,
    "note": "v23 third candidates joined by race_id to frozen v31 first/second candidates.",
}


def standardize(values):
    vector = np.asarray(values, dtype=float)
    standard_deviation = float(vector.std()) or 1.0
    return (vector - float(vector.mean())) / standard_deviation


def compatibility(group, mode: str):
    context = group["context"]
    p1 = {int(no): float(value) for no, value in context["first_probabilities"].items()}
    membership = {
        int(row["no"]): float(row["mass"]) for row in context["top2_ranking"]
    }
    first_candidates = set(map(int, context["first_candidates"]))
    second_candidates = set(map(int, context["second_candidates"]))
    values = []
    for a, b, _ in group["pairs"]:
        if mode == "v21_pair":
            value = p1.get(a, 0.0) + p1.get(b, 0.0)
        elif mode == "soft_cross":
            value = (
                p1.get(a, 0.0) * membership.get(b, 0.0)
                + p1.get(b, 0.0) * membership.get(a, 0.0)
            )
        elif mode == "candidate_cross":
            value = float(
                (a in first_candidates and b in second_candidates)
                or (b in first_candidates and a in second_candidates)
            )
            value += 0.25 * float(a in second_candidates)
            value += 0.25 * float(b in second_candidates)
        else:
            value = 0.0
        values.append(value)
    return standardize(values)


def raw_conditionals(prepared, model):
    raw = model.predict_proba(prepared["x"])[:, 1]
    conditionals = []
    for group in prepared["groups"]:
        branch_probabilities = []
        for branch in group["branches"]:
            branch_probabilities.append(
                v35.normalized(raw[branch["start"] : branch["end"]])
            )
        conditionals.append(branch_probabilities)
    return conditionals


def aggregate(prepared, conditionals, spec):
    rows = []
    for group, branch_conditionals in zip(prepared["groups"], conditionals):
        base_weights = np.asarray(
            [float(branch["probability"]) for branch in group["branches"]], dtype=float
        )
        if spec["mode"] == "global":
            weights = base_weights
        else:
            adjustment = compatibility(group, str(spec["mode"]))
            weights = base_weights * np.exp(float(spec["gamma"]) * adjustment)
        weights = weights / weights.sum()

        scores = {no: 0.0 for no in sorted(group["race"]["base"])}
        for weight, branch, probabilities in zip(
            weights, group["branches"], branch_conditionals
        ):
            for no, probability in zip(branch["candidates"], probabilities):
                scores[no] += float(weight) * float(probability)
        total = sum(scores.values()) or 1.0
        scores = {no: value / total for no, value in scores.items()}
        ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        race = group["race"]
        context = group["context"]
        rows.append(
            {
                "race_id": race["race_id"],
                "race_date": race["race_date"],
                "race_type": race["race_type"],
                "order": race["order"],
                "probabilities": scores,
                "ranking": ranking,
                "pairs": group["pairs"],
                "first_candidates": context["first_candidates"],
                "second_candidates": context["second_candidates"],
                "top2_ranking": context["top2_ranking"],
                "score_spec": spec,
            }
        )
    return rows


def objective(a, b, full, gamma):
    min_third = min(float(a["third_capture"]), float(b["third_capture"]))
    min_complete = min(
        float(a["complete_board_capture"]), float(b["complete_board_capture"])
    )
    min_conditional = min(
        float(a["third_given_first_second"]), float(b["third_given_first_second"])
    )
    drift = abs(float(a["complete_board_capture"]) - float(b["complete_board_capture"]))
    return (
        0.34 * min_third
        + 0.34 * min_complete
        + 0.22 * min_conditional
        + 0.10 * float(full["third_capture"])
        - 0.005 * max(0.0, float(full["avg_third_candidates"]) - 2.0)
        - 0.24 * float(full["four_candidate_rate"])
        - 0.08 * drift
        - 0.002 * float(gamma)
    )


def calibrate(prepared, conditionals):
    grid = []
    for spec in SCORE_SPECS:
        rows = aggregate(prepared, conditionals, spec)
        rows_a = [
            row for row in rows
            if v35.FINAL_CAL_A[0] <= row["race_date"] <= v35.FINAL_CAL_A[1]
        ]
        rows_b = [
            row for row in rows
            if v35.FINAL_CAL_B[0] <= row["race_date"] <= v35.FINAL_CAL_B[1]
        ]
        for policy in v35.policy_specs():
            a = v35.metrics(v35.apply(rows_a, policy, True), True)
            b = v35.metrics(v35.apply(rows_b, policy, True), True)
            full = v35.metrics(v35.apply(rows, policy, True), True)
            if float(full["avg_third_candidates"]) > 2.85:
                continue
            grid.append(
                {
                    "score": spec,
                    "policy": policy,
                    "objective": objective(a, b, full, spec["gamma"]),
                    "cal_a": a,
                    "cal_b": b,
                    "cal_full": full,
                }
            )
    grid.sort(
        key=lambda row: (
            float(row["objective"]),
            min(
                float(row["cal_a"]["complete_board_capture"]),
                float(row["cal_b"]["complete_board_capture"]),
            ),
            min(
                float(row["cal_a"]["third_capture"]),
                float(row["cal_b"]["third_capture"]),
            ),
            -float(row["cal_full"]["avg_third_candidates"]),
        ),
        reverse=True,
    )
    return grid


def monthly(logs):
    output = []
    for month in sorted({row["race_date"][:7] for row in logs}):
        month_rows = [row for row in logs if row["race_date"].startswith(month)]
        output.append({"month": month, **v35.metrics(month_rows, True)})
    return output


def script_hash():
    return hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    races = v32.load_races()
    pair_model, contexts, v31_audit = v35.exact_v31_contexts()

    train = v32.period(races, *v35.FINAL_TRAIN)
    x, y, feature_names = v35.build_training_matrix(train, None)
    model = v35.fit_third_model(x, y, SELECTED_VARIANT)

    calibration_races = [
        race for race in v32.period(races, *v35.FINAL_CALIBRATION)
        if str(race["race_id"]) in contexts
    ]
    diagnostic_races = [
        race for race in v32.period(races, *v35.REOPENED_DIAGNOSTIC)
        if str(race["race_id"]) in contexts
    ]
    calibration_prepared = v35.prepare_prediction(
        calibration_races, pair_model, feature_names, contexts
    )
    calibration_conditionals = raw_conditionals(calibration_prepared, model)
    grid = calibrate(calibration_prepared, calibration_conditionals)
    selected = grid[0]

    diagnostic_prepared = v35.prepare_prediction(
        diagnostic_races, pair_model, feature_names, contexts
    )
    diagnostic_conditionals = raw_conditionals(diagnostic_prepared, model)
    diagnostic_rows = aggregate(diagnostic_prepared, diagnostic_conditionals, selected["score"])
    diagnostic_logs = v35.apply(diagnostic_rows, selected["policy"], True)
    diagnostic_metrics = v35.metrics(diagnostic_logs, True)

    # Post-hoc diagnosis only.  Each score uses the policy selected on the
    # calibration block; these rows are never eligible to alter v36.
    sensitivity = []
    for spec in SCORE_SPECS:
        candidates = [row for row in grid if row["score"] == spec]
        calibration_choice = candidates[0]
        rows = aggregate(diagnostic_prepared, diagnostic_conditionals, spec)
        result = v35.metrics(
            v35.apply(rows, calibration_choice["policy"], True), True
        )
        sensitivity.append(
            {
                "score": spec,
                "calibration_selected_policy": calibration_choice["policy"],
                "calibration_objective": calibration_choice["objective"],
                "calibration_metrics": calibration_choice["cal_full"],
                "diagnostic_metrics": result,
            }
        )

    v35_summary_path = Path("results/keirin_shogi/v35_top2_conditioned_third/summary.json")
    v35_summary = (
        json.loads(v35_summary_path.read_text(encoding="utf-8"))
        if v35_summary_path.exists() else None
    )
    historical_stages = (
        v35_summary.get("historical_policy_walk_forward", []) if v35_summary else []
    )
    if historical_stages:
        stage_captures = [
            float(row["test_metrics"]["third_capture"]) for row in historical_stages
        ]
        stage_counts = [
            float(row["test_metrics"]["avg_third_candidates"]) for row in historical_stages
        ]
        historical_pass = (
            mean(stage_captures) > v32.V23_REFERENCE["third_capture"]
            and min(stage_captures) >= 0.50
            and max(stage_counts) <= v32.V23_REFERENCE["avg_third_candidates"]
        )
    else:
        historical_pass = False

    diagnostic_pass = (
        float(diagnostic_metrics["third_capture"]) > FAIR_V23_ON_V31["third_capture"]
        and float(diagnostic_metrics["avg_third_candidates"]) <= FAIR_V23_ON_V31["avg_third_candidates"]
        and float(diagnostic_metrics["complete_board_capture"])
        > FAIR_V23_ON_V31["complete_board_capture"]
    )
    ready = bool(historical_pass and diagnostic_pass)

    print(
        f"[v36] selected score={selected['score']} policy={selected['policy']} "
        f"cal_complete={selected['cal_full']['complete_board_capture']:.4f}",
        flush=True,
    )
    print(
        f"[v36] diagnostic third={diagnostic_metrics['third_capture']:.4f} "
        f"complete={diagnostic_metrics['complete_board_capture']:.4f} "
        f"conditional={diagnostic_metrics['third_given_first_second']:.4f} "
        f"avg={diagnostic_metrics['avg_third_candidates']:.3f}",
        flush=True,
    )

    summary = {
        "algorithm": "keirin_shogi_v36_board_consistent_third",
        "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST" if ready else "development_failed_not_frozen",
        "concept": (
            "Keep all 21 v31 pair branches, softly reweight them with v21/v31 board "
            "compatibility, then marginalize v35 conditional-third probabilities."
        ),
        "selected_variant": SELECTED_VARIANT,
        "feature_count": len(feature_names),
        "selected_score": selected["score"],
        "selected_policy": selected["policy"],
        "calibration": {
            "period": v35.FINAL_CALIBRATION,
            "split": [v35.FINAL_CAL_A, v35.FINAL_CAL_B],
            "participant_races": len(calibration_races),
            "metrics": selected["cal_full"],
        },
        "historical_model_walk_forward": historical_stages,
        "reopened_2026_h1_diagnostic": {
            "period": v35.REOPENED_DIAGNOSTIC,
            "note": "Diagnostic only; not counted as an untouched future block.",
            "metrics": diagnostic_metrics,
            "monthly": monthly(diagnostic_logs),
            "posthoc_score_sensitivity": sensitivity,
        },
        "fair_baseline_v21_v31_v23third": FAIR_V23_ON_V31,
        "acceptance": {
            "historical_pass": historical_pass,
            "diagnostic_pass": diagnostic_pass,
            "ready_for_true_future": ready,
        },
        "v31_audit": v31_audit,
        "future_lock": {
            "third_training_period": v35.FINAL_TRAIN,
            "true_future_period": v35.TRUE_FUTURE,
        },
        "guards": {
            "odds_or_popularity_used": False,
            "random_shuffle": False,
            "v21_changed": False,
            "v31_changed": False,
            "pair_branch_removed": False,
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
        json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "block_evaluation.json").write_text(
        json.dumps(
            {
                "historical_model_walk_forward": historical_stages,
                "calibration_halves": {
                    "a": selected["cal_a"],
                    "b": selected["cal_b"],
                    "full": selected["cal_full"],
                },
                "diagnostic_monthly": monthly(diagnostic_logs),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "diagnostic_sensitivity.json").write_text(
        json.dumps(sensitivity, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = "# v36 Board-consistent third\n\n"
    readme += "v35の全21組を残し、v21の1着確率とv31 membershipで組確率をソフト再重み付けする。\n"
    readme += "候補除外は行わず、重み係数と候補数は2025-10-27〜12-28のみで校正する。\n\n"
    readme += "| evaluation | third | complete board | third given first+second | avg third |\n"
    readme += "|---|---:|---:|---:|---:|\n"
    readme += (
        f"| calibration | {float(selected['cal_full']['third_capture']):.2%} "
        f"| {float(selected['cal_full']['complete_board_capture']):.2%} "
        f"| {float(selected['cal_full']['third_given_first_second']):.2%} "
        f"| {float(selected['cal_full']['avg_third_candidates']):.3f} |\n"
    )
    readme += (
        f"| reopened 2026 H1 | {float(diagnostic_metrics['third_capture']):.2%} "
        f"| {float(diagnostic_metrics['complete_board_capture']):.2%} "
        f"| {float(diagnostic_metrics['third_given_first_second']):.2%} "
        f"| {float(diagnostic_metrics['avg_third_candidates']):.3f} |\n\n"
    )
    readme += f"判定: **{'完全未来試験へ凍結' if ready else '開発不合格'}**\n\n"
    readme += "```bash\npython scripts/keirin_shogi_v36_board_consistent_third.py\n```\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    freeze_path = OUT / "FREEZE.json"
    if ready:
        freeze_path.write_text(
            json.dumps(
                {
                    "model": "keirin_shogi_v36_board_consistent_third",
                    "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST",
                    "source_script_sha256": script_hash(),
                    "selected_variant": SELECTED_VARIANT,
                    "selected_score": selected["score"],
                    "selected_policy": selected["policy"],
                    "third_training_period": v35.FINAL_TRAIN,
                    "calibration_period": v35.FINAL_CALIBRATION,
                    "v31_pair_variant": "blind_d2a",
                    "v31_pair_params": v35.PAIR_PARAMS,
                    "true_future_period": v35.TRUE_FUTURE,
                    "future_tuning_allowed": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    elif freeze_path.exists():
        freeze_path.unlink()

    print(json.dumps({"ready": ready, "selected": selected, "diagnostic": diagnostic_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

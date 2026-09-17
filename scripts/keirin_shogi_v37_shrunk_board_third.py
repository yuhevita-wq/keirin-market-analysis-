#!/usr/bin/env python3
from __future__ import annotations

"""v37: conservative board-aware shrinkage after the v36 overfit diagnosis.

The score architecture is fixed to candidate_cross with gamma=0.25.  This is
small enough that every v31 pair branch remains close to its original weight.
Only the 2/3/4-rider candidate policy is selected on the pre-2026 calibration
block.  2026 H1 is explicitly developmental; the next test is 2026 Jul-Aug.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v36 = load_module("v36_for_v37", "scripts/keirin_shogi_v36_board_consistent_third.py")
v35 = v36.v35
v32 = v35.v32

OUT = Path("results/keirin_shogi/v37_shrunk_board_third")
SCRIPT = Path(__file__)
FIXED_SCORE = {"mode": "candidate_cross", "gamma": 0.25}


def file_hash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def monthly(logs):
    output = []
    for month in sorted({row["race_date"][:7] for row in logs}):
        rows = [row for row in logs if row["race_date"].startswith(month)]
        output.append({"month": month, **v35.metrics(rows, True)})
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    races = v32.load_races()
    pair_model, contexts, v31_audit = v35.exact_v31_contexts()

    train = v32.period(races, *v35.FINAL_TRAIN)
    x, y, feature_names = v35.build_training_matrix(train, None)
    model = v35.fit_third_model(x, y, v36.SELECTED_VARIANT)

    calibration_races = [
        race for race in v32.period(races, *v35.FINAL_CALIBRATION)
        if str(race["race_id"]) in contexts
    ]
    calibration_prepared = v35.prepare_prediction(
        calibration_races, pair_model, feature_names, contexts
    )
    calibration_conditionals = v36.raw_conditionals(calibration_prepared, model)
    full_grid = v36.calibrate(calibration_prepared, calibration_conditionals)
    grid = [row for row in full_grid if row["score"] == FIXED_SCORE]
    selected = grid[0]

    diagnostic_races = [
        race for race in v32.period(races, *v35.REOPENED_DIAGNOSTIC)
        if str(race["race_id"]) in contexts
    ]
    diagnostic_prepared = v35.prepare_prediction(
        diagnostic_races, pair_model, feature_names, contexts
    )
    diagnostic_conditionals = v36.raw_conditionals(diagnostic_prepared, model)
    diagnostic_rows = v36.aggregate(
        diagnostic_prepared, diagnostic_conditionals, FIXED_SCORE
    )
    diagnostic_logs = v35.apply(diagnostic_rows, selected["policy"], True)
    diagnostic_metrics = v35.metrics(diagnostic_logs, True)

    v35_summary_path = Path("results/keirin_shogi/v35_top2_conditioned_third/summary.json")
    if not v35_summary_path.exists():
        raise RuntimeError("Run v35 first so its chronological walk-forward is immutable.")
    v35_summary = json.loads(v35_summary_path.read_text(encoding="utf-8"))
    historical_stages = v35_summary["historical_policy_walk_forward"]
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
    development_pass = (
        float(diagnostic_metrics["third_capture"])
        > v36.FAIR_V23_ON_V31["third_capture"]
        and float(diagnostic_metrics["avg_third_candidates"])
        <= v36.FAIR_V23_ON_V31["avg_third_candidates"]
        and float(diagnostic_metrics["complete_board_capture"])
        > v36.FAIR_V23_ON_V31["complete_board_capture"]
    )
    ready = bool(historical_pass and development_pass)

    summary = {
        "algorithm": "keirin_shogi_v37_shrunk_board_third",
        "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST" if ready else "development_failed_not_frozen",
        "concept": (
            "Use the v35 conditional-third marginal, then apply only weak gamma=0.25 "
            "soft support to v31 pairs compatible with the frozen first/second boards."
        ),
        "version_reason": (
            "v36 selected gamma=2.0 on calibration and drifted out of candidate budget. "
            "v37 fixes conservative shrinkage as a new version; v36 is not retuned."
        ),
        "selected_variant": v36.SELECTED_VARIANT,
        "fixed_score": FIXED_SCORE,
        "selected_policy": selected["policy"],
        "calibration": {
            "period": v35.FINAL_CALIBRATION,
            "split": [v35.FINAL_CAL_A, v35.FINAL_CAL_B],
            "participant_races": len(calibration_races),
            "metrics": selected["cal_full"],
        },
        "historical_model_walk_forward": historical_stages,
        "reopened_2026_h1_development": {
            "period": v35.REOPENED_DIAGNOSTIC,
            "not_future_evidence": True,
            "metrics": diagnostic_metrics,
            "monthly": monthly(diagnostic_logs),
        },
        "fair_baseline_v21_v31_v23third": v36.FAIR_V23_ON_V31,
        "acceptance": {
            "historical_pass": historical_pass,
            "development_pass": development_pass,
            "ready_for_true_future": ready,
        },
        "v31_audit": v31_audit,
        "future_lock": {
            "third_training_period": v35.FINAL_TRAIN,
            "true_future_period": v35.TRUE_FUTURE,
            "future_tuning_allowed": False,
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
                "development_monthly": monthly(diagnostic_logs),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = "# v37 Shrunk board third\n\n"
    readme += "v36の強い再重み付け過適合を受け、全21組を残したままgamma=0.25へ固定。\n"
    readme += "候補ポリシーだけを2025-10-27〜12-28で選び、2026 H1は開発診断として扱う。\n\n"
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
    readme += "```bash\npython scripts/keirin_shogi_v37_shrunk_board_third.py\n```\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    freeze_path = OUT / "FREEZE.json"
    if ready:
        freeze_path.write_text(
            json.dumps(
                {
                    "model": "keirin_shogi_v37_shrunk_board_third",
                    "status": "FROZEN_BEFORE_TRUE_FUTURE_TEST",
                    "source_hashes": {
                        "v37": file_hash(SCRIPT),
                        "v36_dependency": file_hash(Path("scripts/keirin_shogi_v36_board_consistent_third.py")),
                        "v35_dependency": file_hash(Path("scripts/keirin_shogi_v35_top2_conditioned_third.py")),
                    },
                    "selected_variant": v36.SELECTED_VARIANT,
                    "fixed_score": FIXED_SCORE,
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

    print(
        json.dumps(
            {
                "ready_for_true_future": ready,
                "fixed_score": FIXED_SCORE,
                "selected_policy": selected["policy"],
                "calibration_metrics": selected["cal_full"],
                "development_metrics": diagnostic_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

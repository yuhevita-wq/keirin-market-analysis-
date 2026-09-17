#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate the frozen v37 third row on the already-defined v31 future block.

This file does not calibrate or select anything.  It asserts the v37 freeze,
reconstructs the unchanged v31 pair posterior, and evaluates 2026-07-06 through
2026-08-30 participant races from the committed v31 future log.
"""

import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v37 = load_module("v37_frozen_future", "scripts/keirin_shogi_v37_shrunk_board_third.py")
v36 = v37.v36
v35 = v37.v35
v32 = v37.v32
v31 = v32.v31
v26 = v31.v26
v25 = v31.v25

FUTURE = Path("data/2026_future_block1/s_class_f1_20260701_20260830")
V31_FUTURE = Path("results/keirin_shogi/v31_true_future_block1")
FREEZE_PATH = Path("results/keirin_shogi/v37_shrunk_board_third/FREEZE.json")
OUT = Path("results/keirin_shogi/v37_true_future_block1")

EVAL_START = "2026-07-06"
EVAL_END = "2026-08-30"
FREEZE_COMMIT = "dbe4aa2ae876224927a2b54c2cbcd3b6ff6c701e"


def file_hash(path: str):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(name: str):
    with (FUTURE / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_future_races(v31_logs, pair_model):
    races = {row["race_id"]: row for row in load_csv("races.csv")}
    entries_by = defaultdict(list)
    results_by = defaultdict(list)
    for row in load_csv("entries.csv"):
        entries_by[row["race_id"]].append(row)
    for row in load_csv("results.csv"):
        results_by[row["race_id"]].append(row)

    output = []
    contexts = {}
    membership_differences = []
    for log in v31_logs:
        race_id = str(log["race_id"])
        race = races.get(race_id)
        entries = entries_by.get(race_id, [])
        if race is None or len(entries) != 7:
            raise RuntimeError(f"Missing seven-rider future race {race_id}")
        order = v32.result_order(results_by.get(race_id, []))
        if order is None:
            raise RuntimeError(f"Missing complete future result {race_id}")
        if order[:2] != [int(log["actual_first"]), int(log["actual_second"])]:
            raise RuntimeError(f"v31/raw result mismatch {race_id}")
        base = v32.enrich_base(entries)
        pairs = v31.pair_distribution(base, {"p1_map": {}}, pair_model)
        membership = v31.membership_rank(pairs)
        saved_membership = {
            int(row["no"]): float(row["mass"]) for row in log["second_membership"]
        }
        membership_differences.extend(
            abs(float(value) - saved_membership.get(int(no), 0.0))
            for no, value in membership
        )
        contexts[race_id] = {
            "pairs": pairs,
            "first_candidates": list(map(int, log["first_candidates"])),
            # The frozen v37 score uses candidate_cross only; p1 values are not
            # consulted, but the common context schema remains explicit.
            "first_probabilities": {int(no): 0.0 for no in base},
            "second_candidates": list(map(int, log["second_candidates"])),
            "top2_ranking": [
                {"no": int(no), "mass": float(value)} for no, value in membership
            ],
        }
        output.append(
            {
                "race_id": race_id,
                "race_date": race["race_date"],
                "race_type": race.get("race_type", ""),
                "base": base,
                "order": order,
            }
        )
    output.sort(key=lambda row: (row["race_date"], row["race_id"]))
    return output, contexts, max(membership_differences, default=0.0)


def weekly(logs, week_by_race):
    grouped = defaultdict(list)
    for row in logs:
        grouped[week_by_race[str(row["race_id"])]].append(row)
    return [
        {"week": week, **v35.metrics(rows, True)}
        for week, rows in sorted(grouped.items())
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN_BEFORE_TRUE_FUTURE_TEST"
    assert freeze["true_future_period"] == [EVAL_START, EVAL_END]
    assert freeze["fixed_score"] == v37.FIXED_SCORE
    assert freeze["source_hashes"]["v37"] == file_hash(
        "scripts/keirin_shogi_v37_shrunk_board_third.py"
    )
    assert freeze["source_hashes"]["v36_dependency"] == file_hash(
        "scripts/keirin_shogi_v36_board_consistent_third.py"
    )
    assert freeze["source_hashes"]["v35_dependency"] == file_hash(
        "scripts/keirin_shogi_v35_top2_conditioned_third.py"
    )

    v31_logs = json.loads((V31_FUTURE / "race_log.json").read_text(encoding="utf-8"))
    v31_logs = [
        row for row in v31_logs if EVAL_START <= row["race_date"] <= EVAL_END
    ]
    week_by_race = {str(row["race_id"]): row["week"] for row in v31_logs}

    vrows = v25.load_v19_details()
    v25.fit_v21_selector(vrows)
    v25.mark_v21(vrows)
    _, entries_by, results_by = v25.raw_maps()
    pair_train = v25.make_dataset(
        vrows, entries_by, results_by, v25.TRAIN_START, v25.TRAIN_END, False
    )
    pair_model = v26.fit_pair_model(pair_train, v35.PAIR_PARAMS)

    historical = v32.load_races()
    third_train = v32.period(historical, *freeze["third_training_period"])
    x, y, feature_names = v35.build_training_matrix(third_train, None)
    third_model = v35.fit_third_model(x, y, freeze["selected_variant"])

    future_races, contexts, membership_max_abs_diff = load_future_races(
        v31_logs, pair_model
    )
    prepared = v35.prepare_prediction(
        future_races, pair_model, feature_names, contexts
    )
    conditionals = v36.raw_conditionals(prepared, third_model)
    rows = v36.aggregate(prepared, conditionals, freeze["fixed_score"])
    logs = v35.apply(rows, freeze["selected_policy"], True)
    result = v35.metrics(logs, True)
    weekly_result = weekly(logs, week_by_race)

    reference = v36.FAIR_V23_ON_V31
    benchmark_pass = (
        float(result["third_capture"]) > float(reference["third_capture"])
        and float(result["avg_third_candidates"])
        <= float(reference["avg_third_candidates"])
        and float(result["complete_board_capture"])
        > float(reference["complete_board_capture"])
    )
    status = "TRUE_FUTURE_PASSED" if benchmark_pass else "TRUE_FUTURE_FAILED_NO_RETUNING"

    summary = {
        "algorithm": "keirin_shogi_v37_true_future_block1",
        "status": status,
        "freeze_commit": FREEZE_COMMIT,
        "future_block": [EVAL_START, EVAL_END],
        "participant_races": len(logs),
        "frozen_model": {
            "variant": freeze["selected_variant"],
            "score": freeze["fixed_score"],
            "policy": freeze["selected_policy"],
            "third_training_period": freeze["third_training_period"],
        },
        "metrics": result,
        "weekly": weekly_result,
        "historical_fair_reference_v21_v31_v23third": reference,
        "benchmark_pass": benchmark_pass,
        "audit": {
            "v31_participant_log_races": len(v31_logs),
            "evaluated_races": len(logs),
            "v31_membership_reconstruction_max_abs_diff": membership_max_abs_diff,
        },
        "guards": {
            "parameters_changed_after_future_open": False,
            "candidate_policy_changed_after_future_open": False,
            "odds_or_popularity_used": False,
            "v21_changed": False,
            "v31_changed": False,
            "future_results_used_for_training": False,
            "note": "Future results are read only for labels after the v37 freeze commit was published.",
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "race_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "weekly.json").write_text(
        json.dumps(weekly_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = "# v37 true future block 1\n\n"
    readme += f"凍結コミット: `{FREEZE_COMMIT}`\n\n"
    readme += "2026-07-06〜08-30のv31参加レースへ、凍結済みv37を無変更で適用。\n\n"
    readme += "| races | third | complete board | third given first+second | avg third |\n"
    readme += "|---:|---:|---:|---:|---:|\n"
    readme += (
        f"| {len(logs)} | {float(result['third_capture']):.2%} "
        f"| {float(result['complete_board_capture']):.2%} "
        f"| {float(result['third_given_first_second']):.2%} "
        f"| {float(result['avg_third_candidates']):.3f} |\n\n"
    )
    readme += f"判定: **{status}**\n\n"
    readme += "```bash\npython scripts/keirin_shogi_v37_true_future_block1.py\n```\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

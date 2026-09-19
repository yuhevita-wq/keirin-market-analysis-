#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
WEEK_START = date(2026, 7, 6)
WEEK_END = WEEK_START + timedelta(days=6)


def load_unified():
    path = ROOT / "scripts/keirin_shogi_unified_v1.py"
    spec = importlib.util.spec_from_file_location("unified_v1_week_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


u = load_unified()


def as_of_state(current: date):
    races = u.load_csv("races.csv")
    entries = u.load_csv("entries.csv")
    results = u.load_csv("results.csv")

    entries_by = defaultdict(list)
    results_by = defaultdict(list)
    for row in entries:
        entries_by[row["race_id"]].append(row)
    for row in results:
        results_by[row["race_id"]].append(row)

    history_start = current - timedelta(days=91)
    core_a_end = current - timedelta(days=36)
    policy_start = current - timedelta(days=35)
    policy_end = current - timedelta(days=22)
    final_end = current - timedelta(days=1)

    core_a = u.v19.make_races(
        u.ids_between(races, history_start, core_a_end), entries_by, results_by
    )
    policy_cal = u.v19.make_races(
        u.ids_between(races, policy_start, policy_end), entries_by, results_by
    )
    final_train = u.v19.make_races(
        u.ids_between(races, history_start, final_end), entries_by, results_by
    )
    if min(len(core_a), len(policy_cal), len(final_train)) == 0:
        raise RuntimeError(
            f"missing as-of training data: {len(core_a)},{len(policy_cal)},{len(final_train)}"
        )

    intercept, coefficients = u.frozen_selector()
    return {
        "models": u.v19.v18.fit_ensemble(final_train),
        "candidate_policy": u.v19.candidate_policy_from_cal(core_a, policy_cal),
        "threshold": 0.0,
        "intercept": intercept,
        "coefficients": coefficients,
        "history_end": final_end.isoformat(),
        "state_week": current.isoformat(),
    }, races, entries_by, results_by


def valid_board_orders(first, second, third):
    return sum(
        1
        for a in first
        for b in second
        for c in third
        if len({a, b, c}) == 3
    )


def main():
    state, races, entries_by, results_by = as_of_state(WEEK_START)
    race_by = {row["race_id"]: row for row in races}
    test_ids = u.ids_between(races, WEEK_START, WEEK_END)

    pair_model = u.build_pair_model()
    third_model, feature_names, freeze = u.build_third_model()

    logs = []
    for race_id in test_ids:
        raw = race_by[race_id]
        order = u.v32.result_order(results_by.get(race_id, []))
        if order is None:
            continue

        race = {**raw, "entries": entries_by[race_id]}
        pred = u.unified_joint(
            race, state, pair_model, third_model, feature_names
        )

        actual = tuple(map(int, order[:3]))
        top_orders = pred["top_orders"]
        actual_rank = next(
            (
                i + 1
                for i, row in enumerate(top_orders)
                if (row["first"], row["second"], row["third"]) == actual
            ),
            None,
        )
        first_hit = actual[0] in pred["first_candidates"]
        second_hit = actual[1] in pred["second_candidates"]
        third_hit = actual[2] in pred["third_candidates"]

        logs.append(
            {
                "race_id": race_id,
                "race_date": raw.get("race_date", ""),
                "track": raw.get("track", ""),
                "race_no": raw.get("race_no", ""),
                "actual": list(actual),
                "first_candidates": pred["first_candidates"],
                "second_candidates": pred["second_candidates"],
                "third_candidates": pred["third_candidates"],
                "first_hit": first_hit,
                "second_hit": second_hit,
                "third_hit": third_hit,
                "complete_board_hit": first_hit and second_hit and third_hit,
                "actual_order_rank_top20": actual_rank,
                "valid_board_orders": valid_board_orders(
                    pred["first_candidates"],
                    pred["second_candidates"],
                    pred["third_candidates"],
                ),
                "top_order": top_orders[0],
            }
        )

    if not logs:
        raise RuntimeError("no complete-result races in evaluation week")

    n = len(logs)
    summary = {
        "algorithm": "keirin_shogi_unified_v1",
        "evaluation_week": [WEEK_START.isoformat(), WEEK_END.isoformat()],
        "history_end_for_first_model": state["history_end"],
        "pair_training_end": u.v25.TRAIN_END,
        "third_training_period": freeze["third_training_period"],
        "races": n,
        "first_capture": sum(x["first_hit"] for x in logs) / n,
        "second_capture": sum(x["second_hit"] for x in logs) / n,
        "third_capture": sum(x["third_hit"] for x in logs) / n,
        "complete_board_capture": sum(x["complete_board_hit"] for x in logs) / n,
        "exact_order_top1": sum(x["actual_order_rank_top20"] == 1 for x in logs) / n,
        "exact_order_top5": sum(
            x["actual_order_rank_top20"] is not None
            and x["actual_order_rank_top20"] <= 5
            for x in logs
        ) / n,
        "exact_order_top10": sum(
            x["actual_order_rank_top20"] is not None
            and x["actual_order_rank_top20"] <= 10
            for x in logs
        ) / n,
        "exact_order_top20": sum(
            x["actual_order_rank_top20"] is not None for x in logs
        ) / n,
        "avg_first_candidates": mean(len(x["first_candidates"]) for x in logs),
        "avg_second_candidates": mean(len(x["second_candidates"]) for x in logs),
        "avg_third_candidates": mean(len(x["third_candidates"]) for x in logs),
        "avg_valid_board_orders": mean(x["valid_board_orders"] for x in logs),
        "guards": {
            "odds_used": False,
            "popularity_used": False,
            "participation_filter_used": False,
            "week_results_used_for_training": False,
        },
    }

    print("UNIFIED_WEEK1_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
    for row in logs:
        print("UNIFIED_WEEK1_RACE=" + json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

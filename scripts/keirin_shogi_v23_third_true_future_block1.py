#!/usr/bin/env python3
from __future__ import annotations

"""Fair true-future baseline: v21 row1 + v31 row2 + legacy v23 row3.

The participant IDs and first/second candidates are taken from the committed
v31 future log.  v23's frozen 2025 training and t3=0.55 rule are applied to the
same 153 races without any future tuning.
"""

import csv
import importlib.util
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


future = load_module("v31_future_for_v23", "scripts/keirin_shogi_v31_true_future_block1.py")
v23 = load_module("v23_future_baseline", "scripts/keirin_shogi_v23_conditional_second_third.py")
v19 = future.v19

HIST = future.HIST
FUT = future.FUT
OUT = Path("results/keirin_shogi/v23_third_true_future_block1")
V31_LOG = Path("results/keirin_shogi/v31_true_future_block1/race_log.json")

WARMUP_START = date(2026, 6, 29)
EVAL_START = date(2026, 7, 6)
EVAL_END = date(2026, 8, 30)
FROZEN_T3 = 0.55


def ino(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_csv(name):
    output = []
    for base in (HIST, FUT):
        path = base / name
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                output.extend(csv.DictReader(handle))
    return output


def ids_between(races, start, end):
    start_string = start.isoformat()
    end_string = end.isoformat()
    return [
        row["race_id"]
        for row in races
        if start_string <= row.get("race_date", "") <= end_string
        and ino(row.get("entry_count")) == 7
        and row.get("meeting_grade") == "F1"
        and "Ｓ級" in row.get("race_type", "")
    ]


def metrics(logs):
    count = len(logs)
    first_second = [row for row in logs if row["first_second_hit"]]
    return {
        "races": count,
        "third_capture": sum(row["third_hit"] for row in logs) / count,
        "complete_board_capture": sum(row["complete_board_hit"] for row in logs) / count,
        "third_given_first_second": (
            sum(row["third_hit"] for row in first_second) / len(first_second)
            if first_second else 0.0
        ),
        "avg_third_candidates": mean(len(row["third_candidates"]) for row in logs),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v31_rows = json.loads(V31_LOG.read_text(encoding="utf-8"))
    v31_by_id = {str(row["race_id"]): row for row in v31_rows}

    raw_races = load_csv("races.csv")
    entries_by = defaultdict(list)
    results_by = defaultdict(list)
    for row in load_csv("entries.csv"):
        entries_by[row["race_id"]].append(row)
    for row in load_csv("results.csv"):
        results_by[row["race_id"]].append(row)

    # Rebuild the frozen v23 second/third conditional models exactly.
    historical_race_by, historical_entries, historical_results = v23.raw_maps()
    train_ids = v23.target_ids(
        historical_race_by, v23.RANK_TRAIN_START, v23.RANK_TRAIN_END
    )
    training = v23.make_training_races(
        train_ids, historical_entries, historical_results
    )
    second_model = v23.fit_conditional_model(training, 2)
    third_model = v23.fit_conditional_model(training, 3)

    logs = []
    current = WARMUP_START
    while current <= EVAL_END:
        week_end = current + timedelta(days=6)
        history_start = current - timedelta(days=91)
        core_a_end = current - timedelta(days=36)
        policy_start = current - timedelta(days=35)
        policy_end = current - timedelta(days=22)
        final_end = current - timedelta(days=1)

        core_a = v19.make_races(
            ids_between(raw_races, history_start, core_a_end), entries_by, results_by
        )
        policy_calibration = v19.make_races(
            ids_between(raw_races, policy_start, policy_end), entries_by, results_by
        )
        final_train = v19.make_races(
            ids_between(raw_races, history_start, final_end), entries_by, results_by
        )
        test = v19.make_races(
            ids_between(raw_races, current, week_end), entries_by, results_by
        )
        if min(len(core_a), len(policy_calibration), len(final_train), len(test)) == 0:
            raise RuntimeError(f"Missing v21 walk-forward data for {current}")

        candidate_policy = v19.candidate_policy_from_cal(core_a, policy_calibration)
        models = v19.v18.fit_ensemble(final_train)
        for race_id, race_rows, _winner in test:
            source = v31_by_id.get(str(race_id))
            if current < EVAL_START or source is None:
                continue
            prediction, model_top1s = v19.v18.predict(race_rows, models)
            # Generate the model prediction for its probabilities, but retain
            # the already-committed v31 future first candidates exactly.
            _generated_candidates = v19.v18.choose(
                prediction, model_top1s, candidate_policy
            )
            first_candidates = list(map(int, source["first_candidates"]))
            order = future.result_order(results_by.get(race_id, []))
            if order is None:
                raise RuntimeError(f"Missing future result {race_id}")
            base = v23.race_base_features(entries_by[race_id])
            vrow = {"candidates": first_candidates, "ranking": prediction}
            _second_ranking, third_ranking = v23.predict_second_third(
                base, vrow, second_model, third_model
            )
            third_candidates = v23.choose_by_cum(third_ranking, FROZEN_T3, 3)
            first, second, third = map(int, order[:3])
            second_candidates = list(map(int, source["second_candidates"]))
            first_second_hit = int(
                first in first_candidates and second in second_candidates
            )
            third_hit = int(third in third_candidates)
            logs.append(
                {
                    "race_id": str(race_id),
                    "race_date": entries_by[race_id][0].get("race_date", ""),
                    "week": current.isoformat(),
                    "actual_first": first,
                    "actual_second": second,
                    "actual_third": third,
                    "first_candidates": first_candidates,
                    "second_candidates": second_candidates,
                    "third_candidates": third_candidates,
                    "first_second_hit": first_second_hit,
                    "third_hit": third_hit,
                    "complete_board_hit": int(first_second_hit and third_hit),
                    "third_ranking": [
                        {"no": int(no), "probability": float(probability)}
                        for no, probability in third_ranking
                    ],
                }
            )
        current += timedelta(days=7)

    logs.sort(key=lambda row: (row["race_date"], row["race_id"]))
    if set(row["race_id"] for row in logs) != set(v31_by_id):
        missing = sorted(set(v31_by_id) - set(row["race_id"] for row in logs))
        extra = sorted(set(row["race_id"] for row in logs) - set(v31_by_id))
        raise RuntimeError(f"Future participant mismatch missing={missing} extra={extra}")

    result = metrics(logs)
    weekly = []
    for week in sorted({row["week"] for row in logs}):
        rows = [row for row in logs if row["week"] == week]
        weekly.append({"week": week, **metrics(rows)})
    summary = {
        "algorithm": "keirin_shogi_v23_third_true_future_block1",
        "purpose": "Fair legacy third-row baseline on the exact v31 future participant board.",
        "future_block": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "participant_source": str(V31_LOG),
        "first_row": "frozen v21 candidates from v31 future log",
        "second_row": "frozen v31 candidates from v31 future log",
        "third_row": "v23 frozen conditional model and t3=0.55",
        "v23_training_period": [v23.RANK_TRAIN_START, v23.RANK_TRAIN_END],
        "metrics": result,
        "weekly": weekly,
        "guards": {
            "future_parameter_tuning": False,
            "odds_or_popularity_used": False,
            "future_results_used_for_training": False,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "race_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "weekly.json").write_text(
        json.dumps(weekly, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# v23 third true-future baseline\n\n"
        "v31完全未来153レースの同一1着・2着盤面に、旧v23の3着段だけを適用した公平比較。\n\n"
        f"- 3着捕捉: {float(result['third_capture']):.2%}\n"
        f"- 完全盤面: {float(result['complete_board_capture']):.2%}\n"
        f"- 平均3着候補: {float(result['avg_third_candidates']):.3f}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

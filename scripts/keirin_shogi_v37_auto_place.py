#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic live board builder for Keirin Shogi v21/v31/v37.

This script does not ask an LLM to predict a race. It mechanically reconstructs
and applies the adopted/frozen algorithms:

- row 1: v21 participation + first candidates, using the latest historical
  walk-forward state available in the repository (through 2026-08-30)
- row 2: frozen v31 blind unordered Top2 membership model
- row 3: frozen v37 Top2-conditioned third model with candidate_cross gamma=.25

Live inputs come only from docs/keirin-shogi/live-race-data.json. Odds,
popularity, results and payouts from the target live race are never read.
"""

import csv
import importlib.util
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "docs/keirin-shogi/live-race-data.json"
OUT = ROOT / "docs/keirin-shogi/live-board-data.json"
HIST = ROOT / "data/2026_h1/s_class_f1_all"
FUT = ROOT / "data/2026_future_block1/s_class_f1_20260701_20260830"

V21_TARGET_FRACTION = 0.25
V21_MAX_FIRST_CANDIDATES = 2
V21_LOOKBACK_WEEKS = 4
V31_REL3 = 0.65
V31_MIN3 = 0.30
LATEST_HISTORY_END = date(2026, 8, 30)
PRODUCTION_STATE_WEEK = date(2026, 8, 31)


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v19 = load_module("v19_auto_place", "scripts/keirin_shogi_v19_targeted_participation.py")
v37 = load_module("v37_auto_place", "scripts/keirin_shogi_v37_shrunk_board_third.py")
v36 = v37.v36
v35 = v37.v35
v32 = v37.v32
v31 = v32.v31
v26 = v31.v26
v25 = v31.v25


def ino(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_csv(name: str):
    rows = []
    for base in (HIST, FUT):
        path = base / name
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def ids_between(races, start: date, end: date):
    start_s, end_s = start.isoformat(), end.isoformat()
    return [
        row["race_id"]
        for row in races
        if start_s <= row.get("race_date", "") <= end_s
        and ino(row.get("entry_count")) == 7
        and row.get("meeting_grade") == "F1"
        and "Ｓ級" in row.get("race_type", "")
    ]


def frozen_selector():
    summary = json.loads(
        (ROOT / "results/keirin_shogi/v21_quantile_participation/summary.json").read_text(
            encoding="utf-8"
        )
    )
    model = summary["selector_model"]
    return float(model["intercept"]), np.asarray(model["coefficients"], dtype=float)


def selector_score(features, intercept: float, coefficients: np.ndarray) -> float:
    z = float(intercept + np.dot(np.asarray(features, dtype=float), coefficients))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def threshold_from_history(score_weeks, current_week: str) -> float:
    prior = sorted(week for week in score_weeks if week < current_week)[-V21_LOOKBACK_WEEKS:]
    values = [
        row["score"]
        for week in prior
        for row in score_weeks[week]
        if row["candidate_count"] <= V21_MAX_FIRST_CANDIDATES
    ]
    if not values:
        return 1.0
    count = max(1, int(round(len(values) * V21_TARGET_FRACTION)))
    return float(sorted(values)[-count])


def build_v21_state():
    races = load_csv("races.csv")
    entries = load_csv("entries.csv")
    results = load_csv("results.csv")
    entries_by = defaultdict(list)
    results_by = defaultdict(list)
    for row in entries:
        entries_by[row["race_id"]].append(row)
    for row in results:
        results_by[row["race_id"]].append(row)

    intercept, coefficients = frozen_selector()

    score_weeks = defaultdict(list)
    old = v25.load_v19_details()
    for row in old:
        if row["week"] >= "2026-06-29":
            continue
        score = selector_score(row["x_selector"], intercept, coefficients)
        score_weeks[row["week"]].append(
            {"score": score, "candidate_count": int(row["candidate_count"])}
        )

    current = date(2026, 6, 29)
    while current <= LATEST_HISTORY_END:
        test_end = current + timedelta(days=6)
        history_start = current - timedelta(days=91)
        core_a_end = current - timedelta(days=36)
        policy_start = current - timedelta(days=35)
        policy_end = current - timedelta(days=22)
        final_end = current - timedelta(days=1)

        core_a = v19.make_races(
            ids_between(races, history_start, core_a_end), entries_by, results_by
        )
        policy_cal = v19.make_races(
            ids_between(races, policy_start, policy_end), entries_by, results_by
        )
        final_train = v19.make_races(
            ids_between(races, history_start, final_end), entries_by, results_by
        )
        test_ds = v19.make_races(
            ids_between(races, current, test_end), entries_by, results_by
        )
        if min(len(core_a), len(policy_cal), len(final_train), len(test_ds)) == 0:
            raise RuntimeError(
                f"missing v21 walk-forward data {current}..{test_end}: "
                f"{len(core_a)},{len(policy_cal)},{len(final_train)},{len(test_ds)}"
            )

        candidate_policy = v19.candidate_policy_from_cal(core_a, policy_cal)
        models = v19.v18.fit_ensemble(final_train)
        generated = []
        for _race_id, race_rows, _winner in test_ds:
            ranking, model_top1s = v19.v18.predict(race_rows, models)
            candidates = v19.v18.choose(ranking, model_top1s, candidate_policy)
            features = v19.pred_features(ranking, model_top1s, candidates)
            generated.append(
                {
                    "score": selector_score(features, intercept, coefficients),
                    "candidate_count": len(candidates),
                }
            )
        score_weeks[current.isoformat()].extend(generated)
        current += timedelta(days=7)

    current = PRODUCTION_STATE_WEEK
    history_start = current - timedelta(days=91)
    core_a_end = current - timedelta(days=36)
    policy_start = current - timedelta(days=35)
    policy_end = current - timedelta(days=22)
    final_end = current - timedelta(days=1)
    core_a = v19.make_races(
        ids_between(races, history_start, core_a_end), entries_by, results_by
    )
    policy_cal = v19.make_races(
        ids_between(races, policy_start, policy_end), entries_by, results_by
    )
    final_train = v19.make_races(
        ids_between(races, history_start, final_end), entries_by, results_by
    )
    if min(len(core_a), len(policy_cal), len(final_train)) == 0:
        raise RuntimeError("cannot build latest available v21 production state")

    return {
        "models": v19.v18.fit_ensemble(final_train),
        "candidate_policy": v19.candidate_policy_from_cal(core_a, policy_cal),
        "threshold": threshold_from_history(score_weeks, current.isoformat()),
        "intercept": intercept,
        "coefficients": coefficients,
        "history_end": final_end.isoformat(),
        "state_week": current.isoformat(),
    }


def build_pair_model():
    vrows = v25.load_v19_details()
    v25.fit_v21_selector(vrows)
    v25.mark_v21(vrows)
    _, entries_by, results_by = v25.raw_maps()
    train = v25.make_dataset(
        vrows, entries_by, results_by, v25.TRAIN_START, v25.TRAIN_END, False
    )
    return v26.fit_pair_model(train, v35.PAIR_PARAMS)


def build_third_model():
    freeze = json.loads(
        (ROOT / "results/keirin_shogi/v37_shrunk_board_third/FREEZE.json").read_text(
            encoding="utf-8"
        )
    )
    races = v32.load_races()
    train = v32.period(races, *freeze["third_training_period"])
    x, y, feature_names = v35.build_training_matrix(train, None)
    model = v35.fit_third_model(x, y, freeze["selected_variant"])
    return model, feature_names, freeze


def live_first(race, v21_state):
    entries = race.get("entries", [])
    race_rows = v19.v18.v17.race_rows(entries)
    ranking, model_top1s = v19.v18.predict(race_rows, v21_state["models"])
    candidates = v19.v18.choose(
        ranking, model_top1s, v21_state["candidate_policy"]
    )
    features = v19.pred_features(ranking, model_top1s, candidates)
    score = selector_score(
        features, v21_state["intercept"], v21_state["coefficients"]
    )
    participate = (
        len(candidates) <= V21_MAX_FIRST_CANDIDATES
        and score >= float(v21_state["threshold"])
    )
    p1_map = {ino(row["no"]): float(row["prob"]) for row in ranking}
    return {
        "participate": participate,
        "raw_candidates": [int(no) for no in candidates],
        "candidates": [int(no) for no in candidates] if participate else [],
        "score": score,
        "threshold": float(v21_state["threshold"]),
        "p1_map": p1_map,
        "ranking": [
            {"no": int(row["no"]), "probability": float(row["prob"])}
            for row in ranking
        ],
    }


def target_scope(race) -> tuple[bool, str]:
    entries = race.get("entries", [])
    if len(entries) != 7:
        return False, "7車立てではない"
    return True, "7車なら級別・開催グレード・発走済み/未発走を問わず配置対象"


def place_one(race, v21_state, pair_model, third_model, feature_names, freeze):
    race_id = str(race.get("race_id", ""))
    scope_ok, scope_note = target_scope(race)
    common = {
        "race_id": race_id,
        "race_date": race.get("race_date", ""),
        "track": race.get("track", ""),
        "race_no": race.get("race_no", ""),
        "race_type": race.get("race_type", ""),
        "scope_ok": scope_ok,
        "scope_note": scope_note,
        "versions": {
            "participation_first": "v21_quantile_participation",
            "second": "v31_top2_membership",
            "third": "v37_shrunk_board_third",
        },
    }
    if not scope_ok:
        return {
            **common,
            "participate": False,
            "first_candidates": [],
            "second_candidates": [],
            "third_candidates": [],
            "skip_reason": scope_note,
        }

    first = live_first(race, v21_state)
    if not first["participate"]:
        return {
            **common,
            "participate": False,
            "first_candidates": [],
            "raw_first_candidates": first["raw_candidates"],
            "second_candidates": [],
            "third_candidates": [],
            "v21_selector_score": first["score"],
            "v21_threshold": first["threshold"],
            "first_ranking": first["ranking"],
            "skip_reason": "v21参加判定で見送り",
        }

    entries = race.get("entries", [])
    base = v32.enrich_base(entries)
    vr = {
        "candidates": first["candidates"],
        "p1_map": first["p1_map"],
        "v21_participate": True,
    }
    pairs = v31.pair_distribution(base, vr, pair_model)
    membership = v31.membership_rank(pairs)
    second = [int(no) for no in v31.choose(membership, V31_REL3, V31_MIN3)]

    context = {
        "pairs": pairs,
        "first_candidates": first["candidates"],
        "first_probabilities": first["p1_map"],
        "second_candidates": second,
        "top2_ranking": [
            {"no": int(no), "mass": float(value)} for no, value in membership
        ],
    }
    race_obj = {
        "race_id": race_id,
        "race_date": race.get("race_date", ""),
        "race_type": race.get("race_type", ""),
        "base": base,
        "order": [0, 0, 0],
    }
    prepared = v35.prepare_prediction(
        [race_obj], pair_model, feature_names, {race_id: context}
    )
    conditionals = v36.raw_conditionals(prepared, third_model)
    third_row = v36.aggregate(prepared, conditionals, freeze["fixed_score"])[0]
    third = [int(no) for no in v35.choose(third_row["ranking"], freeze["selected_policy"])]

    return {
        **common,
        "participate": True,
        "first_candidates": first["candidates"],
        "second_candidates": second,
        "third_candidates": third,
        "v21_selector_score": first["score"],
        "v21_threshold": first["threshold"],
        "first_ranking": first["ranking"],
        "second_membership": [
            {"no": int(no), "mass": float(value)} for no, value in membership
        ],
        "top_pairs": [
            {"a": int(a), "b": int(b), "prob": float(probability)}
            for a, b, probability in pairs[:5]
        ],
        "third_ranking": [
            {"no": int(no), "probability": float(probability)}
            for no, probability in third_row["ranking"]
        ],
    }


def main() -> int:
    live = json.loads(LIVE.read_text(encoding="utf-8"))
    v21_state = build_v21_state()
    pair_model = build_pair_model()
    third_model, feature_names, freeze = build_third_model()

    output = []
    failures = []
    for race in live.get("races", []):
        try:
            output.append(
                place_one(
                    race,
                    v21_state,
                    pair_model,
                    third_model,
                    feature_names,
                    freeze,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "race_id": str(race.get("race_id", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = {
        "schema_version": 1,
        "engine": "keirin_shogi_v21_v31_v37_auto_place",
        "mode": "deterministic_no_llm",
        "generated_at_jst": live.get("generated_at_jst", ""),
        "input_source": "live-race-data.json",
        "race_count": len(output),
        "failure_count": len(failures),
        "latest_historical_state": {
            "history_end": v21_state["history_end"],
            "state_week": v21_state["state_week"],
            "note": (
                "v21 is mechanically reconstructed through the latest committed "
                "historical results (2026-08-30), then held fixed for newer live races "
                "until newer chronological history is committed."
            ),
            "candidate_policy": v21_state["candidate_policy"],
            "threshold": v21_state["threshold"],
        },
        "guards": {
            "llm_inference_used": False,
            "odds_used": False,
            "popularity_used": False,
            "live_results_used": False,
            "live_payouts_used": False,
            "v31_parameters_changed": False,
            "v37_parameters_changed": False,
        },
        "races": output,
        "failures": failures,
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out": str(OUT),
                "race_count": len(output),
                "participant_count": sum(bool(row.get("participate")) for row in output),
                "failure_count": len(failures),
                "v21_history_end": v21_state["history_end"],
                "v21_threshold": v21_state["threshold"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if output and not failures else (1 if output else 2)


if __name__ == "__main__":
    raise SystemExit(main())

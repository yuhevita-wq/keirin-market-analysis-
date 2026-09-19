#!/usr/bin/env python3
from __future__ import annotations

"""Unified 7-rider Keirin Shogi predictor.

The old live path exposed three independently selected board rows:
v21 first candidates -> v31 second candidates -> v37 third candidates.

This engine keeps the already-trained signals, but combines them BEFORE any
board row is selected.  The single prediction object is an exact-order
distribution P(first=i, second=j, third=k).  First/second/third board rows are
marginals of that one distribution, so the rows can no longer disagree merely
because three different inference paths were pasted together.

Important scope guard:
The historical component training data are 7-rider F1 S-class races.  Until a
new walk-forward training run proves transfer to other populations, live
prediction is deliberately restricted to that same population.

Odds, popularity, target-race result and target-race payout are never inputs.
The participation flag remains the frozen v21 selector only as a TEMPORARY
safety gate.  It does not choose board riders.  A profit-trained participation
gate is the next separate validation task.
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
LATEST_HISTORY_END = date(2026, 8, 30)
PRODUCTION_STATE_WEEK = date(2026, 8, 31)

# Board extraction happens only AFTER the joint order distribution exists.
# These are row-width policies, not separate prediction models.
FIRST_COUNT = 2
SECOND_REL3 = 0.65
THIRD_REL3 = 0.70


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v19 = load_module(
    "unified_v19",
    "scripts/keirin_shogi_v19_targeted_participation.py",
)
v37 = load_module(
    "unified_v37",
    "scripts/keirin_shogi_v37_shrunk_board_third.py",
)
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
        and ("Ｓ級" in row.get("race_type", "") or "S級" in row.get("race_type", ""))
    ]


def frozen_selector():
    summary = json.loads(
        (
            ROOT
            / "results/keirin_shogi/v21_quantile_participation/summary.json"
        ).read_text(encoding="utf-8")
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
    prior = sorted(week for week in score_weeks if week < current_week)[
        -V21_LOOKBACK_WEEKS:
    ]
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
        score_weeks[row["week"]].append(
            {
                "score": selector_score(row["x_selector"], intercept, coefficients),
                "candidate_count": int(row["candidate_count"]),
            }
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
        (
            ROOT / "results/keirin_shogi/v37_shrunk_board_third/FREEZE.json"
        ).read_text(encoding="utf-8")
    )
    races = v32.load_races()
    train = v32.period(races, *freeze["third_training_period"])
    x, y, feature_names = v35.build_training_matrix(train, None)
    model = v35.fit_third_model(x, y, freeze["selected_variant"])
    return model, feature_names, freeze


def target_scope(race) -> tuple[bool, str]:
    entries = race.get("entries", [])
    if len(entries) != 7:
        return False, "統合v1は7車専用"
    if str(race.get("meeting_grade", "")) != "F1":
        return False, "統合v1は学習母集団と同じF1のみ"
    race_type = str(race.get("race_type", ""))
    if "Ｓ級" not in race_type and "S級" not in race_type:
        return False, "統合v1は学習母集団と同じS級のみ"
    return True, "7車F1 S級"


def first_signal(race, state):
    normalized_entries = [
        {
            **row,
            "line_id": str(row.get("line_id") or ""),
            "style": str(row.get("style") or ""),
        }
        for row in race.get("entries", [])
    ]
    race_rows = v19.v18.v17.race_rows(normalized_entries)
    ranking, model_top1s = v19.v18.predict(race_rows, state["models"])
    legacy_candidates = v19.v18.choose(
        ranking, model_top1s, state["candidate_policy"]
    )
    features = v19.pred_features(ranking, model_top1s, legacy_candidates)
    score = selector_score(features, state["intercept"], state["coefficients"])
    return {
        "p1_map": {ino(row["no"]): float(row["prob"]) for row in ranking},
        "ranking": [
            (ino(row["no"]), float(row["prob"]))
            for row in ranking
        ],
        "legacy_candidates": [ino(no) for no in legacy_candidates],
        "selector_score": score,
        "selector_threshold": float(state["threshold"]),
    }


def normalized_orientation(pa: float, pb: float) -> tuple[float, float]:
    total = max(pa, 0.0) + max(pb, 0.0)
    if total <= 1e-15:
        return 0.5, 0.5
    return max(pa, 0.0) / total, max(pb, 0.0) / total


def rank_map(values: dict[int, float]):
    total = sum(values.values()) or 1.0
    rows = [(int(no), float(value) / total) for no, value in values.items()]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return rows


def choose_top2_plus_ratio(ranking, rel3: float):
    candidates = [int(ranking[0][0]), int(ranking[1][0])]
    if len(ranking) >= 3:
        p2 = float(ranking[1][1])
        p3 = float(ranking[2][1])
        if p3 / max(p2, 1e-12) >= rel3:
            candidates.append(int(ranking[2][0]))
    return candidates


def unified_joint(race, state, pair_model, third_model, feature_names):
    first = first_signal(race, state)

    # Live cache may serialize line_id as an integer, while the historical
    # feature builder expects a string and calls .strip().
    normalized_entries = [
        {
            **row,
            "line_id": str(row.get("line_id") or ""),
            "style": str(row.get("style") or ""),
        }
        for row in race.get("entries", [])
    ]
    base = v32.enrich_base(normalized_entries)

    pair_vr = {
        "p1_map": first["p1_map"],
        "candidates": first["legacy_candidates"],
        "v21_participate": True,
    }
    pairs = v31.pair_distribution(base, pair_vr, pair_model)

    race_id = str(race.get("race_id", ""))
    race_obj = {
        "race_id": race_id,
        "race_date": race.get("race_date", ""),
        "race_type": race.get("race_type", ""),
        "base": base,
        "order": [0, 0, 0],
    }
    prepared = v35.prepare_prediction(
        [race_obj],
        pair_model,
        feature_names,
        {race_id: {"pairs": pairs}},
    )
    conditionals = v36.raw_conditionals(prepared, third_model)[0]
    branches = prepared["groups"][0]["branches"]

    joint = []
    first_mass = defaultdict(float)
    second_mass = defaultdict(float)
    third_mass = defaultdict(float)

    for (a, b, pair_probability), branch, cond in zip(
        pairs, branches, conditionals
    ):
        oa, ob = normalized_orientation(
            first["p1_map"].get(int(a), 0.0),
            first["p1_map"].get(int(b), 0.0),
        )
        for third_no, third_probability in zip(branch["candidates"], cond):
            c = int(third_no)
            third_p = float(third_probability)
            pab = float(pair_probability) * oa * third_p
            pba = float(pair_probability) * ob * third_p
            if pab > 0:
                joint.append((int(a), int(b), c, pab))
                first_mass[int(a)] += pab
                second_mass[int(b)] += pab
                third_mass[c] += pab
            if pba > 0:
                joint.append((int(b), int(a), c, pba))
                first_mass[int(b)] += pba
                second_mass[int(a)] += pba
                third_mass[c] += pba

    total = sum(row[3] for row in joint) or 1.0
    joint = [(a, b, c, p / total) for a, b, c, p in joint]
    joint.sort(key=lambda row: (-row[3], row[0], row[1], row[2]))

    first_ranking = rank_map(first_mass)
    second_ranking = rank_map(second_mass)
    third_ranking = rank_map(third_mass)

    first_candidates = [int(no) for no, _ in first_ranking[:FIRST_COUNT]]
    second_candidates = choose_top2_plus_ratio(second_ranking, SECOND_REL3)
    third_candidates = choose_top2_plus_ratio(third_ranking, THIRD_REL3)

    participate = True

    return {
        "board_generated": True,
        "board_policy": "unified_joint_order_v1",
        "participate": bool(participate),
        "first_candidates": first_candidates,
        "second_candidates": second_candidates,
        "third_candidates": third_candidates,
        "first_ranking": [
            {"no": no, "probability": probability}
            for no, probability in first_ranking
        ],
        "second_ranking": [
            {"no": no, "probability": probability}
            for no, probability in second_ranking
        ],
        "third_ranking": [
            {"no": no, "probability": probability}
            for no, probability in third_ranking
        ],
        "top_orders": [
            {"first": a, "second": b, "third": c, "probability": p}
            for a, b, c, p in joint[:20]
        ],
        "participation_policy": "all_in_scope",
    }


def place_one(race, state, pair_model, third_model, feature_names):
    race_id = str(race.get("race_id", ""))
    scope_ok, scope_note = target_scope(race)
    common = {
        "race_id": race_id,
        "race_date": race.get("race_date", ""),
        "track": race.get("track", ""),
        "race_no": race.get("race_no", ""),
        "race_type": race.get("race_type", ""),
        "meeting_grade": race.get("meeting_grade", ""),
        "scope_ok": scope_ok,
        "scope_note": scope_note,
        "versions": {
            "prediction": "unified_joint_order_v1",
            "component_signals": "v21_first + v31_top2_pair + v35_conditional_third",
            "participation_guard": "none",
        },
    }
    if not scope_ok:
        return {
            **common,
            "board_generated": False,
            "participate": False,
            "first_candidates": [],
            "second_candidates": [],
            "third_candidates": [],
            "skip_reason": scope_note,
        }

    prediction = unified_joint(
        race, state, pair_model, third_model, feature_names
    )
    return {
        **common,
        **prediction,
    }


def main() -> int:
    live = json.loads(LIVE.read_text(encoding="utf-8"))
    state = build_v21_state()
    pair_model = build_pair_model()
    third_model, feature_names, freeze = build_third_model()

    output = []
    failures = []
    for race in live.get("races", []):
        try:
            output.append(
                place_one(
                    race,
                    state,
                    pair_model,
                    third_model,
                    feature_names,
                )
            )
        except Exception as exc:
            failure = {
                "race_id": str(race.get("race_id", "")),
                "failure_stage": "unified_inference",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            output.append(
                {
                    "race_id": failure["race_id"],
                    "race_date": race.get("race_date", ""),
                    "track": race.get("track", ""),
                    "race_no": race.get("race_no", ""),
                    "race_type": race.get("race_type", ""),
                    "board_generated": False,
                    "participate": False,
                    "first_candidates": [],
                    "second_candidates": [],
                    "third_candidates": [],
                    **failure,
                }
            )

    payload = {
        "schema_version": 2,
        "engine": "keirin_shogi_unified_v1",
        "mode": "deterministic_joint_order_no_llm",
        "generated_at_jst": live.get("generated_at_jst", ""),
        "input_source": "live-race-data.json",
        "race_count": len(output),
        "failure_count": len(failures),
        "latest_historical_state": {
            "history_end": state["history_end"],
            "state_week": state["state_week"],
            "training_scope": "7-rider F1 S-class",
            "third_training_period": freeze.get("third_training_period", []),
        },
        "runtime_versions": {
            "seven_rider": "unified_joint_order_v1",
        },
        "guards": {
            "llm_inference_used": False,
            "odds_used": False,
            "popularity_used": False,
            "live_results_used": False,
            "live_payouts_used": False,
            "scope_matches_training_population": True,
            "board_rows_share_one_joint_distribution": True,
            "profit_participation_model_ready": False,
        },
        "races": output,
        "failures": failures,
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(OUT),
                "race_count": len(output),
                "board_count": sum(bool(row.get("board_generated")) for row in output),
                "participant_count": sum(bool(row.get("participate")) for row in output),
                "failure_count": len(failures),
                "failure_samples": failures[:5],
                "engine": payload["engine"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if output and not failures else (1 if output else 2)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

"""Runtime compatibility layer for deterministic v21/v31/v37 placement.

The frozen training/model code remains untouched. This layer normalizes live
KDreams input types, broadens the runtime entrance to variable rider counts,
and keeps v21 participation separate from board generation:

- v21 still decides participate / skip with its frozen selector.
- v21 first candidates are still generated even when the selector says skip.
- v31 and v37 continue from those candidates and probabilities.
- therefore a v21 skip no longer blanks the three-row board.
"""

import importlib.util
import json
import math
import platform
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]
BASE_ENGINE = ROOT / "scripts/keirin_shogi_v37_auto_place.py"
TEMP_LIVE = ROOT / "docs/keirin-shogi/.live-race-data-runtime.json"
BOARD_POLICY = "v21_raw_candidates_through_v31_v37_v1"
REQUIRED_NUMERIC_FIELDS = (
    "score",
    "win_rate",
    "top2_rate",
    "top3_rate",
    "s_count",
    "b_count",
    "nige_count",
    "makuri_count",
    "sashi_count",
    "mark_count",
    "first_count",
    "second_count",
    "third_count",
    "outside_count",
    "line_id",
    "line_position",
)


def load_base_engine():
    spec = importlib.util.spec_from_file_location("keirin_shogi_v37_auto_place_base", BASE_ENGINE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_entry(entry: dict[str, object]) -> dict[str, object]:
    row = dict(entry)
    # Frozen v17/v25 feature builders call .strip() on line_id because the
    # historical CSV stores it as text. KDreams live line parsing stores it as
    # an integer. Normalize only the transport type, not the learned features.
    line_id = row.get("line_id", "")
    row["line_id"] = "" if line_id is None else str(line_id)
    return row


def normalize_live(payload: dict[str, object]) -> dict[str, object]:
    out = dict(payload)
    races = []
    for race in payload.get("races", []):
        if not isinstance(race, dict):
            continue
        rr = dict(race)
        rr["entries"] = [
            normalize_entry(entry)
            for entry in race.get("entries", [])
            if isinstance(entry, dict)
        ]
        races.append(rr)
    out["races"] = races
    return out


def runtime_scope(race) -> tuple[bool, str]:
    entries = race.get("entries", [])
    count = len(entries)
    if count < 3:
        return False, "3着候補を構成できる車数に満たない"
    return True, f"{count}車立て。級別・開催グレード・発走済み/未発走を問わず配置計算対象"


def failure_result(common: dict[str, object], stage: str, exc: Exception) -> dict[str, object]:
    return {
        **common,
        "board_policy": BOARD_POLICY,
        "board_generated": False,
        "participate": False,
        "first_candidates": [],
        "second_candidates": [],
        "third_candidates": [],
        "failure_stage": stage,
        "error": f"{type(exc).__name__}: {exc}",
        "skip_reason": "盤面生成失敗",
    }


def validate_race_input(race: dict[str, object]) -> None:
    entries = race.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("entries is not a list")
    seen: set[int] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entry[{index}] is not an object")
        try:
            car_no = int(float(entry.get("car_no", "")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"entry[{index}].car_no is invalid") from exc
        if car_no <= 0 or car_no in seen:
            raise ValueError(f"entry[{index}].car_no is duplicate or non-positive: {car_no}")
        seen.add(car_no)
        for field in REQUIRED_NUMERIC_FIELDS:
            value = entry.get(field, "")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"car {car_no} missing numeric field {field}") from exc
            if not math.isfinite(number):
                raise ValueError(f"car {car_no} has non-finite field {field}")


def validate_distribution(
    rows: list[dict[str, object]], *, value_key: str, expected_count: int, label: str
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"{label} count {len(rows)} != {expected_count}")
    values = [float(row[value_key]) for row in rows]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"{label} contains invalid probabilities")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(f"{label} probability sum is {sum(values)}")


def validate_candidates(candidates: list[int], rider_numbers: set[int], label: str) -> None:
    if not candidates:
        raise ValueError(f"{label} is empty")
    if len(candidates) != len(set(candidates)) or set(candidates) - rider_numbers:
        raise ValueError(f"{label} contains duplicate or unknown riders: {candidates}")


def runtime_place_one(engine, race, v21_state, pair_model, third_model, feature_names, freeze):
    """Generate the board regardless of the v21 participation decision.

    This intentionally changes only runtime wiring. The frozen v21 score,
    threshold, candidate policy, v31 pair model and v37 third model are reused
    exactly as provided by the base engine.
    """
    race_id = str(race.get("race_id", ""))
    scope_ok, scope_note = runtime_scope(race)
    common = {
        "race_id": race_id,
        "race_date": race.get("race_date", ""),
        "track": race.get("track", ""),
        "race_no": race.get("race_no", ""),
        "race_type": race.get("race_type", ""),
        "scope_ok": scope_ok,
        "scope_note": scope_note,
        "board_policy": BOARD_POLICY,
        "versions": {
            "participation_first": "v21_quantile_participation",
            "second": "v31_top2_membership",
            "third": "v37_shrunk_board_third",
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

    try:
        validate_race_input(race)
    except Exception as exc:
        return failure_result(common, "input_validation", exc)

    # Base live_first still calculates the frozen v21 candidate set before it
    # applies the participation gate. raw_candidates preserves that prediction.
    try:
        first = engine.live_first(race, v21_state)
        first_candidates = [int(no) for no in first["raw_candidates"]]
        rider_numbers = {int(float(row["car_no"])) for row in race.get("entries", [])}
        validate_candidates(first_candidates, rider_numbers, "first_candidates")
        validate_distribution(
            first["ranking"],
            value_key="probability",
            expected_count=len(rider_numbers),
            label="v21 first ranking",
        )
    except Exception as exc:
        return failure_result(common, "v21_prediction", exc)

    entries = race.get("entries", [])
    try:
        base = engine.v32.enrich_base(entries)
    except Exception as exc:
        return failure_result(common, "v31_feature_build", exc)
    vr = {
        "candidates": first_candidates,
        "p1_map": first["p1_map"],
        "v21_participate": bool(first["participate"]),
    }
    try:
        pairs = engine.v31.pair_distribution(base, vr, pair_model)
        expected_pairs = len(rider_numbers) * (len(rider_numbers) - 1) // 2
        pair_rows = [
            {"a": int(a), "b": int(b), "probability": float(probability)}
            for a, b, probability in pairs
        ]
        validate_distribution(
            pair_rows,
            value_key="probability",
            expected_count=expected_pairs,
            label="v31 pair distribution",
        )
        membership = engine.v31.membership_rank(pairs)
        if len(membership) != len(rider_numbers):
            raise ValueError(f"v31 membership count {len(membership)} != {len(rider_numbers)}")
        if not all(math.isfinite(float(value)) for _, value in membership):
            raise ValueError("v31 membership contains non-finite values")
        second = [
            int(no)
            for no in engine.v31.choose(membership, engine.V31_REL3, engine.V31_MIN3)
        ]
        validate_candidates(second, rider_numbers, "second_candidates")
    except Exception as exc:
        return failure_result(common, "v31_prediction", exc)

    context = {
        "pairs": pairs,
        "first_candidates": first_candidates,
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
    try:
        prepared = engine.v35.prepare_prediction(
            [race_obj], pair_model, feature_names, {race_id: context}
        )
        if prepared["x"].size == 0 or not bool(math.prod(prepared["x"].shape)):
            raise ValueError("v37 feature matrix is empty")
        if not bool(engine.np.isfinite(prepared["x"]).all()):
            raise ValueError("v37 feature matrix contains non-finite values")
    except Exception as exc:
        return failure_result(common, "v37_feature_build", exc)
    try:
        conditionals = engine.v36.raw_conditionals(prepared, third_model)
        third_row = engine.v36.aggregate(prepared, conditionals, freeze["fixed_score"])[0]
        third_ranking = [
            {"no": int(no), "probability": float(probability)}
            for no, probability in third_row["ranking"]
        ]
        validate_distribution(
            third_ranking,
            value_key="probability",
            expected_count=len(rider_numbers),
            label="v37 third ranking",
        )
        third = [
            int(no)
            for no in engine.v35.choose(
                third_row["ranking"], freeze["selected_policy"]
            )
        ]
        validate_candidates(third, rider_numbers, "third_candidates")
    except Exception as exc:
        return failure_result(common, "v37_prediction", exc)

    result = {
        **common,
        "board_generated": True,
        "participate": bool(first["participate"]),
        "first_candidates": first_candidates,
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
        "pair_probability_count": len(pairs),
        "third_ranking": third_ranking,
    }
    if not first["participate"]:
        result["skip_reason"] = "v21参加判定で見送り（盤面は配置）"
    return result


def main() -> int:
    engine = load_base_engine()
    original_live = engine.LIVE
    payload = json.loads(original_live.read_text(encoding="utf-8"))
    expected_state = payload.get("board_engine", {}).get("latest_historical_state", {})
    normalized = normalize_live(payload)
    TEMP_LIVE.write_text(
        json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    engine.LIVE = TEMP_LIVE
    engine.target_scope = runtime_scope
    engine.place_one = lambda race, v21_state, pair_model, third_model, feature_names, freeze: runtime_place_one(
        engine,
        race,
        v21_state,
        pair_model,
        third_model,
        feature_names,
        freeze,
    )
    try:
        try:
            engine.OUT.unlink()
        except FileNotFoundError:
            pass
        result = int(engine.main())
        board_payload = json.loads(engine.OUT.read_text(encoding="utf-8"))
        board_payload["runtime_versions"] = {
            "python": platform.python_version(),
            "numpy": engine.np.__version__,
            "scikit_learn": sklearn.__version__,
        }
        actual_state = board_payload.get("latest_historical_state", {})
        expected_policy = expected_state.get("candidate_policy")
        expected_threshold = expected_state.get("threshold")
        state_changed = bool(expected_policy) and (
            actual_state.get("candidate_policy") != expected_policy
            or not math.isclose(
                float(actual_state.get("threshold", math.nan)),
                float(expected_threshold),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if state_changed:
            engine.OUT.unlink()
            print(
                json.dumps(
                    {
                        "error": "frozen v21 production state drift",
                        "failure_stage": "model_runtime_guard",
                        "expected": expected_state,
                        "actual": actual_state,
                        "runtime_versions": board_payload["runtime_versions"],
                    },
                    ensure_ascii=False,
                )
            )
            return 3
        engine.OUT.write_text(
            json.dumps(board_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return result
    finally:
        try:
            TEMP_LIVE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

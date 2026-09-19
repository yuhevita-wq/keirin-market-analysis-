#!/usr/bin/env python3
from __future__ import annotations

"""Runtime compatibility layer for Keirin Shogi live placement.

For seven-car races, v21/v31/v37 are retained as feature generators, then the
v38 KING SEAT model rewrites the visible board and the KING-specific gate
replaces v21 as the final buy/skip decision. Nine-car G1/G2/G3 keeps v3.2.
Target-race odds, popularity, results and payouts are never read.
"""

import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]
BASE_ENGINE = ROOT / "scripts/keirin_shogi_v37_auto_place.py"
NINECAR_ENGINE = ROOT / "scripts/keirin_shogi_ninecar_v32.py"
NINECAR_MODEL = ROOT / "results/keirin_shogi/ninecar_v32/model.joblib"
SEVENCAR_OVERLAY_ENGINE = ROOT / "scripts/keirin_shogi_sevencar_v37_overlay.py"
SEVENCAR_OVERLAY_MODEL = ROOT / "results/keirin_shogi/sevencar_state_transfer/validated_model.joblib"
KING_ENGINE = ROOT / "scripts/keirin_shogi_v38_king_runtime.py"
TEMP_LIVE = ROOT / "docs/keirin-shogi/.live-race-data-runtime.json"
BOARD_POLICY = "v21_raw_candidates_through_v31_v37_v1"
SEVENCAR_POLICY = "v38_king_seat_gate_2024w1"
NINECAR_POLICY = "ninecar_v32_strong_state_overlay"
NINECAR_GRADES = {"G1", "G2", "G3"}
NINECAR_CLASSES = {"SS", "S1", "S2"}
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


def load_ninecar_engine():
    spec = importlib.util.spec_from_file_location("keirin_shogi_ninecar_v32_runtime", NINECAR_ENGINE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_sevencar_overlay_engine():
    spec = importlib.util.spec_from_file_location(
        "keirin_shogi_sevencar_v37_overlay_runtime", SEVENCAR_OVERLAY_ENGINE
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_king_engine():
    spec = importlib.util.spec_from_file_location(
        "keirin_shogi_v38_king_runtime_live", KING_ENGINE
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
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


def is_ninecar_v32_target(race: dict[str, object]) -> bool:
    entries = race.get("entries", [])
    if not isinstance(entries, list) or len(entries) != 9:
        return False
    if str(race.get("meeting_grade", "")).strip().upper() not in NINECAR_GRADES:
        return False
    try:
        cars = {int(float(row.get("car_no", ""))) for row in entries if isinstance(row, dict)}
    except (TypeError, ValueError):
        return False
    if cars != set(range(1, 10)):
        return False
    for row in entries:
        if not isinstance(row, dict):
            return False
        if str(row.get("class", "")).strip().upper() not in NINECAR_CLASSES:
            return False
        try:
            if float(row.get("line_id", 0)) <= 0 or float(row.get("line_position", 0)) <= 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def runtime_place_ninecar(ninecar_engine, race: dict[str, object]) -> dict[str, object]:
    common = {
        "race_id": str(race.get("race_id", "")),
        "race_date": race.get("race_date", ""),
        "track": race.get("track", ""),
        "race_no": race.get("race_no", ""),
        "race_type": race.get("race_type", ""),
        "scope_ok": True,
        "scope_note": "9車S級G1/G2/G3。ninecar v3.2 強者状態補正で配置",
        "board_policy": NINECAR_POLICY,
        "versions": {"ninecar": NINECAR_POLICY},
        "provisional_adoption": False,
        "adoption_note": "9車専用v3.2 採用",
    }
    try:
        validate_race_input(race)
        if not NINECAR_MODEL.exists():
            raise FileNotFoundError(f"ninecar v3.2 model not found: {NINECAR_MODEL}")
        result = ninecar_engine.predict_live(race, model_path=NINECAR_MODEL)
        rider_numbers = set(range(1, 10))
        for key in ("first_candidates", "second_candidates", "third_candidates"):
            validate_candidates([int(no) for no in result.get(key, [])], rider_numbers, key)
        for key in ("first_ranking", "second_ranking", "third_ranking"):
            validate_distribution(
                result.get(key, []),
                value_key="probability",
                expected_count=9,
                label=f"ninecar v3.2 {key}",
            )
        merged = {**common, **result}
        merged["scope_ok"] = True
        merged["scope_note"] = common["scope_note"]
        merged["provisional_adoption"] = False
        merged["adoption_note"] = common["adoption_note"]
        return merged
    except Exception as exc:
        return {
            **common,
            "board_generated": False,
            "participate": False,
            "first_candidates": [],
            "second_candidates": [],
            "third_candidates": [],
            "failure_stage": "ninecar_v32_prediction",
            "error": f"{type(exc).__name__}: {exc}",
            "skip_reason": "9車v3.2盤面生成失敗",
        }


def failure_result(common: dict[str, object], stage: str, exc: Exception) -> dict[str, object]:
    return {
        **common,
        "board_policy": common.get("board_policy", BOARD_POLICY),
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


def runtime_place_one(engine, ninecar_engine, sevencar_overlay_engine, king_engine, race, v21_state, pair_model, third_model, feature_names, freeze):
    """Build the base board, then give seven-car races to v38 KING SEAT.

    v21/v31/v37 remain intact as structural feature generators. For seven-car
    races, their candidate rows are not the final output: v38 projects 24 exact
    KING seats back onto the three board rows and its KING gate decides buy/skip.
    """
    race_id = str(race.get("race_id", ""))
    scope_ok, scope_note = runtime_scope(race)
    is_sevencar = len(race.get("entries", [])) == 7
    common = {
        "race_id": race_id,
        "race_date": race.get("race_date", ""),
        "track": race.get("track", ""),
        "race_no": race.get("race_no", ""),
        "race_type": race.get("race_type", ""),
        "scope_ok": scope_ok,
        "scope_note": scope_note,
        "board_policy": SEVENCAR_POLICY if is_sevencar else BOARD_POLICY,
        "versions": {
            "base_first": "v21_quantile_participation",
            "base_second": "v31_top2_membership",
            "base_third": (
                "v37_shrunk_board_third+softfail_third_append"
                if is_sevencar else "v37_shrunk_board_third"
            ),
            **({"seven_car_primary": "v38_king_seat_gate_2024w1"} if is_sevencar else {}),
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

    if is_ninecar_v32_target(race):
        return runtime_place_ninecar(ninecar_engine, race)

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

    overlay = None
    if is_sevencar:
        try:
            overlay = sevencar_overlay_engine.predict_overlay(
                base=base,
                first_candidates=first_candidates,
                second_candidates=second,
                third_candidates=third,
                membership=membership,
                top_pairs=pairs,
                race_type=race.get("race_type", ""),
                participate=bool(first["participate"]),
                model_path=SEVENCAR_OVERLAY_MODEL,
            )
            third = [int(no) for no in overlay["third_candidates"]]
            validate_candidates(third, rider_numbers, "third_candidates_after_sevencar_overlay")
        except Exception as exc:
            return failure_result(common, "sevencar_v37_state_overlay", exc)

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
    if overlay is not None:
        result.update(
            {
                "sevencar_state_overlay_action": overlay["action"],
                "sevencar_state_strong_rider": overlay["strong_rider"],
                "sevencar_state_probabilities": overlay["state_probabilities"],
                "sevencar_state_added_third": bool(overlay["added"]),
                "sevencar_state_model": overlay.get("model_name"),
                "sevencar_state_validated_through": overlay.get("validated_through"),
            }
        )
    if is_sevencar:
        try:
            king_result = king_engine.predict_king_seat(result, race)
            validate_candidates(
                [int(no) for no in king_result.get("first_candidates", [])],
                rider_numbers,
                "v38_first_candidates",
            )
            validate_candidates(
                [int(no) for no in king_result.get("second_candidates", [])],
                rider_numbers,
                "v38_second_candidates",
            )
            validate_candidates(
                [int(no) for no in king_result.get("third_candidates", [])],
                rider_numbers,
                "v38_third_candidates",
            )
            return king_result
        except Exception as exc:
            return failure_result(common, "v38_king_seat_gate", exc)

    if not first["participate"]:
        result["skip_reason"] = "v21参加判定で見送り（盤面は配置）"
    return result


def main() -> int:
    engine = load_base_engine()
    ninecar_engine = load_ninecar_engine()
    sevencar_overlay_engine = load_sevencar_overlay_engine()
    king_engine = load_king_engine()
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
        ninecar_engine,
        sevencar_overlay_engine,
        king_engine,
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
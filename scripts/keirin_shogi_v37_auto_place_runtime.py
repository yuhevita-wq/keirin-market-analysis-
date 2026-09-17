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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_ENGINE = ROOT / "scripts/keirin_shogi_v37_auto_place.py"
TEMP_LIVE = ROOT / "docs/keirin-shogi/.live-race-data-runtime.json"


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

    # Base live_first still calculates the frozen v21 candidate set before it
    # applies the participation gate. raw_candidates preserves that prediction.
    first = engine.live_first(race, v21_state)
    first_candidates = [int(no) for no in first["raw_candidates"]]

    entries = race.get("entries", [])
    base = engine.v32.enrich_base(entries)
    vr = {
        "candidates": first_candidates,
        "p1_map": first["p1_map"],
        "v21_participate": bool(first["participate"]),
    }
    pairs = engine.v31.pair_distribution(base, vr, pair_model)
    membership = engine.v31.membership_rank(pairs)
    second = [
        int(no)
        for no in engine.v31.choose(membership, engine.V31_REL3, engine.V31_MIN3)
    ]

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
    prepared = engine.v35.prepare_prediction(
        [race_obj], pair_model, feature_names, {race_id: context}
    )
    conditionals = engine.v36.raw_conditionals(prepared, third_model)
    third_row = engine.v36.aggregate(prepared, conditionals, freeze["fixed_score"])[0]
    third = [
        int(no)
        for no in engine.v35.choose(
            third_row["ranking"], freeze["selected_policy"]
        )
    ]

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
        "third_ranking": [
            {"no": int(no), "probability": float(probability)}
            for no, probability in third_row["ranking"]
        ],
    }
    if not first["participate"]:
        result["skip_reason"] = "v21参加判定で見送り（盤面は配置）"
    return result


def main() -> int:
    engine = load_base_engine()
    original_live = engine.LIVE
    payload = json.loads(original_live.read_text(encoding="utf-8"))
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
        return int(engine.main())
    finally:
        try:
            TEMP_LIVE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

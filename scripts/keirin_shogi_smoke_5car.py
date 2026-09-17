#!/usr/bin/env python3
from __future__ import annotations

"""Real KDreams five-rider smoke test for the frozen v21/v31/v37 runtime.

This test fetches one known finished five-rider race, parses only the race card
and published line forecast used by production, then mechanically runs the same
frozen placement engine. Target-race odds, popularity, result and payout fields
are not passed to the engine.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RACE_ID = "2720260610020001"
RACE_URL = f"https://keirin.kdreams.jp/keiokaku/racedetail/{RACE_ID}/"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    cache = load("keirin_shogi_live_cache_for_smoke", ROOT / "scripts/keirin_shogi_live_race_cache.py")
    runtime = load("keirin_shogi_runtime_for_smoke", ROOT / "scripts/keirin_shogi_v37_auto_place_runtime.py")
    engine = runtime.load_base_engine()
    engine.target_scope = runtime.runtime_scope

    session = cache.make_session()
    html = cache.fetch_html(session, RACE_URL)
    meta, entries = cache.parse_entries(html, RACE_ID, RACE_URL)
    line = cache.parse_line_formation_html(html)
    cache.attach_line(entries, line)

    if len(entries) != 5:
        raise RuntimeError(f"expected real five-rider card, got {len(entries)} riders")

    race = {
        **meta,
        "line_status": line.get("status", ""),
        "predicted_line_formation": line.get("formation", ""),
        "line_provider": line.get("provider", ""),
        "entries": [runtime.normalize_entry(entry) for entry in entries],
    }

    v21_state = engine.build_v21_state()
    pair_model = engine.build_pair_model()
    third_model, feature_names, freeze = engine.build_third_model()
    board = engine.place_one(
        race,
        v21_state,
        pair_model,
        third_model,
        feature_names,
        freeze,
    )

    if not board.get("scope_ok"):
        raise RuntimeError(f"five-rider race rejected by runtime scope: {board.get('scope_note')}")

    print(
        json.dumps(
            {
                "smoke": "real_kdreams_5car",
                "race_id": RACE_ID,
                "race_date": race.get("race_date", ""),
                "race_type": race.get("race_type", ""),
                "entry_count": len(entries),
                "scope_ok": bool(board.get("scope_ok")),
                "participate": bool(board.get("participate")),
                "first_candidates": board.get("first_candidates", []),
                "second_candidates": board.get("second_candidates", []),
                "third_candidates": board.get("third_candidates", []),
                "skip_reason": board.get("skip_reason", ""),
                "scope_note": board.get("scope_note", ""),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

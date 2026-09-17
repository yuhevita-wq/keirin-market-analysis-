#!/usr/bin/env python3
from __future__ import annotations

"""Real KDreams runtime smoke tests for deterministic v21/v31/v37 placement.

Checks two production requirements:
1. A known finished five-rider Challenge race can be placed mechanically.
2. A known seven-rider race that v21 marks as skip still receives a full board.

Only race card and published line forecast are passed to the engine. Target-race
odds, popularity, result and payout fields are not passed to the engine.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIVE_RIDER = ("2720260610020001", "https://keirin.kdreams.jp/keiokaku/racedetail/2720260610020001/")
V21_SKIP = ("1320260915030001", "https://keirin.kdreams.jp/iwakitaira/racedetail/1320260915030001/")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_race(cache, runtime, session, race_id: str, race_url: str) -> dict:
    html = cache.fetch_html(session, race_url)
    meta, entries = cache.parse_entries(html, race_id, race_url)
    line = cache.parse_line_formation_html(html)
    cache.attach_line(entries, line)
    return {
        **meta,
        "line_status": line.get("status", ""),
        "predicted_line_formation": line.get("formation", ""),
        "line_provider": line.get("provider", ""),
        "entries": [runtime.normalize_entry(entry) for entry in entries],
    }


def compact(board: dict, race: dict) -> dict:
    return {
        "race_id": board.get("race_id"),
        "race_date": race.get("race_date", ""),
        "race_type": race.get("race_type", ""),
        "entry_count": len(race.get("entries", [])),
        "scope_ok": bool(board.get("scope_ok")),
        "board_generated": bool(board.get("board_generated")),
        "participate": bool(board.get("participate")),
        "first_candidates": board.get("first_candidates", []),
        "second_candidates": board.get("second_candidates", []),
        "third_candidates": board.get("third_candidates", []),
        "skip_reason": board.get("skip_reason", ""),
        "scope_note": board.get("scope_note", ""),
    }


def main() -> int:
    cache = load("keirin_shogi_live_cache_for_smoke", ROOT / "scripts/keirin_shogi_live_race_cache.py")
    runtime = load("keirin_shogi_runtime_for_smoke", ROOT / "scripts/keirin_shogi_v37_auto_place_runtime.py")
    engine = runtime.load_base_engine()
    session = cache.make_session()

    v21_state = engine.build_v21_state()
    pair_model = engine.build_pair_model()
    third_model, feature_names, freeze = engine.build_third_model()

    five = load_race(cache, runtime, session, *FIVE_RIDER)
    if len(five["entries"]) != 5:
        raise RuntimeError(f"expected real five-rider card, got {len(five['entries'])} riders")
    five_board = runtime.runtime_place_one(
        engine, five, v21_state, pair_model, third_model, feature_names, freeze
    )
    if not five_board.get("scope_ok") or not five_board.get("board_generated"):
        raise RuntimeError(f"five-rider placement failed: {five_board}")
    if not all(five_board.get(key) for key in ("first_candidates", "second_candidates", "third_candidates")):
        raise RuntimeError(f"five-rider board has an empty row: {five_board}")

    skipped = load_race(cache, runtime, session, *V21_SKIP)
    if len(skipped["entries"]) != 7:
        raise RuntimeError(f"expected real seven-rider card, got {len(skipped['entries'])} riders")
    skip_board = runtime.runtime_place_one(
        engine, skipped, v21_state, pair_model, third_model, feature_names, freeze
    )
    if skip_board.get("participate"):
        raise RuntimeError(f"regression race no longer exercises v21 skip: {skip_board}")
    if not skip_board.get("board_generated"):
        raise RuntimeError(f"v21 skip blanked board generation: {skip_board}")
    if not all(skip_board.get(key) for key in ("first_candidates", "second_candidates", "third_candidates")):
        raise RuntimeError(f"v21 skip produced an empty board row: {skip_board}")
    if skip_board.get("skip_reason") != "v21参加判定で見送り（盤面は配置）":
        raise RuntimeError(f"unexpected skip status: {skip_board.get('skip_reason')}")

    print(json.dumps({
        "smoke": "keirin_shogi_runtime",
        "five_rider": compact(five_board, five),
        "v21_skip_but_board_present": compact(skip_board, skipped),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

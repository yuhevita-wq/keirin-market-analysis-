#!/usr/bin/env python3
from __future__ import annotations

"""Refresh live race cache through the variable-count v21/v31/v37 runtime."""

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CACHE = ROOT / "scripts/keirin_shogi_live_race_cache.py"
RUNTIME_ENGINE = ROOT / "scripts/keirin_shogi_v37_auto_place_runtime.py"


def load_base_cache():
    spec = importlib.util.spec_from_file_location("keirin_shogi_live_race_cache_base", BASE_CACHE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def add_board_diagnostics(cache) -> dict[str, object]:
    if not cache.OUT.exists() or not cache.BOARD_OUT.exists():
        return {}
    payload = json.loads(cache.OUT.read_text(encoding="utf-8"))
    board = json.loads(cache.BOARD_OUT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(board, dict):
        return {}

    races = [row for row in payload.get("races", []) if isinstance(row, dict)]
    board_rows = [row for row in board.get("races", []) if isinstance(row, dict)]
    board_failures = [row for row in board.get("failures", []) if isinstance(row, dict)]
    by_id = {str(row.get("race_id", "")): row for row in board_rows}
    race_by_id = {str(row.get("race_id", "")): row for row in races}

    summary = defaultdict(lambda: {"input": 0, "board": 0, "participate": 0, "failure": 0})
    for race in races:
        count = str(len(race.get("entries", [])))
        summary[count]["input"] += 1
        board_row = by_id.get(str(race.get("race_id", "")))
        if board_row is not None:
            summary[count]["board"] += 1
            if board_row.get("participate"):
                summary[count]["participate"] += 1
    for failure in board_failures:
        race = race_by_id.get(str(failure.get("race_id", "")), {})
        count = str(len(race.get("entries", []))) if race else "unknown"
        summary[count]["failure"] += 1

    engine = dict(payload.get("board_engine", {}))
    engine.update(
        {
            "runtime_compatibility": "variable_rider_count",
            "input_race_count": len(races),
            "placed_race_count": len(board_rows),
            "missing_auto_board_count": sum(
                1 for race in races if str(race.get("race_id", "")) not in by_id
            ),
            "failures": board_failures[:50],
            "by_entry_count": {key: summary[key] for key in sorted(summary)},
        }
    )
    payload["board_engine"] = engine

    existing_failures = [row for row in payload.get("failures", []) if isinstance(row, dict)]
    known = {(row.get("stage"), row.get("race_id"), row.get("error")) for row in existing_failures}
    for failure in board_failures:
        row = {
            "stage": "auto_place_race",
            "race_id": str(failure.get("race_id", "")),
            "url": "",
            "error": str(failure.get("error", "")),
        }
        key = (row["stage"], row["race_id"], row["error"])
        if key not in known:
            existing_failures.append(row)
            known.add(key)
    payload["failure_count"] = len(existing_failures)
    payload["failures"] = existing_failures[:100]

    cache.OUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return engine


def main() -> int:
    cache = load_base_cache()
    cache.AUTO_PLACE = RUNTIME_ENGINE
    result = int(cache.main())
    engine = add_board_diagnostics(cache)
    if engine:
        print(
            json.dumps(
                {
                    "board_engine_status": engine.get("status", "missing"),
                    "input_race_count": engine.get("input_race_count", 0),
                    "placed_race_count": engine.get("placed_race_count", 0),
                    "missing_auto_board_count": engine.get("missing_auto_board_count", 0),
                    "board_failure_count": engine.get("failure_count", 0),
                    "by_entry_count": engine.get("by_entry_count", {}),
                },
                ensure_ascii=False,
            )
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())

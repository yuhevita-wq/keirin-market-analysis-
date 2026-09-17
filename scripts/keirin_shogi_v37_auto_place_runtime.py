#!/usr/bin/env python3
from __future__ import annotations

"""Runtime compatibility layer for deterministic v21/v31/v37 placement.

The frozen training/model code remains untouched. This layer only normalizes
live KDreams input types and broadens the runtime entrance from seven riders to
any race with enough riders to form a top three.
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
    try:
        return int(engine.main())
    finally:
        try:
            TEMP_LIVE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

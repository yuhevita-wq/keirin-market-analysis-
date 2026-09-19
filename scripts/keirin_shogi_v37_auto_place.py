#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility entry point for the live Keirin Shogi board cache.

The historical filename is kept because keirin_shogi_live_race_cache.py calls
this path.  Live 7-rider prediction itself has moved to the unified joint-order
engine in keirin_shogi_unified_v1.py.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts/keirin_shogi_unified_v1.py"


def load_unified():
    spec = importlib.util.spec_from_file_location("keirin_shogi_unified_v1", TARGET)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(load_unified().main())

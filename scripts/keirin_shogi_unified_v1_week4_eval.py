#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/keirin_shogi_unified_v1_week1_eval.py"

spec = importlib.util.spec_from_file_location("unified_v1_week4_base", BASE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

module.WEEK_START = date(2026, 7, 27)
module.WEEK_END = date(2026, 8, 2)

if __name__ == "__main__":
    module.main()

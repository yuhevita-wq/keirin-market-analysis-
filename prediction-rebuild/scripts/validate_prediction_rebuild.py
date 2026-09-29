#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "raw": ["race_id", "result_seen", "sources"],
    "pre_race": ["race_id", "result_seen", "causal_edges", "cut_edges", "surviving_edges", "reconnected_edges", "final_structure"],
    "results": ["race_id", "finish_order"],
    "evaluations": ["race_id", "core_survival", "cut_edges", "reconnected_edges", "outcome"],
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_file(kind: str, path: Path):
    data = load_json(path)
    errors = []
    for key in REQUIRED[kind]:
        if key not in data:
            errors.append(f"missing key: {key}")

    if kind in {"raw", "pre_race"} and data.get("result_seen") is not False:
        errors.append("result_seen must be false before the race")

    if kind == "pre_race":
        constraints = data.get("constraints", {})
        if constraints.get("free_generation_forbidden") is not True:
            errors.append("free_generation_forbidden must be true")
        if constraints.get("use_only_collected_material") is not True:
            errors.append("use_only_collected_material must be true")
        if constraints.get("preserve_surviving_core") is not True:
            errors.append("preserve_surviving_core must be true")

    return errors


def main():
    failed = 0
    checked = 0

    for kind in REQUIRED:
        base = ROOT / kind
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            checked += 1
            try:
                errors = validate_file(kind, path)
            except Exception as exc:
                errors = [f"invalid JSON: {exc}"]
            if errors:
                failed += 1
                print(f"FAIL {path.relative_to(ROOT)}")
                for err in errors:
                    print(f"  - {err}")
            else:
                print(f"OK   {path.relative_to(ROOT)}")

    print(f"checked={checked} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

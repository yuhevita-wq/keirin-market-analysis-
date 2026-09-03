from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from keirin_prediction_engine import (
    DatasetSplit,
    build_race_table,
    chronological_split,
    engine_manifest,
    read_csv,
    validate_pre_race_frame,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leakage-safe race structures for the prediction engine")
    p.add_argument("--races", required=True)
    p.add_argument("--entries", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    races = read_csv(args.races)
    entries = read_csv(args.entries)
    validate_pre_race_frame(races)
    validate_pre_race_frame(entries)
    race_table = build_race_table(entries, races)
    splits = chronological_split(race_table, DatasetSplit())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    race_table.to_csv(out_dir / "race_structure.csv", index=False)
    for name, df in splits.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(engine_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {name: int(len(df)) for name, df in splits.items()}
    print(json.dumps({"race_rows": int(len(race_table)), "splits": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()

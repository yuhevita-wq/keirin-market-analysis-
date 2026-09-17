#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_LOG = ROOT / "results/keirin_shogi/v37_shrunk_board_third/race_log.json"
FUTURE_LOG = ROOT / "results/keirin_shogi/v37_true_future_block1/race_log.json"
ADOPTION = ROOT / "results/keirin_shogi/v37_adoption/summary.json"
OUT = ROOT / "docs/keirin-shogi/v37-board-data.json"

RACE_FILES = [
    ROOT / "data/2025/s_class_f1_all_parts/2025_q4/races.csv",
    ROOT / "data/2026_h1/s_class_f1_all/races.csv",
    ROOT / "data/2026_future_block1/s_class_f1_20260701_20260830/races.csv",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_meta():
    output = {}
    for path in RACE_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                output[row["race_id"]] = {
                    "race_date": row.get("race_date", ""),
                    "track": row.get("track", ""),
                    "race_no": row.get("race_no", ""),
                    "race_type": row.get("race_type", ""),
                    "meeting_grade": row.get("meeting_grade", ""),
                }
    return output


def main():
    for path in (DEVELOPMENT_LOG, FUTURE_LOG, ADOPTION):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    sources = [
        ("reopened_2026_h1_development", load_json(DEVELOPMENT_LOG)),
        ("true_future_block1", load_json(FUTURE_LOG)),
    ]
    meta = load_meta()
    adoption = load_json(ADOPTION)
    races_by_id = {}
    for source_block, rows in sources:
        for row in rows:
            race_id = str(row["race_id"])
            race_meta = meta.get(race_id, {})
            races_by_id[race_id] = {
                "race_id": race_id,
                "race_date": race_meta.get("race_date") or row.get("race_date", ""),
                "track": race_meta.get("track", ""),
                "race_no": race_meta.get("race_no", ""),
                "race_type": race_meta.get("race_type") or row.get("race_type", ""),
                "meeting_grade": race_meta.get("meeting_grade", ""),
                "participate": True,
                "first_candidates": list(map(int, row.get("first_candidates", []))),
                "second_candidates": list(map(int, row.get("second_candidates", []))),
                "third_candidates": list(map(int, row.get("third_candidates", []))),
                "second_membership": row.get("top2_membership", []),
                "top_pairs": [
                    {
                        "a": int(pair["a"]),
                        "b": int(pair["b"]),
                        "prob": float(pair.get("prob", pair.get("probability", 0.0))),
                    }
                    for pair in row.get("top_pairs", [])
                ],
                "third_ranking": row.get("third_ranking", []),
                "source_block": source_block,
                "versions": {
                    "participation_first": "v21_quantile_participation",
                    "second": "v31_top2_membership",
                    "third": "v37_shrunk_board_third",
                },
            }

    races = sorted(
        races_by_id.values(),
        key=lambda row: (
            row["race_date"],
            str(row["track"]),
            int(row["race_no"] or 0),
            row["race_id"],
        ),
    )
    payload = {
        "schema_version": 2,
        "board_version": "v37",
        "adoption_status": adoption["decision"],
        "freeze_commit": adoption["freeze_commit"],
        "generated_from": {
            "first": "v21 Quantile Participation",
            "second": "v31 Top2 Membership",
            "third": "v37 Top2-conditioned third with gamma=0.25 shrinkage",
        },
        "note": (
            "現行最良の暫定採用盤面。同一完全未来153レースで旧v23より3着・完全盤面を改善。"
            "統計的確証のため次の未開封ブロック検証は継続する。"
        ),
        "race_count": len(races),
        "missing_third_count": sum(not row["third_candidates"] for row in races),
        "future_comparison": {
            "races": adoption["races"],
            "v37": adoption["v37"],
            "legacy_v23": adoption["legacy_v23_same_board"],
            "delta": adoption["v37_minus_v23"],
        },
        "races": races,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(OUT),
                "race_count": len(races),
                "missing_third_count": payload["missing_third_count"],
                "adoption_status": payload["adoption_status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

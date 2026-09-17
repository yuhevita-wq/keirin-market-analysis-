#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V31_LOG = ROOT / "results/keirin_shogi/v31_top2_membership_second/race_log.json"
V23_LOG = ROOT / "results/keirin_shogi/v23_conditional_second_third/race_log.json"
OUT = ROOT / "docs/keirin-shogi/v31-board-data.json"

RACE_FILES = [
    ROOT / "data/2025/s_class_f1_all_parts/2025_q4/races.csv",
    ROOT / "data/2026_h1/s_class_f1_all/races.csv",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_meta():
    meta = {}
    for path in RACE_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                meta[row["race_id"]] = {
                    "race_date": row.get("race_date", ""),
                    "track": row.get("track", ""),
                    "race_no": row.get("race_no", ""),
                    "race_type": row.get("race_type", ""),
                    "meeting_grade": row.get("meeting_grade", ""),
                }
    return meta


def main():
    if not V31_LOG.exists():
        raise SystemExit(f"missing {V31_LOG}")
    if not V23_LOG.exists():
        raise SystemExit(f"missing {V23_LOG}")

    v31 = load_json(V31_LOG)
    v23 = {r["race_id"]: r for r in load_json(V23_LOG)}
    meta = load_meta()

    races = []
    missing_third = 0
    for row in v31:
        rid = row["race_id"]
        old3 = v23.get(rid)
        third = list(old3.get("third_candidates", [])) if old3 else []
        if not third:
            missing_third += 1

        m = meta.get(rid, {})
        membership = row.get("membership", [])
        top_pairs = row.get("top_pairs", [])

        races.append({
            "race_id": rid,
            "race_date": m.get("race_date") or row.get("race_date", ""),
            "track": m.get("track", ""),
            "race_no": m.get("race_no", ""),
            "race_type": m.get("race_type", ""),
            "meeting_grade": m.get("meeting_grade", ""),
            "participate": True,
            "first_candidates": list(row.get("first_candidates", [])),
            "second_candidates": list(row.get("second_candidates", [])),
            "third_candidates": third,
            "second_membership": membership,
            "top_pairs": top_pairs,
            "versions": {
                "participation_first": "v21",
                "second": "v31_top2_membership",
                "third": "v23_legacy_until_v32",
            },
        })

    races.sort(key=lambda r: (r["race_date"], str(r["track"]), int(r["race_no"] or 0), r["race_id"]))

    payload = {
        "schema_version": 1,
        "board_version": "v31",
        "generated_from": {
            "first": "v21 adopted core",
            "second": "v31 Top2 Membership",
            "third": "v23 legacy conditional third (temporary until v32)",
        },
        "note": "v31時点の盤面データ。1着=v21、2着=v31、3着はv32完成までv23を暫定使用。",
        "race_count": len(races),
        "missing_third_count": missing_third,
        "races": races,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "race_count": len(races),
        "missing_third_count": missing_third,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

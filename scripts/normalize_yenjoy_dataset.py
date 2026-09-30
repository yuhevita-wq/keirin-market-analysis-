from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

RACE_RE = re.compile(r"(?P<race>\d+)\s*R", re.IGNORECASE)
LEADING_CAR_RE = re.compile(r"^\s*(\d{1,2})\s+(.*)$")

FIELD_PREFIXES = {
    "レース": "race_label",
    "車番": "car_no",
    "選手名": "racer",
    "府県": "prefecture",
    "ギア": "gear",
    "年齢": "age",
    "期別": "term",
    "級班": "class",
    "脚力": "power_index",
    "ランク": "power_rank",
    "得点": "score",
    "前期": "prev_score",
    "今期": "current_score",
    "勝率": "win_rate",
    "２連": "top2_rate",
    "３連": "top3_rate",
}


def read_jsonl_gz(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["day_id"]] = rec
    return out


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    m = re.search(r"-?\d+", str(value).replace(",", ""))
    return int(m.group()) if m else None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(m.group()) if m else None


def field_key(header: str, index: int) -> str:
    for prefix, key in FIELD_PREFIXES.items():
        if header.startswith(prefix):
            return key
    return f"field_{index:02d}"


def normalize_player(headers: list[str], row: list[str]) -> dict[str, Any]:
    player: dict[str, Any] = {}
    for i, header in enumerate(headers):
        if i >= len(row):
            continue
        player[field_key(header, i)] = row[i]
    if "car_no" in player:
        player["car_no"] = parse_int(player["car_no"])
    if "age" in player:
        player["age"] = parse_int(player["age"])
    if "term" in player:
        player["term"] = parse_int(player["term"])
    for key in ("gear", "score", "prev_score", "current_score"):
        if key in player:
            player[key] = parse_float(player[key])
    for key in ("power_index", "power_rank", "win_rate", "top2_rate", "top3_rate"):
        if key in player:
            player[key] = parse_int(player[key])
    return player


def legacy_entry_map(entries_day: dict[str, Any]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    tables = entries_day.get("tables") or []
    if not tables:
        return {}
    rows = tables[0].get("rows") or []
    if len(rows) < 2:
        return {}
    headers = rows[0]
    race_idx = next((i for i, h in enumerate(headers) if h.startswith("レース")), None)
    if race_idx is None:
        return {}
    for row in rows[1:]:
        if race_idx >= len(row):
            continue
        m = RACE_RE.search(row[race_idx])
        if not m:
            continue
        race_no = int(m.group("race"))
        player = normalize_player(headers, row)
        player["race_no"] = race_no
        grouped[race_no].append(player)
    return {
        race_no: {
            "race_no": race_no,
            "entrants": sorted(
                players,
                key=lambda x: (x.get("car_no") is None, x.get("car_no") or 99),
            ),
            "initial_line_raw": None,
            "final_bs_raw": None,
        }
        for race_no, players in grouped.items()
    }


def entry_map(entries_day: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not entries_day:
        return {}
    races = entries_day.get("races") or []
    if races:
        return {
            int(r["race_no"]): r
            for r in races
            if r.get("race_no") is not None
        }
    return legacy_entry_map(entries_day)


def normalize_finish(finish: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not finish:
        return []
    rows = finish.get("rows") or []
    if len(rows) < 2:
        return []
    labels, values = rows[0], rows[1]
    out: list[dict[str, Any]] = []
    for label, value in zip(labels, values):
        item: dict[str, Any] = {"status": label, "raw": value}
        m = LEADING_CAR_RE.match(value or "")
        if m:
            item["car_no"] = int(m.group(1))
            item["racer_raw"] = m.group(2)
        else:
            item["car_no"] = parse_int(value)
        out.append(item)
    return out


def stable_race_id(day: dict[str, Any], race_no: int) -> str:
    return f"{day['race_date']}_{day['venue_code']}_{day['start_date']}_R{race_no:02d}"


def normalize_day(
    entries_day: dict[str, Any] | None,
    results_day: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    base = results_day or entries_day
    if base is None:
        return []

    entries = entry_map(entries_day)
    results = {
        int(r["race_no"]): r
        for r in (results_day or {}).get("races", [])
        if r.get("race_no") is not None
    }
    race_numbers = sorted(set(entries) | set(results))
    output: list[dict[str, Any]] = []

    for race_no in race_numbers:
        entry = entries.get(race_no) or {}
        entrants = entry.get("entrants") or []
        result = results.get(race_no)
        finish = normalize_finish(result.get("finish") if result else None)
        top3 = [x.get("car_no") for x in finish[:3] if x.get("car_no") is not None]
        output.append(
            {
                "schema_version": 2,
                "source": "yenjoy",
                "race_id": stable_race_id(base, race_no),
                "race_date": base["race_date"],
                "venue_code": base["venue_code"],
                "start_date": base["start_date"],
                "day_id": base["day_id"],
                "race_no": race_no,
                "meeting_title": base.get("title", ""),
                "entrants": entrants,
                "entrant_count": len(entrants),
                "initial_line_raw": entry.get("initial_line_raw"),
                "final_bs_raw": entry.get("final_bs_raw"),
                "finish": finish,
                "top3_car_no": top3,
                "winner_car_no": top3[0] if top3 else None,
                "payout_rows": ((result or {}).get("payout") or {}).get("rows", []),
                "result_detail_url": (result or {}).get("result_detail_url"),
                "quality": {
                    "has_entries": bool(entrants),
                    "has_result": bool(result),
                    "has_top3": len(top3) >= 3,
                    "entrant_result_count_match": bool(entrants)
                    and bool(finish)
                    and len(entrants) == len(finish),
                },
            }
        )
    return output


def main() -> int:
    p = argparse.ArgumentParser(description="Normalize YenJoy day records into race-level joined records")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--start-month", required=True)
    p.add_argument("--end-month", required=True)
    args = p.parse_args()

    root = Path(args.input_dir)
    entries_path = root / f"entries_{args.start_month}_{args.end_month}.jsonl.gz"
    results_path = root / f"results_{args.start_month}_{args.end_month}.jsonl.gz"
    if not entries_path.exists() or not results_path.exists():
        raise SystemExit(f"missing inputs: {entries_path} / {results_path}")

    entries = read_jsonl_gz(entries_path)
    results = read_jsonl_gz(results_path)
    day_ids = sorted(set(entries) | set(results))
    joined_path = root / f"joined_{args.start_month}_{args.end_month}.jsonl.gz"

    counts = defaultdict(int)
    with gzip.open(joined_path, "wt", encoding="utf-8") as out:
        for day_id in day_ids:
            for race in normalize_day(entries.get(day_id), results.get(day_id)):
                out.write(json.dumps(race, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["race_count"] += 1
                for key in (
                    "has_entries",
                    "has_result",
                    "has_top3",
                    "entrant_result_count_match",
                ):
                    counts[key] += int(race["quality"].get(key, False))

    manifest = {
        "schema_version": 2,
        "source": "yenjoy",
        "start_month": args.start_month,
        "end_month": args.end_month,
        "day_count": len(day_ids),
        "race_count": counts["race_count"],
        "races_with_entries": counts["has_entries"],
        "races_with_results": counts["has_result"],
        "races_with_top3": counts["has_top3"],
        "races_with_matching_entry_result_counts": counts["entrant_result_count_match"],
        "joined_file": joined_path.name,
    }
    (root / "normalized_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

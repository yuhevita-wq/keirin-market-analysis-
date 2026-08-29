from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

from .s_yosen_2025 import (
    KDREAMS_DAILY,
    TARGET_RACE_TYPE,
    RaceRef,
    daterange,
    discover_s_yosen_from_daily_html,
    extract_entries,
    fetch_html,
    make_session,
    write_csv,
)

H1_START = date(2026, 1, 1)
H1_END = date(2026, 6, 30)


def collect(start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    """Collect 2026 H1 races using the exact same parsing logic as the frozen 2025 collector."""
    if start != H1_START or end != H1_END:
        raise ValueError("this collector is intentionally restricted to 2026-01-01 through 2026-06-30")
    if start > end:
        raise ValueError("start date must be <= end date")

    session = make_session()
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    discovered: dict[str, RaceRef] = {}
    skipped_outside_window = 0

    for day in daterange(start, end):
        daily_url = KDREAMS_DAILY.format(year=day.year, month=day.month, day=day.day)
        try:
            html = fetch_html(session, daily_url)
            refs = discover_s_yosen_from_daily_html(html, daily_url, day.isoformat())
            for ref in refs:
                discovered.setdefault(ref.url, ref)
        except Exception as exc:
            failures.append(
                {
                    "stage": "daily_discovery",
                    "discovered_on": day.isoformat(),
                    "url": daily_url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    for ref in sorted(discovered.values(), key=lambda item: (item.discovered_on, item.url)):
        try:
            html = fetch_html(session, ref.url)
            race_meta, race_entries = extract_entries(html, ref)
            actual_date = date.fromisoformat(str(race_meta["race_date"]))
            if actual_date < start or actual_date > end:
                skipped_outside_window += 1
                continue
            races.append(race_meta)
            entries.extend(race_entries)
        except Exception as exc:
            failures.append(
                {
                    "stage": "race_parse",
                    "discovered_on": ref.discovered_on,
                    "url": ref.url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    races.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"])))
    entries.sort(
        key=lambda row: (
            str(row["race_date"]),
            str(row["track"]),
            int(row["race_no"]),
            int(row["car_no"]),
        )
    )

    race_fields = [
        "race_id", "race_date", "track", "race_no", "race_type", "start_time",
        "deadline", "entry_count", "source_url", "captured_at_utc",
    ]
    entry_fields = [
        "race_id", "race_date", "track", "race_no", "race_type", "start_time",
        "deadline", "source_url", "captured_at_utc", "car_no", "player_name",
        "player_profile", "prefecture", "age", "term", "class", "style", "gear",
        "score", "s_count", "b_count", "nige_count", "makuri_count", "sashi_count",
        "mark_count", "first_count", "second_count", "third_count", "outside_count",
        "win_rate", "top2_rate", "top3_rate", "prediction_mark", "evaluation", "raw_row_json",
    ]
    failure_fields = ["stage", "discovered_on", "url", "error"]

    write_csv(out_dir / "races.csv", races, race_fields)
    write_csv(out_dir / "entries.csv", entries, entry_fields)
    write_csv(out_dir / "failures.csv", failures, failure_fields)

    summary = {
        "target": TARGET_RACE_TYPE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "daily_source_template": KDREAMS_DAILY,
        "candidate_races": len(discovered),
        "parsed_races": len(races),
        "entry_rows": len(entries),
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "definition_note": "レース種別表記が完全一致する『Ｓ級予選』のみ。Ｓ級予選１/２、一次予選、特別選抜予選等は含めない。",
        "oos_note": "2026 H1 is the forward out-of-sample evaluation window for strategies developed on 2023-2025. Collection/parsing logic is inherited unchanged from the frozen 2025 collector. Strategy thresholds/formations must not be altered after reading 2026 H1 outcomes.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect exact-label S-class preliminary race cards for 2026 H1")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--out-dir", default="data/2026_h1/s_class_yosen")
    parser.add_argument("--sleep", type=float, default=0.35, help="polite delay between requests")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="return success even if discovery/parse failures were recorded",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = collect(
        date.fromisoformat(args.start_date),
        date.fromisoformat(args.end_date),
        Path(args.out_dir),
        args.sleep,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["parsed_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from . import s_yosen_2025 as base


@dataclass(frozen=True)
class SClassRaceRef:
    discovered_on: str
    race_no: int
    race_type: str
    url: str


def discover_s_class_from_daily_html(html: str, daily_url: str, discovered_on: str) -> list[SClassRaceRef]:
    """Return all published race-card links whose label starts with the full-width 'Ｓ級'."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, SClassRaceRef] = {}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row_index, row in enumerate(rows):
            expanded = base.expand_row_cells(row)
            if not expanded:
                continue
            labels = [base.normalize_text(cell.get_text(" ", strip=True)) for cell in expanded]
            target_indexes = [i for i, label in enumerate(labels) if label.startswith("Ｓ級")]
            for target_index in target_indexes:
                race_type = labels[target_index]
                href = None
                for later in rows[row_index + 1 :]:
                    later_cells = base.expand_row_cells(later)
                    if target_index >= len(later_cells):
                        continue
                    for link in later_cells[target_index].find_all("a", href=True):
                        candidate = str(link.get("href", ""))
                        if "/racedetail/" in candidate:
                            href = candidate
                            break
                    if href:
                        break
                if href is None:
                    continue

                url = base.canonical_race_url(daily_url, href)
                import re
                match = re.search(r"/racedetail/(\d{16})/", url)
                race_no = int(match.group(1)[-4:]) if match else target_index + 1
                found[url] = SClassRaceRef(
                    discovered_on=discovered_on,
                    race_no=race_no,
                    race_type=race_type,
                    url=url,
                )

    return sorted(found.values(), key=lambda item: (item.discovered_on, item.url, item.race_no))


def extract_entries_for_ref(html: str, ref: SClassRaceRef):
    """Reuse the frozen race-card parser while supplying the actually published S-class label."""
    base.TARGET_RACE_TYPE = ref.race_type
    frozen_ref = base.RaceRef(discovered_on=ref.discovered_on, race_no=ref.race_no, url=ref.url)
    return base.extract_entries(html, frozen_ref)


def collect(start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    if start.year != 2023 or end.year != 2023:
        raise ValueError("this collector is intentionally restricted to 2023")
    if start > end:
        raise ValueError("start date must be <= end date")

    session = base.make_session()
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    discovered: dict[str, SClassRaceRef] = {}
    skipped_outside_window = 0
    skipped_non_7car = 0

    for day in base.daterange(start, end):
        daily_url = base.KDREAMS_DAILY.format(year=day.year, month=day.month, day=day.day)
        try:
            html = base.fetch_html(session, daily_url)
            for ref in discover_s_class_from_daily_html(html, daily_url, day.isoformat()):
                discovered.setdefault(ref.url, ref)
        except Exception as exc:
            failures.append({
                "stage": "daily_discovery",
                "discovered_on": day.isoformat(),
                "url": daily_url,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if sleep_seconds:
            time.sleep(sleep_seconds)

    for ref in sorted(discovered.values(), key=lambda item: (item.discovered_on, item.url)):
        try:
            html = base.fetch_html(session, ref.url)
            race_meta, race_entries = extract_entries_for_ref(html, ref)
            actual_date = date.fromisoformat(str(race_meta["race_date"]))
            if actual_date < start or actual_date > end:
                skipped_outside_window += 1
                continue
            if int(race_meta["entry_count"]) != 7:
                skipped_non_7car += 1
                continue
            races.append(race_meta)
            entries.extend(race_entries)
        except Exception as exc:
            failures.append({
                "stage": "race_parse",
                "discovered_on": ref.discovered_on,
                "url": ref.url,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if sleep_seconds:
            time.sleep(sleep_seconds)

    races.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"])))
    entries.sort(key=lambda row: (
        str(row["race_date"]), str(row["track"]), int(row["race_no"]), int(row["car_no"])
    ))

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

    base.write_csv(out_dir / "races.csv", races, race_fields)
    base.write_csv(out_dir / "entries.csv", entries, entry_fields)
    base.write_csv(out_dir / "failures.csv", failures, failure_fields)

    type_counts = dict(sorted(Counter(str(r["race_type"]) for r in races).items()))
    summary = {
        "target": "2023 S-class exactly 7-car, all published S-class race labels",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "daily_source_template": base.KDREAMS_DAILY,
        "candidate_s_class_races": len(discovered),
        "parsed_7car_races": len(races),
        "entry_rows": len(entries),
        "skipped_non_7car": skipped_non_7car,
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "race_type_counts": type_counts,
        "definition_note": "日別開催ページでレース種別表記が『Ｓ級』で始まる全レースを候補にし、出走表を解析後、出走人数が正確に7人のレースだけを保存する。予選・一次予選・二次予選・準決勝・決勝・一般・選抜・特選等をレース種別名のまま保持する。",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all 2023 S-class races with exactly seven starters")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--out-dir", default="data/2023/s_class_7car_all")
    parser.add_argument("--sleep", type=float, default=0.20, help="polite delay between requests")
    parser.add_argument("--allow-failures", action="store_true")
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
    if summary["parsed_7car_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

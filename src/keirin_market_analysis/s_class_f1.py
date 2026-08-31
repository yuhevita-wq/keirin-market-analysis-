from __future__ import annotations

import argparse
import json
import re
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


def normalize_track_name(value: str) -> str:
    value = base.normalize_text(value)
    value = re.sub(r"競輪$", "", value)
    return value.strip()


def discover_f1_tracks_from_daily_html(html: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    text = base.normalize_text(soup.get_text(" ", strip=True))
    found = set()
    for match in re.finditer(r"([^\s]+?)Ｆ１", text):
        token = match.group(1)
        token = re.split(r"[、。・|｜:：/／]", token)[-1]
        token = normalize_track_name(token)
        if token:
            found.add(token)
    return found


def discover_s_class_from_daily_html(html: str, daily_url: str, discovered_on: str) -> list[SClassRaceRef]:
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
                match = re.search(r"/racedetail/(\d{16})/", url)
                race_no = int(match.group(1)[-4:]) if match else target_index + 1
                found[url] = SClassRaceRef(discovered_on=discovered_on, race_no=race_no, race_type=race_type, url=url)
    return sorted(found.values(), key=lambda item: (item.discovered_on, item.url, item.race_no))


def extract_entries_for_ref(html: str, ref: SClassRaceRef):
    base.TARGET_RACE_TYPE = ref.race_type
    frozen_ref = base.RaceRef(discovered_on=ref.discovered_on, race_no=ref.race_no, url=ref.url)
    return base.extract_entries(html, frozen_ref)


def collect(start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    if start > end:
        raise ValueError("start date must be <= end date")
    if start.year < 2023 or end.year > 2026:
        raise ValueError("collector is restricted to the 2023-2026 research window")

    session = base.make_session()
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    discovered: dict[str, SClassRaceRef] = {}
    f1_tracks_by_day: dict[str, set[str]] = {}
    skipped_outside_window = 0
    skipped_non_f1 = 0

    for day in base.daterange(start, end):
        daily_url = base.KDREAMS_DAILY.format(year=day.year, month=day.month, day=day.day)
        try:
            html = base.fetch_html(session, daily_url)
            f1_tracks_by_day[day.isoformat()] = discover_f1_tracks_from_daily_html(html)
            for ref in discover_s_class_from_daily_html(html, daily_url, day.isoformat()):
                discovered.setdefault(ref.url, ref)
        except Exception as exc:
            failures.append({"stage": "daily_discovery", "discovered_on": day.isoformat(), "url": daily_url, "error": f"{type(exc).__name__}: {exc}"})
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
            track = normalize_track_name(str(race_meta["track"]))
            f1_tracks = f1_tracks_by_day.get(ref.discovered_on, set())
            if track not in f1_tracks:
                skipped_non_f1 += 1
                continue
            race_meta["meeting_grade"] = "F1"
            for entry in race_entries:
                entry["meeting_grade"] = "F1"
            races.append(race_meta)
            entries.extend(race_entries)
        except Exception as exc:
            failures.append({"stage": "race_parse", "discovered_on": ref.discovered_on, "url": ref.url, "error": f"{type(exc).__name__}: {exc}"})
        if sleep_seconds:
            time.sleep(sleep_seconds)

    races.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"])))
    entries.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"]), int(row["car_no"])))

    race_fields = ["race_id", "race_date", "track", "meeting_grade", "race_no", "race_type", "start_time", "deadline", "entry_count", "source_url", "captured_at_utc"]
    entry_fields = ["race_id", "race_date", "track", "meeting_grade", "race_no", "race_type", "start_time", "deadline", "source_url", "captured_at_utc", "car_no", "player_name", "player_profile", "prefecture", "age", "term", "class", "style", "gear", "score", "s_count", "b_count", "nige_count", "makuri_count", "sashi_count", "mark_count", "first_count", "second_count", "third_count", "outside_count", "win_rate", "top2_rate", "top3_rate", "prediction_mark", "evaluation", "raw_row_json"]
    failure_fields = ["stage", "discovered_on", "url", "error"]

    base.write_csv(out_dir / "races.csv", races, race_fields)
    base.write_csv(out_dir / "entries.csv", entries, entry_fields)
    base.write_csv(out_dir / "failures.csv", failures, failure_fields)

    summary = {
        "target": f"F1 meetings, all S-class races, {start.isoformat()} to {end.isoformat()}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "daily_source_template": base.KDREAMS_DAILY,
        "candidate_s_class_races": len(discovered),
        "parsed_f1_s_class_races": len(races),
        "entry_rows": len(entries),
        "skipped_non_f1": skipped_non_f1,
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "race_type_counts": dict(sorted(Counter(str(r["race_type"]) for r in races).items())),
        "entry_count_counts": dict(sorted(Counter(str(r["entry_count"]) for r in races).items())),
        "definition_note": "楽天Kドリームス日別開催一覧で開催グレードがＦ１と明示された開催のみを対象とし、その開催内でレース種別表記が『Ｓ級』で始まる全レースを保存する。車立て数では絞らない。",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all S-class races from F1 meetings")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.20)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = collect(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date), Path(args.out_dir), args.sleep)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["parsed_f1_s_class_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from . import s_yosen_2025 as base


@dataclass(frozen=True)
class AClass12RaceRef:
    discovered_on: str
    race_no: int
    race_type: str
    url: str


def normalize_track_name(value: str) -> str:
    value = base.normalize_text(value)
    value = re.sub(r"競輪$", "", value)
    return value.strip()


def normalize_class(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).upper()
    return re.sub(r"\s+", "", value)


def is_a1_a2_class(value: object) -> bool:
    return normalize_class(value) in {"A1", "A2"}


def discover_meeting_grades_from_daily_html(html: str) -> dict[str, str]:
    """Return {track: F1/F2} when the daily page exposes meeting grade text."""
    soup = BeautifulSoup(html, "lxml")
    text = base.normalize_text(soup.get_text(" ", strip=True))
    found: dict[str, str] = {}
    for match in re.finditer(r"([^\s]+?)(Ｆ１|Ｆ２)", text):
        token = match.group(1)
        token = re.split(r"[、。・|｜:：/／]", token)[-1]
        track = normalize_track_name(token)
        if not track:
            continue
        grade = unicodedata.normalize("NFKC", match.group(2)).upper()
        found[track] = grade
    return found


def discover_a_class_from_daily_html(
    html: str,
    daily_url: str,
    discovered_on: str,
) -> list[AClass12RaceRef]:
    """Discover race-card links whose published race label starts with 'Ａ級'."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, AClass12RaceRef] = {}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row_index, row in enumerate(rows):
            expanded = base.expand_row_cells(row)
            if not expanded:
                continue
            labels = [base.normalize_text(cell.get_text(" ", strip=True)) for cell in expanded]
            target_indexes = [i for i, label in enumerate(labels) if label.startswith("Ａ級")]
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
                found[url] = AClass12RaceRef(
                    discovered_on=discovered_on,
                    race_no=race_no,
                    race_type=race_type,
                    url=url,
                )

    return sorted(found.values(), key=lambda item: (item.discovered_on, item.url, item.race_no))


def extract_entries_for_ref(html: str, ref: AClass12RaceRef):
    # Reuse the proven generic race-card parser while preserving the actual A-class label.
    base.TARGET_RACE_TYPE = ref.race_type
    frozen_ref = base.RaceRef(
        discovered_on=ref.discovered_on,
        race_no=ref.race_no,
        url=ref.url,
    )
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
    discovered: dict[str, AClass12RaceRef] = {}
    grades_by_day: dict[str, dict[str, str]] = {}
    skipped_outside_window = 0
    skipped_non_a12 = 0
    skipped_class_sets: Counter[str] = Counter()

    for day in base.daterange(start, end):
        daily_url = base.KDREAMS_DAILY.format(year=day.year, month=day.month, day=day.day)
        try:
            html = base.fetch_html(session, daily_url)
            grades_by_day[day.isoformat()] = discover_meeting_grades_from_daily_html(html)
            for ref in discover_a_class_from_daily_html(html, daily_url, day.isoformat()):
                # The same active meeting can appear on several daily pages. Keep first discovery.
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
            html = base.fetch_html(session, ref.url)
            race_meta, race_entries = extract_entries_for_ref(html, ref)
            actual_date = date.fromisoformat(str(race_meta["race_date"]))
            if actual_date < start or actual_date > end:
                skipped_outside_window += 1
                continue

            classes = sorted({normalize_class(entry.get("class", "")) for entry in race_entries})
            if not classes or any(cls not in {"A1", "A2"} for cls in classes):
                skipped_non_a12 += 1
                skipped_class_sets["/".join(classes) if classes else "(empty)"] += 1
                continue

            track = normalize_track_name(str(race_meta["track"]))
            meeting_grade = grades_by_day.get(ref.discovered_on, {}).get(track, "")
            race_meta["meeting_grade"] = meeting_grade
            race_meta["class_scope"] = "A1/A2"
            for entry in race_entries:
                entry["meeting_grade"] = meeting_grade
                entry["class_scope"] = "A1/A2"

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
        "race_id",
        "race_date",
        "track",
        "meeting_grade",
        "class_scope",
        "race_no",
        "race_type",
        "start_time",
        "deadline",
        "entry_count",
        "source_url",
        "captured_at_utc",
    ]
    entry_fields = [
        "race_id",
        "race_date",
        "track",
        "meeting_grade",
        "class_scope",
        "race_no",
        "race_type",
        "start_time",
        "deadline",
        "source_url",
        "captured_at_utc",
        "car_no",
        "player_name",
        "player_profile",
        "prefecture",
        "age",
        "term",
        "class",
        "style",
        "gear",
        "score",
        "s_count",
        "b_count",
        "nige_count",
        "makuri_count",
        "sashi_count",
        "mark_count",
        "first_count",
        "second_count",
        "third_count",
        "outside_count",
        "win_rate",
        "top2_rate",
        "top3_rate",
        "prediction_mark",
        "evaluation",
        "raw_row_json",
    ]
    failure_fields = ["stage", "discovered_on", "url", "error"]

    base.write_csv(out_dir / "races.csv", races, race_fields)
    base.write_csv(out_dir / "entries.csv", entries, entry_fields)
    base.write_csv(out_dir / "failures.csv", failures, failure_fields)

    summary = {
        "target": f"all A-class races whose entrants are exclusively A1/A2, {start.isoformat()} to {end.isoformat()}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "daily_source_template": base.KDREAMS_DAILY,
        "candidate_a_class_races": len(discovered),
        "parsed_a1_a2_races": len(races),
        "entry_rows": len(entries),
        "skipped_non_a1_a2": skipped_non_a12,
        "skipped_non_a1_a2_class_sets": dict(sorted(skipped_class_sets.items())),
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "race_type_counts": dict(sorted(Counter(str(r["race_type"]) for r in races).items())),
        "meeting_grade_counts": dict(sorted(Counter(str(r["meeting_grade"]) or "unknown" for r in races).items())),
        "entry_count_counts": dict(sorted(Counter(str(r["entry_count"]) for r in races).items())),
        "entrant_class_counts": dict(sorted(Counter(normalize_class(e["class"]) for e in entries).items())),
        "definition_note": "日別開催一覧でレース種別表記が『Ａ級』で始まる候補を取得し、実際の出走表に記載された全選手の級班をNFKC正規化したうえでA1/A2のみのレースを保存する。A3チャレンジ等は級班実データで除外する。F1/F2はどちらも対象。車立て数では絞らない。",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect A-class races containing A1/A2 entrants only")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.20)
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
    if summary["parsed_a1_a2_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

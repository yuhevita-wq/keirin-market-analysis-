from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from . import s_yosen_2025 as base


G3_MARKER_RE = re.compile(r"(?:Ｇ３|G3|ＧⅢ|GIII)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class G3Meeting:
    track: str
    venue_code: str
    slug: str
    start: date
    end: date


@dataclass(frozen=True)
class G3RaceRef:
    discovered_on: str
    race_no: int
    race_type: str
    url: str


# 2024 Q2 G3 schedule, taken from the published Rakuten KDreams 2024 GIII schedule.
# Using meeting IDs avoids crawling F1/F2 race-detail pages.
G3_MEETINGS_2024_Q2 = (
    G3Meeting("川崎", "34", "kawasaki", date(2024, 4, 4), date(2024, 4, 7)),
    G3Meeting("高知", "74", "kochi", date(2024, 4, 11), date(2024, 4, 14)),
    G3Meeting("西武園", "26", "seibuen", date(2024, 4, 20), date(2024, 4, 23)),
    G3Meeting("武雄", "84", "takeo", date(2024, 5, 11), date(2024, 5, 14)),
    G3Meeting("函館", "11", "hakodate", date(2024, 5, 16), date(2024, 5, 19)),
    G3Meeting("前橋", "22", "maebashi", date(2024, 6, 1), date(2024, 6, 4)),
    G3Meeting("函館", "11", "hakodate", date(2024, 6, 6), date(2024, 6, 9)),
    G3Meeting("奈良", "53", "nara", date(2024, 6, 6), date(2024, 6, 9)),
    G3Meeting("久留米", "83", "kurume", date(2024, 6, 22), date(2024, 6, 25)),
    G3Meeting("取手", "23", "toride", date(2024, 6, 27), date(2024, 6, 30)),
)


def race_url(meeting: G3Meeting, day_index: int, race_no: int) -> str:
    race_id = (
        f"{meeting.venue_code}{meeting.start:%Y%m%d}"
        f"{day_index:02d}{race_no:04d}"
    )
    return f"https://keirin.kdreams.jp/{meeting.slug}/racedetail/{race_id}/?pageType=result"


def detect_race_type(html: str, expected_race_no: int) -> str:
    soup = BeautifulSoup(html, "lxml")
    title = base.normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    match = re.search(
        rf"(?<!\d){expected_race_no}R\s*([ＳSＡAＬL]級[^\s|]+)",
        title,
    )
    if not match:
        text = base.normalize_text(soup.get_text(" ", strip=True))
        match = re.search(
            rf"(?<!\d){expected_race_no}R\s*([ＳSＡAＬL]級[^\s|]+)",
            text,
        )
    if not match:
        raise base.CollectorError(f"race type not found for {expected_race_no}R")
    return match.group(1)


def extract_entries_for_ref(html: str, ref: G3RaceRef):
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
    if start.year != 2024 or end.year != 2024:
        raise ValueError("this G3 collector is intentionally restricted to 2024")

    meetings = [m for m in G3_MEETINGS_2024_Q2 if m.end >= start and m.start <= end]
    if not meetings:
        raise ValueError("no configured 2024 Q2 G3 meetings overlap the requested window")

    session = base.make_session()
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    candidate_g3_races = 0
    not_found_candidates = 0
    skipped_outside_window = 0
    meeting_race_counts: Counter[str] = Counter()

    for meeting in meetings:
        meeting_key = f"{meeting.track}|{meeting.start.isoformat()}"
        days = (meeting.end - meeting.start).days + 1
        for offset in range(days):
            race_date = meeting.start + timedelta(days=offset)
            if race_date < start or race_date > end:
                skipped_outside_window += 12
                continue
            day_index = offset + 1

            # Standard KDreams daytime/G3 cards have at most 12 races. Probe only
            # this known G3 meeting's 12 deterministic race IDs; a genuine 404
            # simply means that race number was not offered that day.
            for race_no in range(1, 13):
                url = race_url(meeting, day_index, race_no)
                try:
                    html = base.fetch_html(session, url)
                except Exception as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status == 404:
                        not_found_candidates += 1
                        continue
                    failures.append(
                        {
                            "stage": "race_fetch",
                            "discovered_on": race_date.isoformat(),
                            "url": url,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                try:
                    soup = BeautifulSoup(html, "lxml")
                    page_text = base.normalize_text(soup.get_text(" ", strip=True))
                    if not G3_MARKER_RE.search(page_text):
                        raise base.CollectorError("page is not marked G3")

                    race_type = detect_race_type(html, race_no)
                    ref = G3RaceRef(
                        discovered_on=race_date.isoformat(),
                        race_no=race_no,
                        race_type=race_type,
                        url=url,
                    )
                    race_meta, race_entries = extract_entries_for_ref(html, ref)
                    actual_date = date.fromisoformat(str(race_meta["race_date"]))
                    if actual_date != race_date:
                        raise base.CollectorError(
                            f"race date mismatch expected={race_date} actual={actual_date}"
                        )

                    race_meta["meeting_grade"] = "G3"
                    for entry in race_entries:
                        entry["meeting_grade"] = "G3"
                    races.append(race_meta)
                    entries.extend(race_entries)
                    candidate_g3_races += 1
                    meeting_race_counts[meeting_key] += 1
                except Exception as exc:
                    failures.append(
                        {
                            "stage": "race_parse",
                            "discovered_on": race_date.isoformat(),
                            "url": url,
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
        "race_id", "race_date", "track", "meeting_grade", "race_no", "race_type",
        "start_time", "deadline", "entry_count", "source_url", "captured_at_utc",
    ]
    entry_fields = [
        "race_id", "race_date", "track", "meeting_grade", "race_no", "race_type",
        "start_time", "deadline", "source_url", "captured_at_utc", "car_no",
        "player_name", "player_profile", "prefecture", "age", "term", "class",
        "style", "gear", "score", "s_count", "b_count", "nige_count",
        "makuri_count", "sashi_count", "mark_count", "first_count", "second_count",
        "third_count", "outside_count", "win_rate", "top2_rate", "top3_rate",
        "prediction_mark", "evaluation", "raw_row_json",
    ]
    failure_fields = ["stage", "discovered_on", "url", "error"]

    base.write_csv(out_dir / "races.csv", races, race_fields)
    base.write_csv(out_dir / "entries.csv", entries, entry_fields)
    base.write_csv(out_dir / "failures.csv", failures, failure_fields)

    summary = {
        "target": f"G3 meetings, all races, {start.isoformat()} to {end.isoformat()}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "meeting_source": "楽天Kドリームス 2024年GIII開催スケジュール",
        "configured_meetings": len(meetings),
        "candidate_g3_races": candidate_g3_races,
        "parsed_g3_races": len(races),
        "entry_rows": len(entries),
        "not_found_candidate_urls": not_found_candidates,
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "meeting_race_counts": dict(sorted(meeting_race_counts.items())),
        "race_type_counts": dict(sorted(Counter(str(r["race_type"]) for r in races).items())),
        "entry_count_counts": dict(sorted(Counter(str(r["entry_count"]) for r in races).items())),
        "tracks": sorted({str(r["track"]) for r in races}),
        "definition_note": "楽天Kドリームスの2024年GIII開催スケジュールで確定したQ2のG3開催IDだけを使用し、その開催のレース詳細だけを取得する。F1/F2等のレース詳細ページは取得しない。級班・車立て数では絞らない。",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all races from 2024 Q2 G3 meetings")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.05)
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
    if summary["parsed_g3_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

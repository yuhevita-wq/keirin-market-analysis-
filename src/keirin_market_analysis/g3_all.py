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
    max_races: int = 12


@dataclass(frozen=True)
class G3RaceRef:
    discovered_on: str
    race_no: int
    race_type: str
    url: str


# 2024 Q4 G3 schedule, verified against KEIRIN.JP / KDreams published schedules.
# Using deterministic meeting IDs avoids crawling G1/G2/GP/F1/F2 race-detail pages.
G3_MEETINGS_2024_Q4 = (
    G3Meeting("熊本", "87", "kumamoto", date(2024, 10, 3), date(2024, 10, 6)),
    G3Meeting("川崎", "34", "kawasaki", date(2024, 10, 11), date(2024, 10, 14)),
    G3Meeting("別府", "86", "beppu", date(2024, 10, 11), date(2024, 10, 14)),
    G3Meeting("京王閣", "27", "keiokaku", date(2024, 10, 26), date(2024, 10, 29)),
    G3Meeting("防府", "63", "hofu", date(2024, 11, 1), date(2024, 11, 4)),
    G3Meeting("四日市", "48", "yokkaichi", date(2024, 11, 7), date(2024, 11, 10)),
    G3Meeting("松阪", "47", "matsusaka", date(2024, 11, 14), date(2024, 11, 17)),
    G3Meeting("大垣", "44", "ogaki", date(2024, 11, 30), date(2024, 12, 3)),
    G3Meeting("松山", "75", "matsuyama", date(2024, 12, 5), date(2024, 12, 8)),
    G3Meeting("玉野", "61", "tamano", date(2024, 12, 12), date(2024, 12, 15)),
    G3Meeting("佐世保", "85", "sasebo", date(2024, 12, 19), date(2024, 12, 22)),
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

    meetings = [m for m in G3_MEETINGS_2024_Q4 if m.end >= start and m.start <= end]
    if not meetings:
        raise ValueError("no configured 2024 Q4 G3 meetings overlap the requested window")

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
                skipped_outside_window += meeting.max_races
                continue
            day_index = offset + 1

            for race_no in range(1, meeting.max_races + 1):
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
        "meeting_source": "KEIRIN.JP 2024年度グレードレース開催日程",
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
        "definition_note": "2024年度グレードレース開催日程で確定したQ4のG3開催IDだけを使用し、その開催の楽天Kドリームスレース詳細だけを取得する。G1/G2/GP/F1/F2等は取得しない。級班・車立て数では絞らない。取得不能・中止・特殊レース等はfailureとして記録してスキップし、他レースの収集を継続する。",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all races from 2024 Q4 G3 meetings")
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

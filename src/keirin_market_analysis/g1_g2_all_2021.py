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


GRADE_MARKERS = {
    "G1": re.compile(r"(?:Ｇ１|G1|ＧⅠ|GI)(?!I)", flags=re.IGNORECASE),
    "G2": re.compile(r"(?:Ｇ２|G2|ＧⅡ|GII)", flags=re.IGNORECASE),
}


@dataclass(frozen=True)
class GradeMeeting:
    track: str
    venue_code: str
    slug: str
    grade: str
    start: date
    end: date
    max_races: int = 12
    only_race_type_contains: str | None = None


@dataclass(frozen=True)
class GradeRaceRef:
    discovered_on: str
    race_no: int
    race_type: str
    url: str


# Calendar-year 2021 male G1/G2 schedule only.
# Jan-Mar comes from KEIRIN.JP 2020年度 schedule; Apr-Dec from 2021年度.
# Girls races embedded in mixed meetings are excluded at race level.
G1_G2_MEETINGS_2021 = (
    GradeMeeting("川崎", "34", "kawasaki", "G1", date(2021, 2, 20), date(2021, 2, 23)),
    GradeMeeting("松阪", "47", "matsusaka", "G2", date(2021, 3, 25), date(2021, 3, 28)),
    GradeMeeting("京王閣", "27", "keiokaku", "G1", date(2021, 5, 4), date(2021, 5, 9)),
    GradeMeeting("岸和田", "56", "kishiwada", "G1", date(2021, 6, 17), date(2021, 6, 20)),
    GradeMeeting("函館", "11", "hakodate", "G2", date(2021, 7, 16), date(2021, 7, 18)),
    GradeMeeting("いわき平", "13", "iwakitaira", "G1", date(2021, 8, 10), date(2021, 8, 15)),
    GradeMeeting("岐阜", "43", "gifu", "G2", date(2021, 9, 17), date(2021, 9, 20)),
    GradeMeeting("弥彦", "21", "yahiko", "G1", date(2021, 10, 21), date(2021, 10, 24)),
    GradeMeeting("小倉", "81", "kokura", "G1", date(2021, 11, 18), date(2021, 11, 23)),
    # GP series day 2: only Young Grand Prix 2021 is G2.
    GradeMeeting("静岡", "38", "shizuoka", "G2", date(2021, 12, 29), date(2021, 12, 29), 12, "ヤング"),
)


def race_url(meeting: GradeMeeting, day_index: int, race_no: int) -> str:
    race_id = f"{meeting.venue_code}{meeting.start:%Y%m%d}{day_index:02d}{race_no:04d}"
    return f"https://keirin.kdreams.jp/{meeting.slug}/racedetail/{race_id}/?pageType=result"


def detect_race_type(html: str, expected_race_no: int) -> str:
    soup = BeautifulSoup(html, "lxml")
    title = base.normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    match = re.search(rf"(?<!\d){expected_race_no}R\s*([^\s|]+)", title)
    if not match:
        text = base.normalize_text(soup.get_text(" ", strip=True))
        match = re.search(rf"(?<!\d){expected_race_no}R\s*([^\s|]+)", text)
    if not match:
        raise base.CollectorError(f"race type not found for {expected_race_no}R")
    return match.group(1)


def extract_entries_for_ref(html: str, ref: GradeRaceRef):
    base.TARGET_RACE_TYPE = ref.race_type
    frozen_ref = base.RaceRef(discovered_on=ref.discovered_on, race_no=ref.race_no, url=ref.url)
    return base.extract_entries(html, frozen_ref)


def _is_l_class(value: object) -> bool:
    text = base.normalize_text(str(value or "")).upper().replace("Ｌ", "L")
    return text.startswith("L")


def is_girls_race(race_type: str, race_entries: list[dict[str, object]]) -> bool:
    normalized = base.normalize_text(race_type)
    if "ガールズ" in normalized or "女子" in normalized or "パールカップ" in normalized:
        return True
    classes = [entry.get("class") for entry in race_entries if str(entry.get("class") or "").strip()]
    return bool(classes) and all(_is_l_class(value) for value in classes)


def collect(start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    if start > end:
        raise ValueError("start date must be <= end date")
    if start.year != 2021 or end.year != 2021:
        raise ValueError("this collector is intentionally restricted to calendar-year 2021")

    meetings = [m for m in G1_G2_MEETINGS_2021 if m.end >= start and m.start <= end]
    if not meetings:
        raise ValueError("no configured 2021 G1/G2 meetings overlap the requested window")

    session = base.make_session()
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    excluded_girls: list[dict[str, object]] = []
    excluded_non_target: list[dict[str, object]] = []
    not_found_candidates = 0
    skipped_outside_window = 0
    meeting_race_counts: Counter[str] = Counter()

    for meeting in meetings:
        meeting_key = f"{meeting.grade}|{meeting.track}|{meeting.start.isoformat()}"
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
                    failures.append({"stage":"race_fetch","grade":meeting.grade,"discovered_on":race_date.isoformat(),"url":url,"error":f"{type(exc).__name__}: {exc}"})
                    continue

                try:
                    soup = BeautifulSoup(html, "lxml")
                    page_text = base.normalize_text(soup.get_text(" ", strip=True))
                    race_type = detect_race_type(html, race_no)

                    if meeting.only_race_type_contains and meeting.only_race_type_contains not in race_type:
                        excluded_non_target.append({"race_date":race_date.isoformat(),"track":meeting.track,"grade":meeting.grade,"race_no":race_no,"race_type":race_type,"url":url,"reason":"special meeting non-target race"})
                        continue

                    if not GRADE_MARKERS[meeting.grade].search(page_text) and not meeting.only_race_type_contains:
                        raise base.CollectorError(f"page is not marked {meeting.grade}")

                    ref = GradeRaceRef(race_date.isoformat(), race_no, race_type, url)
                    race_meta, race_entries = extract_entries_for_ref(html, ref)
                    actual_date = date.fromisoformat(str(race_meta["race_date"]))
                    if actual_date != race_date:
                        raise base.CollectorError(f"race date mismatch expected={race_date} actual={actual_date}")

                    if is_girls_race(race_type, race_entries):
                        excluded_girls.append({"race_date":race_date.isoformat(),"track":meeting.track,"grade":meeting.grade,"race_no":race_no,"race_type":race_type,"url":url})
                        continue

                    race_meta["meeting_grade"] = meeting.grade
                    for entry in race_entries:
                        entry["meeting_grade"] = meeting.grade
                    races.append(race_meta)
                    entries.extend(race_entries)
                    meeting_race_counts[meeting_key] += 1
                except Exception as exc:
                    failures.append({"stage":"race_parse","grade":meeting.grade,"discovered_on":race_date.isoformat(),"url":url,"error":f"{type(exc).__name__}: {exc}"})

                if sleep_seconds:
                    time.sleep(sleep_seconds)

    races.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"])))
    entries.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"]), int(row["car_no"])))

    race_fields = ["race_id","race_date","track","meeting_grade","race_no","race_type","start_time","deadline","entry_count","source_url","captured_at_utc"]
    entry_fields = ["race_id","race_date","track","meeting_grade","race_no","race_type","start_time","deadline","source_url","captured_at_utc","car_no","player_name","player_profile","prefecture","age","term","class","style","gear","score","s_count","b_count","nige_count","makuri_count","sashi_count","mark_count","first_count","second_count","third_count","outside_count","win_rate","top2_rate","top3_rate","prediction_mark","evaluation","raw_row_json"]
    base.write_csv(out_dir / "races.csv", races, race_fields)
    base.write_csv(out_dir / "entries.csv", entries, entry_fields)
    base.write_csv(out_dir / "failures.csv", failures, ["stage","grade","discovered_on","url","error"])
    base.write_csv(out_dir / "excluded_girls.csv", excluded_girls, ["race_date","track","grade","race_no","race_type","url"])
    base.write_csv(out_dir / "excluded_non_target.csv", excluded_non_target, ["race_date","track","grade","race_no","race_type","url","reason"])

    summary = {
        "target": f"2021 calendar-year G1/G2 meetings, non-girls races, {start.isoformat()} to {end.isoformat()}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "meeting_source": "KEIRIN.JP 2020年度/2021年度 特別競輪等開催日程",
        "configured_meetings": len(meetings),
        "parsed_races": len(races),
        "entry_rows": len(entries),
        "excluded_girls_races": len(excluded_girls),
        "excluded_non_target_races": len(excluded_non_target),
        "not_found_candidate_urls": not_found_candidates,
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "grade_counts": dict(sorted(Counter(str(r["meeting_grade"]) for r in races).items())),
        "meeting_race_counts": dict(sorted(meeting_race_counts.items())),
        "race_type_counts": dict(sorted(Counter(str(r["race_type"]) for r in races).items())),
        "entry_count_counts": dict(sorted(Counter(str(r["entry_count"]) for r in races).items())),
        "tracks": sorted({str(r["track"]) for r in races}),
        "definition_note": "Official calendar-year 2021 male G1/G2 meeting IDs only. Girls races embedded in mixed meetings are excluded by race type and L-class detection. G3/GP/F1/F2 are not crawled. Young Grand Prix 2021 is retained as G2. Individual unavailable/cancelled/special pages are recorded and skipped while the annual run continues.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect non-girls races from calendar-year 2021 G1/G2 meetings")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = collect(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date), Path(args.out_dir), args.sleep)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["parsed_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


# Calendar-year 2026 G3 schedule, H1 only.
# Jan-Mar comes from KEIRIN.JP 2025年度 schedule; Apr-Jun from 2026年度 schedule.
# G1/G2/GP/F1/F2 are intentionally excluded. Girls races inside G3 meetings are skipped.
G3_MEETINGS_2026_H1 = (
    # Q1
    G3Meeting("立川", "28", "tachikawa", date(2026, 1, 4), date(2026, 1, 7)),
    G3Meeting("和歌山", "55", "wakayama", date(2026, 1, 9), date(2026, 1, 12)),
    G3Meeting("大宮", "25", "omiya", date(2026, 1, 15), date(2026, 1, 18)),
    G3Meeting("いわき平", "13", "iwakitaira", date(2026, 1, 22), date(2026, 1, 25)),
    G3Meeting("小松島", "73", "komatsushima", date(2026, 1, 29), date(2026, 2, 1)),
    G3Meeting("奈良", "53", "nara", date(2026, 2, 5), date(2026, 2, 8)),
    G3Meeting("静岡", "38", "shizuoka", date(2026, 2, 12), date(2026, 2, 15)),
    G3Meeting("大垣", "44", "ogaki", date(2026, 2, 28), date(2026, 3, 3)),
    G3Meeting("松山", "75", "matsuyama", date(2026, 3, 5), date(2026, 3, 8)),
    G3Meeting("取手", "23", "toride", date(2026, 3, 12), date(2026, 3, 15)),
    G3Meeting("西武園", "26", "seibuen", date(2026, 3, 12), date(2026, 3, 15)),
    G3Meeting("豊橋", "45", "toyohashi", date(2026, 3, 28), date(2026, 3, 31)),
    # Q2
    G3Meeting("伊東", "37", "ito", date(2026, 4, 2), date(2026, 4, 5)),
    G3Meeting("西武園", "26", "seibuen", date(2026, 4, 9), date(2026, 4, 12)),
    G3Meeting("武雄", "84", "takeo", date(2026, 4, 18), date(2026, 4, 21)),
    G3Meeting("名古屋", "42", "nagoya", date(2026, 4, 23), date(2026, 4, 26)),
    G3Meeting("松阪", "47", "matsusaka", date(2026, 5, 9), date(2026, 5, 12)),
    G3Meeting("函館", "11", "hakodate", date(2026, 5, 14), date(2026, 5, 17)),
    G3Meeting("青森", "12", "aomori", date(2026, 5, 19), date(2026, 5, 21), 9),
    G3Meeting("宇都宮", "24", "utsunomiya", date(2026, 5, 30), date(2026, 6, 2)),
    G3Meeting("久留米", "83", "kurume", date(2026, 6, 4), date(2026, 6, 7)),
    G3Meeting("岐阜", "43", "gifu", date(2026, 6, 11), date(2026, 6, 14)),
    G3Meeting("玉野", "61", "tamano", date(2026, 6, 11), date(2026, 6, 14)),
    G3Meeting("取手", "23", "toride", date(2026, 6, 27), date(2026, 6, 30)),
)


def race_url(meeting: G3Meeting, day_index: int, race_no: int) -> str:
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


def extract_entries_for_ref(html: str, ref: G3RaceRef):
    base.TARGET_RACE_TYPE = ref.race_type
    frozen_ref = base.RaceRef(discovered_on=ref.discovered_on, race_no=ref.race_no, url=ref.url)
    return base.extract_entries(html, frozen_ref)


def collect(start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    if start > end:
        raise ValueError("start date must be <= end date")
    if start.year != 2026 or end.year != 2026:
        raise ValueError("this collector is intentionally restricted to calendar-year 2026")
    if end > date(2026, 6, 30):
        raise ValueError("this collector is intentionally restricted to 2026 H1")

    meetings = [m for m in G3_MEETINGS_2026_H1 if m.end >= start and m.start <= end]
    if not meetings:
        raise ValueError("no configured 2026 H1 G3 meetings overlap the requested window")

    session = base.make_session()
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    excluded_girls: list[dict[str, object]] = []
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
                    failures.append({"stage":"race_fetch","discovered_on":race_date.isoformat(),"url":url,"error":f"{type(exc).__name__}: {exc}"})
                    continue

                try:
                    soup = BeautifulSoup(html, "lxml")
                    page_text = base.normalize_text(soup.get_text(" ", strip=True))
                    if not G3_MARKER_RE.search(page_text):
                        raise base.CollectorError("page is not marked G3")
                    race_type = detect_race_type(html, race_no)
                    if race_type.startswith("ガールズ"):
                        excluded_girls.append({"race_date":race_date.isoformat(),"track":meeting.track,"race_no":race_no,"race_type":race_type,"url":url})
                        continue
                    ref = G3RaceRef(race_date.isoformat(), race_no, race_type, url)
                    race_meta, race_entries = extract_entries_for_ref(html, ref)
                    actual_date = date.fromisoformat(str(race_meta["race_date"]))
                    if actual_date != race_date:
                        raise base.CollectorError(f"race date mismatch expected={race_date} actual={actual_date}")
                    race_meta["meeting_grade"] = "G3"
                    for entry in race_entries:
                        entry["meeting_grade"] = "G3"
                    races.append(race_meta)
                    entries.extend(race_entries)
                    candidate_g3_races += 1
                    meeting_race_counts[meeting_key] += 1
                except Exception as exc:
                    failures.append({"stage":"race_parse","discovered_on":race_date.isoformat(),"url":url,"error":f"{type(exc).__name__}: {exc}"})

                if sleep_seconds:
                    time.sleep(sleep_seconds)

    races.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"])))
    entries.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"]), int(row["car_no"])))

    race_fields = ["race_id","race_date","track","meeting_grade","race_no","race_type","start_time","deadline","entry_count","source_url","captured_at_utc"]
    entry_fields = ["race_id","race_date","track","meeting_grade","race_no","race_type","start_time","deadline","source_url","captured_at_utc","car_no","player_name","player_profile","prefecture","age","term","class","style","gear","score","s_count","b_count","nige_count","makuri_count","sashi_count","mark_count","first_count","second_count","third_count","outside_count","win_rate","top2_rate","top3_rate","prediction_mark","evaluation","raw_row_json"]
    base.write_csv(out_dir / "races.csv", races, race_fields)
    base.write_csv(out_dir / "entries.csv", entries, entry_fields)
    base.write_csv(out_dir / "failures.csv", failures, ["stage","discovered_on","url","error"])
    base.write_csv(out_dir / "excluded_girls.csv", excluded_girls, ["race_date","track","race_no","race_type","url"])

    summary = {
        "target": f"2026 calendar-year G3 meetings, non-girls races, {start.isoformat()} to {end.isoformat()}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "meeting_source": "KEIRIN.JP 2025年度(2026-01 to 2026-03) / 2026年度(2026-04 to 2026-06) グレードレース開催日程",
        "configured_meetings": len(meetings),
        "candidate_g3_races": candidate_g3_races,
        "parsed_g3_races": len(races),
        "entry_rows": len(entries),
        "excluded_girls_races": len(excluded_girls),
        "not_found_candidate_urls": not_found_candidates,
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "meeting_race_counts": dict(sorted(meeting_race_counts.items())),
        "race_type_counts": dict(sorted(Counter(str(r["race_type"]) for r in races).items())),
        "entry_count_counts": dict(sorted(Counter(str(r["entry_count"]) for r in races).items())),
        "tracks": sorted({str(r["track"]) for r in races}),
        "definition_note": "Official 2026 calendar-year H1 G3 meeting IDs only. Girls races are explicitly excluded after race-type detection. G1/G2/GP/F1/F2 are not crawled. Individual unavailable/cancelled/special pages are recorded and skipped while the quarter continues.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect non-girls races from 2026 H1 G3 meetings")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = collect(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date), Path(args.out_dir), args.sleep)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["parsed_g3_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


# Calendar-year schedules use KEIRIN.JP official grade-race schedules.
# Jan-Mar comes from the previous fiscal-year page; Apr-Dec from the same fiscal-year page.
# Meeting IDs are deterministic so unrelated G1/G2/GP/F1/F2 meetings are never crawled.
G3_MEETINGS: dict[int, tuple[G3Meeting, ...]] = {
    2023: (
        G3Meeting("立川", "28", "tachikawa", date(2023, 1, 4), date(2023, 1, 7)),
        G3Meeting("和歌山", "55", "wakayama", date(2023, 1, 12), date(2023, 1, 15)),
        G3Meeting("大宮", "25", "omiya", date(2023, 1, 19), date(2023, 1, 22)),
        G3Meeting("豊橋", "45", "toyohashi", date(2023, 1, 26), date(2023, 1, 29)),
        G3Meeting("奈良", "53", "nara", date(2023, 2, 2), date(2023, 2, 5)),
        G3Meeting("静岡", "38", "shizuoka", date(2023, 2, 9), date(2023, 2, 12)),
        G3Meeting("伊東", "37", "ito", date(2023, 2, 16), date(2023, 2, 19)),
        G3Meeting("大垣", "44", "ogaki", date(2023, 3, 4), date(2023, 3, 7)),
        G3Meeting("松山", "75", "matsuyama", date(2023, 3, 9), date(2023, 3, 12)),
        G3Meeting("玉野", "61", "tamano", date(2023, 3, 26), date(2023, 3, 29)),
        G3Meeting("四日市", "48", "yokkaichi", date(2023, 4, 1), date(2023, 4, 4)),
        G3Meeting("高知", "74", "kochi", date(2023, 4, 6), date(2023, 4, 9)),
        G3Meeting("小田原", "36", "odawara", date(2023, 4, 13), date(2023, 4, 16)),
        G3Meeting("武雄", "84", "takeo", date(2023, 4, 22), date(2023, 4, 25)),
        G3Meeting("久留米", "83", "kurume", date(2023, 4, 27), date(2023, 4, 30)),
        G3Meeting("函館", "11", "hakodate", date(2023, 5, 13), date(2023, 5, 16)),
        G3Meeting("宇都宮", "24", "utsunomiya", date(2023, 5, 18), date(2023, 5, 21)),
        G3Meeting("大垣", "44", "ogaki", date(2023, 6, 3), date(2023, 6, 6)),
        G3Meeting("向日町", "54", "mukomachi", date(2023, 6, 8), date(2023, 6, 11)),
        G3Meeting("久留米", "83", "kurume", date(2023, 6, 24), date(2023, 6, 27)),
        G3Meeting("前橋", "22", "maebashi", date(2023, 6, 29), date(2023, 7, 2)),
        G3Meeting("小松島", "73", "komatsushima", date(2023, 7, 6), date(2023, 7, 9)),
        G3Meeting("福井", "51", "fukui", date(2023, 7, 22), date(2023, 7, 25)),
        G3Meeting("名古屋", "42", "nagoya", date(2023, 7, 27), date(2023, 7, 30)),
        G3Meeting("富山", "46", "toyama", date(2023, 8, 3), date(2023, 8, 6)),
        G3Meeting("京王閣", "27", "keiokaku", date(2023, 8, 10), date(2023, 8, 13)),
        G3Meeting("和歌山", "55", "wakayama", date(2023, 8, 10), date(2023, 8, 13)),
        G3Meeting("松戸", "31", "matsudo", date(2023, 8, 26), date(2023, 8, 29)),
        G3Meeting("向日町", "54", "mukomachi", date(2023, 8, 31), date(2023, 9, 3)),
        G3Meeting("立川", "28", "tachikawa", date(2023, 9, 7), date(2023, 9, 10)),
        G3Meeting("松阪", "47", "matsusaka", date(2023, 9, 23), date(2023, 9, 26)),
        G3Meeting("豊橋", "45", "toyohashi", date(2023, 9, 28), date(2023, 10, 1)),
        G3Meeting("久留米", "83", "kurume", date(2023, 10, 6), date(2023, 10, 9)),
        G3Meeting("小田原", "36", "odawara", date(2023, 10, 12), date(2023, 10, 15)),
        G3Meeting("京王閣", "27", "keiokaku", date(2023, 10, 28), date(2023, 10, 31)),
        G3Meeting("玉野", "61", "tamano", date(2023, 11, 2), date(2023, 11, 5)),
        G3Meeting("四日市", "48", "yokkaichi", date(2023, 11, 9), date(2023, 11, 12)),
        G3Meeting("大垣", "44", "ogaki", date(2023, 11, 16), date(2023, 11, 19)),
        G3Meeting("伊東", "37", "ito", date(2023, 12, 2), date(2023, 12, 5)),
        G3Meeting("別府", "86", "beppu", date(2023, 12, 7), date(2023, 12, 10)),
        G3Meeting("佐世保", "85", "sasebo", date(2023, 12, 14), date(2023, 12, 17)),
        G3Meeting("玉野", "61", "tamano", date(2023, 12, 21), date(2023, 12, 24)),
    ),
    2022: (
        G3Meeting("立川", "28", "tachikawa", date(2022, 1, 4), date(2022, 1, 7)),
        G3Meeting("和歌山", "55", "wakayama", date(2022, 1, 9), date(2022, 1, 12)),
        G3Meeting("大宮", "25", "omiya", date(2022, 1, 15), date(2022, 1, 18)),
        G3Meeting("豊橋", "45", "toyohashi", date(2022, 1, 20), date(2022, 1, 23)),
        G3Meeting("高松", "71", "takamatsu", date(2022, 1, 27), date(2022, 1, 30)),
        G3Meeting("静岡", "38", "shizuoka", date(2022, 2, 3), date(2022, 2, 6)),
        G3Meeting("奈良", "53", "nara", date(2022, 2, 10), date(2022, 2, 13)),
        G3Meeting("高知", "74", "kochi", date(2022, 2, 26), date(2022, 3, 1)),
        G3Meeting("名古屋", "42", "nagoya", date(2022, 3, 3), date(2022, 3, 6)),
        G3Meeting("大垣", "44", "ogaki", date(2022, 3, 10), date(2022, 3, 13)),
        G3Meeting("玉野", "61", "tamano", date(2022, 3, 26), date(2022, 3, 29)),
        G3Meeting("平塚", "35", "hiratsuka", date(2022, 4, 7), date(2022, 4, 10)),
        G3Meeting("川崎", "34", "kawasaki", date(2022, 4, 14), date(2022, 4, 17)),
        G3Meeting("武雄", "84", "takeo", date(2022, 4, 23), date(2022, 4, 26)),
        G3Meeting("青森", "12", "aomori", date(2022, 4, 28), date(2022, 5, 1)),
        G3Meeting("函館", "11", "hakodate", date(2022, 5, 14), date(2022, 5, 17)),
        G3Meeting("宇都宮", "24", "utsunomiya", date(2022, 5, 19), date(2022, 5, 22)),
        G3Meeting("取手", "23", "toride", date(2022, 6, 4), date(2022, 6, 7)),
        G3Meeting("松戸", "31", "matsudo", date(2022, 6, 9), date(2022, 6, 12)),
        G3Meeting("久留米", "83", "kurume", date(2022, 6, 25), date(2022, 6, 28)),
        G3Meeting("小松島", "73", "komatsushima", date(2022, 6, 30), date(2022, 7, 3)),
        G3Meeting("福井", "51", "fukui", date(2022, 7, 7), date(2022, 7, 10)),
        G3Meeting("佐世保", "85", "sasebo", date(2022, 7, 23), date(2022, 7, 26)),
        G3Meeting("弥彦", "21", "yahiko", date(2022, 7, 28), date(2022, 7, 31)),
        G3Meeting("函館", "11", "hakodate", date(2022, 8, 4), date(2022, 8, 7)),
        G3Meeting("岸和田", "56", "kishiwada", date(2022, 8, 4), date(2022, 8, 7)),
        G3Meeting("富山", "46", "toyama", date(2022, 8, 20), date(2022, 8, 23)),
        G3Meeting("小田原", "36", "odawara", date(2022, 8, 25), date(2022, 8, 28)),
        G3Meeting("岐阜", "43", "gifu", date(2022, 9, 1), date(2022, 9, 4)),
        G3Meeting("青森", "12", "aomori", date(2022, 9, 8), date(2022, 9, 11)),
        G3Meeting("向日町", "54", "mukomachi", date(2022, 9, 24), date(2022, 9, 27)),
        G3Meeting("久留米", "83", "kurume", date(2022, 10, 1), date(2022, 10, 4)),
        G3Meeting("松阪", "47", "matsusaka", date(2022, 10, 7), date(2022, 10, 10)),
        G3Meeting("松山", "75", "matsuyama", date(2022, 10, 13), date(2022, 10, 16)),
        G3Meeting("京王閣", "27", "keiokaku", date(2022, 10, 29), date(2022, 11, 1)),
        G3Meeting("防府", "63", "hofu", date(2022, 11, 3), date(2022, 11, 6)),
        G3Meeting("四日市", "48", "yokkaichi", date(2022, 11, 10), date(2022, 11, 13)),
        G3Meeting("富山", "46", "toyama", date(2022, 11, 17), date(2022, 11, 20)),
        G3Meeting("高松", "71", "takamatsu", date(2022, 12, 3), date(2022, 12, 6)),
        G3Meeting("松戸", "31", "matsudo", date(2022, 12, 8), date(2022, 12, 11)),
        G3Meeting("広島", "62", "hiroshima", date(2022, 12, 15), date(2022, 12, 18)),
        G3Meeting("伊東", "37", "ito", date(2022, 12, 22), date(2022, 12, 25)),
    ),
}

MEETING_SOURCE = {
    2023: "KEIRIN.JP 2022年度(2023-01 to 2023-03) / 2023年度(2023-04 to 2023-12) グレードレース開催日程",
    2022: "KEIRIN.JP 2021年度(2022-01 to 2022-03) / 2022年度(2022-04 to 2022-12) グレードレース開催日程",
}


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


def _is_l_class(value: object) -> bool:
    text = base.normalize_text(str(value or "")).upper().replace("Ｌ", "L")
    return text.startswith("L")


def is_girls_race(race_type: str, race_entries: list[dict[str, object]]) -> bool:
    normalized = base.normalize_text(race_type)
    if "ガールズ" in normalized or "女子" in normalized or "パールカップ" in normalized:
        return True
    classes = [entry.get("class") for entry in race_entries if str(entry.get("class") or "").strip()]
    return bool(classes) and all(_is_l_class(value) for value in classes)


def collect(year: int, start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    if year not in G3_MEETINGS:
        raise ValueError(f"unsupported year: {year}")
    if start > end:
        raise ValueError("start date must be <= end date")
    if start.year != year or end.year != year:
        raise ValueError(f"start/end must stay inside calendar-year {year}")

    meetings = [m for m in G3_MEETINGS[year] if m.end >= start and m.start <= end]
    if not meetings:
        raise ValueError(f"no configured {year} G3 meetings overlap the requested window")

    session = base.make_session()
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    excluded_girls: list[dict[str, object]] = []
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
                    failures.append({"stage": "race_fetch", "discovered_on": race_date.isoformat(), "url": url, "error": f"{type(exc).__name__}: {exc}"})
                    continue

                try:
                    soup = BeautifulSoup(html, "lxml")
                    page_text = base.normalize_text(soup.get_text(" ", strip=True))
                    if not G3_MARKER_RE.search(page_text):
                        raise base.CollectorError("page is not marked G3")
                    race_type = detect_race_type(html, race_no)
                    ref = G3RaceRef(race_date.isoformat(), race_no, race_type, url)
                    race_meta, race_entries = extract_entries_for_ref(html, ref)
                    actual_date = date.fromisoformat(str(race_meta["race_date"]))
                    if actual_date != race_date:
                        raise base.CollectorError(f"race date mismatch expected={race_date} actual={actual_date}")
                    if is_girls_race(race_type, race_entries):
                        excluded_girls.append({"race_date": race_date.isoformat(), "track": meeting.track, "race_no": race_no, "race_type": race_type, "url": url})
                        continue
                    race_meta["meeting_grade"] = "G3"
                    for entry in race_entries:
                        entry["meeting_grade"] = "G3"
                    races.append(race_meta)
                    entries.extend(race_entries)
                    meeting_race_counts[meeting_key] += 1
                except Exception as exc:
                    failures.append({"stage": "race_parse", "discovered_on": race_date.isoformat(), "url": url, "error": f"{type(exc).__name__}: {exc}"})

                if sleep_seconds:
                    time.sleep(sleep_seconds)

    races.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"])))
    entries.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"]), int(row["car_no"])))

    race_fields = ["race_id", "race_date", "track", "meeting_grade", "race_no", "race_type", "start_time", "deadline", "entry_count", "source_url", "captured_at_utc"]
    entry_fields = ["race_id", "race_date", "track", "meeting_grade", "race_no", "race_type", "start_time", "deadline", "source_url", "captured_at_utc", "car_no", "player_name", "player_profile", "prefecture", "age", "term", "class", "style", "gear", "score", "s_count", "b_count", "nige_count", "makuri_count", "sashi_count", "mark_count", "first_count", "second_count", "third_count", "outside_count", "win_rate", "top2_rate", "top3_rate", "prediction_mark", "evaluation", "raw_row_json"]
    base.write_csv(out_dir / "races.csv", races, race_fields)
    base.write_csv(out_dir / "entries.csv", entries, entry_fields)
    base.write_csv(out_dir / "failures.csv", failures, ["stage", "discovered_on", "url", "error"])
    base.write_csv(out_dir / "excluded_girls.csv", excluded_girls, ["race_date", "track", "race_no", "race_type", "url"])

    summary = {
        "target": f"{year} calendar-year G3 meetings, non-girls races, {start.isoformat()} to {end.isoformat()}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "meeting_source": MEETING_SOURCE[year],
        "configured_meetings": len(meetings),
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
        "definition_note": f"Official calendar-year {year} G3 meeting IDs only. Girls/L-class races are excluded at race level. G1/G2/GP/F1/F2 are not crawled. Individual unavailable/cancelled/special pages are recorded and skipped while the quarter continues.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect non-girls races from calendar-year 2022/2023 G3 meetings")
    parser.add_argument("--year", type=int, choices=sorted(G3_MEETINGS), required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = collect(args.year, date.fromisoformat(args.start_date), date.fromisoformat(args.end_date), Path(args.out_dir), args.sleep)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["parsed_g3_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

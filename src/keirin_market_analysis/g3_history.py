from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from .s_yosen_2025 import (
    CollectorError,
    RaceRef,
    canonical_race_url,
    cell_text,
    daterange,
    fetch_html,
    make_session,
    normalize_text,
    parse_profile,
    pick_base_racecard_table,
    write_csv,
)

KDREAMS_DAILY = "https://keirin.kdreams.jp/kaisai/{year:04d}/{month:02d}/{day:02d}/"
GRADE = "G3"
GRADE_RE = re.compile(r"(?<![A-Z0-9])(GP|G[123]|F[12])(?![A-Z0-9])")


@dataclass(frozen=True)
class EventCardRef:
    discovered_on: str
    url: str


def nfkc_upper(text: str) -> str:
    return unicodedata.normalize("NFKC", normalize_text(text)).upper()


def canonical_page_url(base_url: str, href: str) -> str:
    absolute = urljoin(base_url, href)
    parts = urlsplit(absolute)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def nearest_published_grades(anchor) -> set[str]:
    """Find the nearest DOM ancestor that explicitly contains a grade label."""
    for ancestor in anchor.parents:
        if getattr(ancestor, "name", None) in {"body", "html"}:
            break
        text = nfkc_upper(ancestor.get_text(" ", strip=True))
        grades = set(GRADE_RE.findall(text))
        if grades:
            return grades
    return set()


def discover_g3_event_cards_from_daily_html(
    html: str, daily_url: str, discovered_on: str
) -> list[EventCardRef]:
    """Discover racecard pages belonging to event blocks explicitly labelled G3."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, EventCardRef] = {}

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", ""))
        if "/racecard/" not in href:
            continue
        grades = nearest_published_grades(anchor)
        if GRADE not in grades:
            continue
        # If a very broad ancestor contains several event grades, do not guess.
        if len(grades) != 1:
            continue
        url = canonical_page_url(daily_url, href)
        found[url] = EventCardRef(discovered_on=discovered_on, url=url)

    return sorted(found.values(), key=lambda item: (item.discovered_on, item.url))


def page_has_g3(soup: BeautifulSoup) -> bool:
    text = nfkc_upper(soup.get_text(" ", strip=True))
    return GRADE in set(GRADE_RE.findall(text))


def discover_races_from_g3_racecard_html(
    html: str, racecard_url: str, discovered_on: str
) -> list[RaceRef]:
    soup = BeautifulSoup(html, "lxml")
    if not page_has_g3(soup):
        raise CollectorError(f"G3 label missing from racecard page: {racecard_url}")

    found: dict[str, RaceRef] = {}
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", ""))
        if "/racedetail/" not in href:
            continue
        url = canonical_race_url(racecard_url, href)
        match = re.search(r"/racedetail/(\d{16})/", url)
        if not match:
            continue
        race_no = int(match.group(1)[-4:])
        found[url] = RaceRef(discovered_on=discovered_on, race_no=race_no, url=url)

    if not found:
        raise CollectorError(f"no race detail links found on G3 racecard page: {racecard_url}")
    return sorted(found.values(), key=lambda item: (item.discovered_on, item.race_no, item.url))


def parse_page_date(soup: BeautifulSoup) -> str:
    text = normalize_text(soup.get_text(" ", strip=True))
    match = re.search(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日\s*レース詳細", text)
    if not match:
        raise CollectorError("race date not found on race page")
    year, month, day = map(int, match.groups())
    return date(year, month, day).isoformat()


def parse_title_meta(soup: BeautifulSoup) -> tuple[str, str, str]:
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    track_match = re.match(r"^(.+?競輪)\s+レース詳細", title)
    track = track_match.group(1) if track_match else ""

    event_name = ""
    race_type = ""
    match = re.search(r"レース詳細\s*\|\s*(.+?)\s+(\d+)R\s+(.+?)\s*\|", title)
    if match:
        event_name = normalize_text(match.group(1))
        race_type = normalize_text(match.group(3))
    return track, event_name, race_type


def extract_meta(soup: BeautifulSoup, ref: RaceRef) -> dict[str, object]:
    if not page_has_g3(soup):
        raise CollectorError(f"G3 label missing from race page: {ref.url}")

    track, event_name, race_type = parse_title_meta(soup)
    if not race_type:
        text = normalize_text(soup.get_text(" ", strip=True))
        type_match = re.search(r"([ＳS]級[^\s|]{1,12})\s+発走予定", text)
        if type_match:
            race_type = normalize_text(type_match.group(1))
    if not race_type:
        raise CollectorError(f"race type not found: {ref.url}")

    text = normalize_text(soup.get_text(" ", strip=True))
    start_match = re.search(r"発走予定\s*(\d{1,2}:\d{2})", text)
    deadline_match = re.search(r"投票締切\s*(\d{1,2}:\d{2})", text)
    race_id_match = re.search(r"/racedetail/(\d{16})/", ref.url)

    return {
        "race_id": race_id_match.group(1) if race_id_match else "",
        "race_date": parse_page_date(soup),
        "track": track,
        "race_no": ref.race_no,
        "race_type": race_type,
        "event_grade": GRADE,
        "event_name": event_name,
        "start_time": start_match.group(1) if start_match else "",
        "deadline": deadline_match.group(1) if deadline_match else "",
        "source_url": ref.url,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def extract_entries(html: str, ref: RaceRef) -> tuple[dict[str, object], list[dict[str, object]]]:
    soup = BeautifulSoup(html, "lxml")
    meta = extract_meta(soup, ref)
    _, entrant_rows = pick_base_racecard_table(soup)

    entries: list[dict[str, object]] = []
    for tr in entrant_rows:
        num_cell = tr.select_one("td.num")
        rider_cell = tr.select_one("td.rider")
        if num_cell is None or rider_cell is None:
            continue

        car_match = re.search(r"([1-9])", normalize_text(num_cell.get_text(" ", strip=True)))
        if not car_match:
            continue
        car_no = int(car_match.group(1))

        home = rider_cell.select_one("span.home")
        profile = normalize_text(home.get_text(" ", strip=True)) if home else ""
        full_rider_text = normalize_text(rider_cell.get_text(" ", strip=True))
        player_name = full_rider_text
        if profile and full_rider_text.endswith(profile):
            player_name = normalize_text(full_rider_text[: -len(profile)])
        prefecture, age, term = parse_profile(profile)

        cells = tr.find_all("td", recursive=False)
        try:
            rider_index = cells.index(rider_cell)
        except ValueError as exc:
            raise CollectorError(f"rider cell index not found: {ref.url}") from exc

        after = cells[rider_index + 1 :]
        stats = [cell_text(after, i) for i in range(17)]
        while len(stats) < 17:
            stats.append("")

        prediction_cell = tr.select_one("td.tip")
        evaluation_cell = tr.select_one("td.evaluation")
        raw_cells = [normalize_text(td.get_text(" ", strip=True)) for td in cells]

        entry = dict(meta)
        entry.update(
            {
                "car_no": car_no,
                "player_name": player_name,
                "player_profile": profile,
                "prefecture": prefecture,
                "age": age,
                "term": term,
                "class": stats[0],
                "style": stats[1],
                "gear": stats[2],
                "score": stats[3],
                "s_count": stats[4],
                "b_count": stats[5],
                "nige_count": stats[6],
                "makuri_count": stats[7],
                "sashi_count": stats[8],
                "mark_count": stats[9],
                "first_count": stats[10],
                "second_count": stats[11],
                "third_count": stats[12],
                "outside_count": stats[13],
                "win_rate": stats[14],
                "top2_rate": stats[15],
                "top3_rate": stats[16],
                "prediction_mark": normalize_text(prediction_cell.get_text(" ", strip=True))
                if prediction_cell
                else "",
                "evaluation": normalize_text(evaluation_cell.get_text(" ", strip=True))
                if evaluation_cell
                else "",
                "raw_row_json": json.dumps(
                    {"row_class": tr.get("class", []), "cells": raw_cells},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        entries.append(entry)

    deduped = {int(entry["car_no"]): entry for entry in entries}
    entries = [deduped[key] for key in sorted(deduped)]
    if len(entries) < 5:
        raise CollectorError(f"too few entrants ({len(entries)}): {ref.url}")

    meta["entry_count"] = len(entries)
    return meta, entries


def collect(start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    if start > end:
        raise ValueError("start date must be <= end date")

    session = make_session()
    event_cards: dict[str, EventCardRef] = {}
    discovered: dict[str, RaceRef] = {}
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    skipped_outside_window = 0

    for day in daterange(start, end):
        daily_url = KDREAMS_DAILY.format(year=day.year, month=day.month, day=day.day)
        try:
            html = fetch_html(session, daily_url)
            for event_ref in discover_g3_event_cards_from_daily_html(
                html, daily_url, day.isoformat()
            ):
                event_cards.setdefault(event_ref.url, event_ref)
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

    for event_ref in sorted(event_cards.values(), key=lambda item: (item.discovered_on, item.url)):
        try:
            html = fetch_html(session, event_ref.url)
            for ref in discover_races_from_g3_racecard_html(
                html, event_ref.url, event_ref.discovered_on
            ):
                discovered.setdefault(ref.url, ref)
        except Exception as exc:
            failures.append(
                {
                    "stage": "racecard_discovery",
                    "discovered_on": event_ref.discovered_on,
                    "url": event_ref.url,
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

    race_map = {str(row["race_id"]): row for row in races if row.get("race_id")}
    races = list(race_map.values())
    valid_ids = set(race_map)
    entry_map = {
        (str(row["race_id"]), int(row["car_no"])): row
        for row in entries
        if str(row.get("race_id", "")) in valid_ids
    }
    entries = list(entry_map.values())

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
        "race_no",
        "race_type",
        "event_grade",
        "event_name",
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
        "race_no",
        "race_type",
        "event_grade",
        "event_name",
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

    write_csv(out_dir / "races.csv", races, race_fields)
    write_csv(out_dir / "entries.csv", entries, entry_fields)
    write_csv(out_dir / "failures.csv", failures, failure_fields)

    summary = {
        "target_grade": GRADE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "daily_source_template": KDREAMS_DAILY,
        "g3_event_cards": len(event_cards),
        "candidate_races": len(discovered),
        "parsed_races": len(races),
        "entry_rows": len(entries),
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "definition_note": "開催ページにG3と明示された開催の配下レースのみ。レース名からG3を推測しない。",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all KDreams race cards from explicitly labelled G3 meetings")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.25, help="polite delay between requests")
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

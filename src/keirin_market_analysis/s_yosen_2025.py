from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TARGET_RACE_TYPE = "Ｓ級予選"
KDREAMS_DAILY = "https://keirin.kdreams.jp/kaisai/{year:04d}/{month:02d}/{day:02d}/"
USER_AGENT = (
    "Mozilla/5.0 (compatible; keirin-market-analysis/0.1; "
    "+https://github.com/yuhevita-wq/keirin-market-analysis-)"
)


@dataclass(frozen=True)
class RaceRef:
    discovered_on: str
    race_no: int
    url: str


class CollectorError(RuntimeError):
    pass


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"})
    return session


def fetch_html(session: requests.Session, url: str, timeout: float = 30.0) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding
    return response.text


def expand_row_cells(row: Tag) -> list[Tag]:
    expanded: list[Tag] = []
    for cell in row.find_all(["th", "td"], recursive=False):
        try:
            colspan = max(1, int(cell.get("colspan", 1)))
        except (TypeError, ValueError):
            colspan = 1
        expanded.extend([cell] * colspan)
    return expanded


def canonical_race_url(base_url: str, href: str) -> str:
    absolute = urljoin(base_url, href)
    parts = urlsplit(absolute)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "pageType=result", ""))


def discover_s_yosen_from_daily_html(html: str, daily_url: str, discovered_on: str) -> list[RaceRef]:
    """Return race-card links whose published label is exactly 'Ｓ級予選'."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, RaceRef] = {}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row_index, row in enumerate(rows):
            expanded = expand_row_cells(row)
            if not expanded:
                continue
            labels = [normalize_text(cell.get_text(" ", strip=True)) for cell in expanded]
            target_indexes = [i for i, label in enumerate(labels) if label == TARGET_RACE_TYPE]
            if not target_indexes:
                continue

            for target_index in target_indexes:
                href = None
                for later in rows[row_index + 1 :]:
                    later_cells = expand_row_cells(later)
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

                url = canonical_race_url(daily_url, href)
                match = re.search(r"/racedetail/(\d{16})/", url)
                race_no = int(match.group(1)[-4:]) if match else target_index + 1
                found[url] = RaceRef(discovered_on=discovered_on, race_no=race_no, url=url)

    return sorted(found.values(), key=lambda item: (item.discovered_on, item.url, item.race_no))


def page_race_date(soup: BeautifulSoup) -> str:
    text = normalize_text(soup.get_text(" ", strip=True))
    match = re.search(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日\s*レース詳細", text)
    if not match:
        raise CollectorError("race date not found on race page")
    year, month, day = map(int, match.groups())
    return date(year, month, day).isoformat()


def extract_race_meta(soup: BeautifulSoup, ref: RaceRef) -> dict[str, object]:
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    track_match = re.match(r"^(.+?競輪)\s+レース詳細", title)
    track = track_match.group(1) if track_match else ""

    text = normalize_text(soup.get_text(" ", strip=True))
    start_match = re.search(r"発走予定\s*(\d{1,2}:\d{2})", text)
    deadline_match = re.search(r"投票締切\s*(\d{1,2}:\d{2})", text)
    race_id_match = re.search(r"/racedetail/(\d{16})/", ref.url)

    captured_at = datetime.now(timezone.utc).isoformat()
    return {
        "race_id": race_id_match.group(1) if race_id_match else "",
        "race_date": page_race_date(soup),
        "track": track,
        "race_no": ref.race_no,
        "race_type": TARGET_RACE_TYPE,
        "start_time": start_match.group(1) if start_match else "",
        "deadline": deadline_match.group(1) if deadline_match else "",
        "source_url": ref.url,
        "captured_at_utc": captured_at,
    }


def parse_profile(profile: str) -> tuple[str, str, str]:
    parts = [normalize_text(part) for part in profile.split("/")]
    if len(parts) != 3:
        return re.sub(r"\s+", "", profile), "", ""
    prefecture = re.sub(r"\s+", "", parts[0])
    return prefecture, parts[1], parts[2]


def pick_base_racecard_table(soup: BeautifulSoup) -> tuple[Tag, list[Tag]]:
    for table in soup.select("table.racecard_table"):
        header_text = normalize_text(" ".join(th.get_text(" ", strip=True) for th in table.find_all("th")))
        entrant_rows = [
            tr
            for tr in table.find_all("tr", recursive=False)
            if tr.select_one("td.num") is not None and tr.select_one("td.rider") is not None
        ]
        if (
            "直近4ヶ月の成績" in header_text
            and "競走得点" in header_text
            and "ギヤ" in header_text
            and len(entrant_rows) >= 5
        ):
            return table, entrant_rows
    raise CollectorError("base race-card table not found")


def cell_text(cells: list[Tag], index: int) -> str:
    if index < 0 or index >= len(cells):
        return ""
    return normalize_text(cells[index].get_text(" ", strip=True))


def extract_entries(html: str, ref: RaceRef) -> tuple[dict[str, object], list[dict[str, object]]]:
    soup = BeautifulSoup(html, "lxml")
    page_text = normalize_text(soup.get_text(" ", strip=True))
    if TARGET_RACE_TYPE not in page_text:
        raise CollectorError(f"target label missing from race page: {ref.url}")

    meta = extract_race_meta(soup, ref)
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
                "prediction_mark": normalize_text(prediction_cell.get_text(" ", strip=True)) if prediction_cell else "",
                "evaluation": normalize_text(evaluation_cell.get_text(" ", strip=True)) if evaluation_cell else "",
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


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    if start.year != 2025 or end.year != 2025:
        raise ValueError("this collector is intentionally restricted to 2025")
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
                # Daily pages show all days of an active meeting. Keep the earliest
                # discovery so day-1 S-class preliminary races are not overwritten by
                # the same card appearing again on day 2/final-day pages.
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
        "race_id",
        "race_date",
        "track",
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
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all exact-label S-class preliminary race cards for 2025")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--out-dir", default="data/2025/s_class_yosen")
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

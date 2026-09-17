#!/usr/bin/env python3
from __future__ import annotations

"""Build a same-origin cache of current KDreams race cards for Keirin Shogi.

This is an ingestion-only step. It deliberately does not read odds, popularity,
results, payouts, or historical race datasets. Existing KDreams HTML parsers are
reused where possible so the GitHub Pages frontend never has to fetch KDreams
cross-origin.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from keirin_market_analysis.line_formation_2025 import parse_line_formation_html
from keirin_market_analysis.s_yosen_2025 import (
    cell_text,
    fetch_html,
    make_session,
    normalize_text,
    parse_profile,
    pick_base_racecard_table,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/keirin-shogi/live-race-data.json"
KDREAMS_DAILY = "https://keirin.kdreams.jp/kaisai/{year:04d}/{month:02d}/{day:02d}/"
JST = ZoneInfo("Asia/Tokyo")


def canonical_race_url(base_url: str, href: str) -> str:
    absolute = urljoin(base_url, href)
    parts = urlsplit(absolute)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def discover_races(html: str, daily_url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, str] = {}
    for link in soup.find_all("a", href=True):
        href = str(link.get("href", ""))
        if "/racedetail/" not in href:
            continue
        url = canonical_race_url(daily_url, href)
        match = re.search(r"/racedetail/(\d{16})/", url)
        if match:
            found[match.group(1)] = url
    return found


def parse_race_type(text: str) -> str:
    patterns = (
        r"[ＳＳSＡAＬL]級\s*(?:初日特選|特選|選抜|予選|準決勝|一般|決勝)",
        r"(?:ガールズ|チャレンジ)\s*(?:予選|一般|準決勝|選抜|決勝)",
        r"(?:特予選|特一般|準決勝|特選|一般|決勝)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_text(match.group(0))
    return ""


def parse_meta(soup: BeautifulSoup, race_id: str, source_url: str) -> dict[str, object]:
    text = normalize_text(soup.get_text(" ", strip=True))
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    track_match = re.match(r"^(.+?競輪)\s+レース詳細", title)
    date_match = re.search(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    start_match = re.search(r"発走予定\s*(\d{1,2}:\d{2})", text)
    deadline_match = re.search(r"投票締切\s*(\d{1,2}:\d{2})", text)
    race_date = ""
    if date_match:
        y, m, d = map(int, date_match.groups())
        race_date = f"{y:04d}-{m:02d}-{d:02d}"
    return {
        "race_id": race_id,
        "race_date": race_date,
        "track": track_match.group(1) if track_match else "",
        "race_no": int(race_id[-4:]),
        "race_type": parse_race_type(text),
        "start_time": start_match.group(1) if start_match else "",
        "deadline": deadline_match.group(1) if deadline_match else "",
        "source_url": source_url,
    }


def parse_entries(html: str, race_id: str, source_url: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    soup = BeautifulSoup(html, "lxml")
    meta = parse_meta(soup, race_id, source_url)
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
        rider_index = cells.index(rider_cell)
        after = cells[rider_index + 1 :]
        stats = [cell_text(after, i) for i in range(17)]
        while len(stats) < 17:
            stats.append("")
        entries.append(
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
            }
        )

    entries = [dict(row) for _, row in sorted({int(row["car_no"]): row for row in entries}.items())]
    if len(entries) < 5:
        raise RuntimeError(f"too few entrants ({len(entries)})")
    meta["entry_count"] = len(entries)
    return meta, entries


def attach_line(entries: list[dict[str, object]], line: dict[str, object]) -> None:
    riders = line.get("riders", {})
    if not isinstance(riders, dict):
        riders = {}
    for entry in entries:
        rider = riders.get(int(entry["car_no"]), {})
        if not isinstance(rider, dict):
            rider = {}
        entry.update(
            {
                "line_id": rider.get("line_id", ""),
                "line_position": rider.get("line_position", ""),
                "line_size": rider.get("line_size", ""),
                "line_role": rider.get("line_role", ""),
            }
        )


def main() -> int:
    session = make_session()
    now = datetime.now(JST)
    # Current day + tomorrow. The workflow refreshes this repeatedly, keeping
    # Pages usable without a cross-origin browser request or a public CORS proxy.
    days = [(now + timedelta(days=offset)).date() for offset in (0, 1)]
    discovered: dict[str, str] = {}
    failures: list[dict[str, str]] = []

    for day in days:
        daily_url = KDREAMS_DAILY.format(year=day.year, month=day.month, day=day.day)
        try:
            discovered.update(discover_races(fetch_html(session, daily_url), daily_url))
        except Exception as exc:
            failures.append({"stage": "daily", "url": daily_url, "error": f"{type(exc).__name__}: {exc}"})

    races: list[dict[str, object]] = []
    for race_id, source_url in sorted(discovered.items()):
        try:
            html = fetch_html(session, source_url)
            meta, entries = parse_entries(html, race_id, source_url)
            line = parse_line_formation_html(html)
            attach_line(entries, line)
            races.append(
                {
                    **meta,
                    "line_status": line.get("status", ""),
                    "predicted_line_formation": line.get("formation", ""),
                    "line_provider": line.get("provider", ""),
                    "entries": entries,
                }
            )
        except Exception as exc:
            failures.append({"stage": "race", "url": source_url, "race_id": race_id, "error": f"{type(exc).__name__}: {exc}"})

    races.sort(key=lambda row: (str(row.get("race_date", "")), str(row.get("track", "")), int(row.get("race_no", 0))))
    payload = {
        "schema_version": 1,
        "generated_at_jst": now.isoformat(),
        "source": "Rakuten KDreams race card + published line forecast",
        "runtime_inputs": "current race card and line only; no odds, popularity, results, payouts, or historical race rows",
        "covered_dates": [day.isoformat() for day in days],
        "race_count": len(races),
        "failure_count": len(failures),
        "races": races,
        "failures": failures[:50],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "race_count": len(races), "failure_count": len(failures)}, ensure_ascii=False))
    return 0 if races else 2


if __name__ == "__main__":
    raise SystemExit(main())

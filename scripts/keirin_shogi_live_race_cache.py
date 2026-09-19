#!/usr/bin/env python3
from __future__ import annotations

"""Build a same-origin KDreams race-card cache for Keirin Shogi.

The fetch stage reads race cards and published line forecasts only. It does not
read target-race odds, popularity, results or payouts. Previously fetched race
cards are retained so finished races remain available for deterministic board
placement after their race day has passed.
"""

import json
import re
import subprocess
import sys
import unicodedata
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
BOARD_OUT = ROOT / "docs/keirin-shogi/live-board-data.json"
AUTO_PLACE = ROOT / "scripts/keirin_shogi_v37_auto_place.py"
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
    """Return the first race-class/stage label appearing in the supplied text.

    KDreams race-detail pages contain links and summaries for many other races.
    The old parser looped over pattern *types* first, so a later "S級予選"
    elsewhere on the page could beat the current race's earlier "S級特一般".
    Collect all supported matches and choose the earliest occurrence instead.
    """
    patterns = (
        r"[ＳＳSＡAＬL]級\s*(?:"
        r"初日特選|初特選|特別選抜予選|一次予選|二次予選[ＡAＢB]?|"
        r"特予選|特一般|優秀|ドリーム|選抜|特選|予選|準決勝|一般|決勝"
        r")",
        r"(?:ガールズ|チャレンジ)\s*(?:"
        r"特予選|特一般|予選|一般|準決勝|選抜|特選|決勝"
        r")",
        r"(?:"
        r"初日特選|初特選|特別選抜予選|一次予選|二次予選[ＡAＢB]?|"
        r"特予選|特一般|優秀|ドリーム|準決勝|特選|選抜|予選|一般|決勝"
        r")",
    )
    matches = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text))
    if not matches:
        return ""
    match = min(matches, key=lambda item: item.start())
    return normalize_text(match.group(0))


def parse_meeting_grade(text: str) -> str:
    """Return the first supported meeting grade appearing on the page.

    Do not prefer G3/G2/G1 by hard-coded order: a G2 race page can contain
    navigation text mentioning G3 later in the document.
    """
    value = unicodedata.normalize("NFKC", text).upper()
    compact = re.sub(r"\s+", "", value)
    pattern = re.compile(
        r"G(?:III|II|I|3|2|1)(?![A-Z])|F(?:II|I|2|1)(?![A-Z])"
    )
    match = pattern.search(compact)
    if not match:
        return ""
    token = match.group(0)
    mapping = {
        "GIII": "G3",
        "G3": "G3",
        "GII": "G2",
        "G2": "G2",
        "GI": "G1",
        "G1": "G1",
        "FII": "F2",
        "F2": "F2",
        "FI": "F1",
        "F1": "F1",
    }
    return mapping.get(token, "")


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
        "meeting_grade": parse_meeting_grade(text),
        "race_no": int(race_id[-4:]),
        # The HTML title identifies the target race and is much less polluted
        # by links to neighbouring races. Fall back to body text only when the
        # title does not expose the stage.
        "race_type": parse_race_type(title) or parse_race_type(text),
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
        # A withdrawn rider remains visible in the KDreams table but is
        # intentionally absent from the published line formation. Treat that
        # as a reduced field, never invent a line_id for the withdrawn rider.
        if "欠車" in full_rider_text or "欠場" in full_rider_text:
            withdrawn = meta.setdefault("withdrawn_car_numbers", [])
            if isinstance(withdrawn, list):
                withdrawn.append(car_no)
            continue
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
    if len(entries) < 3:
        raise RuntimeError(f"too few active entrants ({len(entries)})")
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


def previous_races() -> list[dict[str, object]]:
    if not OUT.exists():
        return []
    try:
        payload = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        return []
    races = payload.get("races", []) if isinstance(payload, dict) else []
    return [row for row in races if isinstance(row, dict) and row.get("race_id")]


def attach_auto_boards(payload: dict[str, object], failures: list[dict[str, str]]) -> None:
    result = subprocess.run(
        [sys.executable, str(AUTO_PLACE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    board_payload = None
    if BOARD_OUT.exists():
        try:
            board_payload = json.loads(BOARD_OUT.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(
                {"stage": "auto_place_read", "url": "", "error": f"{type(exc).__name__}: {exc}"}
            )
    if not isinstance(board_payload, dict):
        failures.append(
            {
                "stage": "auto_place",
                "url": "",
                "error": (result.stderr or result.stdout or f"exit {result.returncode}")[-1000:],
            }
        )
        payload["board_engine"] = {"status": "failed", "exit_code": result.returncode}
        return

    by_id = {str(row.get("race_id", "")): row for row in board_payload.get("races", [])}
    for race in payload.get("races", []):
        board = by_id.get(str(race.get("race_id", "")))
        if board is not None:
            race["auto_board"] = board
    payload["board_engine"] = {
        "status": "ok" if result.returncode == 0 else "partial",
        "engine": board_payload.get("engine", ""),
        "mode": board_payload.get("mode", ""),
        "latest_historical_state": board_payload.get("latest_historical_state", {}),
        "runtime_versions": board_payload.get("runtime_versions", {}),
        "guards": board_payload.get("guards", {}),
        "race_count": board_payload.get("race_count", 0),
        "failure_count": board_payload.get("failure_count", 0),
    }


def main() -> int:
    session = make_session()
    now = datetime.now(JST)
    days = [(now + timedelta(days=offset)).date() for offset in (0, 1)]
    target_dates = {day.isoformat() for day in days}
    discovered: dict[str, str] = {}
    failures: list[dict[str, str]] = []

    retained = previous_races()

    for day in days:
        daily_url = KDREAMS_DAILY.format(year=day.year, month=day.month, day=day.day)
        try:
            discovered.update(discover_races(fetch_html(session, daily_url), daily_url))
        except Exception as exc:
            failures.append({"stage": "daily", "url": daily_url, "error": f"{type(exc).__name__}: {exc}"})

    fresh_races: list[dict[str, object]] = []
    for race_id, source_url in sorted(discovered.items()):
        try:
            html = fetch_html(session, source_url)
            meta, entries = parse_entries(html, race_id, source_url)
            if str(meta.get("race_date", "")) not in target_dates:
                continue
            line = parse_line_formation_html(html)
            attach_line(entries, line)
            fresh_races.append(
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

    # New fetches replace the same race_id, while older finished races remain.
    merged = {str(row.get("race_id", "")): row for row in retained if row.get("race_id")}
    for row in fresh_races:
        merged[str(row.get("race_id", ""))] = row
    races = list(merged.values())
    races.sort(key=lambda row: (str(row.get("race_date", "")), str(row.get("track", "")), int(row.get("race_no", 0))))
    covered_dates = sorted({str(row.get("race_date", "")) for row in races if row.get("race_date")})

    payload = {
        "schema_version": 3,
        "generated_at_jst": now.isoformat(),
        "source": "Rakuten KDreams race card + published line forecast",
        "runtime_inputs": "race card and line only; no target-race odds, popularity, results or payouts",
        "active_fetch_dates": [day.isoformat() for day in days],
        "covered_dates": covered_dates,
        "retains_finished_races": True,
        "race_count": len(races),
        "fresh_race_count": len(fresh_races),
        "failure_count": len(failures),
        "races": races,
        "failures": failures[:50],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    attach_auto_boards(payload, failures)
    payload["failure_count"] = len(failures)
    payload["failures"] = failures[:50]
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "race_count": len(races),
                "fresh_race_count": len(fresh_races),
                "retained_race_count": max(0, len(races) - len(fresh_races)),
                "failure_count": len(failures),
                "board_engine": payload.get("board_engine", {}).get("status", "missing"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if races else 2


if __name__ == "__main__":
    raise SystemExit(main())

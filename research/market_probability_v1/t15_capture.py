"""Capture a KDreams live trifecta/trio market for a strict T-15 decision.

Requires a separately captured race-card JSON and a runner invoked at the right
time by the operator's scheduler. This script DOES NOT submit wagers.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keirin_market_analysis.historical_final_odds import (
    odds_url, parse_trifecta, parse_trio,
)
from keirin_market_analysis.s_yosen_2025 import (
    fetch_html, make_session, normalize_text,
)
from t15_decision import JST, checked_live, timestamp

SOURCE = "楽天Kドリームス / live odds / T-15 research"


def market_status(soup):
    header = soup.select_one("div.odds_header")
    if header is None:
        raise ValueError("odds_header not found; unsupported live page")
    text = normalize_text(header.get_text(" ", strip=True))
    if "確定オッズ" in text:
        raise ValueError("REFUSED: site has already published final odds")
    statuses = header.select("div.status")
    if len(statuses) < 3:
        raise ValueError("odds status blocks not available")
    out = {}
    for name, ix in (("3連単", 0), ("3連複", 2)):
        t = normalize_text(statuses[ix].get_text(" ", strip=True))
        match = re.search(r"(20\d{2})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})\s*現在", t)
        if match is None:
            raise ValueError(f"{name}: no explicit on-page quote timestamp")
        yyyy, mm, dd, hh, minute = (int(v) for v in match.groups())
        out[name] = datetime(yyyy, mm, dd, hh, minute, tzinfo=JST).isoformat()
    return out


def parse_live(html, request, captured):
    race = request["race"]
    if not race.get("source_url", "").startswith("https://keirin.kdreams.jp/"):
        raise ValueError("KDreams race detail URL required")
    entries = request["entries"]
    cars = sorted(int(e["car_no"]) for e in entries)
    if len(cars) < 5 or len(cars) > 7 or len(cars) != len(set(cars)):
        raise ValueError("must have 5–7 unique competitors")
    soup = BeautifulSoup(html, "lxml")
    market_times = market_status(soup)
    trf = parse_trifecta(soup, cars)
    tro = parse_trio(soup, cars)
    orders = list(itertools.permutations(cars, 3))
    subsets = list(itertools.combinations(cars, 3))
    if set(trf) != set(orders) or set(tro) != set(subsets):
        raise ValueError("live odds market combination coverage incomplete")
    for pool in (trf, tro):
        if any(state != "available" for value, state in pool.values()):
            raise ValueError("at least one live quote is unavailable; no backfill")
    result = {
        "race": dict(race),
        "entries": list(entries),
        "odds_phase": "live",
        "source_url": odds_url(race["source_url"]),
        "captured_at_jst": captured.astimezone(JST).isoformat(),
        "market_as_of_jst": market_times,
        "odds": {
            "3連単": {"-".join(map(str, k)): float(trf[k][0]) for k in orders},
            "3連複": {"=".join(map(str, k)): float(tro[k][0]) for k in subsets},
        },
    }
    checked_live(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--race-card", type=Path, required=True,
                   help="JSON with race metadata, confirmed deadline, card_as_of_jst and entries")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--html", type=Path, help="offline HTML source for parser checks only")
    args = p.parse_args()
    request = json.loads(args.race_card.read_text(encoding="utf-8"))
    race = request["race"]
    deadline = timestamp(race["deadline_jst"])
    reference = race.get("t15_reference", "race_start")
    if reference not in ("race_start", "sales_deadline"):
        raise ValueError("invalid T15 reference")
    cutoff = (timestamp(race["start_jst"]) if reference == "race_start" else deadline) - timedelta(minutes=15)
    if cutoff >= deadline:
        raise ValueError("T15 point falls after the sales deadline")
    clock_before = datetime.now(timezone.utc)
    if args.html is None and clock_before > cutoff:
        raise SystemExit("REFUSED: already after the decision cutoff")
    if args.html is None:
        # In live mode, use one request, no slow retry that can cross the cutoff.
        html = fetch_html(make_session(), odds_url(request["race"]["source_url"]))
    else:
        html = args.html.read_text(encoding="utf-8")
    captured = datetime.now(timezone.utc)
    snap = parse_live(html, request, captured)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(snap, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps({"race": request["race"]["race_id"],
                      "captured_at_jst": snap["captured_at_jst"],
                      "cutoff_jst": cutoff.astimezone(JST).isoformat(),
                      "trifecta_quotes": len(snap["odds"]["3連単"]),
                      "trio_quotes": len(snap["odds"]["3連複"]),
                      "path": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

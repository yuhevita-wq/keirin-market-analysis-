"""Live, odds-only race capture. No race results, athlete details or lineup inputs.

Use only with permission to automatically retrieve/store the upstream odds.
Fails closed on stale, incomplete, wrong-date, missing or changed quote tables.
No near-start guarantee is made by GitHub Actions scheduling.
"""
import argparse
import csv
import gzip
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

try:
    from odds_market.collect_all_odds import parse_all, FIELDS, JST, expected
except ModuleNotFoundError:
    from collect_all_odds import parse_all, FIELDS, JST, expected

RACE_ID_RX = re.compile(r"/racedetail/(\d{16})/")
SOURCE_STAMP = re.compile(r"(20\d\d/\d\d/\d\d)\s+(\d\d:\d\d)現在")
ALLOWED_HOST = "keirin.kdreams.jp"


def validate_url(url):
    parsed = urlparse(url)
    match = RACE_ID_RX.search(parsed.path)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST or not match:
        raise ValueError("Only HTTPS KDreams racedetail URLs with 16-digit ID are allowed")
    if parsed.username or parsed.password or parsed.port:
        raise ValueError("Credentials and port in source URL are forbidden")
    return match.group(1)


def request_url(original, counter):
    parsed = urlparse(original)
    query = [(k, v) for k, v in parse_qsl(parsed.query)
             if k not in ("_", "_ts", "pageType")]
    query.extend((("pageType", "odds"), ("_ts", str(counter))))
    return urlunparse(("https", ALLOWED_HOST, parsed.path, "", urlencode(query), ""))


def decode_live(html, race_id, now=None, max_age_minutes=4):
    """All quotes must be explicitly current, on this race's date, same source minute."""
    now = (now or datetime.now(JST)).astimezone(JST)
    soup = BeautifulSoup(html, "html.parser")
    label = soup.get_text(" ", strip=True)
    stamps = sorted(set(SOURCE_STAMP.findall(label)))
    if len(stamps) != 1:
        raise ValueError(f"Source market timestamps missing/inconsistent: {stamps}")
    source_time = datetime.strptime(" ".join(stamps[0]), "%Y/%m/%d %H:%M").replace(tzinfo=JST)
    age = (now - source_time).total_seconds() / 60
    if age < -1.5 or age > max_age_minutes:
        raise ValueError(f"STALE source odds: timestamp={source_time.isoformat()}, age={age:.1f}min")
    if "発売票数" not in label:
        raise ValueError("No live market vote-count headers; invalid odds page")
    if "確定オッズ" in label:
        raise ValueError("Final odds are not a live pre-close observation")
    records = parse_all(html, race_id, require_final=False)
    n = len({int(v) for row in records if row[2] == "3rentan" for v in row[3].split("-")})
    if len(records) != sum(map(len, expected(n).values())):
        raise ValueError("Not all sold combinations were collected")
    times = {row[6] for row in records}
    if len(times) != 1 or source_time.isoformat(timespec="seconds") not in times:
        raise ValueError("Market source timestamp mismatch")
    return records, source_time


def save_private_snapshot(records, root):
    """Immutable complete snapshot with only the seven schema fields; never public commit."""
    rid = records[0][0]
    stamp = records[0][6]
    safe_stamp = stamp.replace("-", "").replace(":", "").replace("+", "_")
    folder = root / rid
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (safe_stamp + ".csv.gz")
    if target.exists():
        return False, target
    tmp = target.with_suffix(".csv.gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(records)
    tmp.replace(target)
    return True, target


def collect_once(session, url, now=None, max_age_minutes=4):
    rid = validate_url(url)
    now = now or datetime.now(JST)
    fresh_url = request_url(url, int(now.timestamp()))
    response = session.get(
        fresh_url, timeout=18, allow_redirects=False,
        headers={"Cache-Control": "no-cache, max-age=0", "Pragma": "no-cache"})
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code} (no redirected/blocked pages accepted)")
    records, stamp = decode_live(response.text, rid, now=now,
                                 max_age_minutes=max_age_minutes)
    return records, stamp


def run(url, close_time, interval, max_age, out_dir, report_path=None):
    """Repeated acquisition until stop time; does not infer data on failure."""
    validate_url(url)
    session = requests.Session()
    session.headers.update({"User-Agent": "Keirin-Research-Odds-Collector/1.0"})
    seen = 0
    errors = 0
    while datetime.now(JST) < close_time:
        now = datetime.now(JST)
        try:
            records, source_time = collect_once(session, url, now, max_age)
            fresh, path = save_private_snapshot(records, out_dir)
            if fresh:
                seen += 1
                from odds_market.market_isolation import scan
                report = scan(records)
                print(json.dumps({
                    "status": "FRESH", "timestamp": source_time.isoformat(),
                    "rows": len(records), "counts": report["counts"],
                    "top": report["top"][:8]}, ensure_ascii=False), flush=True)
                if report_path:
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
            else:
                print(json.dumps({"status": "UNCHANGED", "source_time": source_time.isoformat()},
                                 ensure_ascii=False), flush=True)
        except (requests.RequestException, ValueError) as exc:
            errors += 1
            print(json.dumps({"status": "FAILED", "at": now.isoformat(),
                              "reason": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        time.sleep(min(interval, max(0, (close_time - datetime.now(JST)).total_seconds())))
    print(json.dumps({"status": "CLOSED", "fresh_snapshots": seen, "failures": errors},
                     ensure_ascii=False), flush=True)
    if not seen:
        raise SystemExit("No fresh complete live odds were collected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="KDreams HTTPS race-detail page")
    ap.add_argument("--close-jst", required=True, help="YYYY-MM-DDTHH:MM+09:00")
    ap.add_argument("--interval-seconds", type=int, default=45)
    ap.add_argument("--max-age-minutes", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("/tmp/keirin_live_odds_private"))
    ap.add_argument("--report", type=Path)
    ap.add_argument("--accept-terms", action="store_true")
    args = ap.parse_args()
    if not args.accept_terms:
        ap.error("Source authorization/terms must be explicitly confirmed")
    try:
        close = datetime.fromisoformat(args.close_jst)
        if close.utcoffset() != timedelta(hours=9):
            raise ValueError("Use Japan Standard Time +09:00")
    except ValueError as exc:
        ap.error(str(exc))
    if not 15 <= args.interval_seconds <= 300:
        ap.error("interval must be 15-300 seconds")
    if not 1 <= args.max_age_minutes <= 10:
        ap.error("max age must be 1-10 minutes")
    run(args.url, close, args.interval_seconds, args.max_age_minutes,
        args.out, args.report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Retrieve complete historical final trifecta odds matrices from public race pages.

Requires: pip install requests beautifulsoup4
Usage:
  python odds_market/collect_final.py --start 2024-07-06 --end 2024-07-06

Never treats top-50 popular odds or payouts as a complete odds market.
Check the source's usage/republication conditions before running or sharing data.
"""
import argparse
import csv
import gzip
import itertools
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://keirin.kdreams.jp"
JST = timezone(timedelta(hours=9))
RACE = re.compile(r"/racedetail/(\d{16})/")
DATE = re.compile(r"(20\d\d)年\s*(\d{1,2})月\s*(\d{1,2})日\s*レース詳細")
TIME = re.compile(r"(20\d\d/\d\d/\d\d\s+\d\d:\d\d)現在")
FIELDS = ["race_date","race_id","venue_code","race_no","bet_type","snapshot_type",
          "first","second","third","odds","display_cap","pool_tickets",
          "source_odds_at_jst","retrieved_at_utc","source_url"]


def http_get(session, url, sleep_seconds):
    time.sleep(sleep_seconds)
    response = session.get(url, timeout=35)
    response.raise_for_status()
    if urlparse(response.url).netloc != "keirin.kdreams.jp":
        raise ValueError("Unexpected host redirect")
    return response.text


def cells(tr):
    return [c.get_text(" ", strip=True).replace("\xa0", " ").strip()
            for c in tr.find_all(["th", "td"], recursive=False)]


def car(s):
    return int(s) if re.fullmatch("[1-9]", s) else None


def odd(s):
    s = s.strip().replace(",", "").replace("倍", "")
    return s if re.fullmatch(r"\d+(?:\.\d+)?", s) and float(s) >= 1 else None


def matrix(table, n):
    """Find one n-car first-place odds block using header complement and row labels."""
    trs = table.find_all("tr")
    for i, tr in enumerate(trs):
        head = [car(v) for v in cells(tr) if v]
        if len(head) != n - 1 or None in head or len(set(head)) != n - 1:
            continue
        complement = set(range(1, n + 1)) - set(head)
        if len(complement) != 1:
            continue
        first = complement.pop()
        found = {}
        for tr2 in trs[i + 1:]:
            row = cells(tr2)
            for offset in (0, 1):
                if len(row) < offset + n:
                    continue
                second = car(row[offset])
                if second not in head:
                    continue
                odds_row = row[offset + 1:offset + n]
                candidate = {}
                for j, third in enumerate(head):
                    val = odd(odds_row[j])
                    if second == third:
                        if val is not None:
                            break
                    elif val is None:
                        break
                    else:
                        candidate[(first, second, third)] = val
                else:
                    found.update(candidate)
                    break
        if len(found) == (n - 1) * (n - 2):
            return found
    return None


def parse_final_html(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    if "確定オッズ" not in text:
        raise ValueError("Not confirmed final odds")
    for n in range(9, 2, -1):
        blocks = {}
        for table in soup.find_all("table"):
            result = matrix(table, n)
            if result:
                first = next(iter(result))[0]
                if first in blocks and blocks[first] != result:
                    raise ValueError("Conflicting odds for one first-place car")
                blocks[first] = result
        if set(blocks) != set(range(1, n + 1)):
            continue
        merged = {}
        for result in blocks.values():
            merged.update(result)
        if set(merged) != set(itertools.permutations(range(1, n + 1), 3)):
            continue
        stamps = TIME.findall(text)
        sales = re.search(r"発売票数\s*([\d,]+)", text)
        return n, merged, (stamps[0].replace("/", "-") if stamps else ""), (
            int(sales.group(1).replace(",", "")) if sales else None)
    raise ValueError("Incomplete or unreadable trifecta market")


def discover_races(html, target):
    soup = BeautifulSoup(html, "html.parser")
    urls = {}
    for a in soup.find_all("a", href=True):
        u = urljoin(BASE, a["href"])
        p = urlparse(u)
        m = RACE.search(p.path)
        if p.netloc != "keirin.kdreams.jp" or not m:
            continue
        rid = m.group(1)
        meeting = date.fromisoformat(rid[2:6] + "-" + rid[6:8] + "-" + rid[8:10])
        stage = int(rid[10:12])
        if not 1 <= stage <= 7 or meeting + timedelta(days=stage-1) != target:
            continue
        urls[rid] = p._replace(query="pageType=odds", fragment="").geturl()
    return dict(sorted(urls.items()))


def collect_race(session, d, rid, url, sleep_seconds):
    html = http_get(session, url, sleep_seconds)
    m = DATE.search(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
    if not m or date(*map(int, m.groups())) != d:
        raise ValueError("Actual race date differs from the index")
    n, market, stamp, volume = parse_final_html(html)
    seen = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = []
    for (a,b,c), price in sorted(market.items()):
        result.append(dict(race_date=d.isoformat(), race_id=rid,
            venue_code=rid[:2], race_no=int(rid[-4:]), bet_type="3rentan",
            snapshot_type="final", first=a, second=b, third=c,
            odds=float(price), display_cap=int(price=="9999.9"),
            pool_tickets=volume if volume is not None else "",
            source_odds_at_jst=stamp, retrieved_at_utc=seen, source_url=url))
    assert len(result) == n * (n - 1) * (n - 2)
    return result


def write_partition(root, d, rows, failures, discovered):
    directory = root / str(d.year)
    directory.mkdir(parents=True, exist_ok=True)
    stem = d.isoformat()
    if rows:
        tmp = directory / (stem + ".csv.gz.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(directory / (stem + ".csv.gz"))
    meta = dict(date=stem, bet_type="3rentan", snapshot_type="final",
                discovered_races=discovered,
                complete_races=len(set(r["race_id"] for r in rows)),
                odds_rows=len(rows), failures=failures)
    (directory / (stem + ".manifest.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    return not failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--out", default="data/odds_final")
    ap.add_argument("--sleep", type=float, default=2)
    ap.add_argument("--max-days", type=int, default=7)
    ap.add_argument("--allow-failures", action="store_true")
    args = ap.parse_args()
    if args.end < args.start or (args.end-args.start).days + 1 > args.max_days:
        ap.error("Bad range; split historical collection into <=7-day batches")
    if args.end >= datetime.now(JST).date():
        ap.error("Final odds only: date must be completed before today")
    if args.sleep < 1:
        ap.error("Keep at least 1-second spacing between requests")
    session = requests.Session()
    session.headers.update({"User-Agent":"KeirinMarketResearch/1.0",
                            "Accept-Language":"ja-JP,ja;q=0.9"})
    succeeded = True
    for delta in range((args.end-args.start).days + 1):
        day = args.start + timedelta(days=delta)
        index = BASE + "/kaisai/" + day.strftime("%Y/%m/%d") + "/"
        failures, rows, urls = [], [], {}
        try:
            urls = discover_races(http_get(session, index, args.sleep), day)
            if not urls:
                raise ValueError("No race URL found; not interpreted as zero races")
            for rid, url in urls.items():
                try:
                    rows.extend(collect_race(session, day, rid, url, args.sleep))
                except (ValueError, requests.RequestException) as error:
                    failures.append(dict(race_id=rid, url=url, error=str(error)))
        except (ValueError, requests.RequestException) as error:
            failures.append(dict(race_id="", url=index, error=str(error)))
        succeeded &= write_partition(Path(args.out), day, rows, failures, len(urls))
    if not succeeded and not args.allow_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

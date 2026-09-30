from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.yen-joy.net"
SERIES_RE = re.compile(r"^https://www\.yen-joy\.net/kaisai/race/(\d{6})/(\d{2})/(\d{8})/?$")
DAY_RE = re.compile(
    r"^https://www\.yen-joy\.net/kaisai/race/"
    r"(?:forecast(?:/compare)?|result(?:/detail)?)/"
    r"(\d{6})/(\d{2})/(\d{8})/(\d{8})(?:/\d+)?/?$"
)


@dataclass(frozen=True, order=True)
class RaceDay:
    ym: str
    venue_code: str
    start_date: str
    race_date: str

    @property
    def day_id(self) -> str:
        return f"{self.race_date}_{self.venue_code}_{self.start_date}"

    @property
    def compare_url(self) -> str:
        return (
            f"{BASE}/kaisai/race/forecast/compare/"
            f"{self.ym}/{self.venue_code}/{self.start_date}/{self.race_date}"
        )

    @property
    def result_url(self) -> str:
        return (
            f"{BASE}/kaisai/race/result/"
            f"{self.ym}/{self.venue_code}/{self.start_date}/{self.race_date}"
        )


class Client:
    def __init__(self, delay: float, cache_dir: Path) -> None:
        self.delay = max(0.0, delay)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; keirin-market-analysis/0.1; "
                    "research collector; +https://github.com/yuhevita-wq/keirin-market-analysis-)"
                ),
                "Accept-Language": "ja,en;q=0.7",
            }
        )
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            status=4,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _cache_path(self, url: str) -> Path:
        import hashlib

        h = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.html.gz"

    def get(self, url: str, *, allow_missing: bool = False) -> str | None:
        cache = self._cache_path(url)
        if cache.exists():
            with gzip.open(cache, "rt", encoding="utf-8") as f:
                return f.read()

        wait = self.delay - (time.monotonic() - self.last_request_at)
        if wait > 0:
            time.sleep(wait)

        r = self.session.get(url, timeout=40)
        self.last_request_at = time.monotonic()
        if allow_missing and r.status_code in (404, 410):
            return None
        r.raise_for_status()
        text = r.text
        with gzip.open(cache, "wt", encoding="utf-8") as f:
            f.write(text)
        return text


def month_iter(start_ym: str, end_ym: str) -> Iterable[str]:
    start = datetime.strptime(start_ym, "%Y%m")
    end = datetime.strptime(end_ym, "%Y%m")
    if start > end:
        raise ValueError("start month must be <= end month")
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}{m:02d}"
        m += 1
        if m == 13:
            y += 1
            m = 1


def links_from_html(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        u = urljoin(base_url, a["href"]).split("#", 1)[0]
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def discover_series(client: Client, ym: str) -> list[str]:
    url = f"{BASE}/kaisai/{ym}01"
    html = client.get(url)
    assert html is not None
    series = [u.rstrip("/") for u in links_from_html(html, url) if SERIES_RE.match(u.rstrip("/"))]
    return sorted(set(series))


def discover_days(client: Client, series_url: str) -> list[RaceDay]:
    html = client.get(series_url)
    assert html is not None
    days: set[RaceDay] = set()
    for u in links_from_html(html, series_url):
        m = DAY_RE.match(u.rstrip("/"))
        if m:
            days.add(RaceDay(*m.groups()))
    return sorted(days)


def clean_text(s: str) -> str:
    return " ".join(s.replace("\u3000", " ").split())


def table_record(table, idx: int) -> dict:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    caption = table.find("caption")
    return {
        "index": idx,
        "class": list(table.get("class", [])),
        "caption": clean_text(caption.get_text(" ", strip=True)) if caption else "",
        "rows": rows,
    }


def extract_page(html: str, url: str, day: RaceDay, page_type: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = clean_text(tag.get_text(" ", strip=True))
        if text and text not in headings:
            headings.append(text)
    tables = []
    for i, table in enumerate(soup.find_all("table")):
        rec = table_record(table, i)
        if rec["rows"]:
            tables.append(rec)
    return {
        "schema_version": 1,
        "source": "yenjoy",
        "page_type": page_type,
        "url": url,
        "ym": day.ym,
        "venue_code": day.venue_code,
        "start_date": day.start_date,
        "race_date": day.race_date,
        "day_id": day.day_id,
        "title": title,
        "headings": headings[:40],
        "tables": tables,
    }


def write_jsonl_gz(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def collect(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    client = Client(args.delay, Path(args.cache_dir))

    all_series: list[str] = []
    for ym in month_iter(args.start_month, args.end_month):
        found = discover_series(client, ym)
        print(f"[month] {ym}: series={len(found)}", flush=True)
        all_series.extend(found)

    all_series = sorted(set(all_series))
    if args.max_series:
        all_series = all_series[: args.max_series]

    all_days: set[RaceDay] = set()
    series_errors: list[dict] = []
    for i, series_url in enumerate(all_series, 1):
        try:
            days = discover_days(client, series_url)
            print(f"[series {i}/{len(all_series)}] {series_url} days={len(days)}", flush=True)
            all_days.update(days)
        except Exception as e:  # continue bulk collection and report manifest
            series_errors.append({"url": series_url, "error": repr(e)})
            print(f"[series-error] {series_url}: {e!r}", flush=True)

    days = sorted(all_days)
    if args.max_days:
        days = days[: args.max_days]

    records: list[dict] = []
    page_errors: list[dict] = []
    for i, day in enumerate(days, 1):
        for page_type, url in (("entries_compare", day.compare_url), ("results", day.result_url)):
            try:
                html = client.get(url, allow_missing=True)
                if html is None:
                    print(f"[missing] {url}", flush=True)
                    continue
                rec = extract_page(html, url, day, page_type)
                records.append(rec)
                print(
                    f"[day {i}/{len(days)}] {page_type} {day.day_id} tables={len(rec['tables'])}",
                    flush=True,
                )
            except Exception as e:
                page_errors.append({"url": url, "error": repr(e)})
                print(f"[page-error] {url}: {e!r}", flush=True)

    data_path = output / f"yenjoy_{args.start_month}_{args.end_month}.jsonl.gz"
    n = write_jsonl_gz(data_path, records)
    manifest = {
        "schema_version": 1,
        "source": "yenjoy",
        "start_month": args.start_month,
        "end_month": args.end_month,
        "delay_seconds": args.delay,
        "series_count": len(all_series),
        "day_count": len(days),
        "page_count": n,
        "series_errors": series_errors,
        "page_errors": page_errors,
        "data_file": data_path.name,
        "note": "Only structured table cells/headings are stored; raw HTML is not published in the output.",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk collect factual YenJoy race-entry/result tables")
    p.add_argument("--start-month", required=True, help="YYYYMM")
    p.add_argument("--end-month", required=True, help="YYYYMM")
    p.add_argument("--output", default="results/yenjoy_bulk")
    p.add_argument("--cache-dir", default=".cache/yenjoy")
    p.add_argument("--delay", type=float, default=0.8, help="minimum seconds between network requests")
    p.add_argument("--max-series", type=int, default=0, help="0 = unlimited")
    p.add_argument("--max-days", type=int, default=0, help="0 = unlimited")
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())

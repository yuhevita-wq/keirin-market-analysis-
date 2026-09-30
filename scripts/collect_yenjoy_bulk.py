from __future__ import annotations

import argparse
import gzip
import hashlib
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
EDITORIAL_HEADER_PREFIXES = ("本社", "取材班", "デスク")


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
                    "Mozilla/5.0 (compatible; keirin-market-analysis/0.3; "
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
        r.encoding = "utf-8"
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
    return " ".join(s.replace("\u3000", " ").replace("\xa0", " ").split())


def table_record(table, idx: int) -> dict:
    """Extract only rows/cells whose nearest table is this table, excluding nested mini-tables."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        cells = []
        for c in tr.find_all(["th", "td"]):
            if c.find_parent("tr") is tr and c.find_parent("table") is table:
                cells.append(clean_text(c.get_text(" ", strip=True)))
        if cells:
            rows.append(cells)
    caption = table.find("caption")
    return {
        "index": idx,
        "class": list(table.get("class", [])),
        "caption": clean_text(caption.get_text(" ", strip=True)) if caption else "",
        "rows": rows,
    }


def strip_editorial_columns(item: dict) -> dict:
    rows = item.get("rows", [])
    if not rows:
        return item
    header = rows[0]
    drop = {
        i
        for i, label in enumerate(header)
        if label.startswith(EDITORIAL_HEADER_PREFIXES) or "並び替え選択データ" in label
    }
    if not drop:
        return item
    cleaned = []
    for row in rows:
        cleaned.append([value for i, value in enumerate(row) if i not in drop])
    item = dict(item)
    item["rows"] = cleaned
    item["dropped_columns"] = sorted(drop)
    return item


def page_meta(soup: BeautifulSoup, url: str, day: RaceDay, page_type: str) -> dict:
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    return {
        "schema_version": 3,
        "source": "yenjoy",
        "page_type": page_type,
        "url": url,
        "ym": day.ym,
        "venue_code": day.venue_code,
        "start_date": day.start_date,
        "race_date": day.race_date,
        "day_id": day.day_id,
        "title": title,
    }


def extract_entries(html: str, url: str, day: RaceDay) -> dict:
    soup = BeautifulSoup(html, "lxml")
    rec = page_meta(soup, url, day, "entries_compare")
    tables = []
    for i, table in enumerate(soup.find_all("table")):
        if table.find_parent("table") is not None:
            continue
        classes = set(table.get("class", []))
        if "result-table" in classes:
            continue
        item = table_record(table, i)
        if item["rows"]:
            tables.append(strip_editorial_columns(item))
    rec["tables"] = tables
    return rec


def extract_results(html: str, url: str, day: RaceDay) -> dict:
    soup = BeautifulSoup(html, "lxml")
    rec = page_meta(soup, url, day, "results")
    tables = [t for t in soup.find_all("table") if t.find_parent("table") is None]
    races: list[dict] = []
    race_no = 0
    for i, table in enumerate(tables):
        classes = set(table.get("class", []))
        if "result-table" not in classes:
            continue
        race_no += 1
        finish = table_record(table, i)
        payout = None
        for j in range(i + 1, min(i + 4, len(tables))):
            next_classes = set(tables[j].get("class", []))
            if "result-table" in next_classes:
                break
            candidate = table_record(tables[j], j)
            if candidate["rows"]:
                payout = candidate
                break
        races.append(
            {
                "race_no": race_no,
                "result_detail_url": f"{url}/detail/{race_no}",
                "finish": finish,
                "payout": payout,
            }
        )
    rec["races"] = races
    return rec


def in_requested_month(day: RaceDay, start_month: str, end_month: str) -> bool:
    return start_month <= day.race_date[:6] <= end_month


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
            all_days.update(d for d in days if in_requested_month(d, args.start_month, args.end_month))
        except Exception as e:
            series_errors.append({"url": series_url, "error": repr(e)})
            print(f"[series-error] {series_url}: {e!r}", flush=True)

    days = sorted(all_days)
    if args.max_days:
        days = days[: args.max_days]

    entries_path = output / f"entries_{args.start_month}_{args.end_month}.jsonl.gz"
    results_path = output / f"results_{args.start_month}_{args.end_month}.jsonl.gz"
    page_errors: list[dict] = []
    entries_count = 0
    results_count = 0
    result_race_count = 0

    with gzip.open(entries_path, "wt", encoding="utf-8") as ef, gzip.open(
        results_path, "wt", encoding="utf-8"
    ) as rf:
        for i, day in enumerate(days, 1):
            try:
                html = client.get(day.compare_url, allow_missing=True)
                if html is not None:
                    rec = extract_entries(html, day.compare_url, day)
                    ef.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                    entries_count += 1
                    print(
                        f"[day {i}/{len(days)}] entries {day.day_id} tables={len(rec['tables'])}",
                        flush=True,
                    )
                else:
                    print(f"[missing] {day.compare_url}", flush=True)
            except Exception as e:
                page_errors.append({"url": day.compare_url, "error": repr(e)})
                print(f"[page-error] {day.compare_url}: {e!r}", flush=True)

            try:
                html = client.get(day.result_url, allow_missing=True)
                if html is not None:
                    rec = extract_results(html, day.result_url, day)
                    rf.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                    results_count += 1
                    result_race_count += len(rec["races"])
                    print(
                        f"[day {i}/{len(days)}] results {day.day_id} races={len(rec['races'])}",
                        flush=True,
                    )
                else:
                    print(f"[missing] {day.result_url}", flush=True)
            except Exception as e:
                page_errors.append({"url": day.result_url, "error": repr(e)})
                print(f"[page-error] {day.result_url}: {e!r}", flush=True)

    manifest = {
        "schema_version": 3,
        "source": "yenjoy",
        "start_month": args.start_month,
        "end_month": args.end_month,
        "delay_seconds": args.delay,
        "series_count": len(all_series),
        "day_count": len(days),
        "entries_day_records": entries_count,
        "results_day_records": results_count,
        "result_race_count": result_race_count,
        "series_errors": series_errors,
        "page_errors": page_errors,
        "files": [entries_path.name, results_path.name],
        "note": (
            "Structured entry/player data and result/payout data are stored separately. "
            "Editorial prediction-mark columns are removed. Raw HTML stays only in the job cache."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk collect YenJoy entry and result tables")
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

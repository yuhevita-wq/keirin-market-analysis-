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
    r"(?:forecast(?:/(?:compare|line|detail))?|result(?:/detail)?)/"
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
    def line_url(self) -> str:
        return (
            f"{BASE}/kaisai/race/forecast/line/"
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
                    "Mozilla/5.0 (compatible; keirin-market-analysis/0.4; "
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
        return self.cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.html.gz"

    def get(self, url: str, *, allow_missing: bool = False) -> str | None:
        cache = self._cache_path(url)
        if cache.exists():
            with gzip.open(cache, "rt", encoding="utf-8") as f:
                return f.read()

        wait = self.delay - (time.monotonic() - self.last_request_at)
        if wait > 0:
            time.sleep(wait)
        r = self.session.get(url, timeout=60)
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


def clean_text(s: str) -> str:
    return " ".join(s.replace("\u3000", " ").replace("\xa0", " ").split())


def compact(s: str) -> str:
    return re.sub(r"\s+", "", clean_text(s))


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
    return sorted(
        set(u.rstrip("/") for u in links_from_html(html, url) if SERIES_RE.match(u.rstrip("/")))
    )


def discover_days(client: Client, series_url: str) -> list[RaceDay]:
    html = client.get(series_url)
    assert html is not None
    days: set[RaceDay] = set()
    for u in links_from_html(html, series_url):
        m = DAY_RE.match(u.rstrip("/"))
        if m:
            days.add(RaceDay(*m.groups()))
    return sorted(days)


def table_record(table, idx: int) -> dict:
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
    return {"index": idx, "class": list(table.get("class", [])), "rows": rows}


def parse_int(value: str) -> int | None:
    m = re.search(r"\d+", value or "")
    return int(m.group()) if m else None


def is_basic_lineup_table(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False
    labels = [compact(r[-1]) for r in rows if r]
    return "車" in labels and "選手名" in labels and "府県" in labels and "級班" in labels


def extract_basic_entrants(rows: list[list[str]]) -> list[dict]:
    by_label: dict[str, list[str]] = {}
    for row in rows:
        if len(row) < 2:
            continue
        by_label[compact(row[-1])] = row[:-1]

    cars = by_label.get("車", [])
    names = by_label.get("選手名", [])
    ages = by_label.get("年齢", [])
    prefs = by_label.get("府県", [])
    terms = by_label.get("期別", [])
    classes = by_label.get("級班", [])
    n = min(len(cars), len(names))
    entrants: list[dict] = []
    for i in range(n):
        entrants.append(
            {
                "car_no": parse_int(cars[i]),
                "racer": names[i],
                "age": parse_int(ages[i]) if i < len(ages) else None,
                "prefecture": prefs[i] if i < len(prefs) else None,
                "term": parse_int(terms[i]) if i < len(terms) else None,
                "class": classes[i] if i < len(classes) else None,
            }
        )
    entrants.sort(key=lambda x: (x["car_no"] is None, x["car_no"] or 99))
    return entrants


def extract_entries(html: str, url: str, day: RaceDay) -> dict:
    soup = BeautifulSoup(html, "lxml")
    tables = [t for t in soup.find_all("table") if t.find_parent("table") is None]
    records = [table_record(t, i) for i, t in enumerate(tables)]
    races: list[dict] = []

    for i, item in enumerate(records):
        rows = item["rows"]
        if not is_basic_lineup_table(rows):
            continue
        race_no = len(races) + 1
        initial_line = None
        final_bs = None
        for nxt in records[i + 1 : i + 3]:
            if not nxt["rows"] or not nxt["rows"][0]:
                continue
            label = compact(nxt["rows"][0][0])
            raw = nxt["rows"][0][1] if len(nxt["rows"][0]) > 1 else ""
            if label.startswith("初周"):
                initial_line = raw
            elif label.startswith("最終BS"):
                final_bs = raw
        races.append(
            {
                "race_no": race_no,
                "entrants": extract_basic_entrants(rows),
                "initial_line_raw": initial_line,
                "final_bs_raw": final_bs,
            }
        )

    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    return {
        "schema_version": 4,
        "source": "yenjoy",
        "page_type": "entries_line",
        "url": url,
        "ym": day.ym,
        "venue_code": day.venue_code,
        "start_date": day.start_date,
        "race_date": day.race_date,
        "day_id": day.day_id,
        "title": title,
        "races": races,
    }


def extract_results(html: str, url: str, day: RaceDay) -> dict:
    soup = BeautifulSoup(html, "lxml")
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
            if "result-table" in set(tables[j].get("class", [])):
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
    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    return {
        "schema_version": 4,
        "source": "yenjoy",
        "page_type": "results",
        "url": url,
        "ym": day.ym,
        "venue_code": day.venue_code,
        "start_date": day.start_date,
        "race_date": day.race_date,
        "day_id": day.day_id,
        "title": title,
        "races": races,
    }


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
    entries_count = results_count = entry_race_count = result_race_count = 0

    with gzip.open(entries_path, "wt", encoding="utf-8") as ef, gzip.open(
        results_path, "wt", encoding="utf-8"
    ) as rf:
        for i, day in enumerate(days, 1):
            try:
                html = client.get(day.line_url, allow_missing=True)
                if html is not None:
                    rec = extract_entries(html, day.line_url, day)
                    ef.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
                    entries_count += 1
                    entry_race_count += len(rec["races"])
                    print(
                        f"[day {i}/{len(days)}] entries {day.day_id} races={len(rec['races'])}",
                        flush=True,
                    )
                else:
                    print(f"[missing] {day.line_url}", flush=True)
            except Exception as e:
                page_errors.append({"url": day.line_url, "error": repr(e)})
                print(f"[page-error] {day.line_url}: {e!r}", flush=True)

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
        "schema_version": 4,
        "source": "yenjoy",
        "start_month": args.start_month,
        "end_month": args.end_month,
        "delay_seconds": args.delay,
        "series_count": len(all_series),
        "day_count": len(days),
        "entries_day_records": entries_count,
        "entry_race_count": entry_race_count,
        "results_day_records": results_count,
        "result_race_count": result_race_count,
        "series_errors": series_errors,
        "page_errors": page_errors,
        "files": [entries_path.name, results_path.name],
        "note": (
            "Entry data come from the all-race lineup page, giving complete daily fields for car number, "
            "rider name, age, prefecture, term and class plus initial/final lineup text. "
            "Result/payout data are stored separately. Raw HTML is not uploaded."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk collect complete YenJoy lineups and results")
    p.add_argument("--start-month", required=True, help="YYYYMM")
    p.add_argument("--end-month", required=True, help="YYYYMM")
    p.add_argument("--output", default="results/yenjoy_bulk")
    p.add_argument("--cache-dir", default=".cache/yenjoy")
    p.add_argument("--delay", type=float, default=1.0, help="minimum seconds between network requests")
    p.add_argument("--max-series", type=int, default=0, help="0 = unlimited")
    p.add_argument("--max-days", type=int, default=0, help="0 = unlimited")
    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TARGET_RACE_TYPE = "Ｓ級予選"
KDreamsDaily = "https://keirin.kdreams.jp/kaisai/{year:04d}/{month:02d}/{day:02d}/"
USER_AGENT = (
    "Mozilla/5.0 (compatible; keirin-market-analysis/0.1; "
    "+https://github.com/yuhevita-wq/keirin-market-analysis-)"
)

PREFECTURES = (
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島",
    "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜",
    "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
    "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口",
    "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎",
    "熊本", "大分", "宮崎", "鹿児島", "沖縄",
)


@dataclass(frozen=True)
class RaceRef:
    race_date: str
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


def discover_s_yosen_from_daily_html(html: str, daily_url: str, race_date: str) -> list[RaceRef]:
    """Return only rows whose published race label is exactly 'Ｓ級予選'."""
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
                    links = later_cells[target_index].find_all("a", href=True)
                    for link in links:
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
                if match:
                    race_no = int(match.group(1)[-4:])
                else:
                    race_no = target_index + 1
                found[url] = RaceRef(race_date=race_date, race_no=race_no, url=url)

    return sorted(found.values(), key=lambda item: (item.race_date, item.url, item.race_no))


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns: list[str] = []
    if isinstance(df.columns, pd.MultiIndex):
        for tup in df.columns:
            parts: list[str] = []
            for part in tup:
                text = normalize_text(part)
                if not text or text.startswith("Unnamed:"):
                    continue
                if not parts or parts[-1] != text:
                    parts.append(text)
            columns.append("__".join(parts) or "column")
    else:
        columns = [normalize_text(col) or "column" for col in df.columns]

    seen: dict[str, int] = {}
    unique: list[str] = []
    for col in columns:
        count = seen.get(col, 0)
        seen[col] = count + 1
        unique.append(col if count == 0 else f"{col}__{count + 1}")

    result = df.copy()
    result.columns = unique
    return result


def find_column(columns: Iterable[str], needle: str) -> str | None:
    for col in columns:
        if needle in col:
            return col
    return None


def split_player_profile(raw: str) -> tuple[str, str]:
    raw = normalize_text(raw)
    prefecture_pattern = "|".join(map(re.escape, PREFECTURES))
    match = re.search(rf"^(.*?)(?:\s*)({prefecture_pattern})(?:\s*)[/／]", raw)
    if match:
        return match.group(1).strip(), raw[match.start(2) :].strip()
    return raw, ""


def extract_race_meta(soup: BeautifulSoup, ref: RaceRef) -> dict[str, object]:
    title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    track_match = re.match(r"^(.+?競輪)\s+レース詳細", title)
    track = track_match.group(1) if track_match else ""

    text = normalize_text(soup.get_text(" ", strip=True))
    start_match = re.search(r"発走予定\s*(\d{1,2}:\d{2})", text)
    deadline_match = re.search(r"投票締切\s*(\d{1,2}:\d{2})", text)
    race_id_match = re.search(r"/racedetail/(\d{16})/", ref.url)

    return {
        "race_id": race_id_match.group(1) if race_id_match else "",
        "race_date": ref.race_date,
        "track": track,
        "race_no": ref.race_no,
        "race_type": TARGET_RACE_TYPE,
        "start_time": start_match.group(1) if start_match else "",
        "deadline": deadline_match.group(1) if deadline_match else "",
        "source_url": ref.url,
    }


def extract_entries(html: str, ref: RaceRef) -> tuple[dict[str, object], list[dict[str, object]]]:
    soup = BeautifulSoup(html, "lxml")
    page_text = normalize_text(soup.get_text(" ", strip=True))
    if TARGET_RACE_TYPE not in page_text:
        raise CollectorError(f"target label missing from race page: {ref.url}")

    meta = extract_race_meta(soup, ref)
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError as exc:
        raise CollectorError(f"no tables found: {ref.url}") from exc

    candidate: pd.DataFrame | None = None
    for table in tables:
        flat = flatten_columns(table)
        cols = list(flat.columns)
        if find_column(cols, "選手名") and find_column(cols, "車番"):
            if candidate is None or len(flat) > len(candidate):
                candidate = flat

    if candidate is None:
        raise CollectorError(f"entrant table not found: {ref.url}")

    car_col = find_column(candidate.columns, "車番")
    player_col = find_column(candidate.columns, "選手名")
    class_col = find_column(candidate.columns, "級班")
    style_col = find_column(candidate.columns, "脚質")
    gear_col = find_column(candidate.columns, "ギヤ") or find_column(candidate.columns, "ギア")
    score_col = find_column(candidate.columns, "競走得点")
    if car_col is None or player_col is None:
        raise CollectorError(f"required entrant columns missing: {ref.url}")

    entries: list[dict[str, object]] = []
    for _, row in candidate.iterrows():
        car_raw = normalize_text(row.get(car_col, ""))
        car_match = re.search(r"(?:^|\D)([1-9])(?:\D|$)", car_raw)
        if not car_match:
            continue
        car_no = int(car_match.group(1))
        player_raw = normalize_text(row.get(player_col, ""))
        player_name, profile = split_player_profile(player_raw)
        raw_row = {str(k): normalize_text(v) for k, v in row.to_dict().items()}

        entry = dict(meta)
        entry.update(
            {
                "car_no": car_no,
                "player_name": player_name,
                "player_profile": profile,
                "class": normalize_text(row.get(class_col, "")) if class_col else "",
                "style": normalize_text(row.get(style_col, "")) if style_col else "",
                "gear": normalize_text(row.get(gear_col, "")) if gear_col else "",
                "score": normalize_text(row.get(score_col, "")) if score_col else "",
                "raw_row_json": json.dumps(raw_row, ensure_ascii=False, sort_keys=True),
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

    for day in daterange(start, end):
        daily_url = KDreamsDaily.format(year=day.year, month=day.month, day=day.day)
        try:
            html = fetch_html(session, daily_url)
            refs = discover_s_yosen_from_daily_html(html, daily_url, day.isoformat())
            for ref in refs:
                discovered[ref.url] = ref
        except Exception as exc:  # keep a complete failure ledger instead of silently skipping
            failures.append(
                {
                    "stage": "daily_discovery",
                    "race_date": day.isoformat(),
                    "url": daily_url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    for ref in sorted(discovered.values(), key=lambda item: (item.race_date, item.url)):
        try:
            html = fetch_html(session, ref.url)
            race_meta, race_entries = extract_entries(html, ref)
            races.append(race_meta)
            entries.extend(race_entries)
        except Exception as exc:
            failures.append(
                {
                    "stage": "race_parse",
                    "race_date": ref.race_date,
                    "url": ref.url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    race_fields = [
        "race_id", "race_date", "track", "race_no", "race_type", "start_time",
        "deadline", "entry_count", "source_url",
    ]
    entry_fields = [
        "race_id", "race_date", "track", "race_no", "race_type", "start_time",
        "deadline", "source_url", "car_no", "player_name", "player_profile",
        "class", "style", "gear", "score", "raw_row_json",
    ]
    failure_fields = ["stage", "race_date", "url", "error"]

    write_csv(out_dir / "races.csv", races, race_fields)
    write_csv(out_dir / "entries.csv", entries, entry_fields)
    write_csv(out_dir / "failures.csv", failures, failure_fields)

    captured_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "target": TARGET_RACE_TYPE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": captured_at,
        "source": "楽天Kドリームス 公開レース情報",
        "daily_source_template": KDreamsDaily,
        "discovered_races": len(discovered),
        "parsed_races": len(races),
        "entry_rows": len(entries),
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

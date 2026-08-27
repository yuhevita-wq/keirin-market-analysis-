from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = (
    "Mozilla/5.0 (compatible; keirin-market-analysis/0.1; "
    "+https://github.com/yuhevita-wq/keirin-market-analysis-)"
)
LINE_LABEL = "並び予想"
LINE_SOURCE_LABEL = "楽天Kドリームス 並び予想"


class LineFormationError(RuntimeError):
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


def find_line_container(soup: BeautifulSoup) -> Tag | None:
    for dl in soup.select("dl.racecard_footer-contents"):
        dt = dl.find("dt", recursive=False)
        if dt is None or normalize_text(dt.get_text(" ", strip=True)) != LINE_LABEL:
            continue
        dd = dl.find("dd", recursive=False)
        if dd is None:
            continue
        container = dd.select_one("div.line_position")
        if container is not None:
            return container
    return None


def extract_provider(soup: BeautifulSoup) -> str:
    for dl in soup.select("dl.tipster"):
        children = list(dl.find_all(["dt", "dd"], recursive=False))
        for index, node in enumerate(children):
            if node.name != "dt":
                continue
            if normalize_text(node.get_text(" ", strip=True)) != "並び提供：":
                continue
            for later in children[index + 1 :]:
                if later.name == "dd":
                    return normalize_text(later.get_text(" ", strip=True))
    return ""


def parse_line_formation_html(html: str) -> dict[str, object]:
    """Parse KDreams' published '並び予想'.

    Returns status='not_published' when the page does not publish a line forecast.
    A published but malformed/duplicated formation raises LineFormationError so it
    cannot silently become missing data.
    """
    soup = BeautifulSoup(html, "lxml")
    container = find_line_container(soup)
    provider = extract_provider(soup)
    if container is None:
        return {
            "status": "not_published",
            "formation": "",
            "provider": provider,
            "groups": [],
            "riders": {},
        }

    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for icon in container.find_all("span", class_="icon_p", recursive=False):
        classes = set(icon.get("class", []))
        if "space" in classes:
            flush()
            continue

        nested = icon.find_all("span", recursive=False)
        car_no: int | None = None
        role = ""
        for node in nested:
            text = normalize_text(node.get_text(" ", strip=True))
            if re.fullmatch(r"[1-9]", text):
                car_no = int(text)
            elif text and text != "←":
                role = text

        if car_no is not None:
            current.append({"car_no": car_no, "role": role})

    flush()
    if not groups:
        raise LineFormationError("line container exists but no rider groups were parsed")

    seen: set[int] = set()
    riders: dict[int, dict[str, object]] = {}
    for line_id, group in enumerate(groups, start=1):
        for position, rider in enumerate(group, start=1):
            car_no = int(rider["car_no"])
            if car_no in seen:
                raise LineFormationError(f"duplicate car number in line formation: {car_no}")
            seen.add(car_no)
            riders[car_no] = {
                "line_id": line_id,
                "line_position": position,
                "line_size": len(group),
                "line_role": str(rider.get("role", "")),
            }

    formation = "/".join(
        "-".join(str(int(rider["car_no"])) for rider in group)
        for group in groups
    )
    return {
        "status": "published",
        "formation": formation,
        "provider": provider,
        "groups": groups,
        "riders": riders,
    }


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_fields(existing: list[str], new_fields: list[str]) -> list[str]:
    result = list(existing)
    for field in new_fields:
        if field not in result:
            result.append(field)
    return result


def enrich(data_dir: Path, sleep_seconds: float = 0.25) -> dict[str, object]:
    races_path = data_dir / "races.csv"
    entries_path = data_dir / "entries.csv"
    if not races_path.exists() or not entries_path.exists():
        raise FileNotFoundError("races.csv and entries.csv are required")

    race_fields, races = read_csv(races_path)
    entry_fields, entries = read_csv(entries_path)
    by_race: dict[str, list[dict[str, str]]] = {}
    for entry in entries:
        by_race.setdefault(entry.get("race_id", ""), []).append(entry)

    session = make_session()
    failures: list[dict[str, object]] = []
    published = 0
    not_published = 0
    complete = 0
    incomplete = 0

    for race in races:
        race_id = race.get("race_id", "")
        source_url = race.get("source_url", "")
        race_entries = by_race.get(race_id, [])
        expected = {int(row["car_no"]) for row in race_entries if row.get("car_no", "").isdigit()}

        try:
            html = fetch_html(session, source_url)
            parsed = parse_line_formation_html(html)
            status = str(parsed["status"])
            riders = parsed["riders"]
            if not isinstance(riders, dict):
                raise LineFormationError("riders mapping is invalid")

            if status == "published":
                published += 1
                actual = set(int(value) for value in riders.keys())
                if actual != expected:
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    raise LineFormationError(
                        f"line coverage mismatch missing={missing} extra={extra} expected={sorted(expected)}"
                    )
                complete += 1
            else:
                not_published += 1

            race["predicted_line_formation"] = str(parsed["formation"])
            race["line_status"] = status
            race["line_provider"] = str(parsed["provider"])
            race["line_source"] = LINE_SOURCE_LABEL if status == "published" else ""

            for entry in race_entries:
                car_no = int(entry["car_no"])
                rider = riders.get(car_no, {})
                entry["predicted_line_formation"] = str(parsed["formation"])
                entry["line_status"] = status
                entry["line_provider"] = str(parsed["provider"])
                entry["line_source"] = LINE_SOURCE_LABEL if status == "published" else ""
                entry["line_id"] = rider.get("line_id", "") if isinstance(rider, dict) else ""
                entry["line_position"] = rider.get("line_position", "") if isinstance(rider, dict) else ""
                entry["line_size"] = rider.get("line_size", "") if isinstance(rider, dict) else ""
                entry["line_role"] = rider.get("line_role", "") if isinstance(rider, dict) else ""

        except Exception as exc:
            incomplete += 1
            race["line_status"] = "parse_failure"
            failures.append(
                {
                    "race_id": race_id,
                    "race_date": race.get("race_date", ""),
                    "track": race.get("track", ""),
                    "race_no": race.get("race_no", ""),
                    "source_url": source_url,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if sleep_seconds:
            time.sleep(sleep_seconds)

    race_fields = add_fields(
        race_fields,
        ["predicted_line_formation", "line_status", "line_provider", "line_source"],
    )
    entry_fields = add_fields(
        entry_fields,
        [
            "predicted_line_formation",
            "line_status",
            "line_provider",
            "line_source",
            "line_id",
            "line_position",
            "line_size",
            "line_role",
        ],
    )
    write_csv(races_path, race_fields, races)
    write_csv(entries_path, entry_fields, entries)
    write_csv(
        data_dir / "line_failures.csv",
        ["race_id", "race_date", "track", "race_no", "source_url", "error"],
        failures,
    )

    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "race_count": len(races),
        "published_line_races": published,
        "not_published_line_races": not_published,
        "complete_line_races": complete,
        "line_parse_failures": len(failures),
        "line_source_label": LINE_SOURCE_LABEL,
        "semantics": "predicted_line_formation is the published KDreams '並び予想'; it is not treated as an official fact when the source does not publish it.",
    }
    (data_dir / "line_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich 2025 S-class preliminary data with KDreams line forecasts")
    parser.add_argument("--data-dir", default="data/2025/s_class_yosen")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = enrich(Path(args.data_dir), sleep_seconds=args.sleep)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["line_parse_failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

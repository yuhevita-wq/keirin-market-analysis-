from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from .s_yosen_2025 import fetch_html, make_session, normalize_text

RESULT_SOURCE_LABEL = "楽天Kドリームス 結果・払戻金"
RESULT_PAGE_TYPE = "KS_RACE_CARD_PAGE_TYPE_SHOW_RESULT"

TICKET_TYPES = [
    ("2枠複", "frame_quinella"),
    ("2枠単", "frame_exacta"),
    ("2車複", "quinella"),
    ("2車単", "exacta"),
    ("3連複", "trio"),
    ("3連単", "trifecta"),
    ("ワイド", "wide"),
]


class ResultPayoutError(RuntimeError):
    pass


def result_url(source_url: str) -> str:
    parts = urlsplit(source_url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, f"pageType={RESULT_PAGE_TYPE}", ""))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value))


def find_result_table(soup: BeautifulSoup) -> Tag:
    for table in soup.find_all("table"):
        text = compact(" ".join(table.stripped_strings))
        if all(key in text for key in ("着順", "車番", "選手名", "着差", "上り")):
            return table
    raise ResultPayoutError("result table not found")


def cell_texts(row: Tag) -> list[str]:
    return [normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"], recursive=False)]


def parse_result_rows(soup: BeautifulSoup) -> list[dict[str, object]]:
    table = find_result_table(soup)
    rows = table.find_all("tr", recursive=False)
    if not rows:
        raise ResultPayoutError("result table is empty")

    header_index = None
    header_cells: list[str] = []
    for index, row in enumerate(rows):
        cells = cell_texts(row)
        joined = compact(" ".join(cells))
        if "着順" in joined and "車番" in joined and "選手名" in joined:
            header_index = index
            header_cells = cells
            break
    if header_index is None:
        raise ResultPayoutError("result header row not found")

    normalized_headers = [compact(value) for value in header_cells]

    def find_col(label: str) -> int | None:
        for i, value in enumerate(normalized_headers):
            if value == label or label in value:
                return i
        return None

    idx_finish = find_col("着順")
    idx_car = find_col("車番")
    idx_name = find_col("選手名")
    idx_margin = find_col("着差")
    idx_last200 = find_col("上り")
    idx_method = find_col("決まり手")
    idx_sb = next((i for i, value in enumerate(normalized_headers) if value in {"S/B", "S／B", "SB"}), None)
    idx_comment = find_col("勝敗因")
    required = {"finish": idx_finish, "car": idx_car, "name": idx_name}
    if any(value is None for value in required.values()):
        raise ResultPayoutError(f"required result columns not found: {required} headers={header_cells}")

    parsed: list[dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        cells = cell_texts(row)
        if not cells:
            continue

        def get(index: int | None) -> str:
            if index is None or index >= len(cells):
                return ""
            return cells[index]

        car_text = get(idx_car)
        car_match = re.search(r"[1-9]", car_text)
        if not car_match:
            continue
        car_no = int(car_match.group(0))
        finish_text = get(idx_finish)
        finish_match = re.fullmatch(r"\d+", compact(finish_text))
        finish_position = int(finish_match.group(0)) if finish_match else ""

        parsed.append(
            {
                "car_no": car_no,
                "player_name": get(idx_name),
                "finish_position": finish_position,
                "finish_text": finish_text,
                "margin": get(idx_margin),
                "last200": get(idx_last200),
                "winning_method": get(idx_method),
                "sb": get(idx_sb),
                "result_comment": get(idx_comment),
                "raw_row_json": json.dumps({"cells": cells, "row_class": row.get("class", [])}, ensure_ascii=False),
            }
        )

    if not parsed:
        raise ResultPayoutError("no result rows parsed")
    return parsed


def payout_records_from_cell(cell: Tag, ticket_type: str, bet_code: str) -> list[dict[str, object]]:
    raw = normalize_text(cell.get_text(" ", strip=True))
    compact_raw = compact(raw)
    if "未発売" in compact_raw:
        return [{
            "ticket_type": ticket_type,
            "bet_code": bet_code,
            "combination": "",
            "payout_yen": "",
            "popularity": "",
            "status": "not_offered",
            "raw_text": raw,
        }]

    records: list[dict[str, object]] = []
    for dl in cell.find_all("dl"):
        dt = dl.find("dt")
        dd = dl.find("dd")
        combination = normalize_text(dt.get_text(" ", strip=True)) if dt else ""
        amount_text = normalize_text(dd.get_text(" ", strip=True)) if dd else normalize_text(dl.get_text(" ", strip=True))
        amount_match = re.search(r"([\d,]+)\s*円", amount_text)
        popularity_match = re.search(r"[（(]\s*(\d+)\s*[）)]", amount_text)
        if combination and amount_match:
            records.append({
                "ticket_type": ticket_type,
                "bet_code": bet_code,
                "combination": compact(combination),
                "payout_yen": int(amount_match.group(1).replace(",", "")),
                "popularity": int(popularity_match.group(1)) if popularity_match else "",
                "status": "paid",
                "raw_text": normalize_text(dl.get_text(" ", strip=True)),
            })

    if records:
        return records

    # Fallback for pages where payout details are plain text instead of dl/dt/dd.
    pattern = re.compile(
        r"([1-9](?:[-=][1-9]){1,2})\s*([\d,]+)\s*円(?:\s*[（(]\s*(\d+)\s*[）)])?"
    )
    for combination, amount, popularity in pattern.findall(raw):
        records.append({
            "ticket_type": ticket_type,
            "bet_code": bet_code,
            "combination": compact(combination),
            "payout_yen": int(amount.replace(",", "")),
            "popularity": int(popularity) if popularity else "",
            "status": "paid",
            "raw_text": raw,
        })
    if records:
        return records

    status = "refund" if any(word in compact_raw for word in ("返還", "返金")) else "special"
    return [{
        "ticket_type": ticket_type,
        "bet_code": bet_code,
        "combination": "",
        "payout_yen": "",
        "popularity": "",
        "status": status,
        "raw_text": raw,
    }]


def parse_payout_rows(soup: BeautifulSoup) -> list[dict[str, object]]:
    table = soup.select_one("table.refund_table")
    if table is None:
        raise ResultPayoutError("refund table not found")
    rows = table.find_all("tr", recursive=False)
    if len(rows) < 2:
        raise ResultPayoutError("refund table has fewer than 2 rows")

    first = rows[0].find_all(["th", "td"], recursive=False)
    second = rows[1].find_all(["th", "td"], recursive=False)
    if len(first) < 11 or len(second) < 6:
        raise ResultPayoutError(f"unexpected refund table shape first={len(first)} second={len(second)}")

    mapping = [
        ("2枠複", "frame_quinella", first[2]),
        ("2枠単", "frame_exacta", second[1]),
        ("2車複", "quinella", first[5]),
        ("2車単", "exacta", second[3]),
        ("3連複", "trio", first[8]),
        ("3連単", "trifecta", second[5]),
        ("ワイド", "wide", first[10]),
    ]

    records: list[dict[str, object]] = []
    for ticket_type, bet_code, cell in mapping:
        records.extend(payout_records_from_cell(cell, ticket_type, bet_code))

    present = {str(record["ticket_type"]) for record in records}
    expected = {name for name, _ in TICKET_TYPES}
    if present != expected:
        raise ResultPayoutError(f"ticket coverage mismatch missing={sorted(expected-present)} extra={sorted(present-expected)}")
    return records


def parse_result_page(html: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    soup = BeautifulSoup(html, "lxml")
    return parse_result_rows(soup), parse_payout_rows(soup)


def extract_all(data_dir: Path, sleep_seconds: float = 0.25) -> dict[str, object]:
    races_path = data_dir / "races.csv"
    entries_path = data_dir / "entries.csv"
    if not races_path.exists() or not entries_path.exists():
        raise FileNotFoundError("races.csv and entries.csv are required")

    _, races = read_csv(races_path)
    _, entries = read_csv(entries_path)
    entry_by_race: dict[str, dict[int, str]] = {}
    for row in entries:
        race_id = row.get("race_id", "")
        car = row.get("car_no", "")
        if race_id and car.isdigit():
            entry_by_race.setdefault(race_id, {})[int(car)] = row.get("player_name", "")

    session = make_session()
    results: list[dict[str, object]] = []
    payouts: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    complete_results = 0
    complete_payouts = 0

    for race in races:
        race_id = race.get("race_id", "")
        url = result_url(race.get("source_url", ""))
        captured = datetime.now(timezone.utc).isoformat()
        common = {
            "race_id": race_id,
            "race_date": race.get("race_date", ""),
            "track": race.get("track", ""),
            "race_no": race.get("race_no", ""),
            "race_type": race.get("race_type", ""),
            "result_source_url": url,
            "result_source": RESULT_SOURCE_LABEL,
            "captured_at_utc": captured,
        }
        try:
            html = fetch_html(session, url)
            race_results, race_payouts = parse_result_page(html)

            expected_cars = set(entry_by_race.get(race_id, {}))
            actual_cars = {int(row["car_no"]) for row in race_results}
            if expected_cars and actual_cars != expected_cars:
                raise ResultPayoutError(
                    f"result coverage mismatch missing={sorted(expected_cars-actual_cars)} extra={sorted(actual_cars-expected_cars)}"
                )

            for row in race_results:
                item = dict(common)
                item.update(row)
                results.append(item)
            complete_results += 1

            ticket_names = {str(row["ticket_type"]) for row in race_payouts}
            if ticket_names != {name for name, _ in TICKET_TYPES}:
                raise ResultPayoutError(f"incomplete payout ticket coverage: {sorted(ticket_names)}")
            for row in race_payouts:
                item = dict(common)
                item.update(row)
                payouts.append(item)
            complete_payouts += 1
        except Exception as exc:
            failures.append({
                **common,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if sleep_seconds:
            time.sleep(sleep_seconds)

    results.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"]), int(row["car_no"])))
    payouts.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"]), str(row["bet_code"]), str(row["combination"])))

    result_fields = [
        "race_id", "race_date", "track", "race_no", "race_type", "car_no", "player_name",
        "finish_position", "finish_text", "margin", "last200", "winning_method", "sb", "result_comment",
        "result_source_url", "result_source", "captured_at_utc", "raw_row_json",
    ]
    payout_fields = [
        "race_id", "race_date", "track", "race_no", "race_type", "ticket_type", "bet_code", "combination",
        "payout_yen", "popularity", "status", "result_source_url", "result_source", "captured_at_utc", "raw_text",
    ]
    failure_fields = [
        "race_id", "race_date", "track", "race_no", "race_type", "result_source_url", "result_source", "captured_at_utc", "error",
    ]
    write_csv(data_dir / "results.csv", result_fields, results)
    write_csv(data_dir / "payouts.csv", payout_fields, payouts)
    write_csv(data_dir / "result_failures.csv", failure_fields, failures)

    status_counts = Counter(str(row["status"]) for row in payouts)
    ticket_counts = Counter(str(row["ticket_type"]) for row in payouts)
    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": RESULT_SOURCE_LABEL,
        "race_count": len(races),
        "complete_result_races": complete_results,
        "complete_payout_races": complete_payouts,
        "result_rows": len(results),
        "payout_rows": len(payouts),
        "parse_failures": len(failures),
        "payout_status_counts": dict(sorted(status_counts.items())),
        "ticket_row_counts": dict(sorted(ticket_counts.items())),
        "ticket_types": [name for name, _ in TICKET_TYPES],
        "semantics": "payouts.csv stores every published winning payout row for all seven standard ticket types; not-offered ticket types are retained with status=not_offered. Results and payouts are outcome labels and must not be used as pre-race decision features.",
    }
    (data_dir / "result_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 2025 S-class preliminary results and payouts from KDreams")
    parser.add_argument("--data-dir", default="data/2025/s_class_yosen")
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = extract_all(Path(args.data_dir), sleep_seconds=args.sleep)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["parse_failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

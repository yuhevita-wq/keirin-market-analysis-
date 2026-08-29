from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from .s_yosen_2025 import fetch_html, make_session, normalize_text

SOURCE = "楽天Kドリームス 確定オッズ"
_thread_local = threading.local()


class OddsParseError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def odds_url(source_url: str) -> str:
    parts = urlsplit(source_url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "pageType=odds", ""))


def session():
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = make_session()
        _thread_local.session = s
    return s


def compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value))


def parse_odd(raw: str) -> tuple[float | str, str]:
    value = normalize_text(raw)
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value), "available"
    if not value or value in {"-", "--", "---", "----"}:
        return "", "unavailable"
    if any(x in value for x in ("取消", "欠場", "未発売", "発売なし")):
        return "", "unavailable"
    return "", f"unparsed:{value}"


def direct_texts(row: Tag) -> list[str]:
    return [normalize_text(c.get_text(" ", strip=True)) for c in row.find_all(["th", "td"], recursive=False)]


def parse_fixed_first_trifecta_table(table: Tag, entrants: list[int]) -> tuple[int, dict[tuple[int,int,int], tuple[float | str, str]]] | None:
    rows = table.find_all("tr", recursive=False)
    if not rows:
        return None
    title = normalize_text(rows[0].get_text(" ", strip=True))
    m = re.match(r"^([1-9])(?:\s|$)", title)
    if not m:
        return None
    first = int(m.group(1))
    if first not in entrants:
        return None
    columns = [x for x in entrants if x != first]
    parsed: dict[tuple[int,int,int], tuple[float | str, str]] = {}
    for tr in rows[1:]:
        cells = direct_texts(tr)
        if not cells:
            continue
        # Data rows have the third-place car at the left (and often repeated at right).
        if not re.fullmatch(r"[1-9]", cells[0]):
            continue
        third = int(cells[0])
        if third not in columns:
            continue
        core = cells[1:]
        if core and core[-1] == str(third):
            core = core[:-1]
        if len(core) != len(columns):
            continue
        usable = 0
        for second, raw in zip(columns, core):
            if second == third:
                continue
            odd, status = parse_odd(raw)
            if status == "available" or status == "unavailable":
                usable += 1
            parsed[(first, second, third)] = (odd, status)
        if usable < max(1, len(columns) - 2):
            # Not a trifecta data row; discard keys tentatively added from it.
            for second in columns:
                if second != third:
                    parsed.pop((first, second, third), None)
    expected = (len(entrants)-1) * (len(entrants)-2)
    if len(parsed) != expected:
        return None
    return first, parsed


def parse_trifecta(soup: BeautifulSoup, entrants: list[int]) -> dict[tuple[int,int,int], tuple[float | str, str]]:
    by_first: dict[int, dict[tuple[int,int,int], tuple[float | str, str]]] = {}
    for table in soup.select("table.odds_table"):
        got = parse_fixed_first_trifecta_table(table, entrants)
        if got is None:
            continue
        first, block = got
        # The page also contains other odds matrices. Only retain one complete fixed-first block per car.
        by_first.setdefault(first, block)
    missing = sorted(set(entrants) - set(by_first))
    if missing:
        raise OddsParseError(f"trifecta fixed-first tables missing for cars {missing}; found={sorted(by_first)}")
    merged: dict[tuple[int,int,int], tuple[float | str, str]] = {}
    for first in entrants:
        merged.update(by_first[first])
    expected = len(entrants) * (len(entrants)-1) * (len(entrants)-2)
    if len(merged) != expected:
        raise OddsParseError(f"trifecta coverage {len(merged)} != {expected}")
    return merged


def parse_trio(soup: BeautifulSoup, entrants: list[int]) -> dict[tuple[int,int,int], tuple[float | str, str]]:
    # KDreams exposes popularity/high-payout tables. Their union contains every trio
    # even when one table alone is capped (e.g. larger fields). We only use combination+odds.
    found: dict[tuple[int,int,int], tuple[float | str, str]] = {}
    pat = re.compile(r"^\s*\d+\s+([1-9])=([1-9])=([1-9])\s+([^\s]+)")
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            text = normalize_text(tr.get_text(" ", strip=True))
            m = pat.search(text)
            if not m:
                continue
            cars = tuple(sorted(map(int, m.groups()[:3])))
            if len(set(cars)) != 3 or any(c not in entrants for c in cars):
                continue
            odd, status = parse_odd(m.group(4))
            if status == "available":
                found[cars] = (odd, status)
    expected_combos = list(itertools.combinations(sorted(entrants), 3))
    if not found:
        raise OddsParseError("no trio combination/odds rows found")
    # If scratches made some combinations unavailable, retain them explicitly.
    return {combo: found.get(combo, ("", "unavailable")) for combo in expected_combos}


def parse_market_header(soup: BeautifulSoup) -> dict[str, dict[str, object]]:
    header = soup.select_one("div.odds_header")
    if header is None:
        raise OddsParseError("odds_header not found")
    header_text = normalize_text(header.get_text(" ", strip=True))
    if "確定オッズ" not in header_text:
        raise OddsParseError("page is not marked 確定オッズ")
    statuses = header.select("div.status")
    if len(statuses) < 5:
        raise OddsParseError(f"expected >=5 odds status blocks; got {len(statuses)}")

    def one(tag: Tag) -> dict[str, object]:
        text = normalize_text(tag.get_text(" ", strip=True))
        vm = re.search(r"発売票数\s*([\d,]+)", text)
        tm = re.search(r"(20\d{2}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2})現在", text)
        return {
            "total_votes": int(vm.group(1).replace(",", "")) if vm else "",
            "odds_as_of": tm.group(1) if tm else "",
        }

    return {"3連単": one(statuses[0]), "3連複": one(statuses[2])}


def assign_ranks(rows: list[dict[str, object]]) -> None:
    available = sorted(float(r["odds"]) for r in rows if r["odds_status"] == "available")
    first_rank: dict[float, int] = {}
    for i, odd in enumerate(available, 1):
        first_rank.setdefault(odd, i)
    for r in rows:
        r["market_rank"] = first_rank.get(float(r["odds"]), "") if r["odds_status"] == "available" else ""


def scrape_race(race: dict[str, str], entrants: list[int], sleep_seconds: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    url = odds_url(race.get("source_url", ""))
    html = fetch_html(session(), url)
    if sleep_seconds:
        time.sleep(sleep_seconds)
    soup = BeautifulSoup(html, "lxml")
    header = parse_market_header(soup)
    tri3 = parse_trifecta(soup, entrants)
    trio = parse_trio(soup, entrants)
    captured = datetime.now(timezone.utc).isoformat()
    common = {
        "race_id": race.get("race_id", ""), "race_date": race.get("race_date", ""),
        "track": race.get("track", ""), "race_no": race.get("race_no", ""),
        "race_type": race.get("race_type", ""), "odds_phase": "final",
        "odds_source": SOURCE, "odds_source_url": url, "captured_at_utc": captured,
    }
    out3: list[dict[str, object]] = []
    for combo, (odd, status) in sorted(tri3.items()):
        out3.append({**common, "ticket_type":"3連単", "combination":"-".join(map(str,combo)),
                     "odds":odd, "odds_status":status, **header["3連単"]})
    outt: list[dict[str, object]] = []
    for combo, (odd, status) in sorted(trio.items()):
        outt.append({**common, "ticket_type":"3連複", "combination":"=".join(map(str,combo)),
                     "odds":odd, "odds_status":status, **header["3連複"]})
    assign_ranks(out3)
    assign_ranks(outt)
    return out3, outt


def collect(data_dir: Path, workers: int, sleep_seconds: float, limit: int | None = None) -> dict[str, object]:
    races = read_csv(data_dir / "races.csv")
    entries = read_csv(data_dir / "entries.csv")
    entrants_by_race: dict[str, list[int]] = {}
    for e in entries:
        if e.get("race_id") and str(e.get("car_no","")).isdigit():
            entrants_by_race.setdefault(e["race_id"], []).append(int(e["car_no"]))
    for rid in entrants_by_race:
        entrants_by_race[rid] = sorted(set(entrants_by_race[rid]))
    if limit is not None:
        races = races[:limit]

    trifecta_rows: list[dict[str, object]] = []
    trio_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    complete = 0

    def task(race):
        rid = race.get("race_id", "")
        entrants = entrants_by_race.get(rid, [])
        if len(entrants) < 3:
            raise OddsParseError(f"too few entrants in entries.csv: {entrants}")
        return scrape_race(race, entrants, sleep_seconds)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(task, race): race for race in races}
        for i, future in enumerate(as_completed(futures), 1):
            race = futures[future]
            try:
                r3, rt = future.result()
                trifecta_rows.extend(r3)
                trio_rows.extend(rt)
                complete += 1
            except Exception as exc:
                failures.append({
                    "race_id":race.get("race_id",""), "race_date":race.get("race_date",""),
                    "track":race.get("track",""), "race_no":race.get("race_no",""),
                    "race_type":race.get("race_type",""), "odds_source_url":odds_url(race.get("source_url","")),
                    "error":f"{type(exc).__name__}: {exc}",
                })
            if i % 100 == 0 or i == len(races):
                print(f"progress {i}/{len(races)} complete={complete} failures={len(failures)}", flush=True)

    key = lambda r: (str(r["race_date"]), str(r["track"]), int(r["race_no"]), str(r["combination"]))
    trifecta_rows.sort(key=key)
    trio_rows.sort(key=key)
    failures.sort(key=lambda r: (str(r["race_date"]), str(r["track"]), int(r["race_no"] or 0)))
    fields = ["race_id","race_date","track","race_no","race_type","ticket_type","combination","odds",
              "market_rank","odds_status","total_votes","odds_as_of","odds_phase","odds_source","odds_source_url","captured_at_utc"]
    write_csv(data_dir / "trifecta_final_odds.csv", fields, trifecta_rows)
    write_csv(data_dir / "trio_final_odds.csv", fields, trio_rows)
    write_csv(data_dir / "odds_failures.csv", ["race_id","race_date","track","race_no","race_type","odds_source_url","error"], failures)
    summary = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "odds_phase": "final",
        "race_count_attempted": len(races),
        "complete_races": complete,
        "failed_races": len(failures),
        "coverage_rate": complete/len(races) if races else 0.0,
        "trifecta_rows": len(trifecta_rows),
        "trio_rows": len(trio_rows),
        "available_trifecta_rows": sum(r["odds_status"]=="available" for r in trifecta_rows),
        "available_trio_rows": sum(r["odds_status"]=="available" for r in trio_rows),
        "note": "Historical KDreams final odds. This is not T-10-minute odds and must not be presented as pre-deadline snapshot data.",
    }
    (data_dir / "odds_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--limit", type=int)
    a=p.parse_args()
    collect(a.data_dir,a.workers,a.sleep,a.limit)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

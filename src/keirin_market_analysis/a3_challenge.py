from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup

from . import s_yosen_2025 as base


@dataclass(frozen=True)
class ChallengeRaceRef:
    discovered_on: str
    race_no: int
    race_type: str
    url: str


def normalize_track_name(value: str) -> str:
    value = base.normalize_text(value)
    value = re.sub(r"競輪$", "", value)
    return value.strip()


def normalize_class(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).upper()
    return re.sub(r"\s+", "", value)


def is_a3_class(value: object) -> bool:
    return normalize_class(value) == "A3"


def detail_url(daily_url: str, href: str, page_type: str = "odds") -> str:
    # Reuse the canonical resolver for host/path normalization, but choose a
    # race-card page that retains entrant stats on modern historical pages.
    canonical = base.canonical_race_url(daily_url, href)
    parts = urlsplit(canonical)
    query = f"pageType={page_type}" if page_type else ""
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def detail_url_variants(url: str) -> list[str]:
    parts = urlsplit(url)
    variants = [
        urlunsplit((parts.scheme, parts.netloc, parts.path, "pageType=odds", "")),
        urlunsplit((parts.scheme, parts.netloc, parts.path, "pageType=yoso", "")),
        urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")),
        urlunsplit((parts.scheme, parts.netloc, parts.path, "pageType=result", "")),
    ]
    return list(dict.fromkeys(variants))


def discover_meeting_grades_from_daily_html(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    text = base.normalize_text(soup.get_text(" ", strip=True))
    found: dict[str, str] = {}
    for match in re.finditer(r"([^\s]+?)(Ｇ１|Ｇ２|Ｇ３|Ｆ１|Ｆ２)", text):
        token = match.group(1)
        token = re.split(r"[、。・|｜:：/／]", token)[-1]
        track = normalize_track_name(token)
        if not track:
            continue
        grade = unicodedata.normalize("NFKC", match.group(2)).upper()
        found[track] = grade
    return found


def discover_challenge_from_daily_html(
    html: str,
    daily_url: str,
    discovered_on: str,
) -> list[ChallengeRaceRef]:
    """Discover racedetail links whose published label contains チャレンジ."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, ChallengeRaceRef] = {}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row_index, row in enumerate(rows):
            expanded = base.expand_row_cells(row)
            if not expanded:
                continue
            labels = [base.normalize_text(cell.get_text(" ", strip=True)) for cell in expanded]
            target_indexes = [
                i for i, label in enumerate(labels)
                if "チャレンジ" in unicodedata.normalize("NFKC", label)
            ]
            for target_index in target_indexes:
                race_type = labels[target_index]
                href = None
                for later in rows[row_index + 1 :]:
                    later_cells = base.expand_row_cells(later)
                    if target_index >= len(later_cells):
                        continue
                    for link in later_cells[target_index].find_all("a", href=True):
                        candidate = str(link.get("href", ""))
                        if "/racedetail/" in candidate:
                            href = candidate
                            break
                    if href:
                        break
                if href is None:
                    continue

                url = detail_url(daily_url, href, "odds")
                match = re.search(r"/racedetail/(\d{16})/", url)
                race_no = int(match.group(1)[-4:]) if match else target_index + 1
                found[url] = ChallengeRaceRef(
                    discovered_on=discovered_on,
                    race_no=race_no,
                    race_type=race_type,
                    url=url,
                )

    return sorted(found.values(), key=lambda item: (item.discovered_on, item.url, item.race_no))


def extract_entries_for_ref(session, ref: ChallengeRaceRef):
    errors: list[str] = []
    for candidate_url in detail_url_variants(ref.url):
        frozen_ref = base.RaceRef(
            discovered_on=ref.discovered_on,
            race_no=ref.race_no,
            url=candidate_url,
        )
        try:
            html = base.fetch_html(session, candidate_url)
            # Daily discovery determines the challenge label. Some historical
            # detail pages omit that duplicated label, so entrant-table parsing
            # must not depend on it.
            old_target = base.TARGET_RACE_TYPE
            base.TARGET_RACE_TYPE = ref.race_type
            try:
                meta, entries = base.extract_entries(
                    html,
                    frozen_ref,
                    require_target_label=False,
                )
            finally:
                base.TARGET_RACE_TYPE = old_target
            return meta, entries
        except Exception as exc:
            errors.append(f"{candidate_url} -> {type(exc).__name__}: {exc}")
    raise RuntimeError("all detail variants failed | " + " | ".join(errors))


def collect(start: date, end: date, out_dir: Path, sleep_seconds: float) -> dict[str, object]:
    if start > end:
        raise ValueError("start date must be <= end date")
    if start.year < 2023 or end.year > 2026:
        raise ValueError("collector is restricted to the 2023-2026 research window")

    session = base.make_session()
    races: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    discovered: dict[str, ChallengeRaceRef] = {}
    grades_by_day: dict[str, dict[str, str]] = {}
    skipped_outside_window = 0
    skipped_non_a3 = 0
    skipped_class_sets: Counter[str] = Counter()

    for day in base.daterange(start, end):
        daily_url = base.KDREAMS_DAILY.format(year=day.year, month=day.month, day=day.day)
        try:
            html = base.fetch_html(session, daily_url)
            grades_by_day[day.isoformat()] = discover_meeting_grades_from_daily_html(html)
            for ref in discover_challenge_from_daily_html(html, daily_url, day.isoformat()):
                discovered.setdefault(ref.url, ref)
        except Exception as exc:
            failures.append({
                "stage": "daily_discovery",
                "discovered_on": day.isoformat(),
                "url": daily_url,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if sleep_seconds:
            time.sleep(sleep_seconds)

    for ref in sorted(discovered.values(), key=lambda item: (item.discovered_on, item.url)):
        try:
            race_meta, race_entries = extract_entries_for_ref(session, ref)
            actual_date = date.fromisoformat(str(race_meta["race_date"]))
            if actual_date < start or actual_date > end:
                skipped_outside_window += 1
                continue

            classes = sorted({normalize_class(entry.get("class", "")) for entry in race_entries})
            if classes != ["A3"]:
                skipped_non_a3 += 1
                skipped_class_sets["/".join(classes) if classes else "(empty)"] += 1
                continue

            track = normalize_track_name(str(race_meta["track"]))
            meeting_grade = grades_by_day.get(ref.discovered_on, {}).get(track, "")
            race_meta["race_type"] = ref.race_type
            race_meta["meeting_grade"] = meeting_grade
            race_meta["class_scope"] = "A3 challenge"
            for entry in race_entries:
                entry["race_type"] = ref.race_type
                entry["meeting_grade"] = meeting_grade
                entry["class_scope"] = "A3 challenge"

            races.append(race_meta)
            entries.extend(race_entries)
        except Exception as exc:
            failures.append({
                "stage": "race_parse",
                "discovered_on": ref.discovered_on,
                "url": ref.url,
                "error": f"{type(exc).__name__}: {exc}",
            })
        if sleep_seconds:
            time.sleep(sleep_seconds)

    races.sort(key=lambda row: (str(row["race_date"]), str(row["track"]), int(row["race_no"])))
    entries.sort(key=lambda row: (
        str(row["race_date"]), str(row["track"]), int(row["race_no"]), int(row["car_no"])
    ))

    race_fields = [
        "race_id", "race_date", "track", "meeting_grade", "class_scope",
        "race_no", "race_type", "start_time", "deadline", "entry_count",
        "source_url", "captured_at_utc",
    ]
    entry_fields = [
        "race_id", "race_date", "track", "meeting_grade", "class_scope",
        "race_no", "race_type", "start_time", "deadline", "source_url",
        "captured_at_utc", "car_no", "player_name", "player_profile",
        "prefecture", "age", "term", "class", "style", "gear", "score",
        "s_count", "b_count", "nige_count", "makuri_count", "sashi_count",
        "mark_count", "first_count", "second_count", "third_count",
        "outside_count", "win_rate", "top2_rate", "top3_rate",
        "prediction_mark", "evaluation", "raw_row_json",
    ]
    failure_fields = ["stage", "discovered_on", "url", "error"]

    base.write_csv(out_dir / "races.csv", races, race_fields)
    base.write_csv(out_dir / "entries.csv", entries, entry_fields)
    base.write_csv(out_dir / "failures.csv", failures, failure_fields)

    summary = {
        "target": f"all challenge-labelled races whose entrants are exclusively A3, {start.isoformat()} to {end.isoformat()}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "楽天Kドリームス 公開レース情報",
        "daily_source_template": base.KDREAMS_DAILY,
        "candidate_challenge_races": len(discovered),
        "parsed_a3_challenge_races": len(races),
        "entry_rows": len(entries),
        "skipped_non_a3": skipped_non_a3,
        "skipped_non_a3_class_sets": dict(sorted(skipped_class_sets.items())),
        "skipped_outside_window": skipped_outside_window,
        "failures": len(failures),
        "race_type_counts": dict(sorted(Counter(str(r["race_type"]) for r in races).items())),
        "meeting_grade_counts": dict(sorted(Counter(str(r["meeting_grade"]) or "unknown" for r in races).items())),
        "entry_count_counts": dict(sorted(Counter(str(r["entry_count"]) for r in races).items())),
        "entrant_class_counts": dict(sorted(Counter(normalize_class(e["class"]) for e in entries).items())),
        "definition_note": "日別開催一覧でレース種別表記に『チャレンジ』を含む候補を取得し、実際の出走表に記載された全選手の級班がA3のレースだけを保存する。予選・一般・選抜・準決勝・決勝・Ａ級チャレンジＦ等を含む。車立て数では絞らない。",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect A3 challenge races")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sleep", type=float, default=0.20)
    parser.add_argument("--allow-failures", action="store_true")
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
    if summary["parsed_a3_challenge_races"] == 0:
        return 3
    if summary["failures"] and not args.allow_failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build an ODDS + RACE TYPE ONLY data mart, excluding women's keirin.

Data records have EXACTLY three keys:
  race_id: immutable join/dedup key (never a predictive feature)
  race_type: race category with meeting grade if known
  odds: { "3連単": { "1-2-3": 54.3, ... }, "3連複": {"1=2=3": 17.8,...} }

No results, payouts, line forecasts, player names, scores, ranks, dates, tracks,
riders or event labels are copied into the output.

Includes the original archived CSVs, original graded ZIPs, the new immutable
odds backfill shards, and recovered previously unarchived races. Only
completely numeric, internally consistent whole-book *final* markets enter.
All source inspection metadata goes into a SEPARATE audit JSON.

Run at repo root:
    python research/odds_universe_v1/build_odds_only.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import re
import sqlite3
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from scope import START, END, in_scope

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/odds_only"
AUDIT = ROOT / "data/audits/odds_only_export.json"
LABEL_FIELDS = {"race_id", "race_type", "odds"}
SUPPORTED = ("3連単", "3連複")
RACE_ID = re.compile(r"^\d{16}$")
GIRLS = re.compile(r"ガールズ|女子|L級|ガ予|ガ決|ガ一般|ガ選", re.I)
PLACEHOLDER = {"UNIVERSE_REPAIR", "UNKNOWN", "UNKNOWN_RACE", "ODDS", "?"}


def csv_plain(p):
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def csv_gzip(p):
    with gzip.open(p, "rt", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def csv_from_zip(z, name):
    names = [p for p in z.namelist() if p.rsplit("/", 1)[-1] == name]
    if len(names) != 1:
        return
    with z.open(names[0]) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)


def source_dirs():
    for family in ("2023", "2024", "2025", "2026_h1",
                   "2026_future_block1", "a_class_12", "a3_challenge"):
        path = ROOT / "data" / family
        if not path.exists():
            continue
        for p in sorted(path.rglob("races.csv")):
            if (p.parent / "entries.csv").exists():
                yield p.parent


def year_of(meta, rid):
    day = str(meta.get("race_date", "")).strip()
    if re.fullmatch(r"20\d\d-\d\d-\d\d", day) and in_scope(day):
        return day[:4]
    # Race-ID encodes meeting START date, not the date of the race itself.
    # Do not silently infer the race date from that starting date.
    return None


def compact_class(value):
    return unicodedata.normalize("NFKC", str(value or "")).upper().replace(" ", "")


def construct_race_type(meta, rider_classes):
    rt = unicodedata.normalize("NFKC", str(meta.get("race_type", "") or "")).strip()
    grade = compact_class(meta.get("meeting_grade") or meta.get("event_grade") or "")
    if GIRLS.search(rt) or any(compact_class(x).startswith("L") for x in rider_classes):
        return None, "girls"
    if not rt or rt.upper() in PLACEHOLDER:
        fam = {compact_class(c) for c in rider_classes if c}
        if fam and all(c.startswith("S") for c in fam):
            rt = "S級/詳細不明"
        elif fam and all(c.startswith("A3") for c in fam):
            rt = "A級チャレンジ/詳細不明"
        elif fam and all(c.startswith("A") for c in fam):
            rt = "A級/詳細不明"
        else:
            rt = "種別不明"
        return (f"{grade}/{rt}" if grade else rt), "imprecise"
    return (f"{grade}/{rt}" if grade else rt), "ok"


def source_identity(meta):
    return str(meta.get("race_id", "")).strip()


def read_metadata(races, entries):
    meta = {}
    for row in races:
        rid = source_identity(row)
        if RACE_ID.fullmatch(rid) and in_scope(row.get("race_date")):
            meta[rid] = row
    runners = defaultdict(set)
    classes = defaultdict(list)
    for row in entries:
        rid = source_identity(row)
        if rid not in meta:
            continue
        try:
            c = int(row.get("car_no", ""))
            if 1 <= c <= 9:
                runners[rid].add(c)
                classes[rid].append(row.get("class", ""))
        except (TypeError, ValueError):
            pass
    return meta, runners, classes


def market_parts(filenames):
    """Accept entire trifecta market or all of its partitioned files, never both."""
    paths = sorted(p for p in filenames if p.startswith("trifecta_final_odds_part") and p.endswith(".csv"))
    if "trifecta_final_odds.csv" in filenames:
        return ["trifecta_final_odds.csv"]
    return paths


def normalize_combi(raw, kind):
    parts = str(raw or "").split("-" if kind == "3連単" else "=")
    if len(parts) != 3 or any(not re.fullmatch(r"[1-9]", x) for x in parts):
        return None
    nums = tuple(map(int, parts))
    if len(set(nums)) != 3:
        return None
    if kind == "3連複":
        nums = tuple(sorted(nums))
    sep = "-" if kind == "3連単" else "="
    return sep.join(map(str, nums))


def market_rows_to_record(rid, market, meta, runners, classes):
    if not in_scope(meta[rid].get("race_date")):
        return None, "out_of_scope"
    count = len(runners.get(rid, ()))
    if not 3 <= count <= 9:
        return None, "missing_or_bad_entries"
    rt, status = construct_race_type(meta[rid], classes[rid])
    if rt is None:
        return None, "girls_excluded"
    odds = market["odds"]
    three = odds["3連単"]
    trio = odds["3連複"]
    if market["bad"]:
        return None, "invalid_or_missing_price"
    if len(three) != count * (count - 1) * (count - 2) or len(trio) != math.comb(count, 3):
        return None, "incomplete_price_grid"
    cars = sorted(runners[rid])
    exp3 = {"-".join(map(str, t)) for t in itertools.permutations(cars, 3)}
    expt = {"=".join(map(str, t)) for t in itertools.combinations(cars, 3)}
    if set(three) != exp3 or set(trio) != expt:
        return None, "combination_universe_mismatch"
    obj = {"race_id": rid, "race_type": rt,
           "odds": {"3連単": dict(sorted(three.items())), "3連複": dict(sorted(trio.items()))}}
    if set(obj) != LABEL_FIELDS:
        raise AssertionError("unexpected data field")
    return obj, ("ok" if status == "ok" else "imprecise_race_type")


def consolidate_source(db, source, races, entries, trifecta, trio, summary):
    meta, runners, classes = read_metadata(races, entries)
    if not meta:
        summary["empty_metadata"] += 1
        return
    market = defaultdict(lambda: {"odds": {"3連単": {}, "3連複": {}}, "bad": 0})
    for kind, iterator in (("3連単", trifecta), ("3連複", trio)):
        for row in iterator:
            rid = source_identity(row)
            if rid not in meta:
                continue
            bucket = market[rid]
            if row.get("ticket_type") and row["ticket_type"] != kind:
                bucket["bad"] += 1
                continue
            if row.get("odds_phase") not in ("final", None, ""):
                bucket["bad"] += 1
                continue
            key = normalize_combi(row.get("combination"), kind)
            if key is None:
                bucket["bad"] += 1
                continue
            if row.get("odds_status") not in ("available", None, ""):
                bucket["bad"] += 1
                continue
            try:
                value = float(row["odds"])
                if not math.isfinite(value) or value < 1:
                    raise ValueError("invalid numeric price")
            except (ValueError, TypeError, KeyError):
                bucket["bad"] += 1
                continue
            if key in bucket["odds"][kind]:
                bucket["bad"] += 1
                continue
            bucket["odds"][kind][key] = value
    summary["sources_read"] += 1
    for rid, data in market.items():
        record, state = market_rows_to_record(rid, data, meta, runners, classes)
        summary[state] += 1
        if record is None:
            continue
        y = year_of(meta[rid], rid)
        if y is None:
            summary["missing_year"] += 1
            continue
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        old = db.execute("SELECT checksum,race_type,source,payload FROM races WHERE race_id=?", (rid,)).fetchone()
        if old is not None:
            summary["duplicate_race"] += 1
            if old[0] != fingerprint:
                original = json.loads(old[3])
                if original["odds"] == record["odds"]:
                    summary["duplicate_type_label_variation"] += 1
                else:
                    summary["conflicting_duplicate_odds"] += 1
                    if len(summary["conflict_examples"]) < 25:
                        summary["conflict_examples"].append({"race_id": rid,
                            "first_source": old[2], "new_source": source,
                            "first_race_type": old[1], "new_race_type": record["race_type"]})
            continue
        db.execute("INSERT INTO races(race_id, year, race_type, payload, checksum, source) "
                   "VALUES(?,?,?,?,?,?)",
                   (rid, y, record["race_type"], payload, fingerprint, source))
        summary["unique_complete_races"] += 1
        summary["year_counts"][y] += 1
        if state == "imprecise_race_type":
            summary["imprecise_race_type_count"] += 1
    db.commit()


def standard_csv_sources(db, summary):
    for directory in source_dirs():
        names = {x.name for x in directory.iterdir() if x.is_file()}
        parts = market_parts(names)
        if not parts or "trio_final_odds.csv" not in names:
            summary["skipped_csv_source_without_both_market_files"] += 1
            continue
        def lines():
            return itertools.chain.from_iterable(csv_plain(directory / n) for n in parts)
        consolidate_source(db, str(directory.relative_to(ROOT)),
                           csv_plain(directory / "races.csv"),
                           csv_plain(directory / "entries.csv"),
                           lines(), csv_plain(directory / "trio_final_odds.csv"), summary)


def grade_zip_sources(db, summary):
    for zpath in sorted((ROOT / "data/grade_races").rglob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            names = set(p.rsplit("/", 1)[-1] for p in z.namelist())
            if not {"races.csv", "entries.csv", "trifecta_final_odds.csv", "trio_final_odds.csv"}.issubset(names):
                summary["zip_without_both_markets"] += 1
                continue
            consolidate_source(db, str(zpath.relative_to(ROOT)),
                csv_from_zip(z,"races.csv"), csv_from_zip(z,"entries.csv"),
                csv_from_zip(z,"trifecta_final_odds.csv"),
                csv_from_zip(z,"trio_final_odds.csv"), summary)


def recovered_shards(db, summary):
    for batch in sorted((ROOT / "data/odds_universe_v1").rglob("summary.json")):
        parent = batch.parent
        if not (parent / "trifecta_final_odds.csv.gz").exists() or not (parent / "trio_final_odds.csv.gz").exists():
            continue
        report = json.loads(batch.read_text(encoding="utf-8"))
        source = report.get("batch_source")
        if source:
            original = ROOT / source
            if not ((original / "races.csv").exists() and (original / "entries.csv").exists()):
                summary["recovery_source_missing"] += 1
                continue
            consolidate_source(db, str(batch.relative_to(ROOT)),
                csv_plain(original / "races.csv"), csv_plain(original / "entries.csv"),
                csv_gzip(parent / "trifecta_final_odds.csv.gz"),
                csv_gzip(parent / "trio_final_odds.csv.gz"), summary)
            continue
        archive = report.get("source_archive")
        if archive:
            path = ROOT / archive
            if not path.exists():
                summary["recovery_archive_missing"] += 1
                continue
            with zipfile.ZipFile(path) as z:
                consolidate_source(db, str(batch.relative_to(ROOT)),
                    csv_from_zip(z, "races.csv"), csv_from_zip(z, "entries.csv"),
                    csv_gzip(parent / "trifecta_final_odds.csv.gz"),
                    csv_gzip(parent / "trio_final_odds.csv.gz"), summary)


def absent_race_shards(db, summary):
    for batch in sorted((ROOT / "data/unarchived_races_v1").glob("batch_*/summary.json")):
        folder = batch.parent
        needed = ("races.csv.gz","entries.csv.gz","trifecta_final_odds.csv.gz","trio_final_odds.csv.gz")
        if not all((folder / n).exists() for n in needed):
            summary["missing_absent_race_shard"] += 1
            continue
        consolidate_source(db,str(batch.relative_to(ROOT)),
            csv_gzip(folder/"races.csv.gz"),csv_gzip(folder/"entries.csv.gz"),
            csv_gzip(folder/"trifecta_final_odds.csv.gz"),
            csv_gzip(folder/"trio_final_odds.csv.gz"),summary)


def export(db, summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    products = {}
    years = [r[0] for r in db.execute("SELECT DISTINCT year FROM races ORDER BY year")]
    for y in years:
        filename = f"odds_{y}.jsonl.gz"
        path = output_dir / filename
        count = 0
        with path.open("wb") as binary:
            with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0, compresslevel=6) as gz:
                for (payload,) in db.execute(
                    "SELECT payload FROM races WHERE year=? ORDER BY race_id", (y,)
                ):
                    gz.write((payload+"\n").encode("utf-8"))
                    count += 1
        products[y]={"file":filename,"complete_races":count,"bytes":path.stat().st_size,
                     "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.stat().st_size >= 95_000_000:
            raise RuntimeError(f"{filename} exceeds safe GitHub file size; partition year first")
    return products


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--audit",type=Path,default=AUDIT)
    args = parser.parse_args()
    tally = Counter()
    summary = {"year_counts": defaultdict(int), "conflict_examples": []}
    class Meter(dict):
        def __missing__(self, key):
            self[key] = 0
            return 0
    summary = Meter(summary)
    with tempfile.TemporaryDirectory(prefix="odds_only_export_") as tmp:
        db = sqlite3.connect(str(Path(tmp)/"markets.sqlite"))
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("CREATE TABLE races (race_id TEXT PRIMARY KEY, year TEXT NOT NULL, "
                   "race_type TEXT NOT NULL, payload TEXT NOT NULL, checksum TEXT NOT NULL,"
                   "source TEXT NOT NULL)")
        db.execute("CREATE INDEX idx_races_year ON races(year)")
        print("SOURCE csv", flush=True)
        standard_csv_sources(db,summary)
        print("SOURCE grade archives",flush=True)
        grade_zip_sources(db,summary)
        print("SOURCE recovered odds",flush=True)
        recovered_shards(db,summary)
        print("SOURCE entirely absent races",flush=True)
        absent_race_shards(db,summary)
        products=export(db,summary,args.output)
        db.close()
    # A single portable file for this request. Concatenated gzip streams
    # are standard gzip and gzip.open reads every member transparently.
    combined_path = args.output / "odds_all.jsonl.gz"
    combined_size = sum((args.output / p["file"]).stat().st_size for p in products.values())
    if combined_size <= 90_000_000:
        with combined_path.open("wb") as dest:
            for year in sorted(products):
                with (args.output / products[year]["file"]).open("rb") as src:
                    import shutil
                    shutil.copyfileobj(src, dest)
        combined = {"file":combined_path.name,
                    "complete_races":sum(p["complete_races"] for p in products.values()),
                    "bytes":combined_path.stat().st_size,
                    "sha256":hashlib.sha256(combined_path.read_bytes()).hexdigest(),
                    "compression":"concatenated_gzip_members"}
    else:
        if combined_path.exists():
            combined_path.unlink()
        combined = {"file":None,"reason":"all-years compressed file would approach GitHub per-file size limit; use yearly files"}
    report={
        "schema":"odds_and_race_type_only_v1",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "years":products,
        "all_years":combined,
        "unique_complete_races":summary.get("unique_complete_races",0),
        "supported_odds_types":list(SUPPORTED),
        "field_names":["race_id","race_type","odds"],
        "girls_included":False,
        "price_phase":"final",
        "eligible_date_range":[START.isoformat(),END.isoformat()],
        "excluded_2026_second_half":True,
        "sources_loaded":summary.get("sources_read",0),
        "duplicates_deduplicated":summary.get("duplicate_race",0),
        "conflicting_duplicate_odds":summary.get("conflicting_duplicate_odds",0),
        "duplicate_type_label_variation":summary.get("duplicate_type_label_variation",0),
        "conflict_examples":summary["conflict_examples"],
        "incomplete_or_invalid_by_reason":{k:v for k,v in summary.items()
            if k not in ("conflict_examples","year_counts") and isinstance(v,int)
            and k not in ("unique_complete_races","sources_read","duplicate_race","conflicting_duplicate_odds","duplicate_type_label_variation")},
        "notes":[
            "Output record keys are strictly race_id/race_type/odds, no result or rider fields.",
            "Only completely available numeric 3連単 and 3連複 books are exported.",
            "The dataset does not yet contain all wager types, only 3連単 and 3連複.",
            "Original sources and integrity criteria are checked, never included as model predictors.",
            "File presence alone never means nationwide full historic coverage."
        ]
    }
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ODDS_ONLY_EXPORT",json.dumps(
        {"races":report["unique_complete_races"],"years":{k:v["complete_races"] for k,v in products.items()},
         "skipped":report["incomplete_or_invalid_by_reason"],
         "duplicates":report["duplicates_deduplicated"],
         "odds_conflicts":report["conflicting_duplicate_odds"],
         "type_label_variations":report["duplicate_type_label_variation"]},
         ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()

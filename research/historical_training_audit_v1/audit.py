#!/usr/bin/env python3
"""Audit the EXISTING 2021..2026H1 keirin archives, without scraping or decrypting.

Population: all strictly verified podium race IDs from the merged result archive.
Categories are exhaustive and disjoint:
  two_market_complete: racecard + real result + validated final 3連単 and 3連複 grids
  partial: result + at least a usable racecard OR a complete two-market odds record
  insufficient: result only, neither of the above
The encrypted seven-market archive is independent and cannot be ID-joined without
authorized PRIVATE CMS decryption. Never claim its manifest proves complete training.
This audit does not train a model or modify source data.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "ai_nursery"))
from source_bridge import load_cards_year, stream_full_odds
from train_week import eligible, numeric

DATA = ROOT / "data"
OUTPUT = DATA / "audits" / "historical_training_coverage_v1.json"
LEDGER = DATA / "audits" / "historical_training_coverage_v1_missing.csv.gz"
RACE_ID = re.compile(r"^[0-9]{16}$")
SCORE_KEYS = ("competitive_score", "competition_score", "race_score", "score",
              "average_score", "points", "rating_points", "kyoso_tokuten")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def result_rows(year: int, manifest: dict) -> dict:
    path = DATA / "result_universe_v2" / "combined" / f"results_{year}.jsonl.gz"
    expected = manifest["years"][str(year)]["sha256"]
    if digest(path) != expected:
        raise RuntimeError(f"RESULT_HASH_MISMATCH: {path} (refresh verified manifest)")
    found = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rid = str(item.get("race_id") or "")
            try:
                cars = tuple(int(x) for x in item["finish_order"])
            except (ValueError, TypeError, KeyError):
                raise RuntimeError("Malformed verified result: " + rid)
            if not RACE_ID.fullmatch(rid) or len(cars) != 3 or len(set(cars)) != 3:
                raise RuntimeError("Invalid podium: " + rid)
            if rid in found:
                raise RuntimeError("Duplicate result ID: " + rid)
            found[rid] = cars
    if len(found) != manifest["years"][str(year)]["unique_verified_results"]:
        raise RuntimeError("Result source count mismatch: " + str(year))
    return found


def usable_card(obj: dict | None, result: tuple) -> bool:
    if not obj:
        return False
    entries = obj["entries"]
    if not (5 <= len(entries) <= 9 and set(result).issubset(entries)):
        return False
    raw = obj["race"].get("entry_count")
    if raw not in (None, ""):
        try:
            if int(raw) != len(entries):
                return False
        except (ValueError, TypeError):
            return False
    return True


def published_line(obj: dict | None) -> bool:
    if not obj:
        return False
    race = obj["race"]
    if str(race.get("line_status") or "").lower() != "published":
        return False
    text = str(race.get("predicted_line_formation") or "")
    if not text:
        return False
    try:
        cars = [int(x) for part in text.split("/") for x in part.split("-")]
    except (TypeError, ValueError):
        return False
    return bool(cars) and len(cars) == len(set(cars)) and set(cars) == set(obj["entries"])


def score_coverage(obj: dict | None) -> tuple[bool, set]:
    if not obj:
        return False, set()
    discovered = set()
    for ent in obj["entries"].values():
        valid = False
        for key, raw in ent.items():
            low = key.lower()
            if (key in SCORE_KEYS or "score" in low or "競走得点" in key):
                val = numeric(raw)
                if val is not None and math.isfinite(val) and 30 <= val <= 150:
                    discovered.add(key)
                    valid = True
        if not valid:
            return False, discovered
    return True, discovered


def payout_source_rows(sources: set[str], wanted: set[str]) -> tuple[set[str], set[str]]:
    """Recognize actual saved payout rows. Do NOT mistake row presence for paid amounts."""
    rows, amounts = set(), set()
    for name in sorted(sources):
        src = ROOT / name
        try:
            if src.is_dir():
                f = src / "payouts.csv"
                if not f.is_file():
                    continue
                with f.open("r", encoding="utf-8-sig", newline="") as h:
                    sources_rows = list(csv.DictReader(h))
            elif src.suffix == ".zip":
                with zipfile.ZipFile(src) as z:
                    found = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == "payouts.csv"]
                    if len(found) != 1:
                        continue
                    with z.open(found[0]) as raw:
                        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as h:
                            sources_rows = list(csv.DictReader(h))
            else:
                continue
            for row in sources_rows:
                rid = str(row.get("race_id") or "").strip()
                if rid not in wanted:
                    continue
                rows.add(rid)
                for key, value in row.items():
                    k = str(key).lower()
                    if ("payout" in k or "dividend" in k or k in
                        {"return_yen", "refund_yen", "amount_yen", "払戻金", "払戻"}):
                        v = numeric(value)
                        if v is not None and v > 0:
                            amounts.add(rid)
                            break
        except (OSError, zipfile.BadZipFile, csv.Error) as e:
            raise RuntimeError("Cannot read saved payout source " + name) from e
    return rows, amounts


def audit():
    manifest = json.loads((DATA / "audits" / "result_universe_merged_v2.json").read_text())
    odds_manifest = json.loads((DATA / "audits" / "odds_only_export.json").read_text())
    sealed = json.loads((DATA / "audits" / "odds_seven_bet_archive_count.json").read_text())
    year_rows = []
    grand = Counter()
    score_fields = Counter()
    details = []
    for year in range(2021, 2027):
        results = result_rows(year, manifest)
        ids = set(results)
        cards, card_source_counts, selected_sources = load_cards_year(year)
        odds, market_meta = stream_full_odds(year, ids, verify_sha=True)
        payout_rows, payout_amounts = payout_source_rows(set(selected_sources), ids)
        counts = Counter()
        counts["verified_results_population"] = len(results)
        counts["racecards_with_entries_source"] = len(cards)
        counts["two_market_odds_source"] = odds_manifest["years"][str(year)]["complete_races"]
        counts["encrypted_seven_bet_races_lower_bound"] = sealed["year"][str(year)]["confirmed_distinct_race_lower_bound"]
        for rid, finish in results.items():
            card = cards.get(rid)
            card_ok = usable_card(card, finish)
            price_ok = rid in odds
            if card_ok:
                counts["verified_label_and_usable_card"] += 1
            if price_ok:
                counts["verified_label_and_complete_two_market_odds"] += 1
            if rid in payout_rows:
                counts["payout_row_present"] += 1
            if rid in payout_amounts:
                counts["numeric_payout_amount_present"] += 1
            line_ok = card_ok and published_line(card)
            if line_ok:
                counts["published_line_covers_entrants"] += 1
            score_ok, fields = score_coverage(card) if card_ok else (False, set())
            if score_ok:
                counts["rider_score_present_for_all_entries"] += 1
            score_fields.update(fields)
            combined = card_ok and price_ok
            if combined:
                candidate = {"race_id":rid, **card, "odds":odds[rid], "outcome":finish}
                combined = eligible(candidate)
                if not combined:
                    counts["card_odds_podium_validation_failed"] += 1
            if combined:
                category = "two_market_complete"
                counts["two_market_complete"] += 1
                if line_ok:
                    counts["two_market_complete_with_line"] += 1
                if score_ok:
                    counts["two_market_complete_with_score"] += 1
                if rid in payout_amounts:
                    counts["two_market_complete_with_numeric_payout"] += 1
                if line_ok and score_ok and rid in payout_amounts:
                    counts["two_market_complete_with_line_score_numeric_payout"] += 1
            elif card_ok or price_ok:
                category = "partial"
                counts["partial"] += 1
            else:
                category = "insufficient"
                counts["insufficient"] += 1
            if category != "two_market_complete":
                reasons = []
                if not card_ok:
                    reasons.append("missing_or_invalid_entries")
                if not price_ok:
                    reasons.append("missing_complete_two_market_odds")
                if card_ok and price_ok:
                    reasons.append("inconsistent_entries_prices_or_podium")
                if not line_ok:
                    reasons.append("missing_or_invalid_published_line")
                if not score_ok:
                    reasons.append("rider_scores_incomplete")
                if rid not in payout_amounts:
                    reasons.append("numeric_payout_unverified")
                details.append((rid, year, category, ";".join(reasons)))
        assert (counts["two_market_complete"] + counts["partial"] +
                counts["insufficient"] == counts["verified_results_population"])
        row = {"year":year, **dict(counts),
               "racecard_archive_provenance":card_source_counts,
               "odds_verified_sha256":market_meta["archive_sha256"]}
        year_rows.append(row)
        grand.update(counts)
        print("HISTORICAL_AUDIT_YEAR", year,
              "results", len(results), "two_market_complete", counts["two_market_complete"],
              "partial", counts["partial"], "insufficient", counts["insufficient"], flush=True)
        del cards, odds, results
    assert grand["verified_results_population"] == manifest["verified_unique_results"]
    report = {
        "schema":"historical_training_coverage_v1",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"2021-01-01 to 2026-06-30; saved non-girls archives",
        "population_definition":"verified unique podium race IDs, NOT exact union with sealed prices",
        "categories":{
            "two_market_complete":"verified result + usable entrants + full final trifecta and trio grids",
            "partial":"verified result plus usable entrants or complete two-market odds, but not both consistently",
            "insufficient":"verified result without usable entrants and without complete two-market odds"
        },
        "seven_bet_exact_id_join_status":"NOT VERIFIED: encrypted CMS archive; no private key used",
        "seven_bet_complete_with_all_requested_fields_verified":None,
        "seven_bet_manifest_lower_bound":sealed["confirmed_distinct_race_lower_bound"],
        "prediction_cutoff_verified":False,
        "price_phase":"FINAL (not confirmed T-minus-15)",
        "payout_row_does_not_prove_numeric_amount":True,
        "predicted_lines_are_historical_published_forecasts_not_capture_time_verified":True,
        "score_field_candidates":dict(score_fields),
        "grand_totals":dict(grand),
        "years":year_rows,
        "source_manifest_files":[
            "data/audits/result_universe_merged_v2.json",
            "data/audits/odds_only_export.json",
            "data/audits/odds_seven_bet_archive_count.json"
        ],
        "source_data_modified":False,
        "trained_models_modified":False
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with LEDGER.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as h:
                writer = csv.writer(h)
                writer.writerow(("race_id", "year", "category", "missing_reasons"))
                writer.writerows(sorted(details))
    print("AUDIT_SUMMARY",json.dumps(report["grand_totals"],ensure_ascii=False),flush=True)


if __name__ == "__main__":
    audit()

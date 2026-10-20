"""Secure Phase-1 source audit of ALL existing seven-market CMS odds 2021-2026H1.

NO training, NO buying, NO upload of decrypted odds, NO key publication.
Counts every authenticated ciphertext (including retries) and verifies all
available complete bet-type grids, calendar days, exact official result labels.
Persists only year-level aggregate coverage and exclusion reason counts.

Run inside an ephemeral GitHub Actions runner:
  KEIRIN_SEVEN_BET_PRIVATE_KEY_PEM=... python research/deletion_ai/audit_full_sealed_v3me_2021_2026.py

The manifest counts are lower bounds, not estimates of supervised usable races.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

from secure_seven_bet_bridge import ROOT, CERT, iter_year, secure_key_temp

CENSUS=ROOT/"data/audits/odds_seven_bet_archive_count.json"
OUTPUT=ROOT/"data/audits/v3me_full_sealed_source_integrity_2021_2026.json"
YEAR=range(2021,2027)
EXPECTED_LOWER_BOUND=134336
VALID_REASONS=(
    "no_verified_podium", "contradictory_price_races_quarantined",
    "race_date_id_conflict", "race_type_missing", "girls_excluded",
    "unknown_market", "bad_odds", "duplicate_price_combination",
    "out_of_5_to_9_car_model", "invalid_car_grid", "incomplete_seven_bet_types",
    "non_point_price_for_model_pool", "duplicate_normalized_selection",
    "podium_cars_not_in_market",
)

def check_local_private_key(key):
    """Compare RSA public bytes to repository's published X509 cert; no PEM output."""
    key_pub=subprocess.run(
        ["openssl","pkey","-in",str(key),"-pubout","-outform","DER"],
        capture_output=True,check=True)
    cert_pub=subprocess.run(
        ["openssl","x509","-in",str(CERT),"-pubkey","-noout"],
        capture_output=True,check=True)
    cert_der=subprocess.run(
        ["openssl","pkey","-pubin","-outform","DER"],
        input=cert_pub.stdout,capture_output=True,check=True)
    if hashlib.sha256(key_pub.stdout).digest()!=hashlib.sha256(cert_der.stdout).digest():
        raise RuntimeError("PRIVATE_KEY_DOES_NOT_MATCH_ARCHIVE_PUBLIC_CERT")
    return hashlib.sha256(CERT.read_bytes()).hexdigest()

def run():
    if OUTPUT.exists():
        raise RuntimeError("IMMUTABLE_FULL_SOURCE_AUDIT_EXISTS")
    manifest=json.loads(CENSUS.read_text(encoding="utf-8"))
    if manifest["confirmed_distinct_race_lower_bound"]<EXPECTED_LOWER_BOUND:
        raise RuntimeError("SOURCE_CENSUS_REGRESSED")
    if manifest["distinct_days"]<2000:
        raise RuntimeError("SOURCE_DAYS_REGRESSED")
    years={}
    total_supervised=0
    total_days=0
    total_original_files=0
    total_noncontradictory=0
    keymanager,keypath=secure_key_temp()
    try:
        cert_fingerprint=check_local_private_key(keypath)
        for year in YEAR:
            y=str(year)
            begin=f"{year}-01-01"
            end=(f"{year}-06-30" if year==2026 else f"{year}-12-31")
            c=Counter()
            bets=Counter()
            entries=Counter()
            races_seen_for_year=0
            for row in iter_year(year,begin,end,keypath,c):
                races_seen_for_year+=1
                books=row["odds"]
                for market in ("3連単","3連複"):
                    bets[market]+=len(books[market])
                entries[len(row["entries"])]+=1
            original=manifest["year"][y]
            year_count=original["confirmed_distinct_race_lower_bound"]
            if c["sealed_files_scanned"]<1 or c["days_decrypted"]<1:
                raise RuntimeError(f"EMPTY_ENCRYPTED_ARCHIVE_{year}")
            if c["fully_supervised_usable_races"] != races_seen_for_year:
                raise RuntimeError(f"COUNT_MISMATCH_{year}")
            if c["distinct_same_day_full_books"] < c["fully_supervised_usable_races"]:
                raise RuntimeError(f"SUPERVISED_EXCEEDS_DISTINCT_BOOKS_{year}")
            # One verified price book may legitimately lack a result label.
            # Lower-bound manifests are per-day max; full day unions can be larger.
            # Incompatible rerun prices are quarantined rather than fabricated.
            years[y]={
              "manifest_days_with_verified_ciphertext":original["distinct_days_with_verified_ciphertext"],
              "actual_decrypted_days":c["days_decrypted"],
              "original_manifest_distinct_races_lower_bound":year_count,
              "authenticated_cipher_files_scanned":c["sealed_files_scanned"],
              "verified_complete_books_including_retry_duplicates":c["verified_sealed_books_including_retries"],
              "unique_day_books_before_normalization":c["distinct_same_day_full_books"],
              "valid_official_result_labels_universe":c["valid_official_result_labels"],
              "fully_supervised_usable_races":races_seen_for_year,
              "missing_verified_labels":c["no_verified_podium"],
              "rejected_girls_races":c["girls_excluded"],
              "contradictory_price_races_quarantined":c["contradictory_price_races_quarantined"],
              "other_rejection_reason_counts":{
                  key:int(c[key]) for key in VALID_REASONS
                  if key not in {"no_verified_podium","girls_excluded",
                                 "contradictory_price_races_quarantined"} and c[key]},
              "modeled_3rentan_odds_rows":bets["3連単"],
              "modeled_3renpuku_odds_rows":bets["3連複"],
              "usable_races_by_car_count":{str(n):v for n,v in sorted(entries.items())},
              "published_odds_are_historical_FINAL":True,
            }
            total_supervised+=races_seen_for_year
            total_days+=c["days_decrypted"]
            total_original_files+=c["sealed_files_scanned"]
            total_noncontradictory+=c["distinct_same_day_full_books"]
            print("V3ME_SECURE_AUDIT_YEAR_FINISHED",json.dumps({
              "year":year,"days":c["days_decrypted"],"auth_files":c["sealed_files_scanned"],
              "manifest_lower_bound":year_count,"distinct_books":c["distinct_same_day_full_books"],
              "supervised":races_seen_for_year,"unmatched_results":c["no_verified_podium"],
              "conflicting_retries":c["contradictory_price_races_quarantined"],
            },ensure_ascii=False),flush=True)
        assert len(years)==6
    finally:
        keymanager.cleanup()
    report={
        "schema":"v3me_full_sealed_source_integrity_v1",
        "status":"COMPLETE",
        "source":"data/odds_ciphertext/2021..2026",
        "source_census":"data/audits/odds_seven_bet_archive_count.json",
        "all_six_years_sealed_source_decrypted":True,
        "key_matches_repository_public_certificate":True,
        "repository_public_cert_sha256":cert_fingerprint,
        "all_source_cipher_sha256_verified":True,
        "each_saved_market_full_grid_validated":True,
        "all_original_market_types_authenticated_not_all_modeled":True,
        "modeled_markets":["3連単","3連複"],
        "racecard_not_required_for_odds_only_learning":True,
        "exact_result_join_by_race_id":True,
        "girls_races_explicitly_excluded":True,
        "archive_manifest_distinct_race_lower_bound":manifest["confirmed_distinct_race_lower_bound"],
        "authenticated_days_total":total_days,
        "original_sealed_files_scanned":total_original_files,
        "distinct_day_books_before_normalization":total_noncontradictory,
        "complete_and_supervised_usable_races_total":total_supervised,
        "year":years,
        "encrypted_secret_or_plaintext_published":False,
        "only_aggregate_evidence_published":True,
        "no_fitting_or_predictive_model_training":True,
        "this_is_phase_one_source_audit_not_model_evaluation":True,
        "warning":"Manifest counts are per-day confirmed lower bounds, not the exact number of distinct valid joined races. Raw seven types include frame markets only for nine-car races. This is a source completeness audit, not training or profit verification."
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V3ME_SECURE_AUDIT_COMPLETE",json.dumps({
        "verified_days":total_days,
        "original_files":total_original_files,
        "complete_joined_races":total_supervised,
        "result_filepath":str(OUTPUT.relative_to(ROOT)),
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    run()

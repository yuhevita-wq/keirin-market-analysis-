#!/usr/bin/env python3
"""Secure FULL original 7-market odds + exact-result connection for Type Lab research.

No sampling. Every SHA-verified ciphertext archive covering a specified complete year
is opened. All seven saleable markets stay in the private encrypted output.
Every no-result, duplicate, quarantine, and unsupported car-count is counted.
DOES NOT generate p1/p3 or infer unavailable rider/line features.
"""
from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "research" / "deletion_ai"))
from secure_seven_bet_bridge import (
    CERT, catalog, decrypt_sealed, normalize_book, race_day,
    secure_key_temp, verified_results
)

YEARS = tuple(range(2021, 2027))
CAR_SUPPORTED = (7, 9)
FINAL_DAY_2026 = "2026-06-30"


class SealedWriter:
    """Streaming GZIP -> OpenSSL CMS encryption; never stores plaintext at rest."""
    def __init__(self, target):
        self.target = Path(target)
        self.proc = None
        self.z = None
        self.entries = 0

    def __enter__(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "openssl", "cms", "-encrypt", "-binary", "-aes-256-gcm",
            "-stream", "-outform", "DER", "-out", str(self.target), str(CERT)
        ]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.z = gzip.GzipFile(fileobj=self.proc.stdin, mode="wb", mtime=0)
        return self

    def write(self, record):
        assert self.z is not None
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self.z.write((payload + "\n").encode("utf-8"))
        self.entries += 1

    def __exit__(self, typ, value, tb):
        if self.z is not None:
            self.z.close()
        if self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.close()
        if self.proc is not None:
            try:
                rc = self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                raise RuntimeError("Secure writer did not finish")
            if rc:
                self.target.unlink(missing_ok=True)
                raise RuntimeError("CMS output encryption failed")
        if typ is not None:
            self.target.unlink(missing_ok=True)
        return False


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", required=True, type=int, choices=YEARS)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    year = args.year
    first, last = f"{year}-01-01", (FINAL_DAY_2026 if year == 2026 else f"{year}-12-31")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Preflight verifies ALL per-date manifests and checks there is no skipped date.
    days, manifest_totals = catalog(first, last)
    calendar_days = 181 if year == 2026 else (366 if year == 2024 else 365)
    if len(days) != calendar_days and not (year == 2022 and len(days) == 363):
        raise RuntimeError(f"Historical day catalog unexpectedly incomplete: {year} {len(days)}")
    unique_cipher = {str(p): digest for group in days.values() for p, digest in group}
    if not unique_cipher:
        raise RuntimeError("empty verified ciphertext manifest")
    # Each year must inspect every manifest-listed file, not a per-day largest-run sample.
    print("ARCHIVE_YEAR_START", json.dumps({
        "year":year, "days":len(days), "unique_cipher_files":len(unique_cipher),
        "source_rev":subprocess.check_output(["git","rev-parse","HEAD"],cwd=SRC,text=True).strip()
    }), flush=True)

    outcomes, result_stats = verified_results(year)
    counted = collections.Counter()
    bycar = collections.Counter()
    malformed_examples = []
    written_rids = set()
    conn_odds = out / f"all_seven_bet_odds_{year}.jsonl.gz.cms"
    conn_labels = out / f"verified_result_labels_{year}.jsonl.gz.cms"

    # External secret never appears in log/artifact or destination CSV.
    tmp, key = secure_key_temp()
    try:
        with tmp, SealedWriter(conn_odds) as odds_out, SealedWriter(conn_labels) as label_out:
            for day, files in days.items():
                candidates = {}
                conflicts = set()
                with tempfile.TemporaryDirectory(prefix="type_lab_private_day_") as scratch:
                    for sealed, digest in files:
                        source = decrypt_sealed(sealed, digest, key, scratch)
                        counted["archives_decrypted"] += 1
                        counted["race_books_including_retries"] += len(source)
                        for rid, records in source.items():
                            if rid in conflicts:
                                continue
                            canonical = sorted(records)
                            if rid in candidates and candidates[rid] != canonical:
                                candidates.pop(rid, None)
                                conflicts.add(rid)
                                continue
                            candidates[rid] = canonical

                counted["price_conflict_races"] += len(conflicts)
                counted["days_processed"] += 1
                counted["race_books_after_retry_dedup"] += len(candidates)

                for rid, rows in sorted(candidates.items()):
                    if rid in written_rids:
                        raise RuntimeError("SAME_RACE_ID_ACROSS_DAYS: source record collision")
                    written_rids.add(rid)
                    book, reason = normalize_book(rid, day, rows)
                    if reason:
                        counted[f"excluded_{reason}"] += 1
                        if len(malformed_examples) < 12:
                            malformed_examples.append({"reason":reason,"id_sha256":hashlib.sha256(rid.encode()).hexdigest()})
                        continue
                    n = len(book["entries"])
                    bycar[str(n)] += 1
                    counted["all_seven_market_prices_verified"] += 1
                    if n not in CAR_SUPPORTED:
                        counted["not_supported_by_original_type_lab_5_6_8_cars"] += 1
                    label = outcomes.get(rid)
                    if label is None:
                        counted["missing_verified_result"] += 1
                    elif not set(label).issubset(book["entries"]):
                        counted["result_car_outside_odds_lineup"] += 1
                        label = None
                    else:
                        counted["odds_result_race_id_joined"] += 1
                        if n in CAR_SUPPORTED:
                            counted["odds_result_joined_type_lab_7_9_cars"] += 1

                    # Full seven-market odds are preserved in the sealed artifact.
                    grouped = collections.defaultdict(list)
                    for raw in rows:
                        grouped[raw[2]].append([raw[3], raw[4], raw[5]])
                    record = {
                        "race_id":rid, "race_date":day, "race_type":rows[0][1],
                        "n_cars":n, "price_phase":"historical_final",
                        "market_count":len(grouped),
                        "markets":dict(grouped),
                    }
                    odds_out.write(record)
                    if label is not None:
                        label_out.write({"race_id":rid, "finish_order":list(label)})

                if counted["days_processed"] % 31 == 0:
                    print("TYPE_LAB_FULL_CONNECT_PROGRESS", json.dumps({
                        "year":year, "days":counted["days_processed"],
                        "archives":counted["archives_decrypted"],
                        "verified_books":counted["all_seven_market_prices_verified"],
                        "joined_outcomes":counted["odds_result_race_id_joined"],
                    }), flush=True)
            if counted["archives_decrypted"] != len(unique_cipher):
                raise RuntimeError("archive count mismatch: not all manifest-listed sources read")
            if counted["days_processed"] != len(days):
                raise RuntimeError("day catalog under-read")
    finally:
        # Ephemeral RSA private key and decrypted temporary CSV were not persisted.
        pass

    report = {
        "schema":"type_lab_full_seven_bet_secure_join_v1",
        "year":year, "first_day":first, "last_day":last,
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=SRC,text=True).strip(),
        "encrypted_files_expected":len(unique_cipher),
        "encrypted_files_verified_decrypted":counted["archives_decrypted"],
        "all_catalog_days":len(days),
        "days_processed":counted["days_processed"],
        "counts":dict(sorted(counted.items())),
        "by_car_count":dict(sorted(bycar.items())),
        "historical_result_table_count":len(outcomes),
        "result_table_validation":result_stats,
        "encrypted_artifacts": [
            {"name":conn_odds.name, "sha256":sha256(conn_odds), "bytes":conn_odds.stat().st_size,
             "records":counted["all_seven_market_prices_verified"]},
            {"name":conn_labels.name, "sha256":sha256(conn_labels), "bytes":conn_labels.stat().st_size,
             "records":counted["odds_result_race_id_joined"]},
        ],
        "malformed_race_hashes_sample":malformed_examples,
        "plaintext_odds_in_artifact":False,
        "owner_private_key_in_artifact":False,
        "type_lab_rider_features_connected":False,
        "type_lab_p1_p3_model_predictions_connected":False,
        "train_started":False,
        "notes":[
            "Preserves every market from all verified ciphertext files; output encrypted with repository's public CMS certificate.",
            "Results are stored in a separate encrypted artifact; no hindsight features or bet placements.",
            "Missing or conflicting result labels are reported and never fabricated.",
            "7/9-car market and result join does not imply complete racecard/p1/p3/line inputs for the original Type Lab.",
        ],
    }
    (out/"connection_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("TYPE_LAB_FULL_YEAR_DONE",json.dumps({
        "year":year,"files":len(unique_cipher),"all_seven_bet_books":counted["all_seven_market_prices_verified"],
        "joined":counted["odds_result_race_id_joined"],
        "joined_type_lab_7_9":counted["odds_result_joined_type_lab_7_9_cars"],
        "outcome_missing":counted["missing_verified_result"],
        "status":"COMPLETE_ONE_YEAR"
    }),flush=True)


if __name__=="__main__":
    main()

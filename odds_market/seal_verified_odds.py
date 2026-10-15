"""Encrypt verified full odds books before storing them in a public GitHub repo.

The recipient certificate is public, but the matching RSA private key must NEVER
be uploaded to GitHub. Each output is an encrypted CMS envelope, not plain odds.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from .collect_all_odds import complete_saved_races, GIRLS_RX


def seal_all(source, target, run_id, cert):
    if not cert.is_file():
        raise FileNotFoundError("Missing public encryption certificate")
    summary = {"run_id": str(run_id), "dates": {}, "complete_races": 0,
               "quote_rows": 0, "encrypted_files": []}
    for path in sorted(source.rglob("*.csv.gz")):
        day = date.fromisoformat(path.name.removesuffix(".csv.gz"))
        if not date(2021, 1, 1) <= day <= date(2026, 6, 30):
            raise ValueError("Date outside approved archival scope")
        records = complete_saved_races(path)
        if not records:
            continue
        if any(GIRLS_RX.search(r[1]) for rows in records.values() for r in rows):
            raise ValueError("Girls race in odds archive")
        # Every persisted CSV is already built from complete individual race books.
        # Require that a resumable integrity re-read accepts ALL of its records.
        import csv, gzip
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            source_rows = list(csv.DictReader(f))
        row_count = sum(map(len, records.values()))
        if row_count != len(source_rows):
            raise ValueError("Refusing incomplete quotes in source archive")
        relative = Path(str(day.year)) / day.isoformat() / (str(run_id) + ".p7m")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError("Immutable encrypted archive already exists")
        tmp = destination.with_suffix(".p7m.tmp")
        try:
            subprocess.run(["openssl", "cms", "-encrypt", "-binary",
                            "-aes-256-cbc", "-in", str(path),
                            "-out", str(tmp), "-outform", "DER", str(cert)],
                           check=True, capture_output=True, text=True)
            if not tmp.is_file() or tmp.stat().st_size < 500:
                raise ValueError("Encryption returned an unexpectedly short file")
            tmp.replace(destination)
        finally:
            tmp.unlink(missing_ok=True)
        summary["dates"][day.isoformat()] = {"complete_races": len(records),
                                           "quote_rows": row_count}
        summary["complete_races"] += len(records)
        summary["quote_rows"] += row_count
        summary["encrypted_files"].append({
            "path": str(destination),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()
        })
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--cert", type=Path, default=Path("odds_market/vault_public_cert.pem"))
    p.add_argument("--audit", type=Path, required=True)
    args = p.parse_args()
    manifest = seal_all(args.source, args.target, args.run_id, args.cert)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n",
                          encoding="utf-8")
    print("ENCRYPTED_COMPLETE_RACES", manifest["complete_races"], flush=True)
    print("ENCRYPTED_ODDS_ROWS", manifest["quote_rows"], flush=True)


if __name__ == "__main__":
    main()

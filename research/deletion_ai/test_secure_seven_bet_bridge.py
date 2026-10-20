"""Synthetic 7-bet CMS cryptography fixture. No production secret required."""
from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import os
import subprocess
import tempfile
from pathlib import Path

from odds_market.collect_all_odds import FIELDS,expected,ALL_TYPES
import secure_seven_bet_bridge as bridge


def check():
    assert bridge.race_day("7320240704010001")=="2024-07-04"
    assert bridge.race_day("7320240704020001")=="2024-07-05"
    assert bridge.race_day("incorrect-race-id") is None
    with tempfile.TemporaryDirectory(prefix="synthetic_seven_bet_") as root:
        folder=Path(root)
        cert=folder/"certificate.pem"
        key=folder/"private.pem"
        odds=folder/"private.csv.gz"
        sealed=folder/"sealed.p7m"
        subprocess.run(["openssl","req","-newkey","rsa:2048","-nodes","-x509",
            "-keyout",str(key),"-out",str(cert),"-days","1","-subj",
            "/CN=synthetic-unit-test"],check=True,capture_output=True)
        day="2024-07-04";rid="7320240704010001"
        types=expected(7)
        assert set(types)==set(ALL_TYPES)-{"2wakutan","2wakufuku"}
        with gzip.open(odds,"wt",encoding="utf-8",newline="") as file:
            writer=csv.writer(file)
            writer.writerow(FIELDS)
            for market,combos in types.items():
                for permutation in sorted(combos):
                    k="-".join(map(str,permutation))
                    writer.writerow([rid,"A級予選",market,k,"37.0","37.0","2024-07-04 11:00"])
        subprocess.run(["openssl","cms","-encrypt","-binary","-aes-256-gcm",
            "-in",str(odds),"-out",str(sealed),"-outform","DER",str(cert)],
            check=True,capture_output=True)
        exact_sha=hashlib.sha256(sealed.read_bytes()).hexdigest()
        old_cert=bridge.CERT
        try:
            bridge.CERT=cert
            with tempfile.TemporaryDirectory(prefix="sealed_output_") as plain_dir:
                records=bridge.decrypt_sealed(sealed,exact_sha,key,plain_dir)
                assert len(records)==1 and rid in records
                rec,error=bridge.normalize_book(rid,day,records[rid])
                assert error is None,error
                assert rec["seven_bet_market_types"]==5
                assert len(rec["odds"]["3連単"])==210
                assert len(rec["odds"]["3連複"])==35
                assert all(p==37. for p in rec["odds"]["3連単"].values())
                assert all(p==37. for p in rec["odds"]["3連複"].values())
                assert not any(Path(plain_dir).iterdir()),"Cleartext residue"
                mismatch,why=bridge.normalize_book(rid,"2024-07-05",records[rid])
                assert mismatch is None and why=="race_date_id_conflict"
                try:
                    bridge.decrypt_sealed(sealed,"0"*64,key,plain_dir)
                except RuntimeError as e:
                    assert "Cipher digest mismatch" in str(e)
                else:raise AssertionError("Must reject checksum tampering")
        finally:
            bridge.CERT=old_cert
    assert not os.environ.get(bridge.SECRET_ENV),"Synthetic tests must not need real private key"
    print("SECURE_SEVEN_BET_CRYPTOGRAPHY_AND_FULL_BOOK_UNIT_TESTS_PASS")


if __name__=="__main__":
    check()

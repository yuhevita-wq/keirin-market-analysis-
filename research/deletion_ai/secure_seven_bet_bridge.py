"""SECURE seven-bet historical archive reader for deletion-model training.

Never scrape. Never export plaintext odds to Git. Decryption requires the
owner's RSA private key injected as an Actions secret and read ONLY in an
ephemeral runner. Archives are fully checked by ciphertext SHA-256 and the
collector's original 7-market completeness validator before any price is used.

Return only verified, distinct race-ID/date/cars/final 3連単+3連複 markets
to the existing two-market deletion engine. Validate availability of ALL
seven published kinds but DO NOT claim that all seven kinds are modeled.
Results come from official recorded podiums by exact race ID, not scraped
from odds or inferred. No arbitrary racecard join: this odds-only model does
not use racecard features, and requiring saved racecards would silently
discard otherwise valid encrypted markets.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter,defaultdict
from datetime import date,timedelta
from pathlib import Path

from odds_market.collect_all_odds import complete_saved_races,FIELDS,ALL_TYPES

ROOT=Path(__file__).resolve().parents[2]
SEALED=ROOT/"data/odds_ciphertext"
MANIFESTS=ROOT/"data/audits/odds_seven_bet_runs"
CERT=ROOT/"odds_market/vault_public_cert.pem"
RESULTS=ROOT/"data/result_universe_v2/combined"
SECRET_ENV="KEIRIN_SEVEN_BET_PRIVATE_KEY_PEM"
MODELED={"3rentan":"3連単","3renpuku":"3連複"}
SEVEN=set(ALL_TYPES)
GIRLS_RX=re.compile(r"ガールズ|女子|[ＬL]級",re.I)


def race_day(rid):
    if len(rid)!=16 or not rid.isascii() or not rid.isdigit():
        return None
    try:
        start=date(int(rid[2:6]),int(rid[6:8]),int(rid[8:10]))
        d=int(rid[10:12])
        if not 1<=d<=7:return None
        return (start+timedelta(days=d-1)).isoformat()
    except ValueError:
        return None


def catalog(first,last):
    """Include ALL authenticated run files per day, not just a max-size one.

    A single run can contain different race IDs than another retry. Catalog
    deduplicates only by exact encrypted file path+SHA, never by run count.
    """
    day_files=defaultdict(dict)
    counters=Counter()
    for path in sorted(MANIFESTS.glob("*.json")):
        report=json.loads(path.read_text(encoding="utf-8"))
        runid=str(report["run_id"])
        dates=report.get("dates",{})
        assert isinstance(dates,dict)
        if sum(int(v["complete_races"]) for v in dates.values())!=int(report["complete_races"]):
            raise RuntimeError("Corrupt sealed odds run manifest "+path.name)
        seen_dates=set()
        for x in report.get("encrypted_files",[]):
            fp=ROOT/x["path"]
            day=fp.parent.name
            if not first<=day<=last:continue
            if day not in dates:
                raise RuntimeError("File not in run's verified date listing")
            if not x["path"].startswith("data/odds_ciphertext/"):
                raise RuntimeError("Source path outside authorized sealed archive")
            if not fp.exists():
                raise FileNotFoundError("Missing verified sealed archive: "+str(fp.relative_to(ROOT)))
            sha=x["sha256"]
            old=day_files[day].get(fp)
            if old and old!=sha:
                raise RuntimeError("Conflicting digests for same archived cipher")
            day_files[day][fp]=sha
            seen_dates.add(day)
            counters["sealed_files"]+=1
        if any(first<=d<=last and d not in seen_dates for d in dates):
            raise RuntimeError("Manifest dates without verified archive")
    assert day_files,("Empty ciphertext catalog",first,last)
    return {day:sorted(values.items(),key=lambda v:str(v[0]))
            for day,values in sorted(day_files.items())},dict(counters)


def verified_results(year):
    """Read exact race IDs and finish order once, without any racecard filter."""
    path=RESULTS/f"results_{year}.jsonl.gz"
    known={}
    invalid=set()
    stats=Counter()
    with gzip.open(path,"rt",encoding="utf-8") as file:
        for line in file:
            if not line.strip():continue
            obj=json.loads(line)
            rid=str(obj.get("race_id",""))
            if rid in invalid:continue
            order=obj.get("finish_order")
            try:
                if not isinstance(order,list) or len(order)!=3:
                    stats["missing_order"]+=1;continue
                cars=tuple(int(c) for c in order)
                if len(set(cars))!=3 or min(cars)<1 or max(cars)>9:
                    stats["invalid_order"]+=1;continue
            except (TypeError,ValueError):
                stats["invalid_order"]+=1;continue
            old=known.get(rid)
            if old and old!=cars:
                invalid.add(rid)
                del known[rid]
                stats["conflicting_results"]+=1
                continue
            known[rid]=cars
    stats["verified_distinct_results"]=len(known)
    return known,dict(stats)


def secure_key_temp():
    """Caller removes secret before process returns. NEVER log its bytes."""
    pem=os.environ.pop(SECRET_ENV,"")
    if not pem:
        raise RuntimeError(
          "PRIVATE_KEY_NOT_CONFIGURED: set GitHub Actions secret "
          +SECRET_ENV+" via repository Settings; do not upload or commit it."
        )
    tmp=tempfile.TemporaryDirectory(prefix="keirin_private_key_")
    keypath=Path(tmp.name)/"key.pem"
    fd=os.open(keypath,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,"wb") as file:
            file.write(pem.encode("utf-8").replace(b"\r\n",b"\n"))
    finally:
        pem=""
    return tmp,keypath


def decrypt_sealed(src,sha,key,tmpdir):
    """Never persist any plaintext outside a private temporary directory."""
    check=hashlib.sha256()
    with src.open("rb") as file:
        for block in iter(lambda:file.read(1024*1024),b""):
            check.update(block)
    if check.hexdigest()!=sha:
        raise RuntimeError("Cipher digest mismatch for "+src.name)
    outfile=Path(tmpdir)/"day_cleartext.csv.gz"
    if outfile.exists():outfile.unlink()
    cmd=["openssl","cms","-decrypt","-inform","DER","-in",str(src),
         "-recip",str(CERT),"-inkey",str(key),"-out",str(outfile)]
    result=subprocess.run(cmd,capture_output=True,text=True,check=False,timeout=120)
    if result.returncode!=0:
        outfile.unlink(missing_ok=True)
        raise RuntimeError("CMS_DECRYPT_FAILED: check privately installed RSA key/certificate match")
    if not outfile.is_file() or outfile.stat().st_size<50:
        raise RuntimeError("CMS_DECRYPT_EMPTY_OUTPUT")
    try:
        verified=complete_saved_races(outfile)
    except Exception:
        outfile.unlink(missing_ok=True)
        raise RuntimeError("ARCHIVE_FULL_SEVEN_MARKET_VERIFICATION_FAILED")
    finally:
        outfile.unlink(missing_ok=True)
    return verified


def normalize_book(rid,day,rows):
    """Reject odds that are not mathematically exact single-price outcomes."""
    if race_day(rid)!=day:return None,"race_date_id_conflict"
    rtype={r[1] for r in rows}
    if len(rtype)!=1 or not next(iter(rtype)):
        return None,"race_type_missing"
    if GIRLS_RX.search(next(iter(rtype))):
        return None,"girls_excluded"
    bytype=defaultdict(dict)
    cars=set()
    for r in rows:
        market=r[2]
        selection=r[3]
        if market not in SEVEN:
            return None,"unknown_market"
        try:
            lo=float(r[4]);hi=float(r[5])
        except (TypeError,ValueError):
            return None,"bad_odds"
        if not(math.isfinite(lo) and math.isfinite(hi) and 1<=lo<=hi):
            return None,"bad_odds"
        if selection in bytype[market]:
            return None,"duplicate_price_combination"
        bytype[market][selection]=(lo,hi)
        if market=="3rentan":
            cars.update(int(c) for c in selection.split("-"))
    n=len(cars)
    if not 5<=n<=9:return None,"out_of_5_to_9_car_model"
    if cars!=set(range(1,n+1)):
        return None,"invalid_car_grid"
    expected=set(SEVEN if n==9 else SEVEN-{"2wakutan","2wakufuku"})
    if set(bytype)!=expected:
        return None,"incomplete_seven_bet_types"
    three={}
    trio={}
    for original,target in (("3rentan",three),("3renpuku",trio)):
        for k,(low,high) in bytype[original].items():
            # These two pools have point prices; WIDE's range is not modeled.
            if not math.isclose(low,high,rel_tol=0,abs_tol=1e-9):
                return None,"non_point_price_for_model_pool"
            sel=k if original=="3rentan" else "=".join(
                    sorted(k.split("-"),key=int))
            if sel in target:return None,"duplicate_normalized_selection"
            target[sel]=low
    return {
        "race_id":rid,
        "race":{"race_date":day},
        "entries":{c:{} for c in range(1,n+1)},
        "odds":{"3連単":three,"3連複":trio},
        "seven_bet_market_types":len(bytype),
        "all_seven_bet_rows":sum(len(x) for x in bytype.values()),
    },None


def iter_year(year,first,last,keypath,stats):
    """One decrypted calendar DAY held in memory, never entire multi-year odds.

    All retry files are unioned by *race_id*, and contradictory quoted odds
    quarantine that exact race rather than choosing according to results.
    """
    first=max(first,f"{year}-01-01")
    last=min(last,f"{year}-12-31")
    source,census=catalog(first,last)
    stats["sealed_files_scanned"]+=census["sealed_files"]
    label,label_stats=verified_results(year)
    stats["valid_official_result_labels"]=label_stats["verified_distinct_results"]
    for day,files in source.items():
        union={}
        conflicted=set()
        with tempfile.TemporaryDirectory(prefix="keirin_private_day_") as d:
            for path,sha in files:
                obtained=decrypt_sealed(path,sha,keypath,d)
                stats["verified_sealed_books_including_retries"]+=len(obtained)
                for rid,raw in obtained.items():
                    if rid in conflicted:continue
                    canon=sorted(raw)
                    if rid in union and union[rid]!=canon:
                        union.pop(rid,None)
                        conflicted.add(rid)
                        continue
                    union[rid]=canon
        stats["contradictory_price_races_quarantined"]+=len(conflicted)
        stats["days_decrypted"]+=1
        stats["distinct_same_day_full_books"]+=len(union)
        for rid,raw in sorted(union.items()):
            normalized,reason=normalize_book(rid,day,raw)
            if reason:
                stats[reason]+=1
                continue
            finish=label.get(rid)
            if finish is None:
                stats["no_verified_podium"]+=1
                continue
            if not set(finish).issubset(normalized["entries"]):
                stats["podium_cars_not_in_market"]+=1
                continue
            stats["fully_supervised_usable_races"]+=1
            # Only the two market feature books are constructed from the
            # authenticated SEVEN-market source. All other types validated.
            normalized["outcome"]=finish
            yield normalized
        if stats["days_decrypted"]%31==0:
            print("SECURE_SEVEN_BET_DAY_PROGRESS",json.dumps({
                "year":year,"days":stats["days_decrypted"],
                "usable":stats["fully_supervised_usable_races"],
                "sealed_races_seen":stats["distinct_same_day_full_books"]},
                ensure_ascii=False),flush=True)

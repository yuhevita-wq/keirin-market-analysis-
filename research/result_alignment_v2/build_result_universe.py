#!/usr/bin/env python3
"""Recover independently verified historic top-three results from existing GitHub archives.

Independent of the old 32,643-race two-market odds export, and independent of
the private decryption key of the nationwide seven-bet odds archive.

The published result archive is a separate training-LABEL universe. Never
claim these IDs match encrypted seven-bet odds until a secure exact-ID join.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/result_universe_v2"
REPORT = ROOT / "data/audits/result_universe_v2.json"
LOW = date(2021, 1, 1)
HIGH = date(2026, 6, 30)
RX = re.compile(r"\d{16}")


def actual_day(rid: str):
    if not RX.fullmatch(rid):
        return None
    try:
        if not 1 <= int(rid[-4:]) <= 12 or not 1 <= int(rid[10:12]) <= 7:
            return None
        value = date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10]))
        value += timedelta(days=int(rid[10:12]) - 1)
        return value if LOW <= value <= HIGH else None
    except ValueError:
        return None


def accepted_podium(rows):
    values=defaultdict(list)
    for r in rows:
        if "ガールズ" in str(r.get("race_type", "")) or "Ｇ級" in str(r.get("race_type", "")):
            return None
        if str(r.get("finish_position","")).strip() in {"1","2","3"}:
            car=str(r.get("car_no","")).strip()
            if not re.fullmatch("[1-9]", car):
                return None
            values[int(r["finish_position"])].append(int(car))
    if set(values)!={1,2,3} or any(len(values[k]) != 1 for k in (1,2,3)):
        return None
    order=tuple(values[k][0] for k in (1,2,3))
    return order if len(set(order))==3 else None


def consume(handle, seen, stats):
    rows=defaultdict(list)
    for r in csv.DictReader(handle):
        rid=str(r.get("race_id","")).strip()
        if actual_day(rid) is not None:
            rows[rid].append(r)
    stats["source_files"]+=1
    stats["candidate_races"]+=len(rows)
    for rid, chunk in rows.items():
        podium=accepted_podium(chunk)
        if podium is None:
            stats["invalid_or_excluded_races"]+=1
            continue
        seen[rid].add(podium)
        stats["valid_candidates"]+=1


def scan():
    seen=defaultdict(set)
    stats=Counter()
    for p in sorted((ROOT/"data").rglob("results.csv")):
        with p.open("r",encoding="utf-8-sig",newline="") as handle:
            consume(handle,seen,stats)
    for p in sorted((ROOT/"data").rglob("results.csv.gz")):
        with gzip.open(p,"rt",encoding="utf-8-sig",newline="") as handle:
            consume(handle,seen,stats)
    for p in sorted((ROOT/"data/grade_races").rglob("*.zip")):
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.rsplit("/",1)[-1]=="results.csv":
                    with z.open(name) as b:
                        with io.TextIOWrapper(b,encoding="utf-8-sig",newline="") as handle:
                            consume(handle,seen,stats)
    conflicts={rid:sorted(list(o) for o in choices) for rid,choices in seen.items() if len(choices)>1}
    complete={rid:next(iter(choices)) for rid,choices in seen.items() if len(choices)==1}
    OUT.mkdir(parents=True,exist_ok=True)
    by_year={}
    for y in range(2021,2027):
        ids=sorted(rid for rid in complete if actual_day(rid).year == y)
        path=OUT/f"results_{y}.jsonl.gz"
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0,filename="") as gz:
                for rid in ids:
                    gz.write((json.dumps({"race_id":rid,"finish_order":complete[rid]},separators=(",",":"))+"\n").encode())
        by_year[str(y)]={"verified_results":len(ids),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    report={
        "schema":"historical_verified_results_independent_of_odds_v2",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "scope":[LOW.isoformat(), HIGH.isoformat()],
        "verified_podium_races":len(complete),
        "conflicting_race_count":len(conflicts),
        "conflicts_sample":dict(list(sorted(conflicts.items()))[:20]),
        "source_counts":dict(stats),
        "years":by_year,
        "warning":"These are historical outcome LABELS, NOT yet joined to encrypted seven-bet odds race IDs.",
        "no_synthetic_results":True,
        "source_odds_unchanged":True,
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("HISTORIC_RESULT_UNIVERSE",json.dumps({
        "verified_results":len(complete),"conflicts":len(conflicts),
        "by_year":{k:v["verified_results"] for k,v in by_year.items()},
        "files":stats["source_files"]
    }),flush=True)
    return report


if __name__=="__main__":
    scan()

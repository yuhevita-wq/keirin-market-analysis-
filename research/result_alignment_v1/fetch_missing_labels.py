#!/usr/bin/env python3
"""Recover real missing podium labels ONLY for races in the verified odds universe.

Uses the existing result-table parser; keeps response failures and stops early on
a source-wide outage. Never generates or interpolates an outcome.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from keirin_market_analysis.s_yosen_2025 import make_session, fetch_html, page_race_date
from keirin_market_analysis.results_payouts_2025 import parse_result_rows

from build_result_targets import ROOT, odds_universe, parse_podium

BASE = ROOT / "data/recovered_result_labels_v1/odds_matched"
LEDGER = ROOT / "data/audits/results_to_odds_fetch_ledger.json"
COLS = ["race_id", "car_no", "finish_position", "finish_text", "result_source_url", "captured_at_utc"]
ERR_COLS = ["race_id", "error", "attempted_at_utc"]


def unresolved(root=ROOT):
    path = root / "data/results_only/unmatched_race_ids.csv.gz"
    if not path.exists():
        raise FileNotFoundError("First run build_result_targets.py to get exact missing IDs")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        items = list(csv.DictReader(handle))
    return [r["race_id"] for r in items if re.fullmatch(r"\d{16}", r.get("race_id",""))]


def choose(ids, attempts, limit, max_attempts):
    """Prioritize fewer attempts and alternate years, not only 2023."""
    universe = odds_universe(ROOT)
    buckets = defaultdict(list)
    for rid in ids:
        if rid not in universe:
            raise ValueError(f"Missing ledger includes a race outside odds: {rid}")
        if attempts.get(rid, 0) < max_attempts:
            buckets[(attempts.get(rid, 0), universe[rid][0])].append(rid)
    for rows in buckets.values():
        rows.sort()
    ordered = []
    keys = sorted(buckets)
    while keys and len(ordered) < limit:
        remaining = []
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
            if len(ordered) == limit:
                break
        keys = remaining
    return ordered, universe


def request_one(rid, cars, delay):
    # A single source request with a polite per-worker delay.
    time.sleep(delay)
    url = f"https://keirin.kdreams.jp/racedetail/{rid}/?pageType=KS_RACE_CARD_PAGE_TYPE_SHOW_RESULT"
    session = make_session()
    html = fetch_html(session, url)
    soup = BeautifulSoup(html, "lxml")
    first = date.fromisoformat(rid[2:6] + "-" + rid[6:8] + "-" + rid[8:10])
    expected_day = first + timedelta(days=int(rid[10:12]) - 1)
    published_day = page_race_date(soup)
    if published_day != expected_day.isoformat():
        raise ValueError("Published result day differs from odds race ID")
    parsed = parse_result_rows(soup)
    order = parse_podium(parsed, cars)
    if order is None:
        raise ValueError("Result has no unique verified 1st-3rd podium")
    car_numbers = [int(r["car_no"]) for r in parsed]
    if len(set(car_numbers)) != len(car_numbers) or not set(car_numbers).issubset(cars):
        raise ValueError("Results have unexpected or duplicate riders")
    now = datetime.now(timezone.utc).isoformat()
    return [{ "race_id": rid, "car_no": str(r["car_no"]),
              "finish_position": str(r.get("finish_position", "")),
              "finish_text": str(r.get("finish_text", "")),
              "result_source_url": url, "captured_at_utc": now }
            for r in parsed]


def save_csv_gz(path, rows, fields):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as out:
        w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(limit, workers, pause, max_attempts):
    ids = unresolved()
    state = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"schema":1, "attempts":{}}
    attempts = {k:int(v) for k,v in state.get("attempts",{}).items()}
    selection, universe = choose(ids, attempts, limit, max_attempts)
    if not selection:
        print("NO_ELIGIBLE_MISSING_RESULTS", len(ids), "unmatched; capped retries may remain", flush=True)
        return
    rows, fails, attempted = [], [], []
    halted = False
    # Small batches allow a circuit breaker when the publisher is unavailable.
    for start in range(0,len(selection),20):
        chunk = selection[start:start+20]
        outcomes=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(request_one,rid,universe[rid][1],pause):rid for rid in chunk}
            for future in as_completed(jobs):
                rid=jobs[future]
                try:
                    value=future.result()
                    outcomes.append((rid,value,None))
                except Exception as exc:
                    outcomes.append((rid,None,f"{type(exc).__name__}: {exc}"[:500]))
        for rid, records, error in sorted(outcomes):
            attempts[rid] = attempts.get(rid,0)+1
            attempted.append(f"{rid}:{attempts[rid]}")
            if error is not None:
                fails.append({"race_id":rid,"error":error,
                              "attempted_at_utc":datetime.now(timezone.utc).isoformat()})
            else:
                rows.extend(records)
        if len(fails)==len(attempted) and len(attempted)>=20:
            halted=True
            print("SOURCE_CIRCUIT_BREAKER", "first requests have all failed; remaining skipped",flush=True)
            break
    rid_to_rows=defaultdict(list)
    for row in rows:
        rid_to_rows[row["race_id"]].append(row)
    saved_ids = sorted(rid_to_rows)
    name="batch_"+hashlib.sha256("\n".join(attempted).encode()).hexdigest()[:16]
    folder=BASE/name
    if folder.exists():
        raise FileExistsError("Immutable batch already exists: "+str(folder))
    folder.mkdir(parents=True,exist_ok=False)
    save_csv_gz(folder/"results.csv.gz", rows, COLS)
    with (folder/"failures.csv").open("w", encoding="utf-8", newline="") as out:
        writer=csv.DictWriter(out,fieldnames=ERR_COLS)
        writer.writeheader()
        writer.writerows(fails)
    report={"attempted":len(attempted),"completed":len(saved_ids),"failed":len(fails),
            "completed_ids":saved_ids,"attempted_race_ids":[s.split(":")[0] for s in attempted],
            "source_outage_breaker":halted,
            "matched_against_only_saved_odds":True,
            "results_never_model_features":True}
    (folder/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    state["attempts"] = attempts
    state["last_batch"] = name
    state["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    LEDGER.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MISSING_RESULTS_RECOVERY",json.dumps({
        "unmatched_start":len(ids), "attempted":len(attempted),
        "recovered":len(saved_ids), "failed":len(fails),
        "outage_breaker":halted, "batch":name
    },ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--limit",type=int,default=240)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--pause",type=float,default=2.0)
    p.add_argument("--max-attempts",type=int,default=3)
    args=p.parse_args()
    if not (1 <= args.limit <= 400 and 1 <= args.workers <= 2
            and args.pause >= 2.0 and 1 <= args.max_attempts <= 5):
        p.error("Respect bounds: limit 1..400, workers 1..2, pause >=2sec, max-attempts 1..5")
    run(args.limit, args.workers, args.pause, args.max_attempts)


if __name__ == "__main__":
    main()

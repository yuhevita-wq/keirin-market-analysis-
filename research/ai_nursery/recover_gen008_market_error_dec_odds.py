"""Restore December 2023 GEN-008 V3 market-error holdout odds from archived books then KDreams.

Read only public historical FINAL prices. Never use outcomes to select or derive odds.
Leave original odds mart and earlier frozen exam untouched.
"""
from __future__ import annotations
import csv,gzip,hashlib,itertools,json,math,sys
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlsplit
from train_week import ROOT
from source_bridge import load_cards_year,stream_full_odds,_archive_paths,_archive_streams
from keirin_market_analysis.historical_final_odds import scrape_race

FIRST,LAST="2023-12-01","2023-12-14"
BASE=ROOT/"data/holdout_odds_recovery/gen008-2023-12-01-to-14"
PRICES=BASE/"books.jsonl.gz"
REPORT=BASE/"coverage.json"
S_SOURCE=ROOT/"data/2023/s_class_f1_all_parts/2023_q4"
MARKETS=("3連単","3連複")

def fingerprint(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def expected(cars,pool):
    return ({"-".join(map(str,c)) for c in itertools.permutations(cars,3)}
            if pool=="3連単" else
            {"=".join(map(str,c)) for c in itertools.combinations(sorted(cars),3)})

def validate_books(rid,cars,record):
    assert set(record)==set(MARKETS),rid
    assert 3<=len(cars)<=9,len(cars)
    for market in MARKETS:
        book=record[market]
        if set(book)!=expected(cars,market):
            raise ValueError(f"wrong full price grid {rid} {market}: got {len(book)} expected {len(expected(cars,market))}")
        if any(not isinstance(p,(int,float)) or not math.isfinite(p) or p<1 for p in book.values()):
            raise ValueError(f"invalid numeric market {rid} {market}")
    return True

def from_rows(rid,cars,three,trio):
    odds={}
    for market,rs in zip(MARKETS,(three,trio)):
        b={}
        for r in rs:
            if str(r.get("race_id",""))!=rid:raise ValueError("wrong race ID")
            if r.get("ticket_type")!=market or r.get("odds_phase")!="final":
                raise ValueError("not specified final-market odds")
            if r.get("odds_status")!="available":
                raise ValueError(f"unavailable price {market} {r.get('combination')}")
            price=float(r["odds"])
            k=r["combination"]
            if k in b:raise ValueError("duplicate odds "+k)
            b[k]=price
        odds[market]=b
    validate_books(rid,cars,odds)
    return odds

def source_races(wanted):
    """Find provider URLs in raw cards; sanitized model cards omit URLs."""
    found={}
    for kind,path in _archive_paths(2023):
        for source,races,entries in _archive_streams(kind,path):
            for r in races:
                rid=str(r.get("race_id") or "").strip()
                if rid not in wanted or str(r.get("race_date"))<FIRST or str(r.get("race_date"))>LAST:
                    continue
                url=str(r.get("source_url") or "")
                if urlsplit(url).netloc.lower()!="keirin.kdreams.jp":
                    continue
                selected=source==wanted[rid].get("source")
                if rid not in found or selected:
                    found[rid]=dict(r)
    return found

def load_saved_sclass(ids,cards):
    """Existing full S F1 market CSVs, not fabricated substitutes."""
    local=defaultdict(lambda:{"3連単":[],"3連複":[]})
    for market,file in (("3連単","trifecta_final_odds.csv"),("3連複","trio_final_odds.csv")):
        with (S_SOURCE/file).open(encoding="utf-8-sig",newline="") as handle:
            for row in csv.DictReader(handle):
                rid=str(row.get("race_id") or "")
                if rid in ids:
                    local[rid][market].append(row)
    valid={};failed={}
    for rid,books in local.items():
        try:
            price=from_rows(rid,sorted(cards[rid]["entries"]),books["3連単"],books["3連複"])
            valid[rid]={"race_id":rid,"odds":price,"source":"KDreams archived original S F1 Q4 CSV",
                        "phase":"FINAL","source_file":str((S_SOURCE/"trifecta_final_odds.csv").relative_to(ROOT))}
        except Exception as e:failed[rid]=f"{type(e).__name__}: {str(e)[:280]}"
    return valid,failed

def fetch_one(raw,cars):
    rid=str(raw["race_id"])
    three,trio=scrape_race(raw,cars,0.5)
    price=from_rows(rid,cars,three,trio)
    return {"race_id":rid,"odds":price,"source":"KDreams verified historical original page",
            "phase":"FINAL","source_url":three[0].get("odds_source_url",""),
            "captured_at_utc":three[0].get("captured_at_utc","")}

def collect():
    if BASE.exists():
        raise RuntimeError("Holdout recovery already exists. Immutable: refuse overwrite")
    cards,counts,src=load_cards_year(2023,FIRST,LAST)
    prior,market_audit=stream_full_odds(2023,set(cards))
    missing=set(cards)-set(prior)
    print("DEC_MARKET_ODDS_BASELINE",json.dumps({"cards":len(cards),"prior_odds":len(prior),"missing":len(missing)}),flush=True)
    raw=source_races({rid:cards[rid] for rid in missing})
    recovered,local_invalid=load_saved_sclass(missing,cards)
    remaining=missing-set(recovered)
    print("DEC_MARKET_ODDS_LOCAL_ARCHIVE",json.dumps({"local_restored":len(recovered),"remaining":len(remaining),"provider_urls":len(set(raw)&remaining)}),flush=True)
    failures={}
    for rid in sorted(remaining-set(raw)):
        failures[rid]="Missing valid exact race-id KDreams URL in original racecard"
    requests=sorted(remaining&set(raw),key=lambda rid:(raw[rid]["race_date"],rid))
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(fetch_one,raw[rid],sorted(cards[rid]["entries"])):rid for rid in requests}
        for i,fut in enumerate(as_completed(pending),1):
            rid=pending[fut]
            try:recovered[rid]=fut.result()
            except Exception as exc:failures[rid]=f"{type(exc).__name__}: {str(exc)[:400]}"
            if i%50==0:
                print("DEC_MARKET_ODDS_FETCH_PROGRESS",json.dumps({"attempted":i,"of":len(requests),"recovered":len(recovered),"failures":len(failures)}),flush=True)
    assert not(set(recovered)&set(prior))
    assert set(recovered)|set(failures)==missing
    BASE.mkdir(parents=True,exist_ok=False)
    with gzip.open(PRICES,"wt",encoding="utf-8",compresslevel=6) as f:
        for rid,obj in sorted(recovered.items()):
            validate_books(rid,sorted(cards[rid]["entries"]),obj["odds"])
            f.write(json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    failure_freq=Counter(v.split(":",1)[0] for v in failures.values())
    report={
      "stage":"GEN-008-DEC-2023-MARKET-ERROR-ODDS-RECOVERY",
      "target_dates":[FIRST,LAST],"racecards":len(cards),
      "already_complete_in_odds_only":len(prior),
      "missing_from_odds_only":len(missing),
      "raw_S_class_F1_recovered":sum("archived original S" in x["source"] for x in recovered.values()),
      "web_original_recovered":sum("historical original page" in x["source"] for x in recovered.values()),
      "recovered_total":len(recovered),"remaining_unrecovered":len(failures),
      "new_coverage":len(prior)+len(recovered),
      "previous_original_odds_archive_verified":market_audit["hash_verified"],
      "source_count":counts,"selected_source":src,
      "price_type":"historical FINAL; NOT pre-close",
      "no_imputed_odds":True,"no_result_labels_read":True,
      "errors_by_category":dict(failure_freq),"failures":failures,
      "local_invalid_before_fallback":local_invalid,
      "books_sha256":fingerprint(PRICES),
      "run_utc":datetime.now(timezone.utc).isoformat(),
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DEC_MARKET_ODDS_RECOVERY_COMPLETE",json.dumps({k:report[k] for k in ("racecards","already_complete_in_odds_only","missing_from_odds_only","raw_S_class_F1_recovered","web_original_recovered","recovered_total","remaining_unrecovered","new_coverage","errors_by_category")},ensure_ascii=False),flush=True)

if __name__=="__main__":
    collect()

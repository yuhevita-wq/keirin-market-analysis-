#!/usr/bin/env python3
"""2023 COMPLETE SOURCE CENSUS: raw trifecta odds across all archived sources.

This is an AUDIT, not a model fit or a market-strategy test. Avoids the
incorrect reduction to complete (3連単 AND 3連複) odds_only books.
Includes original CSV, grade ZIP, odds backfill, unarchived shards.
Dedup by exact 2023 verified race ID + exact 3連単 ticket; never invent odds.
Record 2023 results universe and any-odds/full-trifecta coverage separately.
"""
from __future__ import annotations
import csv,gzip,io,itertools,json,math,zipfile
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/"data/result_universe_v2/combined/results_2023.jsonl.gz"
OUT=ROOT/"data/audits/trifecta_raw_source_census_2023.json"
EXAMPLE=ROOT/"data/audits/trifecta_raw_source_census_2023_missing.json"
def crows(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)
def ziprows(path,filename):
    with zipfile.ZipFile(path) as z:
        files=[n for n in z.namelist() if n.rsplit("/",1)[-1]==filename]
        if len(files)!=1:return
        with z.open(files[0]) as bf:
            with io.TextIOWrapper(bf,encoding="utf-8-sig",newline="") as f:
                yield from csv.DictReader(f)
def main():
    truth={}
    with gzip.open(RESULT,"rt",encoding="utf-8") as f:
        for line in f:
            j=json.loads(line);t=j.get("finish_order")
            if isinstance(t,list) and len(t)>=3:
                t=tuple(map(str,t[:3]))
                if len(set(t))==3 and all(s.isdigit() for s in t):
                    truth[str(j["race_id"])]=t
    wanted=set(truth)
    # canonical observed tickets by race ID
    prices=defaultdict(dict)
    conflicts=Counter();coverage=Counter();source_counts={};samples={}
    sources=[]
    for p in sorted((ROOT/"data/2023").rglob("trifecta_final_odds*.csv")):
        sources.append(("original_s",str(p.relative_to(ROOT)),lambda p=p:crows(p)))
    for p in sorted((ROOT/"data/grade_races").rglob("*.zip")):
        if "2023" in str(p):
            sources.append(("grade_zip",str(p.relative_to(ROOT)),lambda p=p:ziprows(p,"trifecta_final_odds.csv")))
    for p in sorted((ROOT/"data/odds_universe_v1/data").rglob("trifecta_final_odds.csv.gz")):
        if "/2023/" in str(p) or "/2023_" in str(p):
            sources.append(("backfill",str(p.relative_to(ROOT)),lambda p=p:crows(p)))
    for p in sorted((ROOT/"data/odds_universe_v1/grade_archive_repairs").rglob("trifecta_final_odds.csv.gz")):
        if "2023" in str(p):
            sources.append(("grade_repair",str(p.relative_to(ROOT)),lambda p=p:crows(p)))
    for p in sorted((ROOT/"data/unarchived_races_v1").glob("batch_*/trifecta_final_odds.csv.gz")):
        sources.append(("previously_unarchived",str(p.relative_to(ROOT)),lambda p=p:crows(p)))
    # Some source paths are aliases; exact ticket dedup accounts for it.
    for typ,path,reader in sources:
        stats=Counter();seen_races=set()
        for row in reader():
            stats["rows"]+=1
            rid=str(row.get("race_id","")).strip()
            if rid not in wanted:
                stats["not_in_2023_results"]+=1;continue
            stats["target_rows"]+=1;seen_races.add(rid)
            key=str(row.get("combination","")).strip()
            k=key.split("-")
            if len(k)!=3 or len(set(k))!=3 or not all(x in "123456789" and len(x)==1 for x in k):
                stats["invalid_combination"]+=1;continue
            if row.get("odds_phase") not in ("final","",None):
                stats["nonfinal"]+=1;continue
            if row.get("odds_status") not in ("available","",None):
                stats["unavailable"]+=1;continue
            if row.get("ticket_type") not in ("3連単","",None):
                stats["invalid_ticket_type"]+=1;continue
            try:price=float(row.get("odds") or "")
            except (ValueError,TypeError):stats["nonnumeric"]+=1;continue
            if not math.isfinite(price) or price<1:
                stats["invalid_price"]+=1;continue
            if key in prices[rid]:
                stats["duplicates"]+=1
                if prices[rid][key]!=price:conflicts[rid]+=1
                continue
            prices[rid][key]=price
            stats["unique_tickets_added"]+=1
        stats["races_matched"]=len(seen_races)
        source_counts[path]={"type":typ,**dict(stats)}
        coverage[f"sources_{typ}"]+=1
        coverage[f"raw_rows_{typ}"]+=stats["rows"]
        print("SOURCE",typ,path,"races",len(seen_races),"unique_added",stats["unique_tickets_added"],flush=True)
    # Also include already-valid 2023 consolidated books; dedup exact tickets
    path=ROOT/"data/odds_only/odds_2023.jsonl.gz"
    consolidated=set();odds_without_raw=set()
    if path.exists():
        with gzip.open(path,"rt",encoding="utf-8") as f:
            for line in f:
                d=json.loads(line);rid=str(d["race_id"])
                if rid not in wanted:continue
                consolidated.add(rid)
                if not prices.get(rid):odds_without_raw.add(rid)
                for ticket,price in d["odds"]["3連単"].items():
                    old=prices[rid].get(ticket)
                    if old is None:prices[rid][ticket]=float(price)
                    elif old!=float(price):conflicts[rid]+=1
    complete=set();partial=set();invalid_grid=set()
    diagnostics={}
    for rid,tickets in prices.items():
        cars=sorted(set(c for ticket in tickets for c in ticket.split("-")),key=int)
        n=len(cars)
        if n<3 or n>9:invalid_grid.add(rid);continue
        expected={"-".join(z) for z in itertools.permutations(cars,3)}
        if len(tickets)==len(expected) and set(tickets)==expected:complete.add(rid)
        else:
            partial.add(rid)
            if len(diagnostics)<200:diagnostics[rid]={"observed":len(tickets),"expected":len(expected),"runners":n}
    empty=sorted(wanted-set(prices))
    report={
        "status":"CENSUS_COMPLETE_2023_ALL_RAW_TRIFECTA_SOURCES",
        "year":2023,
        "verified_result_races":len(wanted),
        "raw_source_files_scanned":len(sources),
        "raw_trifecta_price_any_races":len(prices),
        "complete_trifecta_price_grid_races":len(complete),
        "partial_trifecta_races":len(partial),
        "invalid_ticket_universe_races":len(invalid_grid),
        "results_missing_all_trifecta_prices":len(empty),
        "consolidated_3trifecta_and_trio_books_with_2023_result":len(consolidated),
        "consolidated_only_races_before_fallback":len(odds_without_raw),
        "total_exact_distinct_available_trifecta_ticket_prices":sum(map(len,prices.values())),
        "race_ids_with_conflicting_duplicate_prices":len(conflicts),
        "conflicting_race_examples":list(conflicts.items())[:30],
        "sources":source_counts,
        "source_counters":dict(coverage),
        "method":"Union ALL independently saved original 2023 3連単 prices; exact result race IDs, exact ticket dedup, accepts 3連単 even without 3連複.",
        "important_limit":"No odds were fetched from external websites. The report measures actual GitHub archived 2023 final-odds coverage only, not hypothetical nationwide availability.",
        "no_market_modelling_or_profit_claims":True
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    EXAMPLE.write_text(json.dumps({
        "verified_races_without_any_3trifecta_odds_count":len(empty),
        "first_300_missing_race_ids":empty[:300],
        "first_200_partial_races":diagnostics,
        "first_100_conflicting_race_ids":list(conflicts)[:100]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FINAL_2023_RAW_ODDS_CENSUS",json.dumps({k:v for k,v in report.items()
        if k in ("verified_result_races","raw_source_files_scanned","raw_trifecta_price_any_races",
                 "complete_trifecta_price_grid_races","partial_trifecta_races",
                 "results_missing_all_trifecta_prices","consolidated_3trifecta_and_trio_books_with_2023_result",
                 "total_exact_distinct_available_trifecta_ticket_prices")},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

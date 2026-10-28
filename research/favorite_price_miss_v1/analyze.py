#!/usr/bin/env python3
"""Market favorite trifecta misses by the favorite's final odds.

Uses the complete published odds-only archive and exact race-ID matching
with independently validated finishing orders. No inferred labels.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ODDS = ROOT / "data/odds_only"
RESULTS = ROOT / "data/result_universe_v2/combined"
OUTPUT = ROOT / "data/audits"
EDGES = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, float("inf")]

def band(value):
    for a,b in zip(EDGES[:-1], EDGES[1:]):
        if a <= value < b:
            return (f"{a:g}-" + (f"{b:g}" if math.isfinite(b) else "inf"))
    raise ValueError(value)

def records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in f:
            if row.strip():
                yield json.loads(row)

def load_results():
    out={}
    invalid=Counter()
    for p in sorted(RESULTS.glob("results_20??.jsonl.gz")):
        for rec in records(p):
            rid=str(rec.get("race_id",""))
            finish=tuple(rec.get("finish_order") or ())
            if len(finish)!=3 or len(set(finish))!=3 or any(not isinstance(c,int) or not 1<=c<=9 for c in finish):
                invalid["invalid_finish"]+=1
                continue
            if rid in out and out[rid]!=finish:
                invalid["conflicts"]+=1
                out[rid]=None
            elif rid not in out:
                out[rid]=finish
    return out,dict(invalid)

def market_rows():
    seen=set()
    for p in sorted(ODDS.glob("odds_20??.jsonl.gz")):
        for r in records(p):
            rid=str(r["race_id"])
            if rid in seen:
                raise RuntimeError("Duplicate market race ID "+rid)
            seen.add(rid)
            # Archive partition year is the actual race date year; the race ID encodes meeting start year.
            r["_archive_year"] = int(p.name[5:9])
            yield r

def compute_row(record, actual):
    prices=record["odds"]["3連単"]
    if len(prices)<6:
        raise ValueError("Incomplete odds")
    # Complete market already checked by the odds_only exporter.
    best_combo, best_price=min(prices.items(), key=lambda t:(float(t[1]),t[0]))
    price=float(best_price)
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Invalid odds")
    inv_total=sum(1.0/float(x) for x in prices.values())
    if not math.isfinite(inv_total) or inv_total<=0:
        raise ValueError("Invalid sum of inverse prices")
    cars={int(ch) for c in prices for ch in c.split("-")}
    ordered_key="-".join(map(str,actual))
    if ordered_key not in prices:
        raise ValueError("Actual finish not in odds book")
    n=len(cars)
    return {
      "race_id":record["race_id"],"year":record["_archive_year"],
      "cars":n,"race_type":record["race_type"],"band":band(price),
      "price":price,"favorite":best_combo,"actual":ordered_key,
      "favorite_tied":sum(float(o)==price for o in prices.values())>1,
      "hit":int(best_combo==ordered_key),"market_p":(1/price)/inv_total,
      "overround":inv_total
    }

def wilson_ci(n,k,z=1.95996398454):
    if not n:return [None,None]
    p=k/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [100*(mid-d),100*(mid+d)]

def summarize(rows):
    n=len(rows)
    if not n:return {"races":0}
    hits=sum(r["hit"] for r in rows)
    expected=sum(r["market_p"] for r in rows)
    var=sum(r["market_p"]*(1-r["market_p"]) for r in rows)
    p=hits/n
    e=expected/n
    z=(hits-expected)/math.sqrt(var) if var else 0.0
    # 'excess miss' > 0 means the favorite underperformed the full-market baseline.
    return {
      "races":n, "favorite_hits":hits, "favorite_misses":n-hits,
      "favorite_hit_pct":round(100*p,4),
      "favorite_miss_pct":round(100*(1-p),4),
      "market_expected_hit_pct":round(100*e,4),
      "market_expected_miss_pct":round(100*(1-e),4),
      "excess_miss_pp":round(100*(e-p),4),
      "underperformance_z":round(-z,4),
      "favorite_hit_pct_wilson_95": [round(x,4) for x in wilson_ci(n,hits)],
      "favorite_100yen_stake_yen":100*n,
      "favorite_100yen_payout_yen":round(100*sum(r["price"]*r["hit"] for r in rows)),
      "favorite_roi_pct":round(100*sum(r["price"]*r["hit"] for r in rows)/n,4),
      "favorite_mean_odds":round(sum(r["price"] for r in rows)/n,4),
      "market_mean_overround":round(sum(r["overround"] for r in rows)/n,5)
    }

def aggregate(rows, fields):
    groups=defaultdict(list)
    for r in rows:
        k=tuple(str(r[f]) for f in fields)
        groups[k].append(r)
    return [{"group":dict(zip(fields,k)),**summarize(v)} for k,v in sorted(groups.items())]

def main():
    labels,label_diagnostics=load_results()
    counters=Counter()
    rows=[]
    bad=[]
    for item in market_rows():
        counters["market_total"]+=1
        rid=str(item["race_id"])
        actual=labels.get(rid)
        if actual is None:
            counters["no_verified_result"]+=1
            continue
        try:
            x=compute_row(item,actual)
        except (ValueError,TypeError,KeyError) as e:
            counters["invalid_matched_race"]+=1
            if len(bad)<20:bad.append([rid,str(e)])
            continue
        x["period"]="development_2021_2024" if x["year"]<=2024 else "forward_2025_2026H1"
        x["cars_group"]=str(x["cars"])
        rows.append(x)
    assert len(rows)>30000, f"Unexpectedly low exact-ID join: {len(rows)}"
    counters["joined_valid_races"]=len(rows)
    counters["all_outcomes"]=len(labels)
    tables={
       "all_bands":aggregate(rows,["band"]),
       "development_bands":aggregate([r for r in rows if r["year"]<=2024],["band"]),
       "forward_bands":aggregate([r for r in rows if r["year"]>=2025],["band"]),
       "year_bands":aggregate(rows,["year","band"]),
       "car_count_bands":aggregate(rows,["cars_group","band"]),
       "period_car_bands":aggregate(rows,["period","cars_group","band"]),
       "car_15_25_by_period":aggregate([r for r in rows if 15<=r["price"]<25],["period","cars_group"]),
       "car_15_25_all":aggregate([r for r in rows if 15<=r["price"]<25],["cars_group"]),
       "unique_favorite_bands":aggregate([r for r in rows if not r["favorite_tied"]],["band"]),
       "by_year":aggregate(rows,["year"]),
       "by_car_count":aggregate(rows,["cars_group"]),
    }
    # Preserve numeric bin ordering for use as price chart data.
    order={band((a+b)/2 if math.isfinite(b) else a+1):i for i,(a,b) in enumerate(zip(EDGES[:-1],EDGES[1:]))}
    for t in tables.values():
        if t and "band" in t[0]["group"]:
            t.sort(key=lambda r:(*[str(v) for k,v in r["group"].items() if k!="band"],order[r["group"]["band"]]))
    # Price band beyond random variance: scan as exploratory and require directional repetition.
    d={r["group"]["band"]:r for r in tables["development_bands"]}
    f={r["group"]["band"]:r for r in tables["forward_bands"]}
    candidates=[]
    for b in d:
        x,y=d[b],f.get(b)
        if not y or min(x["races"],y["races"])<100:continue
        if x["excess_miss_pp"]>0 and y["excess_miss_pp"]>0:
            candidates.append({"band":b,
                "development_races":x["races"],"development_excess_miss_pp":x["excess_miss_pp"],
                "forward_races":y["races"],"forward_excess_miss_pp":y["excess_miss_pp"],
                "forward_z":y["underperformance_z"]})
    candidates.sort(key=lambda t:(-t["forward_z"],-t["forward_excess_miss_pp"]))
    report={
       "status":"COMPLETE",
       "generated_at_utc":datetime.now(timezone.utc).isoformat(),
       "source_odds":"data/odds_only/odds_2021.jsonl.gz ... odds_2026.jsonl.gz",
       "source_results":"data/result_universe_v2/combined/results_2021.jsonl.gz ... results_2026.jsonl.gz",
       "scope":"Men's keirin, completely priced trifecta and trio, years 2021-2025 plus 2026H1; final odds (NOT actionable T-minus odds)",
       "favorite_definition":"minimum finalized 3連単 decimal odds within the complete race market, lexical tie-break",
       "hit_definition":"exact ordered winning first-second-third matches the favorite ticket",
       "market_probability":"(1/favorite_odds)/sum(1/all_trifecta_odds) ; expected miss = 1-market_probability",
       "meaning":"excess_miss_pp > 0 implies favorites missed more than full-market normalized odds predicted. Mere high miss percent is not evidence of mispricing.",
       "limitations":["Only matched fully priced markets with genuine unambiguous results. Missing races excluded, selection bias possible.",
            "Final odds are not guaranteed available at bet placement time.",
            "Grouped-bin significance exploratory; many price bands, no multiplicity correction.",
            "Check forward 2025-2026H1 rather than just in-sample 2021-2024. No individual ticket or formation profit proven."],
       "counters":dict(counters),"favorite_tie_races":sum(int(r["favorite_tied"]) for r in rows),
       "result_input_diagnostics":label_diagnostics,
       "invalid_examples":bad,"total":summarize(rows),"price_band_edges":EDGES[:-1]+["inf"],
       "repeated_excess_miss_candidates_exploratory":candidates,
       "tables":tables
    }
    OUTPUT.mkdir(exist_ok=True,parents=True)
    p=OUTPUT/"favorite_trifecta_price_bands_2021_2026h1.json"
    p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUTPUT/"favorite_trifecta_price_bands_2021_2026h1.csv").open("w",encoding="utf-8",newline="") as out:
        writer=csv.writer(out)
        writer.writerow(["period","band","races","hits","miss_pct","market_expected_miss_pct","excess_miss_pp","underperformance_z","favorite_roi_pct"])
        for section in ["all_bands","development_bands","forward_bands"]:
            for v in tables[section]:
                writer.writerow([section,v["group"]["band"],v["races"],v["favorite_hits"],v["favorite_miss_pct"],
                    v["market_expected_miss_pct"],v["excess_miss_pp"],v["underperformance_z"],v["favorite_roi_pct"]])
    print("FAVORITE_TRIFECTA_PRICE_BANDS",json.dumps({
        "matched":len(rows),"source_markets":counters["market_total"],"unmatched":counters["no_verified_result"],
        "total":report["total"],"candidates":candidates[:10],"all_bands":tables["all_bands"]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Low-priced 3連単 top favorite: exact 0.1x/0.5x odds bands and misses.

Read-only re-analysis of validated odds/result archives 2021-2026H1.
Does not modify user predictive systems or source data.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

from analyze import ROOT, load_results, market_rows, compute_row, summarize

DEST = ROOT / "data/audits/favorite_trifecta_low_odds_fine_v1"
COHORTS=("all","development_2021_2024","forward_2025_2026H1")
CAR_GROUPS=("all","7","9")
def dec(x): return Decimal(str(x))
def fmt(x): return f"{x:g}"
def by_cohort(rows, name): return rows if name=="all" else [r for r in rows if r["period"]==name]

def extra(rows):
    if not rows:
        return {"races":0}
    v=summarize(rows)
    n=v["races"]
    # Standard deviation under individual race-specific normalized market probabilities.
    var=sum(r["market_p"]*(1-r["market_p"]) for r in rows)
    se=100*math.sqrt(var)/n
    gap=v["excess_miss_pp"]
    v["excess_miss_se_pp"]=round(se,4)
    v["excess_miss_ci95_pp"]=[round(gap-1.96*se,4),round(gap+1.96*se,4)]
    v["avg_full_market_implied_hit_pct"]=v["market_expected_hit_pct"]
    return v

def main():
    labels,_=load_results()
    counts=Counter()
    buckets=defaultdict(list)
    rows=[]
    for record in market_rows():
        counts["total_odds"]+=1
        actual=labels.get(str(record["race_id"]))
        if actual is None:
            counts["no_verified_results"]+=1
            continue
        try: r=compute_row(record,actual)
        except (KeyError,ValueError,TypeError):
            counts["bad_records"]+=1
            continue
        counts["matched"]+=1
        r["period"]=COHORTS[1] if r["year"]<=2024 else COHORTS[2]
        r["bin_tenth"]=int((dec(r["price"])*10).to_integral_value(rounding=ROUND_FLOOR))
        r["bin_half"]=int((dec(r["price"])*2).to_integral_value(rounding=ROUND_FLOOR))
        rows.append(r)
        for group in ("all",str(r["cars"])):
            for period in ("all",r["period"]):
                buckets[(group,period,"tenth",r["bin_tenth"])].append(r)
                buckets[(group,period,"half",r["bin_half"])].append(r)
    assert len(rows)>40000, counts
    # Return each exact decimal 0.1x band from 1.0 to <10.0,
    # and each half point from 1.0 to <12.0. The 0.1x bands for 1.0-5.0 are primary.
    tenth,half=[],[]
    for group in CAR_GROUPS:
        for period in COHORTS:
            for idx in range(10,101):
                records=buckets.get((group,period,"tenth",idx),[])
                tenth.append({"cars":group,"period":period,
                    "band":f"{idx/10:.1f}-{(idx+1)/10:.1f}","lower":idx/10,"upper":(idx+1)/10,
                    **extra(records)})
            for idx in range(2,25):
                records=buckets.get((group,period,"half",idx),[])
                half.append({"cars":group,"period":period,
                    "band":f"{idx/2:.1f}-{(idx+1)/2:.1f}","lower":idx/2,"upper":(idx+1)/2,
                    **extra(records)})
    # Cumulative favorite thresholds; visible raw risk vs market-implied risk.
    cumulative=[]
    for group in CAR_GROUPS:
        group_rows=[r for r in rows if group=="all" or str(r["cars"])==group]
        for period in COHORTS:
            rr=by_cohort(group_rows,period)
            for threshold_tenth in range(15,101,5):
                hi=Decimal(threshold_tenth)/10
                sel=[r for r in rr if dec(r["price"])<hi]
                cumulative.append({"cars":group,"period":period,"market_favorite_odds_below":float(hi),**extra(sel)})
    # Unit bins one to twelve, all with time cohorts.
    units=[]
    for group in CAR_GROUPS:
        for period in COHORTS:
            for low in range(1,13):
                subset=[]
                for k in range(low*10,(low+1)*10):
                    subset.extend(buckets.get((group,period,"tenth",k),[]))
                units.append({"cars":group,"period":period,"lower":low,"upper":low+1,**extra(subset)})
    # Tie sensitivity for cheap odds 1 to 5; no special selection of favorite ties.
    unique=[]
    for group in CAR_GROUPS:
        gr=[r for r in rows if (group=="all" or str(r["cars"])==group) and not r["favorite_tied"]]
        for period in COHORTS:
            rr=by_cohort(gr,period)
            for low in range(1,5):
                sub=[r for r in rr if low<=r["price"]<low+1]
                unique.append({"cars":group,"period":period,"lower":low,"upper":low+1,**extra(sub)})
    # Yearly cheap band for trend stability.
    yearly=[]
    for year in range(2021,2027):
        for group in ("all","7","9"):
            rr=[r for r in rows if r["year"]==year and (group=="all" or str(r["cars"])==group)]
            for lo in (1,2,3,4,5):
                sub=[r for r in rr if lo<=r["price"]<lo+1]
                yearly.append({"year":year,"cars":group,"lower":lo,"upper":lo+1,**extra(sub)})
    document={
        "status":"complete",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "study":"Low-odds-range miss of single lowest-odds (1番人気) 3連単, all matching real races 2021-2026H1",
        "units":"final decimal odds; all bins [lower,upper); exact decimal 0.1 via Decimal to avoid float imprecision",
        "null_hypothesis":"normalized whole-race trifecta market 1/odds divided by sum of all 1/odds; excess_miss_pp=observed_miss_minus_market_expected_miss",
        "note":"negative excess_miss_pp means favorite won MORE often than normalized entire market predicted; this does not imply positive ROI. 9-car low-odds bins can be sparse. Odds final, not guaranteed purchasable pre-close. Exploratory multiple bins not corrected. Overlapping accumulations not independent.",
        "sources":"data/odds_only/odds_20??.jsonl.gz and data/result_universe_v2/combined/results_20??.jsonl.gz",
        "counts":dict(counts),
        "low_one_tenth":tenth,
        "half_point":half,
        "unit_bands":units,
        "all_odds_below_threshold":cumulative,
        "untied_favorite_unit_bands":unique,
        "yearly_low_unit_bands":yearly
    }
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/"details.json").write_text(json.dumps(document,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
    with (DEST/"low_odds_bands.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["cars","period","lower_inclusive","upper_exclusive","races","hits","misses","actual_miss_pct","expected_miss_pct","excess_miss_pp","underperformance_z","roi_pct"])
        for v in tenth:
            writer.writerow([v["cars"],v["period"],v["lower"],v["upper"],
                v["races"],v.get("favorite_hits",0),v.get("favorite_misses",0),
                v.get("favorite_miss_pct",""),v.get("market_expected_miss_pct",""),
                v.get("excess_miss_pp",""),v.get("underperformance_z",""),v.get("favorite_roi_pct","")])
    print("LOW_ODDS_FINE_AUDIT",json.dumps({
       "matched":counts["matched"],
       "unit_all":[v for v in units if v["cars"]=="all" and v["period"]=="all" and v["lower"]<=8],
       "half_all":[v for v in half if v["cars"]=="all" and v["period"]=="all" and v["lower"]<5],
       "unit_seven":[v for v in units if v["cars"]=="7" and v["period"]=="all" and v["lower"]<=5]
    },ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()

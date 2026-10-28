#!/usr/bin/env python3
"""Fine-grained LOWER bound audit of 3連単 minimum-odds favorite.

Non-invasive follow-up to analyze.py. Exact race-ID match; same data, same odds,
no model modifications. Only historical settled ("final") odds are analyzed.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze import ROOT, load_results, market_rows, compute_row, summarize

OUT = ROOT / "data/audits/favorite_trifecta_lower_bound_v1"
OUT.mkdir(parents=True,exist_ok=True)
YEARS = ("development_2021_2024", "forward_2025_2026H1")
LOWERS = [i / 2 for i in range(16, 43)]    # 8.0 .. 21.0 half-step
HALF_STEPS = [i / 2 for i in range(20, 41)]  # 10.0 .. 20.0 half-step
ONE_STEPS = list(range(7, 26))
FIXED_UPPERS = [20.0, 25.0]
SCOPE = ["2021", "2022", "2023", "2024", "2025", "2026"]

def fmt(row):
    if not row:
        return {"n": 0}
    s = summarize(row)
    n = len(row)
    variance=sum(r["market_p"]*(1-r["market_p"]) for r in row)
    se=100*math.sqrt(variance)/n
    x=s["excess_miss_pp"]
    s["excess_miss_ci95_pp"]=[round(x-1.96*se,4), round(x+1.96*se,4)]
    s["excess_miss_se_pp"]=round(se,4)
    s["n"]=n
    return s

def pack(rows, predicate, label):
    out=[]
    for cohort in ("all",*YEARS):
        subset=[r for r in rows if (cohort=="all" or r["period"]==cohort) and predicate(r)]
        out.append({"period":cohort,"window":label,**fmt(subset)})
    return out

def main():
    labels, diag=load_results()
    rows=[]
    counts=Counter()
    for record in market_rows():
        counts["market_races"]+=1
        actual=labels.get(record["race_id"])
        if actual is None:
            counts["missing_verified_result"]+=1
            continue
        try:
            row=compute_row(record,actual)
        except (KeyError,ValueError,TypeError):
            counts["invalid"]+=1
            continue
        row["period"]=YEARS[0] if row["year"]<=2024 else YEARS[1]
        rows.append(row)
    counts["matched"]=len(rows)
    assert len(rows)>=40000, counts
    seven=[r for r in rows if r["cars"]==7]
    nine=[r for r in rows if r["cars"]==9]
    # Half-point distribution; avoids hiding the onset within a broad threshold.
    half_bins=[]
    for lo in HALF_STEPS:
        hi=lo+0.5
        half_bins.extend(pack(seven,lambda r,lo=lo,hi=hi: lo<=r["price"]<hi,f"{lo:g}-{hi:g}"))
    # 1x bins: higher precision than original 5x spread, adequate sample sizes.
    unit_bins=[]
    for lo in ONE_STEPS:
        hi=lo+1
        unit_bins.extend(pack(seven,lambda r,lo=lo,hi=hi:lo<=r["price"]<hi,f"{lo:g}-{hi:g}"))
    # Fixed-end windows: varied lower limit only, the exact requested sensitivity.
    fixed_upper={}
    for up in FIXED_UPPERS:
        fixed_upper[f"{up:g}"]=[]
        for lo in LOWERS:
            if lo>=up:
                continue
            fixed_upper[f"{up:g}"].extend(pack(seven,lambda r,lo=lo,up=up:lo<=r["price"]<up,f"{lo:g}-{up:g}"))
    # Parallel placebo/negative-control: nine-car race at exactly same bounds.
    nine_control={}
    for lo in (10,11,12,13,14,15,16,17,18,19,20):
        nine_control[str(lo)]=pack(nine,lambda r,lo=lo: lo<=r["price"]<25,f"{lo:g}-25")
    # Evaluate only long windows with sufficient development sample and forward sample.
    candidates=[]
    for upper,table in fixed_upper.items():
        ds={v["window"]:v for v in table if v["period"]==YEARS[0]}
        fs={v["window"]:v for v in table if v["period"]==YEARS[1]}
        for key,dev in ds.items():
            forward=fs[key]
            if dev["n"]<500 or forward["n"]<300:continue
            candidates.append({
                "window":key, "development_n":dev["n"],"forward_n":forward["n"],
                "development_excess_miss_pp":dev["excess_miss_pp"],
                "forward_excess_miss_pp":forward["excess_miss_pp"],
                "development_z":dev["underperformance_z"],
                "forward_z":forward["underperformance_z"],
                "forward_ci95_pp":forward["excess_miss_ci95_pp"],
                "forward_roi_pct":forward["favorite_roi_pct"],
                "same_direction":dev["excess_miss_pp"]>0 and forward["excess_miss_pp"]>0
            })
    # Report, do not imply optimizing a cutoff on test period is out-of-sample.
    candidates.sort(key=lambda x:(-x["development_z"],x["window"]))
    years=[{"year":year,**fmt([r for r in seven if str(r["year"])==year and 15<=r["price"]<25])} for year in SCOPE]
    unique_seven=[r for r in seven if not r["favorite_tied"]]
    unique_check=[]
    for lo in (10,11,12,13,14,15,16,17,18,19,20):
        unique_check.extend(pack(unique_seven,lambda r,lo=lo:lo<=r["price"]<25,f"{lo:g}-25"))
    output={
        "status":"complete",
        "date_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"7車立て男子競輪・3連単1番人気の確定オッズ下限探索、9車比較対照、2021-2026上期",
        "base_source":"data/odds_only/odds_2021..2026.jsonl.gz, data/result_universe_v2/combined/results_2021..2026.jsonl.gz",
        "definition":"1番人気=当該レースの三連単最小オッズ、同率なら組番昇順。オッズの範囲は下限を含み上限を含まない",
        "baseline":"normalized full-market implied probability (1/odds)/SUM(1/all-trifecta-odds), not inverse of favorite alone",
        "warning":"These are final odds, no pre-bet availability. Multiple tested cutoffs are exploratory; later cohort was previously examined at coarse 15-25 bounds, so it is NOT a pristine untouched confirmatory holdout. Overlapping cumulative windows are highly correlated. Do not declare significance from selecting the maximum z.",
        "counts":dict(counts),
        "seven_races":len(seven),"nine_races":len(nine),
        "seven_favorite_ties":sum(r["favorite_tied"] for r in seven),
        "seven_15_25_by_year":years,
        "half_point_7car":half_bins,
        "one_point_7car":unit_bins,
        "fixed_upper_7car":fixed_upper,
        "nine_car_fixed_upper_25":nine_control,
        "no_tied_favorite_7car_upper25":unique_check,
        "exploratory_cutoff_ranking_by_development_only":candidates,
    }
    (OUT/"details.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUT/"cutoff_table.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["scope","upper","window","period","races","miss_pct","expected_miss_pct","excess_miss_pp","excess_miss_ci95_low","excess_miss_ci95_high","z","roi_pct"])
        for upper,values in fixed_upper.items():
            for v in values:
                w.writerow(["7car",upper,v["window"],v["period"],v["n"],v["favorite_miss_pct"],v["market_expected_miss_pct"],v["excess_miss_pp"],*v["excess_miss_ci95_pp"],v["underperformance_z"],v["favorite_roi_pct"]])
        for lower,vals in nine_control.items():
            for v in vals:
                w.writerow(["9car","25",v["window"],v["period"],v["n"],v["favorite_miss_pct"],v["market_expected_miss_pct"],v["excess_miss_pp"],*v["excess_miss_ci95_pp"],v["underperformance_z"],v["favorite_roi_pct"]])
    print("LOWER_BOUND_AUDIT",json.dumps({"matched":counts["matched"],"seven":len(seven),
          "top_dev":candidates[:8],
          "windows_25": [v for v in fixed_upper["25"] if v["period"]=="all" and v["window"].split("-")[0] in ("10","11","12","13","14","15","16","17","18","19","20")],
          "year_15_25":years},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

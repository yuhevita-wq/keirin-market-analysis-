#!/usr/bin/env python3
"""Search real 1-4-point ordered wager formations from unconditional market residuals.

No field / favorite-odds-band conditioning; all valid 2021-2026H1 race books.
2021-2023 discover, pre-fixed mask 2024 and 2025-26 evaluate.
Any valid formation is <=4 3連単 tickets at every 5-9-car field, with favorite
exact wager removed by decomposition, not by cherry-picking the race.
No TRAIN ROI is used for choosing candidates: select on surplus realized
frequency over normalized market-implied expectation, repeated over years.
"""
from __future__ import annotations
import itertools,json,math
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
from research.unconditional_formation_residuals_20261028 import discover as u
OUT=Path(__file__).resolve().parent
SIZES=((1,1,1),(2,1,1),(2,2,1),(2,2,2),(3,1,1),(3,2,1),
       (3,3,1),(4,1,1),(4,2,1))
POS=range(7)
YEARS=u.YEARS
TRAIN=u.TRAIN
EXPOSURE_MIN=10000
EXP_MIN=25
EXP_YEAR_MIN=2.5
KEEP=20

def load():
    labels=u.load_labels()
    seen=set()
    for path in sorted(u.ODDS.glob("odds_20??.jsonl.gz")):
        yr=int(path.name[5:9])
        for row in u.raw_lines(path):
            rid=str(row.get("race_id",""))
            if rid in seen:raise RuntimeError("Duplicate market race id")
            seen.add(rid)
            u.AUDIT["quoted_races"]+=1
            winner=labels.get(rid)
            if winner is None:
                u.AUDIT["unmatched_result"]+=1
                continue
            u.observe(rid,yr,row,winner)
    if u.AUDIT["accepted_races"]<45000:raise RuntimeError("Full market data missing")

def yearly_arrays():
    agg={}
    for y in YEARS:
        a={k:np.zeros((9,9,9),dtype=np.float64) for k in ("actual","expected","payout","purchase_points")}
        for n in u.CARS:
            r=u.DATA.get((y,n,"ALL"))
            if not r:continue
            a["actual"]+=r["actual"]
            a["expected"]+=r["expected"]-r["fav_expected"]
            a["payout"]+=r["payout"]
            a["purchase_points"]+=r["race_count"]*u.VALID[n].astype(float)-r["fav_frequency"]
        agg[y]={k:v.reshape(-1) for k,v in a.items()}
    return agg

def possible_masks():
    masks_by_k={k:[(sum(1<<x for x in p),p) for p in itertools.combinations(POS,k)] for k in range(1,5)}
    seen=set()
    counted=0
    for sizes in SIZES:
        for arrangement in sorted(set(itertools.permutations(sizes))):
            for triple in itertools.product(*(masks_by_k[k] for k in arrangement)):
                raw_mask=tuple(x[0] for x in triple)
                xs=[x[1] for x in triple]
                # Real valid 3 distinct riders; no fake ticket with repeated rider.
                cells=tuple(sorted((a*81+b*9+c for a,b,c in itertools.product(*xs)
                                    if len({a,b,c})==3)))
                if not (1<=len(cells)<=4):continue
                counted+=1
                if cells in seen:continue
                seen.add(cells)
                yield raw_mask, cells
    print("COMPLETE_CANDIDATE_SETS",len(seen),"RAW_CANDIDATES",counted,flush=True)

def scoring(mask,cells,agg):
    ar={}
    total_actual=total_expected=0
    for y in TRAIN:
        a=agg[y]
        actual=float(a["actual"][list(cells)].sum())
        exp=float(a["expected"][list(cells)].sum())
        exposure=float(a["purchase_points"][list(cells)].sum())
        ar[y]=(actual,exp,exposure)
        total_actual+=actual
        total_expected+=exp
    if total_expected<EXP_MIN or ar[2021][1]<EXP_YEAR_MIN or ar[2022][1]<EXP_YEAR_MIN or ar[2023][1]<EXP_YEAR_MIN:
        return None
    if sum(v[2] for v in ar.values())<EXPOSURE_MIN:
        return None
    z=[(v[0]-v[1])/math.sqrt(v[1]) for v in ar.values()]
    # Minimum year is primary: does not reward one fluke year.
    pooled=(total_actual-total_expected)/math.sqrt(total_expected)
    return min(z)+0.2*pooled,ar

def main():
    load()
    agg=yearly_arrays()
    ranked=[]
    eligible=0
    for mask,cells in possible_masks():
        scored=scoring(mask,cells,agg)
        if scored is None:continue
        eligible+=1
        score,yrs=scored
        if min(v[0]-v[1] for v in yrs.values())<=0:continue
        ranked.append((score,mask,cells))
    ranked.sort(key=lambda x:(-x[0],x[2]))
    reports=[]
    for i,(score,m,cells) in enumerate(ranked[:KEEP],1):
        reports.append({
            "rank":i,"train_signal_score":round(score,5),
            "market_rank_masks_only_internal":list(m),
            "max_valid_points_any_field":len(cells),
            "training":u.accumulate(m,TRAIN),
            "training_by_year":{str(y):u.accumulate(m,(y,)) for y in TRAIN},
            "validation_2024":u.accumulate(m,(2024,)),
            "forward_2025_2026H1":u.accumulate(m,u.FORWARD),
            "forward_by_year":{str(y):u.accumulate(m,(y,)) for y in u.FORWARD},
            "numbered_real_formations":u.real_examples(m),
            "favorite_band_diagnostic_only":{b:{
                "train":u.accumulate(m,TRAIN,b),
                "validation":u.accumulate(m,(2024,),b),
                "forward":u.accumulate(m,u.FORWARD,b)} for b in u.BIN_NAMES}
        })
    report={
        "status":"COMPLETE_SMALL_REAL_FORMATIONS",
        "utc":datetime.now(timezone.utc).isoformat(),
        "data_join":dict(u.AUDIT),
        "scope":"All 45,521 valid 2021-2026H1 5-9 car races, favorite odds not used for discovery.",
        "discovery":"2021-23 unconditional frequency minus normalized implied full-market price; no training ROI in score",
        "eligibility":"Every actual 3連単 Cartesian formation has <=4 genuine distinct-car tickets at every n; remove exact favorite by partitioning.",
        "train_min_all_form_stake_yen":100*EXPOSURE_MIN,
        "market_expected_hits_min":EXP_MIN,
        "training_year_market_expected_min":EXP_YEAR_MIN,
        "candidate_formation_shapes":"Unrestricted actual rank subsets from first 7 common ranks; 1-4 valid tickets, positional mask shapes 1x1x1 through 4x2x1. This is an exhaustive search of these constrained shapes, not all possible masks.",
        "qualified_candidates":eligible,
        "training_each_year_positive_residual":len(ranked),
        "forms":reports,
        "limitations":[
            "Within rank 1-7, limited mask shapes; NOT all possible zero-to-four-point subsets or multi-formation unions.",
            "Discovered within previously analyzed 2021-23, held-out chronology not pristine.",
            "Historical final odds not available to ensure stake at race -15min.",
            "All-form ROI can stay below 100 despite occurrence frequency surplus.",
            "Compared with thousands of discovered candidates, multiple comparison risk."
        ]
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"compact.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("RESULTS",json.dumps({
        "universe":u.AUDIT["accepted_races"],
        "qualified":eligible,"positive_train_in_each_year":len(ranked),
        "top":[{"rank":r["rank"],"max_pts":r["max_valid_points_any_field"],
                 "dev_hit":r["training"]["hits"],"dev_expect":r["training"]["market_expected_hits"],
                 "dev_roi":r["training"]["roi_pct"],"val_roi":r["validation_2024"]["roi_pct"],
                 "forward_hit":r["forward_2025_2026H1"]["hits"],
                 "forward_exp":r["forward_2025_2026H1"]["market_expected_hits"],
                 "forward_roi":r["forward_2025_2026H1"]["roi_pct"],
                 "example":r["numbered_real_formations"][0]["formations_union"]
                 if r["numbered_real_formations"] else None}
               for r in reports[:10]]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()

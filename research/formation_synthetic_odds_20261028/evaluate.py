"""Composite price vs actual payouts for TWO complete trifecta formations.

Built as an entirely separate research experiment. No production strategy edits.
Reads existing verified sealed 4-pool odds + outcomes and applies identical
formation constructor from the completed prior experiment.
"""
from __future__ import annotations
import json,math,os,sys
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
from research.picking_winners_true_formations_20261028.observe import create_base,tickets

OUT=ROOT/"research/formation_synthetic_odds_20261028"
AUDIT=Counter()
# Before outcome inspection: all grids fixed. Thresholds denote LOWER bound.
ODDS_CUTS=(0.,1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,8.,10.,15.,20.,30.,50.,100.,float("inf"))
ENTRY_THRESHOLDS=(3.,3.5,4.,5.,6.,8.,10.,12.,15.,20.,30.,50.)
POINT_CAPS=(6,8,10,12,16,20,24,32,48,64,128,1000)
POINT_BINS=(0,4,6,8,10,12,16,20,24,32,48,64,128,float("inf"))
METHODS=("base_only","change_1","change_2","change_3","change_12","change_13","change_23","change_123")
METRICS=("races","points","stake","payout","hits","dutch_virtual_return_units","dutch_market_implied_hits")
CUBE=defaultdict(lambda:np.zeros(7,dtype=np.float64))
SUMMARY=defaultdict(lambda:np.zeros(7,dtype=np.float64))
GRID=defaultdict(lambda:np.zeros(7,dtype=np.float64))
SAMPLES=defaultdict(list)

def rec(target,key,pts,hit,odds,synthetic,market_expected):
    a=target[key]
    a[0]+=1
    a[1]+=pts
    a[2]+=100*pts
    a[3]+=100*odds*hit
    a[4]+=hit
    # Ideal fractional dutching stakes: B/O_i / sum(1/O_i).  B=100 yen total.
    # This is NOT a valid 100-yen-increment executable betting scheme.
    a[5]+=synthetic*hit
    a[6]+=market_expected

def bucket(x,cuts):
    # [cut[i],cut[i+1]) except final +infty bin.
    return next((i for i in range(len(cuts)-1) if cuts[i]<=x<cuts[i+1]),len(cuts)-2)

def observe(rid,day,books,finish,rtype,source):
    AUDIT["callbacks"]+=1
    if source!="sealed_four_markets_orientation_verified":
        raise RuntimeError("Odds orientation is not independently verified")
    book=books["3rentan"]
    cars=tuple(sorted({c for p in book for c in p}))
    n=len(cars)
    outcome=tuple(finish)
    if outcome not in book or len(book)!=n*(n-1)*(n-2):
        AUDIT["invalid_market_or_result"]+=1;return
    if any(not math.isfinite(float(o)) or float(o)<=1 for o in book.values()):
        AUDIT["bad_price"]+=1;return
    base,ranks,ks=create_base(cars,book)
    base_t=tickets(base)
    if len(base_t)<4:
        raise AssertionError("Not a multi-point formation")
    v=[("base_only",base,base_t)]
    for mask in ("1","2","3","12","13","23","123"):
        alt=list(base);valid=True
        for ch in mask:
            pos=int(ch)-1
            shift=tuple(cars[int(i)] for i in np.roll(ranks[pos],-1)[:ks[pos]])
            if set(shift)==set(base[pos]):
                valid=False;break
            alt[pos]=shift
        if not valid:
            AUDIT["unchangeable_"+mask]+=1;continue
        alt=tuple(alt);new=tickets(alt)
        if len(new)<4 or new==base_t:
            AUDIT["invalid_second_"+mask]+=1;continue
        v.append(("change_"+mask,alt,new))
    final_odds=float(book[outcome])
    year=day[:4]
    full_market_sum=sum(1/float(x) for x in book.values())
    for policy,alt,alt_t in v:
        union=base_t|alt_t
        pts=len(union)
        inv_price_mass=sum(1/float(book[x]) for x in union)
        if inv_price_mass<=0:raise AssertionError("Zero synthetic inverse mass")
        syn=1/inv_price_mass
        hit=int(outcome in union)
        market_mass=inv_price_mass/full_market_sum
        rec(SUMMARY,(policy,year),pts,hit,final_odds,syn,market_mass)
        rec(CUBE,(policy,year,bucket(pts,POINT_BINS),bucket(syn,ODDS_CUTS)),
            pts,hit,final_odds,syn,market_mass)
        if syn>=ENTRY_THRESHOLDS[0]:
            for cap in POINT_CAPS:
                if pts>cap:continue
                for threshold in ENTRY_THRESHOLDS:
                    if syn<threshold:break
                    rec(GRID,(policy,year,cap,threshold),pts,hit,final_odds,syn,market_mass)
        if len(SAMPLES[policy])<8 and year in ("2025","2026") and pts<=16 and syn>=3:
            SAMPLES[policy].append({
                "race_id":rid,"day":day,"cars":n,
                "base":[list(s) for s in base],"second":[list(s) for s in alt] if policy!="base_only" else None,
                "base_points":len(base_t),"second_points":len(alt_t) if policy!="base_only" else None,
                "shared_exact_points":len(base_t&alt_t) if policy!="base_only" else 0,
                "unique_points":pts,"synthetic_odds":round(syn,5),
                "finish":list(outcome),"hit":bool(hit),
                "stake_100_each":100*pts,"payout_100_each":100*final_odds*hit
            })
        AUDIT["policy_events_"+policy]+=1
    AUDIT["races_analyzed"]+=1
    AUDIT["year_"+year]+=1

def fmt(a):
    races=int(a[0]);amount=float(a[2]);payout=float(a[3])
    return {"races":races,"total_points":int(a[1]),
        "mean_points":float(a[1]/a[0]) if races else None,
        "stake_yen":round(amount),"payout_yen":round(payout),
        "profit_yen":round(payout-amount),
        "roi":payout/amount if amount else None,
        "hits":int(a[4]),
        "hit_rate":float(a[4]/a[0]) if races else None,
        "fractional_dutch_ideal_roi":float(a[5]/a[0]) if races else None,
        "market_expected_hits":float(a[6]) if races else None}

def totals_for(dct,policy,years,cap,thr):
    v=np.zeros(7)
    for y in years:v+=dct.get((policy,str(y),cap,thr),np.zeros(7))
    return v

def choose(training_years,test_year):
    minimum=200
    choices=[]
    for pol in METHODS:
        for cap in POINT_CAPS:
            for threshold in ENTRY_THRESHOLDS:
                total=totals_for(GRID,pol,training_years,cap,threshold)
                if total[0]<minimum:continue
                # minimum independent calendar-year exposure. Stop one odd year from dominating.
                if any(GRID.get((pol,str(y),cap,threshold),np.zeros(7))[0]<25 for y in training_years):
                    continue
                b=fmt(total)
                choices.append((b["roi"],pol,cap,threshold,b))
    choices.sort(key=lambda x:(-x[0],x[2],x[3],METHODS.index(x[1])))
    top=[{"method":x[1],"point_cap":x[2],"minimum_synthetic_odds":x[3],
          "train":x[4],"test":fmt(totals_for(GRID,x[1],[test_year],x[2],x[3]))}
         for x in choices[:8]]
    if not choices:return {"training_years":training_years,"test_year":test_year,"status":"NO_CANDIDATE_AT_MINIMUM_EXPOSURE"}
    # No wager is a proper decision when even best training strategy is negative.
    winner=choices[0]
    approved=bool(winner[0]>1.0)
    return {
        "training_years":training_years,"test_year":test_year,
        "eligible_candidate_count":len(choices),
        "minimum_training_exposure_races":minimum,
        "minimum_per_training_year":25,
        "best_training_candidate":{
            "method":winner[1],"point_cap":winner[2],"minimum_synthetic_odds":winner[3],
            "training":winner[4],"test_even_if_rejected":fmt(totals_for(GRID,winner[1],[test_year],winner[2],winner[3])),
            "was_authorized_by_training_profit":approved},
        "actual_rule": "BUY" if approved else "SKIP",
        "actual_test":fmt(totals_for(GRID,winner[1],[test_year],winner[2],winner[3])) if approved else {"races":0,"stake_yen":0,"payout_yen":0,"profit_yen":0,"roi":None},
        "top_eight_training_candidates":top,
        "controls":"Parametric grid fixed in code; all candidate training ROC selection retrospective. No test outcome used to pick winner."
    }

def main():
    if not os.environ.get(corrected.old.bridge.SECRET_ENV):
        raise RuntimeError("Missing sealed archive secret: do not switch sources")
    corrected.base_observe=observe
    corrected.old.save=lambda state:print("CHECKPOINT",state,dict(AUDIT),flush=True)
    corrected.old.sealed()
    if AUDIT["races_analyzed"]<100000:raise RuntimeError("Unexpectedly small archive coverage")
    price_by_policy=[]
    for pol in METHODS:
        for pb in range(len(POINT_BINS)-1):
            for ob in range(len(ODDS_CUTS)-1):
                v=np.zeros(7)
                for y in range(2021,2027):v+=CUBE.get((pol,str(y),pb,ob),np.zeros(7))
                if v[0]:
                    price_by_policy.append({"method":pol,"point_range":[POINT_BINS[pb],POINT_BINS[pb+1]],
                      "synthetic_range":[ODDS_CUTS[ob],ODDS_CUTS[ob+1]],**fmt(v)})
    overall=[]
    for pol in METHODS:
        v=np.zeros(7)
        for y in range(2021,2027):v+=SUMMARY.get((pol,str(y)),np.zeros(7))
        overall.append({"method":pol,**fmt(v)})
    threshold_grid=[{"method":pol,"year":year,"point_cap":cap,"minimum_synthetic_odds":thr,**fmt(v)}
        for (pol,year,cap,thr),v in sorted(GRID.items())]
    years=[{"method":pol,"year":str(y),**fmt(SUMMARY.get((pol,str(y)),np.zeros(7)))}
        for pol in METHODS for y in range(2021,2027)]
    report={
       "status":"COMPOSITE_ODDS_FORMATION_ROI_FULL_ARCHIVE_COMPLETE",
       "generated_utc":datetime.now(timezone.utc).isoformat(),
       "source_commit":os.environ.get("GITHUB_SHA","unknown"),
       "audit":dict(AUDIT),
       "definition":{
           "formation":"Cartesian 1st x 2nd x 3rd excluding repeated riders; two sets may overlap, unique 3連単 combinations purchased once",
           "synthetic_odds":"1 / sum(1 / final_odds_of_each_unique_ticket)",
           "synthetic_price_interpretation":"Common decimal payout multiple only under EXACT fractional dutching; this experiment's actual primary ROI buys EACH unique ticket for 100 yen, not equal-payout stakes",
           "true_evaluation":"100 yen x |union| stakes, 100 yen x outcome odds if exact order is in union; no partial-match score",
           "baseline":"Unmodified earlier market-marginal top-k base, plus seven original fixed shift/hold masks; all sizes generated by old algorithm. This does not prove global formation optimality.",
           "threshold_policy":"Only bet on historical final-odds synthetic_odds >= threshold AND total unique points <= point cap, no post-result conditions",
           "price_analysis_bins":ODDS_CUTS,"point_bins":POINT_BINS,
           "entry_minimum_odds":ENTRY_THRESHOLDS,"point_caps":POINT_CAPS,
           "train_selections":"Most profitable historical ROI among preregistered grid with at least 200 training entries and 25/year. Skip if selected training ROI <=100%. Evaluate next year untouched by THIS selection."},
       "overall":overall,"years":years,
       "price_point_cells":price_by_policy,
       "threshold_grid":threshold_grid,
       "chronological_selections":[choose([2021,2022,2023,2024],2025),
                                   choose([2021,2022,2023,2024,2025],2026)],
       "examples":dict(SAMPLES),
       "limits":[
          "All prices are historical FINAL odds, not T-15 observable odds; profitable segment does NOT establish live tradability.",
          "Past research has already examined these historical years. Even test years are not a genuinely untouched dataset.",
          "No betting decisions use outcomes; but group selection from the same historical outcomes is susceptible to multiple-testing and post-hoc bias.",
          "Market-only formation construction cannot by itself create an independent predictive edge; test is for observed conditional mispricing.",
          "The minimum threshold 3.0 is a tested entry limit; bins below 3.0 included only for descriptive controls.",
          "No risk-adjusted capital allocation or executable integer-yen dutching is validated.",
          "A strong theoretical synthetic odds multiplier is not the actual payout multiple for equal stakes on unequal-odds combinations.",
          "No model, purchased ticket strategy, or production code updated."
       ]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    slim={k:v for k,v in report.items() if k not in ("price_point_cells","threshold_grid","examples","years")}
    (OUT/"compact.json").write_text(json.dumps(slim,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DONE",json.dumps({"audit":dict(AUDIT),"selections":report["chronological_selections"],
      "overall":overall},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

"""Results-first discovery of repeatably mispriced DECISION SETS as 3-slot formations.

Research protocol is locked in this file BEFORE its workflow is first launched.
Uses 2021-23 only to discover positional-rank Cartesian sets. Reports 2024,
2025 and 2026H1 unchanged. BOTH directions (over/under market) and EVERY
budget/car-classification are reported without selecting a convenient result.

No individual winning tickets are selected or evaluated in isolation. Exact
3rentan hits/payoffs are measured at the complete formation level.
"""
from __future__ import annotations
import json,math,os,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected

OUT=ROOT/"research/result_first_subset_scan_20261028"
DISCOVERY=("2021","2022","2023")
VALIDATION=("2024",)
HOLDOUT=("2025","2026")
CARS=(5,6,7,8,9)
CAPS=(8,16,32,None)  # predeclared; None = no imposed points ceiling
DIRECTIONS=("underpriced","overpriced") # symmetric positive and negative controls
MIN_DISCOVERY_MARKET_EXPECTED=20.0
RNG_SEED=20261028
STARTS=64
SWEEPS=6
AUDIT=Counter()
MATS={}
SAMPLES=defaultdict(list)

def arr(year,n):
    k=(year,n)
    if k not in MATS:
        MATS[k]={"races":0,"expected":np.zeros((n,n,n),np.float64),
            "observed":np.zeros((n,n,n),np.float64),
            "ticket_count":np.zeros((n,n,n),np.float64),
            "payout_odds":np.zeros((n,n,n),np.float64)}
    return MATS[k]

def observe(rid,day,books,finish,rtype,source):
    AUDIT["callbacks"]+=1
    if source!="sealed_four_markets_orientation_verified":
        raise RuntimeError("Refuse unverified ordered trifecta odds")
    odds=books.get("3rentan",{})
    cars=tuple(sorted({v for t in odds for v in t}))
    n=len(cars)
    if n not in CARS or len(odds)!=n*(n-1)*(n-2) or tuple(finish) not in odds:
        AUDIT["excluded_invalid_odds_or_result"]+=1
        return
    q={t:1/float(o) for t,o in odds.items()}
    if any(not math.isfinite(v) or v<=0 for v in q.values()):
        AUDIT["excluded_invalid_price"]+=1;return
    normalizer=sum(q.values())
    marginals=[Counter(),Counter(),Counter()]
    for t,v in q.items():
        for i in range(3):marginals[i][t[i]]+=v
    # Rank is a pre-event deterministic label derived from *odds*, not outcome.
    car_rank=[sorted(cars,key=lambda c:(-marginals[i][c],c)) for i in range(3)]
    rank_index=[{car:i for i,car in enumerate(car_rank[k])} for k in range(3)]
    m=arr(day[:4],n)
    m["races"]+=1
    target=tuple(finish)
    for (a,b,c),inv in q.items():
        pos=(rank_index[0][a],rank_index[1][b],rank_index[2][c])
        m["expected"][pos]+=inv/normalizer
        m["ticket_count"][pos]+=1
        if (a,b,c)==target:
            m["observed"][pos]+=1
            m["payout_odds"][pos]+=float(odds[(a,b,c)])
    if day[:4] in VALIDATION+HOLDOUT and len(SAMPLES[(day[:4],n)])<3:
        SAMPLES[(day[:4],n)].append({
            "rid":rid,"day":day,"ranked_cars":car_rank,
            "finish":list(target),"winning_odds":float(odds[target])
        })
    AUDIT["usable"]+=1
    AUDIT["year_"+day[:4]]+=1
    AUDIT["cars_"+str(n)]+=1

def combine(years,n):
    out={"races":0}
    for key in ("expected","observed","ticket_count","payout_odds"):
        out[key]=np.zeros((n,n,n),np.float64)
    for year in years:
        z=MATS.get((year,n))
        if z is None:continue
        out["races"]+=z["races"]
        for k in ("expected","observed","ticket_count","payout_odds"):
            out[k]+=z[k]
    return out

def bits(n):
    codes=np.arange(1,2**n,dtype=np.int32)
    mat=((codes[:,None]>>np.arange(n))&1).astype(np.float64)
    return codes,mat,mat.sum(axis=1)

def pattern_score(obs,expected,direction):
    if expected<MIN_DISCOVERY_MARKET_EXPECTED or not math.isfinite(obs+expected):
        return float("-inf")
    if direction=="underpriced" and obs<=expected:return float("-inf")
    if direction=="overpriced" and obs>=expected:return float("-inf")
    # One-sided Poisson log likelihood ratio, discovery only, not a p-value.
    return float(obs*math.log(max(obs,1e-250)/expected)-(obs-expected))

def axes_sum(t,a,b,c):
    return float(np.einsum("ijk,i,j,k->",t,a,b,c,optimize=True))

def find_pattern(train,n,cap,direction):
    codes,mat,sizes=bits(n)
    obs_cube=train["observed"]; exp_cube=train["expected"]
    rng=np.random.default_rng(RNG_SEED+n*100+(cap or 9999)+(0 if direction=="underpriced" else 5000))
    starts=[]
    full=2**n-1
    starts.extend([(full,full,full),(1,1,1)])
    for sz in (2,3,4,n):
        s=(1<<sz)-1
        starts.append((s,s,s))
        starts.extend([(s,full,1),(full,s,1),(1,s,full),(1,full,s)])
    for i in range(n):
        starts.extend([(1<<i,full,1),(full,1<<i,1),(1,full,1<<i)])
    while len(starts)<STARTS:
        starts.append(tuple(int(rng.integers(1,2**n)) for _ in range(3)))
    best=None
    for init in starts:
        masks=list(init)
        for _ in range(SWEEPS):
            changed=False
            for axis in (0,1,2):
                vectors=[mat[masks[j]-1] for j in range(3)]
                if axis==0:
                    ov=np.einsum("ijk,j,k->i",obs_cube,vectors[1],vectors[2],optimize=True)
                    ev=np.einsum("ijk,j,k->i",exp_cube,vectors[1],vectors[2],optimize=True)
                elif axis==1:
                    ov=np.einsum("ijk,i,k->j",obs_cube,vectors[0],vectors[2],optimize=True)
                    ev=np.einsum("ijk,i,k->j",exp_cube,vectors[0],vectors[2],optimize=True)
                else:
                    ov=np.einsum("ijk,i,j->k",obs_cube,vectors[0],vectors[1],optimize=True)
                    ev=np.einsum("ijk,i,j->k",exp_cube,vectors[0],vectors[1],optimize=True)
                o=mat@ov;e=mat@ev
                selected_count=sizes
                for fixed_axis in range(3):
                    if fixed_axis!=axis:selected_count=selected_count*sizes[masks[fixed_axis]-1]
                valid=(e>=MIN_DISCOVERY_MARKET_EXPECTED)
                if cap is not None:valid&=(selected_count<=cap)
                valid &= (o>e) if direction=="underpriced" else (o<e)
                if not valid.any():continue
                ix=np.where(valid)[0]
                # Safe: positive observed for underpricing; negative may have zero observed.
                llr=o[ix]*np.log(np.maximum(o[ix],1e-250)/e[ix])-(o[ix]-e[ix])
                win=ix[int(np.argmax(llr))]
                nextmask=int(codes[win])
                if masks[axis]!=nextmask:
                    masks[axis]=nextmask;changed=True
            if not changed:break
        vectors=[mat[x-1] for x in masks]
        actual_o=axes_sum(obs_cube,*vectors)
        market_e=axes_sum(exp_cube,*vectors)
        score=pattern_score(actual_o,market_e,direction)
        raw_points=int(np.prod([int(sizes[x-1]) for x in masks]))
        if cap is not None and raw_points>cap:continue
        if not math.isfinite(score):continue
        cand=(score,tuple(masks),actual_o,market_e,raw_points)
        if best is None or cand[0]>best[0]+1e-9 or (abs(cand[0]-best[0])<=1e-9 and cand[1]<best[1]):
            best=cand
    if best is None:return None
    return {"score":best[0],"masks":list(best[1]),"discovery_hits":best[2],
        "discovery_expected_hits":best[3],"raw_cartesian_upper_bound":best[4]}

def assess(mats,pattern,n):
    if pattern is None:return {"races":mats["races"],"status":"NO_DISCOVERED_PATTERN"}
    _,bits_mat,_=bits(n)
    vectors=[bits_mat[int(x)-1] for x in pattern["masks"]]
    obs=axes_sum(mats["observed"],*vectors)
    exp=axes_sum(mats["expected"],*vectors)
    points=axes_sum(mats["ticket_count"],*vectors)
    payoff=axes_sum(mats["payout_odds"],*vectors)
    N=mats["races"]
    return {"races":N,"hits":round(obs),"market_expected_hits":exp,
      "observed_div_market_expected":obs/exp if exp>0 else None,
      "mean_points":points/N if N else None,
      "total_points":round(points),"stake_yen":round(points*100),
      "payout_yen":round(payoff*100),
      "profit_yen":round((payoff-points)*100),
      "roi":payoff/points if points>0 else None,
      "actual_hit_rate":obs/N if N else None,
      "market_implied_hit_rate":exp/N if N else None}

def mk_example(rec,pattern):
    if pattern is None:return {}
    n=len(rec["ranked_cars"][0])
    slots=[]
    for axis,mask in enumerate(pattern["masks"]):
        slots.append([rec["ranked_cars"][axis][pos] for pos in range(n) if mask>>pos&1])
    combos={(a,b,c) for a in slots[0] for b in slots[1] for c in slots[2] if len({a,b,c})==3}
    hit=tuple(rec["finish"]) in combos
    return {"race_id":rec["rid"],"date":rec["day"],"formation":slots,
       "actual_points":len(combos),"finish":rec["finish"],
       "hit":hit,"stake_yen":100*len(combos),
       "payout_yen":round(100*rec["winning_odds"]*hit)}

def main():
    if not os.environ.get(corrected.old.bridge.SECRET_ENV):
        raise RuntimeError("Sealed archival private key missing; no silent reduced dataset")
    corrected.base_observe=observe
    corrected.old.save=lambda status:print("CHECKPOINT",status,dict(AUDIT),flush=True)
    corrected.old.sealed()
    if AUDIT["usable"]<130000:raise RuntimeError("Full archive insufficient")
    records=[]
    for n in CARS:
        train=combine(DISCOVERY,n)
        for cap in CAPS:
            for direction in DIRECTIONS:
                pattern=find_pattern(train,n,cap,direction)
                report={"cars":n,"max_raw_cartesian_points":cap,"direction":direction,
                  "selection_basis":"ONE SIDED observed/market Poisson LLR on discovery 2021-23 ONLY",
                  "pattern":pattern,
                  "year_results":{y:assess(combine([y],n),pattern,n) for y in YEAR_RANGE},
                  "discovery":assess(train,pattern,n),
                  "validation":assess(combine(VALIDATION,n),pattern,n),
                  "holdout_2025":assess(combine(["2025"],n),pattern,n),
                  "holdout_2026H1":assess(combine(["2026"],n),pattern,n),
                  "holdout_combined":assess(combine(HOLDOUT,n),pattern,n),
                  "examples":[mk_example(s,pattern) for y in ("2025","2026") for s in SAMPLES.get((y,n),[])][:6]}
                # This buy-or-skip decision does NOT change masks / select another pattern.
                if pattern is None:
                    report["decision_from_validation"]="NO_DISCOVERED_PATTERN"
                else:
                    v=report["validation"]
                    report["decision_from_validation"]="PASS_GATE" if (
                        v["races"]>=100 and v["roi"] is not None
                        and v["roi"]>1.0 and
                        v["observed_div_market_expected"] is not None
                        and v["observed_div_market_expected"]>1.0 and
                        direction=="underpriced") else "SKIP"
                records.append(report)
    warnings=[
       "No favorite positive result is chosen after observing validation/holdout; all patterns and control directions appear.",
       "The scan is HEURISTIC coordinate ascent, not exhaustive global subset search; rank-based positioning, not all free rider-subsets.",
       "Reference market probability is normalized reciprocal final 3連単 odds, not ground truth.",
       "Discovery 2021-23 / validation 2024 / holdout 2025,2026H1. These later years were used previously in unrelated project experiments, not a virgin unseen dataset.",
       "Training scans thousands of correlated subsets: any training deviation has substantial multiple-testing/overfitting risk.",
       "A candidate exceeding market implied frequency can still lose money due to the market takeout and nonlinear ticket odds.",
       "All ticket prices are FINAL, not confirmed executable 15 minutes before cutoff.",
       "Formation point caps limit the raw Cartesian product; actual valid combination count may be smaller due to repeated rider removals.",
       "2026 only through June. Small 5/8-car groups may be statistically noisy.",
       "The only direct payoff is winning exact 3連単 result, with each valid ticket bought once at 100 yen.",
       "No existing model, purchase rule, or production website was changed."
    ]
    report={"status":"RESULT_FIRST_FORMATION_SUBSET_SCAN_COMPLETE",
      "timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "source_commit":os.getenv("GITHUB_SHA","local"),
      "audit":dict(AUDIT),
      "locked_protocol":{"discovery_years":DISCOVERY,"validation_years":VALIDATION,"holdout_years":HOLDOUT,
        "car_counts":CARS,"max_cartesian_point_budgets":CAPS,
        "opposite_direction_control":True,
        "minimum_training_market_expected_hits":MIN_DISCOVERY_MARKET_EXPECTED,
        "seed":RNG_SEED,"random_starts":STARTS,"coordinate_sweeps":SWEEPS,
        "scan_statistic":"One-sided Poisson deviance from actual result counts vs normalized market-implied outcome counts across whole CARTESIAN formation.",
        "purchase":"100 yen each eligible actual ordered 3連単 ticket; count/pay full-set ROI, not individual ticket EV.",
        "no_posthoc_rule":"All size/budget/direction cells published, no selecting holdout best"},
      "all_discovered_sets":records,"limitations":warnings}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    compact={k:v for k,v in report.items() if k!="all_discovered_sets"}
    compact["selected_results"]=[{k:v for k,v in r.items() if k!="year_results" and k!="examples"} for r in records]
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
    print("COMPLETE",json.dumps({
       "status":report["status"],"audit":dict(AUDIT),
       "results":[{"cars":r["cars"],"cap":r["max_raw_cartesian_points"],
         "direction":r["direction"],"gate":r["decision_from_validation"],
         "train_roi":r["discovery"]["roi"],"val_roi":r["validation"]["roi"],
         "holdout_roi":r["holdout_combined"]["roi"],
         "holdout_races":r["holdout_combined"]["races"]} for r in records]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()

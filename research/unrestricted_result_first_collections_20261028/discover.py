#!/usr/bin/env python3
"""Results first: form actual nonfavorite ordered-trifecta SETS without preset templates.

All ordered prices and actual finish are used. 2021-2023 train ONLY.
2024 checks the frozen masks; 2025-2026H1 replay them without retraining.
Every candidate is a true Cartesian 1着×2着×3着 formation, with the exact
market favorite excluded by skipping the ENTIRE race if it is within the set.
No existing buy/prediction model is imported or changed.
"""
from __future__ import annotations
import gzip,itertools,json,math,random
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ODDS=ROOT/"data/odds_only"
RESULTS=ROOT/"data/result_universe_v2/combined"
OUT=ROOT/"research/unrestricted_result_first_collections_20261028"
YEARS=(2021,2022,2023,2024,2025,2026)
TRAIN=(2021,2022,2023)
FORWARD=(2025,2026)
FIELD=(5,6,7,8,9)
PRICE_EDGES=(0,3,5,8,12,15,20,25,float("inf"))
MIN_RACES=60
MIN_EXPECTED=8
SEARCH_SEED=20261028
STEPS=4
RESTARTS=12
DATA={}
RACE_COUNTS=Counter()
AUDIT=Counter()
EXAMPLES=defaultdict(list)

def rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if line.strip(): yield json.loads(line)

def result_labels():
    labels={}
    conflicting=set()
    for path in sorted(RESULTS.glob("results_20??.jsonl.gz")):
        for item in rows(path):
            rid=str(item.get("race_id",""))
            finish=tuple(item.get("finish_order") or ())
            if len(finish)!=3 or len(set(finish))!=3 or any(not isinstance(c,int) or c<1 or c>9 for c in finish):
                AUDIT["invalid_finish"]+=1
                continue
            if rid in labels and labels[rid]!=finish: conflicting.add(rid)
            else: labels[rid]=finish
    for rid in conflicting: labels.pop(rid,None)
    AUDIT["verified_result_labels"]=len(labels)
    AUDIT["result_conflicts"]=len(conflicting)
    return labels

def band(odd):
    return next(f"{a:g}-{b:g}" if math.isfinite(b) else f"{a:g}+"
        for a,b in zip(PRICE_EDGES[:-1],PRICE_EDGES[1:]) if a<=odd<b)

def new_bucket(year,price_band,n):
    key=(year,price_band,n)
    if key not in DATA:
        triple=list(itertools.permutations(range(n),3))
        index={v:i for i,v in enumerate(triple)}
        size=len(triple)
        DATA[key]={
            "triple":triple,"index":index,
            "n":np.zeros(size,dtype=np.int64),
            "inverse_book":np.zeros(size),
            "market":np.zeros((size,size)),
            "wins":np.zeros((size,size)),
            "pay":np.zeros((size,size)),
            "biggest_single_payout":np.zeros((size,size))
        }
    return DATA[key]

def observe(rid,year,row,actual):
    quoted=row.get("odds",{}).get("3連単") or {}
    try: prices={tuple(int(v) for v in t.split("-")):float(o) for t,o in quoted.items()}
    except (ValueError,TypeError):
        AUDIT["malformed_quotes"]+=1
        return
    cars=sorted({c for t in prices for c in t})
    n=len(cars)
    if n not in FIELD:
        AUDIT["excluded_other_car_counts"]+=1
        return
    if len(prices)!=n*(n-1)*(n-2) or tuple(actual) not in prices:
        AUDIT["invalid_complete_book_or_result"]+=1
        return
    if any(not math.isfinite(v) or v<=0 for v in prices.values()):
        AUDIT["invalid_price"]+=1
        return
    fav=min(prices,key=lambda t:(prices[t],t))
    b=band(prices[fav])
    invsum=sum(1/v for v in prices.values())
    support=Counter()
    for t,v in prices.items():
        for car in t: support[car]+=1/v
    ordered=sorted(cars,key=lambda c:(-support[c],c))
    rank={car:i for i,car in enumerate(ordered)}
    cube=new_bucket(year,b,n)
    index=cube["index"]
    favorite_index=index[tuple(rank[c] for c in fav)]
    win_index=index[tuple(rank[c] for c in actual)]
    q=np.zeros(len(index))
    for t,v in prices.items():
        q[index[tuple(rank[c] for c in t)]]=1/v/invsum
    cube["n"][favorite_index]+=1
    cube["inverse_book"][favorite_index]+=1/invsum
    cube["market"][favorite_index]+=q
    cube["wins"][favorite_index,win_index]+=1
    cube["pay"][favorite_index,win_index]+=prices[tuple(actual)]
    cube["biggest_single_payout"][favorite_index,win_index]=max(cube["biggest_single_payout"][favorite_index,win_index],100*prices[tuple(actual)])
    RACE_COUNTS[(year,b,n)]+=1
    AUDIT["usable_races"]+=1
    if year in FORWARD and len(EXAMPLES[(year,b,n)])<15:
        EXAMPLES[(year,b,n)].append({
            "id":rid,"year":year,"ordered":ordered,"fav":fav,
            "fav_odds":prices[fav],"result":tuple(actual),"prices":prices
        })

def merge(years,b,n):
    out=None
    for year in years:
        x=DATA.get((year,b,n))
        if x is None: continue
        if out is None:
            out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in x.items()}
        else:
            for k in ("n","inverse_book","market","wins","pay"):out[k]+=x[k]
            out["biggest_single_payout"]=np.maximum(out["biggest_single_payout"],x["biggest_single_payout"])
    return out

def selected(cube,mask):
    a,b,c=mask
    return np.array([i for i,(x,y,z) in enumerate(cube["triple"])
        if (a>>x)&1 and (b>>y)&1 and (c>>z)&1],dtype=np.int32)

def evaluate(cube,mask):
    if cube is None: return None
    points=selected(cube,mask)
    if not len(points):return None
    available=np.ones(len(cube["triple"]),dtype=bool)
    # If an actual race's favorite is included, buy NOTHING in that race.
    available[points]=False
    entered=int(cube["n"][available].sum())
    if not entered:return None
    wager=100*entered*len(points)
    ix=np.ix_(available,points)
    hits=float(cube["wins"][ix].sum())
    implied=float(cube["market"][ix].sum())
    paid=100*float(cube["pay"][ix].sum())
    largest=float(cube["biggest_single_payout"][ix].max()) if paid else 0
    return {
        "races":entered,"points":len(points),"hits":int(hits),
        "market_expected_hits":round(implied,6),
        "observed_over_market":round(hits/implied,5) if implied else None,
        "stake_yen":wager,"payout_yen":round(paid),
        "profit_yen":round(paid-wager),
        "roi_pct":round(100*paid/wager,4),
        "largest_single_paid_ticket_yen":round(largest),
        "largest_hit_share_of_payout_pct":round(100*largest/paid,4) if paid else None,
        "market_reference_roi_pct":round(100*cube["inverse_book"][available].sum()/entered,4)
    }

def training_quality(mask,train_cubes):
    a=[evaluate(c,mask) for c in train_cubes]
    if any(v is None for v in a):return -1e10
    if sum(x["races"] for x in a)<MIN_RACES: return -1e10
    if sum(x["market_expected_hits"] for x in a)<MIN_EXPECTED:return -1e10
    if all(x["hits"]==0 for x in a):return -1e10
    r=[x["roi_pct"]/100 for x in a]
    e=[x["observed_over_market"] or 0 for x in a]
    return min(r)+0.12*min(e)+0.01*sum(r)/len(r)

def search(price_band,n):
    cubes=[DATA.get((year,price_band,n)) for year in TRAIN]
    if any(c is None for c in cubes):return [],{"status":"missing_train_period"}
    if sum(int(c["n"].sum()) for c in cubes)<MIN_RACES:
        return [],{"status":"insufficient_train_races"}
    rng=random.Random(SEARCH_SEED+n+sum(ord(c) for c in price_band))
    singles=list(itertools.permutations(range(n),3))
    rng.shuffle(singles)
    starts=[tuple(1<<x for x in s) for s in singles[:RESTARTS]]
    full=(1<<n)-1
    starts.append((full,full,full))
    for _ in range(RESTARTS):
        starts.append(tuple(rng.randint(1,full) for _ in range(3)))
    cache={}
    def quality(mask):
        if mask not in cache:cache[mask]=training_quality(mask,cubes)
        return cache[mask]
    converged={}
    for start in starts:
        current=start
        value=quality(current)
        for _ in range(STEPS):
            best=current;best_value=value
            # Arbitrary rider subset; change 1 bit in ANY of three slots.
            for position in range(3):
                for i in range(n):
                    t=list(current);t[position]^=1<<i
                    if t[position]==0:continue
                    cand=tuple(t)
                    v=quality(cand)
                    if v>best_value+1e-10 or (abs(v-best_value)<1e-10 and cand<best):
                        best=cand;best_value=v
            if best==current:break
            current=best;value=best_value
        if value>-1e9:converged[current]=value
    winners=sorted(converged,key=lambda m:(-converged[m],m))[:8]
    return winners,{"status":"completed",
        "candidate_masks_examined":len(cache),
        "discovered_distinct_local_structures":len(converged),
        "optimizer_is_heuristic":True}

def concrete(mask,ex):
    ordered=ex["ordered"]
    places=[[ordered[i] for i in range(len(ordered)) if (v>>i)&1] for v in mask]
    t=sorted(x for x in itertools.product(*places) if len(set(x))==3)
    if ex["fav"] in t:raise AssertionError("Favorite included in purchase")
    pay=100*ex["prices"][ex["result"]] if ex["result"] in t else 0
    return {
        "race_id":ex["id"],
        "example_selection":"first eligible archive race in forward period, NOT picked for result or payout",
        "formation":"-".join("".join(map(str,slot)) for slot in places),
        "1着":places[0],"2着":places[1],"3着":places[2],
        "valid_points":len(t),"favorite":"-".join(map(str,ex["fav"])),
        "favorite_odds":ex["fav_odds"],
        "ticket_odds":[{"numbers":"-".join(map(str,x)),
                       "odds":ex["prices"][x],"stake_yen":100} for x in t],
        "result":"-".join(map(str,ex["result"])),
        "total_stake_yen":100*len(t),
        "payout_yen":round(pay),"profit_yen":round(pay-100*len(t))
    }

def example(mask,price_band,n):
    for year in FORWARD:
        for ex in EXAMPLES[(year,price_band,n)]:
            rank={c:i for i,c in enumerate(ex["ordered"])}
            if all((mask[j]>>rank[ex["fav"][j]])&1 for j in range(3)):
                continue
            return concrete(mask,ex)
    return None

def research_report():
    groups=[]
    atoms=[]
    for a,b in zip(PRICE_EDGES[:-1],PRICE_EDGES[1:]):
        price_band=f"{a:g}-{b:g}" if math.isfinite(b) else f"{a:g}+"
        for n in FIELD:
            masks,search_audit=search(price_band,n)
            candidates=[]
            for mask in masks:
                candidates.append({
                    "mask_for_reproduction":list(mask),
                    "development_2021_2023":{str(y):evaluate(DATA.get((y,price_band,n)),mask) for y in TRAIN},
                    "development_combined":evaluate(merge(TRAIN,price_band,n),mask),
                    "validation_2024":evaluate(DATA.get((2024,price_band,n)),mask),
                    "forward_2025_2026H1":evaluate(merge(FORWARD,price_band,n),mask),
                    "forward_by_year":{str(y):evaluate(DATA.get((y,price_band,n)),mask) for y in FORWARD},
                    "concrete_actual_car_formation":example(mask,price_band,n)
                })
            groups.append({"band":price_band,"cars":n,
                           "search":search_audit,"frozen_training_ranked_candidates":candidates})
            for tri in itertools.permutations(range(n),3):
                mask=tuple(1<<i for i in tri)
                atoms.append({
                    "band":price_band,"cars":n,"internal_atom_rank_zero_based":list(tri),
                    "train":evaluate(merge(TRAIN,price_band,n),mask),
                    "validation_2024":evaluate(DATA.get((2024,price_band,n)),mask),
                    "forward":evaluate(merge(FORWARD,price_band,n),mask)
                })
    audit={
        "status":"RESEARCH_COMPLETE_RESULTS_FIRST",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"7/9 car exact-result matched complete 3連単 archive 2021-2025, 2026H1",
        "audit":dict(AUDIT),
        "races_by_year_band_field":{f"{y}:{b}:{n}":int(c) for (y,b,n),c in sorted(RACE_COUNTS.items())},
        "method":{
            "no_predefined_line_cross_or_buy_form":True,
            "actual_finish_plus_all_price_quotes":True,
            "candidate":"Cartesian actual car subsets of arbitrary size in each finish position; no point cap",
            "rank_is_only_intermediate":"car ordering by total support across all ordered market prices; never present market-rank codes as purchasable formation",
            "favorite_exclusion":"skip WHOLE race if proposed complete formation contains exact most popular trifecta",
            "chronology":"2021-23 discover, frozen compare 2024, replay 2025 and 2026H1",
            "point_stake_yen":100,
            "discovery":"deterministic multistart bit-flip hill climb, NOT exhaustive global optimization",
            "starting_conditions":"rank-triplet single-bit seeds chosen at random plus random three-slot subsets and all-cars subset",
            "train_objective":"worst year real whole-formation ROI + weakest year's observed/market residual with minimum support",
            "min_train_market_expected_hits":MIN_EXPECTED,
            "min_train_entered_races":MIN_RACES,
            "historical_prices":"FINAL, not T-minus-15 odds"
        },
        "groups":groups,
        "all_actual_vs_market_atoms":atoms,
        "limitations":[
            "Research only, not a purchased system or proven market advantage.",
            "Train best may overfit; hundreds of correlated candidate masks explored.",
            "2024 onward not used to select patterns, but previously seen in other analyses.",
            "No independent pre-close executable-price audit.",
            "All 5-9 field counts included; sparse field/odds cohorts can return no discovered pattern.",
            "One Cartesian formation only; multi-component union research later."
        ]
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    compact={k:v for k,v in audit.items() if k not in ("all_actual_vs_market_atoms","groups")}
    compact["groups"]=[
        {"band":g["band"],"cars":g["cars"],"search":g["search"],
         "top3":[{"development":r["development_combined"],
                  "validation":r["validation_2024"],"forward":r["forward_2025_2026H1"],
                  "train_years":r["development_2021_2023"],
                  "forward_years":r["forward_by_year"],
                  "formation":r["concrete_actual_car_formation"]}
                 for r in g["frozen_training_ranked_candidates"][:3]]}
        for g in groups]
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
    print("RESEARCH_COMPLETED",json.dumps({"audit":dict(AUDIT),
          "groups":[{"band":g["band"],"cars":g["cars"],
                 "candidates":len(g["frozen_training_ranked_candidates"]),
                 "best_train_roi":g["frozen_training_ranked_candidates"][0]["development_combined"]["roi_pct"]
                     if g["frozen_training_ranked_candidates"] else None,
                 "val_roi":g["frozen_training_ranked_candidates"][0]["validation_2024"]["roi_pct"]
                     if g["frozen_training_ranked_candidates"] and g["frozen_training_ranked_candidates"][0]["validation_2024"] else None,
                 "forward_roi":g["frozen_training_ranked_candidates"][0]["forward_2025_2026H1"]["roi_pct"]
                     if g["frozen_training_ranked_candidates"] and g["frozen_training_ranked_candidates"][0]["forward_2025_2026H1"] else None
                 } for g in groups]},ensure_ascii=False),flush=True)

def main():
    labels=result_labels()
    seen=set()
    for p in sorted(ODDS.glob("odds_20??.jsonl.gz")):
        year=int(p.name[5:9])
        for r in rows(p):
            rid=str(r.get("race_id",""))
            if rid in seen:raise RuntimeError("Duplicate race ID, stop rather than double count")
            seen.add(rid)
            AUDIT["all_quoted_races"]+=1
            actual=labels.get(rid)
            if actual is None:
                AUDIT["unmatched_result"]+=1
                continue
            observe(rid,year,r,actual)
    if AUDIT["usable_races"]<40000:
        raise RuntimeError("Full archival dataset unavailable; refuse tiny proxy")
    research_report()

if __name__=="__main__":main()

#!/usr/bin/env python3
"""Unconditional results-first 3rentan decision-SET discovery.

Stage 1: 2021-2026H1, every valid 5-9 car race in one unconditional population.
Stage 2: discover all position rank-triple residuals; optimize arbitrary
         nonempty 1st x 2nd x 3rd rider subsets using 2021-2023 ONLY.
Stage 3: translate frozen subsets into actual numbered formations, splitting
         around the forbidden most-popular EXACT 3rentan ticket where needed.
Stage 4: evaluate whole-set cost/payoff on fixed 2024, 2025, 2026H1.
Stage 5: only AFTER selection, split the same frozen shapes by favorite price.
Never select candidates using held-out periods or favorite price groups.
No existing predictor, formation, purchase logic, or website is modified.
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
OUT=ROOT/"research/unconditional_formation_residuals_20261028"
YEARS=(2021,2022,2023,2024,2025,2026)
TRAIN=(2021,2022,2023)
FORWARD=(2025,2026)
CARS=(5,6,7,8,9)
EDGES=(0,3,5,8,12,15,20,25,float("inf"))
BIN_NAMES=("0-3","3-5","5-8","8-12","12-15","15-20","20-25","25+")
D=9
RANK_TRIPLES=list(itertools.permutations(range(D),3))
VALID={n:np.zeros((D,D,D),dtype=bool) for n in CARS}
for n in CARS:
    for a,b,c in itertools.permutations(range(n),3):
        VALID[n][a,b,c]=True
DATA={}
EXAMPLES=defaultdict(list)
COUNT=Counter()
AUDIT=Counter()
SEARCH_SEED=20261028
SEARCH_STARTS=80
SEARCH_SWEEPS=6
MIN_TRAIN_EXPECTED_HITS=90
MIN_YEAR_EXPECTED_HITS=8
MIN_TRAIN_BET_RACES=5000

def raw_lines(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def load_labels():
    labels={};conflict=set()
    for p in sorted(RESULTS.glob("results_20??.jsonl.gz")):
        for r in raw_lines(p):
            rid=str(r.get("race_id",""))
            x=tuple(r.get("finish_order") or ())
            if len(x)!=3 or len(set(x))!=3 or any(type(c)!=int or c<1 or c>9 for c in x):
                AUDIT["invalid_label"]+=1;continue
            if rid in labels and labels[rid]!=x:conflict.add(rid)
            else:labels[rid]=x
    for k in conflict:labels.pop(k,None)
    AUDIT["label_conflicts"]=len(conflict)
    AUDIT["valid_labels"]=len(labels)
    return labels

def market_band(v):
    return next(label for label,lo,hi in zip(BIN_NAMES,EDGES[:-1],EDGES[1:]) if lo<=v<hi)

def new_cube():
    return {"actual":np.zeros((D,D,D),dtype=np.float64),
        "expected":np.zeros((D,D,D),dtype=np.float64),
        "payout":np.zeros((D,D,D),dtype=np.float64),
        "top_hit":np.zeros((D,D,D),dtype=np.float64),
        "fav_frequency":np.zeros((D,D,D),dtype=np.float64),
        "fav_expected":np.zeros((D,D,D),dtype=np.float64),
        "fav_inv_overround":np.zeros((D,D,D),dtype=np.float64),
        "race_count":0,"inv_overround_sum":0.0}

def cube(key):
    if key not in DATA:DATA[key]=new_cube()
    return DATA[key]

def observe(rid,year,row,finish):
    prices_str=(row.get("odds") or {}).get("3連単") or {}
    try:
        p={tuple(map(int,k.split("-"))):float(v) for k,v in prices_str.items()}
    except (TypeError,ValueError):
        AUDIT["bad_price_parse"]+=1;return
    numbers=sorted({c for t in p for c in t})
    n=len(numbers)
    if n not in CARS or len(p)!=n*(n-1)*(n-2) or tuple(finish) not in p:
        AUDIT["incomplete_book_or_result"]+=1;return
    if not all(math.isfinite(v) and v>=1 for v in p.values()):
        AUDIT["bad_prices"]+=1;return
    favorite=min(p,key=lambda t:(p[t],t))
    price_band=market_band(p[favorite])
    inv={t:1/v for t,v in p.items()}
    z=sum(inv.values())
    support=Counter()
    for t,v in inv.items():
        for car in t:support[car]+=v
    order=sorted(numbers,key=lambda c:(-support[c],c))
    ranks={c:i for i,c in enumerate(order)}
    fav_rank=tuple(ranks[c] for c in favorite)
    result_rank=tuple(ranks[c] for c in finish)
    vals=np.zeros((D,D,D),dtype=np.float64)
    for t,v in inv.items():vals[tuple(ranks[c] for c in t)]=v/z
    for group in ("ALL",price_band):
        a=cube((year,n,group))
        a["race_count"]+=1
        a["inv_overround_sum"]+=1/z
        a["expected"]+=vals
        a["fav_frequency"][fav_rank]+=1
        a["fav_expected"][fav_rank]+=inv[favorite]/z
        a["fav_inv_overround"][fav_rank]+=1/z
        if tuple(finish)!=favorite:
            a["actual"][result_rank]+=1
            a["payout"][result_rank]+=100*p[tuple(finish)]
            a["top_hit"][result_rank]=max(a["top_hit"][result_rank],100*p[tuple(finish)])
    AUDIT["accepted_races"]+=1
    COUNT[(year,n)]+=1
    if year>=2025 and len(EXAMPLES[(year,n)])<20:
        EXAMPLES[(year,n)].append({
            "rid":rid,"year":year,"n":n,"order":order,
            "favorite":favorite,"favorite_odds":p[favorite],
            "finish":tuple(finish),"book":p})

def mask_array(mask):
    a,b,c=mask
    aa=np.array([(a>>i)&1 for i in range(D)],dtype=bool)
    bb=np.array([(b>>i)&1 for i in range(D)],dtype=bool)
    cc=np.array([(c>>i)&1 for i in range(D)],dtype=bool)
    return aa[:,None,None]&bb[None,:,None]&cc[None,None,:]

def accumulate(mask,years,group="ALL"):
    select=mask_array(mask)
    entries=points=hit=expected=payout=baseline=0.
    biggest=0.
    by_n={}
    for n in CARS:
        k=int(np.count_nonzero(select&VALID[n]))
        n_races=0;stake_points=0.;f_exp=o_exp=o_hit=o_payout=0.
        for y in years:
            r=DATA.get((y,n,group))
            if not r:continue
            n_races+=r["race_count"]
            f=float(r["fav_frequency"][select].sum())
            count_points=k*r["race_count"]-f
            stake_points+=count_points
            o_hit+=float(r["actual"][select].sum())
            o_exp+=float((r["expected"]-r["fav_expected"])[select].sum())
            o_payout+=float(r["payout"][select].sum())
            biggest=max(biggest,float(r["top_hit"][select].max()) if k else 0.)
            baseline+=k*r["inv_overround_sum"]-float(r["fav_inv_overround"][select].sum())
        if stake_points<0:raise AssertionError("Negative valid point count")
        if stake_points>0:
            entries+=n_races
            points+=stake_points
            hit+=o_hit
            expected+=o_exp
            payout+=o_payout
            by_n[str(n)]={"races":n_races,"purchased_points":int(stake_points),
                "actual_hits":int(o_hit)}
    if points<=0:return None
    return {"eligible_races":int(entries),
        "mean_points":round(points/entries,3) if entries else None,
        "purchased_points":int(points),
        "hits":int(hit),"market_expected_hits":round(expected,5),
        "observed_vs_market":round(hit/expected,5) if expected else None,
        "stake_yen":int(points*100),"payout_yen":round(payout),
        "profit_yen":round(payout-points*100),
        "roi_pct":round(100*payout/(points*100),4),
        "largest_single_hit_yen":round(biggest),
        "largest_hit_share_pct":round(100*biggest/payout,3) if payout else None,
        "market_baseline_roi_pct":round(100*baseline/points,4) if points else None,
        "cars_detail":by_n}

def signal(mask, direction):
    vals=[accumulate(mask,(y,)) for y in TRAIN]
    if any(x is None for x in vals):return -1e9
    total=accumulate(mask,TRAIN)
    if total["eligible_races"]<MIN_TRAIN_BET_RACES:return -1e9
    if total["market_expected_hits"]<MIN_TRAIN_EXPECTED_HITS:return -1e9
    if any(v["market_expected_hits"]<MIN_YEAR_EXPECTED_HITS for v in vals):return -1e9
    delta=[direction*(v["hits"]-v["market_expected_hits"]) for v in vals]
    per=[d/math.sqrt(max(v["market_expected_hits"],1)) for d,v in zip(delta,vals)]
    overall=direction*(total["hits"]-total["market_expected_hits"])/math.sqrt(total["market_expected_hits"])
    # Select only based on observed-minus-market frequency, never training payout.
    return min(per)+0.25*overall

def learn(direction):
    rng=random.Random(SEARCH_SEED+(0 if direction==1 else 1000))
    full=(1<<D)-1
    seeds=[tuple(1<<x for x in t) for t in RANK_TRIPLES[:50]]
    seeds += [(full,full,full),(1,full,full),(full,1,full),(full,full,1)]
    seeds += [tuple((1<<rng.sample(range(D),rng.randint(1,D))[0]) for _ in range(3)) for _ in range(8)]
    for _ in range(SEARCH_STARTS):
        seeds.append(tuple(rng.randint(1,full) for _ in range(3)))
    cache={}
    def objective(m):
        if m not in cache:cache[m]=signal(m,direction)
        return cache[m]
    found={}
    for start in seeds:
        m=start;score=objective(m)
        for _ in range(SEARCH_SWEEPS):
            best=m;bestscore=score
            for pos in range(3):
                for bit in range(D):
                    t=list(m);t[pos]^=1<<bit
                    if not t[pos]:continue
                    candidate=tuple(t)
                    v=objective(candidate)
                    if v>bestscore+1e-10:
                        best,bestscore=candidate,v
            if best==m:break
            m,score=best,bestscore
        if score>-1e8:found[m]=score
    choices=sorted(found,key=lambda m:(-found[m],m))
    return choices[:12],{"direction":"underpriced" if direction==1 else "overpriced",
        "searched_masks":len(cache),"distinct_feasible_local_optima":len(found),
        "optimizer":"multi-start single-rider-slot bit flips; non-exhaustive, no point cap",
        "objective":"min(year-by-year Poisson-style observed-market z) plus 0.25 pooled z",
        "selection_data":"2021-2023 actual vs market normalized full-book quotes only",
        "no_odds_band_or_field_segment_in_training":True}

def numbered_formations(mask,example):
    n=example["n"]
    order=example["order"]
    x=[{order[i] for i in range(n) if mask[pos]&(1<<i)} for pos in range(3)]
    originals=set(t for t in itertools.product(*x) if len(set(t))==3)
    favorite=example["favorite"]
    wanted=originals-{favorite}
    if not wanted:return None
    if favorite not in originals:
        candidate=[x]
    else:
        fa,fb,fc=favorite
        candidate=[
            [x[0]-{fa},x[1],x[2]],
            [{fa},x[1]-{fb},x[2]],
            [{fa},{fb},x[2]-{fc}]]
    forms=[];got=set()
    for triple_sets in candidate:
        if not all(triple_sets):continue
        ts={t for t in itertools.product(*triple_sets) if len(set(t))==3}
        if not ts:continue
        if ts&got:raise AssertionError("Duplicated ticket in partition")
        got|=ts
        p=[sorted(a) for a in triple_sets]
        forms.append({"1着":p[0],"2着":p[1],"3着":p[2],
            "formation":"-".join("".join(map(str,a)) for a in p),
            "valid_points":len(ts)})
    if got!=wanted or favorite in got:raise AssertionError("Incorrect favorite-excluding formation")
    return forms,got

def real_examples(mask):
    answers=[]
    for n in (7,9):
        chosen=None
        for year in FORWARD:
            for ex in EXAMPLES[(year,n)]:
                candidate=numbered_formations(mask,ex)
                if not candidate:continue
                forms,tickets=candidate
                chosen=(ex,forms,sorted(tickets))
                break
            if chosen:break
        if chosen:
            ex,forms,tickets=chosen
            pay=100*ex["book"][ex["finish"]] if ex["finish"] in tickets else 0
            answers.append({"sample_policy":"First eligible 2025 or 2026 race among first 20 archived for this n, not picked by outcome",
                "race_id":ex["rid"],"cars":n,
                "favorite":"-".join(map(str,ex["favorite"])),
                "favorite_price":ex["favorite_odds"],
                "formations_union":forms,
                "unique_points":len(tickets),
                "all_tickets_and_prices":[{"ticket":"-".join(map(str,t)),
                    "decimal_odds":ex["book"][t]} for t in tickets],
                "result":"-".join(map(str,ex["finish"])),
                "stake_yen":100*len(tickets),
                "payout_yen":round(pay),
                "profit_yen":round(pay-100*len(tickets))})
    return answers

def candidate_report(mask,rank,direction,score):
    r={"direction":direction,"discovery_rank":rank,"mask_for_reproduction":list(mask),
       "training_signal_score":round(score,5),
       "development_combined":accumulate(mask,TRAIN),
       "development_yearly":{str(y):accumulate(mask,(y,)) for y in TRAIN},
       "validation_2024":accumulate(mask,(2024,)),
       "forward_combined":accumulate(mask,FORWARD),
       "forward_yearly":{str(y):accumulate(mask,(y,)) for y in FORWARD},
       "concrete_real_car_examples":real_examples(mask)}
    # Fifth research step, diagnostic only; never used for choosing masks.
    r["favorite_price_band_diagnostics"]={b:{
        "train":accumulate(mask,TRAIN,b),
        "validation":accumulate(mask,(2024,),b),
        "forward":accumulate(mask,FORWARD,b)} for b in BIN_NAMES}
    return r

def main():
    labels=load_labels()
    visited=set()
    for path in sorted(ODDS.glob("odds_20??.jsonl.gz")):
        year=int(path.name[5:9])
        for row in raw_lines(path):
            rid=str(row.get("race_id",""))
            if rid in visited:raise RuntimeError("Duplicate quoted race ID")
            visited.add(rid)
            AUDIT["quoted_races"]+=1
            actual=labels.get(rid)
            if actual is None:
                AUDIT["unmatched_result"]+=1
                continue
            observe(rid,year,row,actual)
    if AUDIT["accepted_races"]<45000:
        raise RuntimeError("Full archival universe missing; refusing restricted sample")
    atom_rows=[]
    for t in RANK_TRIPLES:
        m=tuple(1<<p for p in t)
        dev=accumulate(m,TRAIN)
        atom_rows.append({"market_rank_positions_one_based":[i+1 for i in t],
            "development":dev,"validation":accumulate(m,(2024,)),
            "forward":accumulate(m,FORWARD)})
    under,under_audit=learn(1)
    over,over_audit=learn(-1)
    under_report=[candidate_report(m,i+1,"underpriced",signal(m,1)) for i,m in enumerate(under)]
    over_report=[candidate_report(m,i+1,"overpriced",signal(m,-1)) for i,m in enumerate(over)]
    # Union candidates: discover only via underpriced TRAIN scores from top
    # masks (not post-hoc holdout). Exact union payoff needs race-level ticket
    # overlaps, so do not publish a false additive ROI.
    report={"status":"UNCONDITIONAL_RESULT_FIRST_FORMATION_SCAN_COMPLETE",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source":"Verified exact result labels joined to COMPLETE final 3連単 price books",
        "audit":dict(AUDIT),
        "unconditional_races_by_year_and_car_count":{
            f"{y}:{n}":int(v) for (y,n),v in sorted(COUNT.items())},
        "unconditional_total":sum(COUNT.values()),
        "protocol":{"stages":[
            "1 full race universe, do not segment by car count or favorite band for search",
            "2 compare realized outcome set frequency to normalized market-implied frequency",
            "3 translate arbitrary ordered rider-slot subsets to actual numbered multi-component favorite-excluding formations",
            "4 replay frozen candidates across distinct calendar periods and settle all tickets at 100 yen",
            "5 only then inspect favorite-price-band residuals"],
            "train_years":TRAIN,"validation_year":2024,"forward_years":FORWARD,
            "no_fixed_point_cap":True,
            "discovery_exposure_min_train_races":MIN_TRAIN_BET_RACES,
            "discovery_market_expected_min_train":MIN_TRAIN_EXPECTED_HITS,
            "favorite_rule":"Never buy first favorite exact combo. If favorite lies in rectangle, decompose A×B×C minus that combo into at most three valid cartesian formations. Buy all races, not cherry-picked subset.",
            "rank_labels":"Market support ranking used as invariant FEATURE only, never displayed as car numbers",
            "price_bands_diagnostic_only":BIN_NAMES,
            "payout":"100yen each unique 3連単 ticket; actual ordered winner pays if included",
            "odds_availability":"final historical odds, no T-minus-15 verification"},
        "search_audits":[under_audit,over_audit],
        "underpriced_discovery_candidates":under_report,
        "negative_control_overpriced_candidates":over_report,
        "all_unconditional_rank_triplet_atoms":atom_rows,
        "limitations":["Heuristic mask search, not exhaustive global optimum; includes arbitrary slot subsets but no learned union optimization.",
           "Market inverse-odds normalization is only a market-price proxy, not an independent true-probability estimate.",
           "Multiple testing creates discovery bias; no claim of superiority unless fixed rules work outside search sample.",
           "Prior projects inspected some later data, so time holdouts are not totally pristine.",
           "All archived prices are final quotes and not certified 15 min before purchase.",
           "No existing production model, AI, purchase rule or website changed."]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    compact={k:v for k,v in report.items() if k not in ("all_unconditional_rank_triplet_atoms","negative_control_overpriced_candidates")}
    compact["top_market_residual_atoms"] = sorted(
        [v for v in atom_rows if v["development"] and v["development"]["market_expected_hits"]>=20],
        key=lambda v: (-(v["development"]["hits"]-v["development"]["market_expected_hits"]),v["market_rank_positions_one_based"]))[:30]
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
    print("COMPLETE",json.dumps({"universe":report["unconditional_total"],
       "search":report["search_audits"],
       "top_underpriced":[{"rank":r["discovery_rank"],
            "mask":r["mask_for_reproduction"],
            "development_roi":r["development_combined"]["roi_pct"] if r["development_combined"] else None,
            "validation_roi":r["validation_2024"]["roi_pct"] if r["validation_2024"] else None,
            "forward_roi":r["forward_combined"]["roi_pct"] if r["forward_combined"] else None,
            "development_residual_ratio":r["development_combined"]["observed_vs_market"] if r["development_combined"] else None,
            "forward_residual_ratio":r["forward_combined"]["observed_vs_market"] if r["forward_combined"] else None}
            for r in under_report[:12]]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

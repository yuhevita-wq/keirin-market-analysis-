#!/usr/bin/env python3
"""Compare equal 100yen-per-ticket vs 1000yen/race stake allocation in genuine
small-cartesian 3連単 formations, using all verified 2021-2026H1 books.

The 20 fixed formation masks were discovered in the separate outcome-first
experiment using 2021-2023, no re-selection against 2024+.

Training 2021-2023: estimate 27 coarse positional support-rank cell calibration
factors with 100-market-expected-hit shrinkage. Do not optimize stake strategies
on test outcomes. Fixed 1000yen bankroll PER ENTERED RACE for all comparisons.
2024 and 2025-2026H1: time-separated out-of-sample tests.
One row = same race, same tickets, only cash allocation changes.
"""
from __future__ import annotations
import gzip,itertools,json,math
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
ODDS=ROOT/"data/odds_only"
RESULTS=ROOT/"data/result_universe_v2/combined"
FORM=ROOT/"research/unconditional_small_point_formation_20261028/compact.json"
OUT=ROOT/"research/formation_capital_allocation_20261028"
TRAIN=(2021,2022,2023)
METHODS=("equal","dutch","high_odds","learned_residual","positive_ev_only")
BUDGET_UNITS=10
SMOOTH_EXPECTED=100.0

def read(p):
    with gzip.open(p,"rt",encoding="utf-8") as fh:
        for s in fh:
            if s.strip():yield json.loads(s)

def labels():
    d={};conflicts=set()
    for p in sorted(RESULTS.glob("results_20??.jsonl.gz")):
        for row in read(p):
            rid=str(row.get("race_id",""))
            v=tuple(row.get("finish_order") or ())
            if len(v)!=3 or len(set(v))!=3 or any(type(c)!=int or c<1 or c>9 for c in v):continue
            if rid in d and d[rid]!=v:conflicts.add(rid)
            else:d[rid]=v
    for rid in conflicts:d.pop(rid,None)
    return d

def odds_rows(label_data):
    seen=set()
    for p in sorted(ODDS.glob("odds_20??.jsonl.gz")):
        year=int(p.name[5:9])
        for row in read(p):
            rid=str(row.get("race_id",""))
            if rid in seen:raise RuntimeError("Duplicate race id")
            seen.add(rid)
            finish=label_data.get(rid)
            if finish is None:continue
            raw=(row.get("odds") or {}).get("3連単") or {}
            try:prices={tuple(map(int,k.split("-"))):float(v) for k,v in raw.items()}
            except (TypeError,ValueError):continue
            cars=sorted({v for t in prices for v in t})
            n=len(cars)
            if n not in (5,6,7,8,9) or len(prices)!=n*(n-1)*(n-2) or finish not in prices:continue
            if any(not math.isfinite(v) or v<=0 for v in prices.values()):continue
            support={c:0.0 for c in cars}
            invsum=0.0
            for ticket,odd in prices.items():
                s=1/odd
                invsum+=s
                for c in ticket:support[c]+=s
            ordered=sorted(cars,key=lambda c:(-support[c],c))
            rank={c:i for i,c in enumerate(ordered)}
            favorite=min(prices,key=lambda t:(prices[t],t))
            yield year,rid,finish,prices,rank,favorite,invsum

def bucket(ticket,rank):
    return tuple((0 if rank[c]<2 else 1 if rank[c]<4 else 2) for c in ticket)

def learn(label_data):
    expected=defaultdict(float);hits=Counter();audit=Counter()
    for year,rid,finish,prices,rank,favorite,invsum in odds_rows(label_data):
        if year not in TRAIN:continue
        audit["training_races"]+=1
        for t,o in prices.items():
            expected[bucket(t,rank)]+=(1/o)/invsum
        hits[bucket(finish,rank)]+=1
    factors={}
    for cell in itertools.product(range(3),repeat=3):
        factor=(hits[cell]+SMOOTH_EXPECTED)/(expected[cell]+SMOOTH_EXPECTED)
        factors[cell]=factor
    return factors,hits,expected,dict(audit)

def valid_tickets(mask,rank,prices,favorite):
    slots=[{c for c,k in rank.items() if (mask[i]>>k)&1} for i in range(3)]
    if not all(slots):return []
    chosen=[t for t in itertools.product(*slots) if len(set(t))==3 and t!=favorite]
    assert len(chosen)==len(set(chosen))
    return sorted(chosen)

def assign_units(weights,total=10,minimum=1):
    n=len(weights)
    if n==0 or (minimum*n>total):return [0]*n
    m=total-minimum*n
    v=[max(0,float(x)) for x in weights]
    if sum(v)<=0:v=[1.0]*n
    ratio=[m*w/sum(v) for w in v]
    out=[minimum+int(math.floor(x)) for x in ratio]
    remaining=total-sum(out)
    order=sorted(range(n),key=lambda i:(-(ratio[i]-math.floor(ratio[i])),i))
    for i in order[:remaining]:out[i]+=1
    assert sum(out)==total
    return out

def allocation(method,tickets,prices,rank,invsum,factors,precomputed_norm):
    n=len(tickets)
    if not n:return [0]*n
    if method=="equal":return assign_units([1]*n)
    if method=="dutch":return assign_units([1/prices[t] for t in tickets])
    if method=="high_odds":return assign_units([math.sqrt(prices[t]) for t in tickets])
    if method=="learned_residual":
        return assign_units([factors[bucket(t,rank)]**3 for t in tickets])
    if method=="positive_ev_only":
        # Calibrate all outcomes, not just selected tickets. Raw implied
        # normalized market p(t) = (1/odds(t))/invsum.
        ev={t:factors[bucket(t,rank)]/(invsum*precomputed_norm) for t in tickets}
        eligible=[(t,max(0,(ev[t]-1.05))) for t in tickets if ev[t]>1.05]
        if not eligible:return [0]*n
        avail={t:i for i,t in enumerate(tickets)}
        active=[0]*n
        selected=assign_units([v for _,v in eligible],minimum=0)
        for (t,_),units in zip(eligible,selected):
            active[avail[t]]=units
        return active
    raise KeyError(method)

def blank():
    return {"races_in_formation":0,"races_bought":0,"stake_yen":0,
        "payout_yen":0,"hits":0,"max_hit_payout_yen":0,
        "total_selected_points":0,"total_zero_stake_points":0}

def evaluate():
    d=json.loads(FORM.read_text(encoding="utf-8"))
    candidates=[(x["rank"],tuple(x["market_rank_masks_only_internal"])) for x in d["forms"]]
    assert len(candidates)>=20
    label_data=labels()
    factors,hits,expected,train_audit=learn(label_data)
    results=defaultdict(blank)
    audit=Counter()
    examples=defaultdict(list)
    for year,rid,finish,prices,rank,favorite,invsum in odds_rows(label_data):
        audit["verified_complete_races"]+=1
        # This all-market normalization is shared by all 20 candidate sets.
        # Recomputing it within every candidate would be wasteful and identical.
        calibrated_norm=sum((1/o)/invsum*factors[bucket(t,rank)] for t,o in prices.items())
        for cand,mask in candidates:
            tickets=valid_tickets(mask,rank,prices,favorite)
            if not tickets:continue
            for method in METHODS:
                u=allocation(method,tickets,prices,rank,invsum,factors,calibrated_norm)
                stat=results[(cand,year,method)]
                stat["races_in_formation"]+=1
                stat["total_selected_points"]+=len(tickets)
                stat["total_zero_stake_points"]+=sum(1 for v in u if v==0)
                stake=100*sum(u)
                if stake==0:continue
                stat["races_bought"]+=1
                stat["stake_yen"]+=stake
                hit_stake=dict(zip(tickets,u)).get(finish,0)
                if hit_stake:
                    pay=int(round(hit_stake*100*prices[finish]))
                    stat["payout_yen"]+=pay
                    stat["hits"]+=1
                    stat["max_hit_payout_yen"]=max(stat["max_hit_payout_yen"],pay)
                if year>=2025 and cand in (3,5,7) and len(examples[(cand,method)])<2:
                    if method in ("equal","dutch","high_odds","learned_residual","positive_ev_only"):
                        examples[(cand,method)].append({
                            "race_id":rid,"year":year,
                            "formation_1着":[c for c,k in rank.items() if (mask[0]>>k)&1],
                            "formation_2着":[c for c,k in rank.items() if (mask[1]>>k)&1],
                            "formation_3着":[c for c,k in rank.items() if (mask[2]>>k)&1],
                            "favorite":"-".join(map(str,favorite)),
                            "actual_finish":"-".join(map(str,finish)),
                            "ticket_stakes":[{"ticket":"-".join(map(str,t)),"odds":prices[t],
                                            "stake_yen":100*units} for t,units in zip(tickets,u)],
                            "total_stake_yen":stake,
                            "payout_yen":int(round(hit_stake*100*prices[finish])) if hit_stake else 0})
    if audit["verified_complete_races"]<45000:raise RuntimeError("Full universe unavailable")
    def summary(cand,years,method):
        s=blank()
        for y in years:
            r=results.get((cand,y,method))
            if not r:continue
            for k in s:
                if k=="max_hit_payout_yen":s[k]=max(s[k],r[k])
                else:s[k]+=r[k]
        wager=s["stake_yen"];pay=s["payout_yen"]
        return {**s,"profit_yen":pay-wager,
                "roi_pct":round(100*pay/wager,4) if wager else None,
                "largest_hit_share_pct":round(100*s["max_hit_payout_yen"]/pay,3) if pay else None}
    out=[]
    for cand,mask in candidates:
        item={"candidate_rank":cand,"frozen_rank_masks":list(mask)}
        for method in METHODS:
            item[method]={
                "development_2021_23":summary(cand,TRAIN,method),
                "validation_2024":summary(cand,(2024,),method),
                "forward_2025_26H1":summary(cand,(2025,2026),method),
                "forward_2025":summary(cand,(2025,),method),
                "forward_2026H1":summary(cand,(2026,),method),
                "samples":examples.get((cand,method),[])
            }
        out.append(item)
    document={
        "status":"ALLOCATION_ANALYSIS_COMPLETE",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source_small_formations":"research/unconditional_small_point_formation_20261028/compact.json",
        "protocol":{"sample":"All 5-9 car races with exact 3連単 price book and verified results 2021-2026H1",
          "formation_masks":"20 historical TRAIN-only ranked models held FIXED",
          "all_comparisons":"Same race/tickets; change ONLY weights, or skip if no estimated +EV",
          "fixed_budget_yen_per_entered_race":1000,
          "ticket_min_yen_for_non_ev_methods":100,
          "allocation_units":"100 yen",
          "train_corrections":"2021-2023 all priced races, posterior correction for 27 rank-cell bias",
          "train_smoothing_market_expected_hits":SMOOTH_EXPECTED,
          "equal":"1000 yen distributed as evenly as units permit",
          "dutch":"More money on lower odds in inverse proportion to odds",
          "high_odds":"More money on high odds proportional to sqrt(odds)",
          "learned_residual":"More money on historical market error factor raised to power 3",
          "positive_ev_only":"Buy only ticket(s) with modeled 5%+ net expected profit, skip races without; model uncertainty remains",
          "model_p":"p_market(t)*coarse_cell_factor / sum_market[p_market(cell)*factor(cell)]",
          "historical_price":"FINAL odds not T-minus-15, training and verification time-separated but previously inspected in other studies",
          "no_modification":"No existing predictor or live buying policies changed"},
        "audit":{**train_audit,**dict(audit)},
        "trained_factors":[{"rank_cell":[x+1 for x in cell],"wins":hits[cell],
            "market_expected":round(expected[cell],4),"smoothed_bias_ratio":round(v,5)}
             for cell,v in sorted(factors.items())],
        "candidates":out,
        "warnings":["Reallocating cash cannot turn all negative-EV ticket options positive.",
             "Fixed 1000 yen race budget differs from previous flat 100 yen per point; compare ROI, not raw profit.",
             "Training on the same outcomes used to discover candidate masks creates overfit risk.",
             "An ex-post final odds book is not a pre-purchase executable quote.",
             "Positive-EV calibration is hypothetical and needs market close risk/bet-timing verification."]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    compact={**{k:v for k,v in document.items() if k not in ("trained_factors","candidates")},"candidates":[]}
    for i in out:
        compact["candidates"].append({"candidate_rank":i["candidate_rank"],"frozen_rank_masks":i["frozen_rank_masks"],
          **{method:{p:v for p,v in i[method].items() if p!="samples"} for method in METHODS},
          "examples":{method:i[method]["samples"][:1] for method in METHODS}})
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ALLOCATION_RESULTS",json.dumps({"audit":document["audit"],
         "method_comparison":[{"rank":i["candidate_rank"],
           **{k:{"dev":i[k]["development_2021_23"]["roi_pct"],
                 "val":i[k]["validation_2024"]["roi_pct"],
                 "forward":i[k]["forward_2025_26H1"]["roi_pct"],
                 "races":i[k]["forward_2025_26H1"]["races_bought"]}
              for k in METHODS}} for i in out]},ensure_ascii=False),flush=True)

if __name__=="__main__":evaluate()

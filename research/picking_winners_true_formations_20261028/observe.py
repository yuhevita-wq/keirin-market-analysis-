"""Two actual 3-slot trifecta formations: share/replace slot sets, evaluate exact payout.
Liu et al. (2023) inspired controlled observation, NOT a proof of paper transfer.
No hand-authored car-number formation templates and no changes to purchase models.
"""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
OUT=ROOT/"research/picking_winners_true_formations_20261028"
NAMES=("base_only","change_1","change_2","change_3","change_12","change_13","change_23","change_123")
AUDIT=Counter()
STATS=defaultdict(Counter)
EXAMPLES=[]

def tickets(slots):
    a,b,c=slots
    return {(i,j,k) for i in a for j in b for k in c if len({i,j,k})==3}

def create_base(cars,book):
    n=len(cars); ids={v:i for i,v in enumerate(cars)}
    p=np.zeros((n,n,n),dtype=float)
    z=sum(1/float(v) for v in book.values())
    for (a,b,c),odd in book.items():
        p[ids[a],ids[b],ids[c]]=(1/float(odd))/z
    ranks=(np.argsort(-p.sum(axis=(1,2)),kind="stable"),
           np.argsort(-p.sum(axis=(0,2)),kind="stable"),
           np.argsort(-p.sum(axis=(0,1)),kind="stable"))
    grid=p[np.ix_(*ranks)]
    count=np.zeros_like(grid)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                count[i,j,k]=int(len({int(ranks[0][i]),int(ranks[1][j]),int(ranks[2][k])})==3)
    mass=grid.cumsum(0).cumsum(1).cumsum(2)
    sizes=count.cumsum(0).cumsum(1).cumsum(2)
    # Experimental market-only penalty, NOT a learned edge or stake recommendation.
    threshold=float(np.quantile(p[p>0],0.90))
    value=mass-threshold*sizes
    value[sizes<4]=-np.inf
    a,b,c=np.unravel_index(np.argmax(value),value.shape)
    ks=(a+1,b+1,c+1)
    base=tuple(tuple(cars[int(j)] for j in ranks[pos][:ks[pos]]) for pos in range(3))
    return base,ranks,ks

def sample(name,group,base_set,second_set,win,odd):
    union=base_set|second_set
    extra=second_set-base_set
    hit=int(win in union)
    extra_hit=int(win in extra)
    x=STATS[(name,*group)]
    x["races"]+=1
    x["unique_points"]+=len(union)
    x["base_points"]+=len(base_set)
    x["second_points"]+=len(second_set) if name!="base_only" else 0
    x["shared_exact_points"]+=len(base_set&second_set) if name!="base_only" else 0
    x["extra_points"]+=len(extra) if name!="base_only" else 0
    x["stake_yen"]+=100*len(union)
    x["payout_final_odds_yen"]+=100*odd*hit
    x["hit_count"]+=hit
    x["incremental_hits"]+=extra_hit if name!="base_only" else 0
    x["incremental_stake_yen"]+=100*len(extra) if name!="base_only" else 0
    x["incremental_payout_final_odds_yen"]+=100*odd*extra_hit if name!="base_only" else 0

def observe(rid,day,books,finish,rtype,source):
    AUDIT["callback"]+=1
    if source!="sealed_four_markets_orientation_verified":
        raise RuntimeError("Unverified price orientation")
    book=books["3rentan"]
    cars=tuple(sorted({c for w in book for c in w}))
    n=len(cars)
    win=tuple(finish)
    if win not in book or len(book)!=n*(n-1)*(n-2) or any(not math.isfinite(float(v)) or v<1 for v in book.values()):
        AUDIT["invalid_book_or_result"]+=1
        return
    base,ranks,ks=create_base(cars,book)
    t1=tickets(base)
    if len(t1)<4:raise AssertionError("Formation cardinality corruption")
    odd=float(book[win])
    candidates=[("base_only",base,t1)]
    for mask in ("1","2","3","12","13","23","123"):
        alternative=list(base);valid=True
        for ch in mask:
            pos=int(ch)-1
            shifted=tuple(cars[int(i)] for i in np.roll(ranks[pos],-1)[:ks[pos]])
            if set(shifted)==set(base[pos]):
                valid=False;break
            alternative[pos]=shifted
        if not valid:
            AUDIT["unshiftable_"+mask]+=1
            continue
        alternative=tuple(alternative)
        t2=tickets(alternative)
        if len(t2)<4 or t2==t1:
            AUDIT["invalid_candidate_"+mask]+=1
            continue
        candidates.append(("change_"+mask,alternative,t2))
    groups=(("all",),("year",day[:4]),("cars",str(n)),("race_type",str(rtype)))
    for name,form,t2 in candidates:
        for group in groups:
            sample(name,group,t1,t2,win,odd)
        if day[:4]>="2023" and name!="base_only" and len(EXAMPLES)<56:
            EXAMPLES.append({"race_id":rid,"date":day,"policy":name,
                "first":[list(v) for v in base],"first_points":len(t1),
                "second":[list(v) for v in form],"second_points":len(t2),
                "shared_exact_points":len(t1&t2),
                "union_points":len(t1|t2),"actual_finish":list(win),
                "hit":win in (t1|t2),
                "stake_yen":100*len(t1|t2),
                "payout_yen":100*odd*int(win in (t1|t2))})
    AUDIT["analyzed"]+=1
    AUDIT["year_"+day[:4]]+=1
    AUDIT["cars_"+str(n)]+=1

def report_rows():
    rows=[]
    for key,s in sorted(STATS.items()):
        d={"policy":key[0],"group":list(key[1:]),**dict(s)}
        d["roi"]=s["payout_final_odds_yen"]/s["stake_yen"]
        d["hit_rate"]=s["hit_count"]/s["races"]
        d["mean_unique_points"]=s["unique_points"]/s["races"]
        d["mean_base_points"]=s["base_points"]/s["races"]
        d["mean_second_points"]=s["second_points"]/s["races"]
        d["mean_shared_exact_points"]=s["shared_exact_points"]/s["races"]
        d["incremental_roi"]=s["incremental_payout_final_odds_yen"]/s["incremental_stake_yen"] if s["incremental_stake_yen"] else None
        rows.append(d)
    return rows

def chronological(rows):
    by={(r["policy"],r["group"][1]):r for r in rows if r["group"][0]=="year"}
    out=[]
    for train_years,test_year in [(["2021","2022","2023","2024"],"2025"),
                                  (["2021","2022","2023","2024","2025"],"2026")]:
        ranks=[]
        for policy in NAMES:
            train=[by.get((policy,y)) for y in train_years]
            if any(r is None for r in train):continue
            cost=sum(r["stake_yen"] for r in train)
            pay=sum(r["payout_final_odds_yen"] for r in train)
            ranks.append((pay/cost,policy))
        chosen=max(ranks,key=lambda x:(x[0],-NAMES.index(x[1]))) if ranks else None
        result=by.get((chosen[1],test_year)) if chosen else None
        base=by.get(("base_only",test_year))
        out.append({"train_years":train_years,"test_year":test_year,
                    "trained_best":chosen[1] if chosen else None,
                    "training_roi":chosen[0] if chosen else None,
                    "test_roi":result["roi"] if result else None,
                    "base_test_roi":base["roi"] if base else None,
                    "test_stake_yen":result["stake_yen"] if result else None,
                    "test_payout_yen":result["payout_final_odds_yen"] if result else None})
    return out

def main():
    if not os.environ.get(corrected.old.bridge.SECRET_ENV):
        raise RuntimeError("Existing sealed archive key missing; no partial fallback")
    corrected.base_observe=observe
    corrected.old.save=lambda status:print("CHECKPOINT",status,dict(AUDIT),flush=True)
    corrected.old.sealed()
    if AUDIT["analyzed"]<1000:
        raise RuntimeError("Insufficient data; not a full archive study")
    rows=report_rows()
    summary={"status":"TWO_REAL_FORMATION_OBSERVATION_COMPLETE",
       "generated_utc":datetime.now(timezone.utc).isoformat(),
       "source_commit":os.environ.get("GITHUB_SHA","local"),
       "audit":dict(AUDIT),"rows":rows,
       "chronological_policy_selection":chronological(rows),
       "actual_formation_examples":EXAMPLES,
       "methods":{
          "formation":"3 ordered sets A(first), B(second), C(third); expand full Cartesian product excluding repeated riders",
          "base":"For each race compute 1st/2nd/3rd market marginal probabilities from all exact-ordered odds. Optimize EVERY size triple 1..N for this rank-top-k family, using mass minus 90th-percentile implied-ticket-probability times points. Require >=4 actual points.",
          "second":"Hold or rank-shift exact vehicle set in 1st, 2nd, 3rd positions; seven sharing masks. Reject unchanged or <4-point second formations.",
          "money":"100 yen per unique 3連単 combination in F1 union F2. Overlaps are not double purchased. Actual full-match payout, zero credit for partial placement matches.",
          "observation":"Compare all seven changed-slot patterns and base-only. Train-year highest-ROI policy is evaluated in the subsequent historical year.",
          "data":"Existing verified four-market sealed odds/results 2021..2026H1, including independent orientation correction. FINAL odds."
       },
       "limits":[
         "This is a controlled first-stage test of full formation structure; not the paper's original multiple-event payoff theorem.",
         "The top-k rank family is a search restriction; not all 2^(3N) possible rider subsets are searched.",
         "No point count is fixed, but >=4 distinct tickets is required to exclude a disguised two-singleton experiment.",
         "90th-percentile penalty and shift-by-one rank are experimental design parameters, not empirically selected optimal values.",
         "Uses market odds only; cannot establish an independent pricing edge.",
         "Final odds payout proxy, not certified purchase at T-15, and not official settlement.",
         "All evaluation years were historically available in prior research; not genuine future untouched data.",
         "Existing models, purchase rules and websites unchanged."
       ]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("FINISHED",json.dumps({"audit":dict(AUDIT),
          "overall":[r for r in rows if r["group"]==["all"]],
          "chronological":summary["chronological_policy_selection"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()

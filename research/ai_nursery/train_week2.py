"""GEN-001 / week 2: add FORMATION mark sheets, with no fixed number of sheets.

Continue exactly from frozen 2023-01-07 zero-born learner's checkpoint.
Input to learner: racecard + full available odds + only the instruction
for interpreting formation marks (3連単 order / 3連複 unordered).
Outcome is not accessed until ALL sheets for the race are flushed.
The stop decision is a neutral stochastic mechanism, not handicapping advice.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import os
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from train_week import ROOT, D, LR, OUT as W1, WEB, load_cards, load_results
from train_week import load_odds, eligible, feature_matrix, softmax

OUT=ROOT/"research/ai_nursery/outputs/GEN-001/week2"
WEEK_FIRST=date(2023,1,8)
WEEK_LAST=WEEK_FIRST+timedelta(days=6)
SEED=20261020
COST_PER_MARK=100


def form_expand(pool, sets, odds):
    """Pure mark-sheet semantics: cartesian product; distinct riders; trio unordered."""
    if pool not in ("3連単","3連複") or len(sets)!=3:
        raise ValueError("invalid formation")
    delim="-" if pool=="3連単" else "="
    tickets={}
    for a,b,c in itertools.product(*sets):
        if len({a,b,c})!=3:continue
        values=(a,b,c) if pool=="3連単" else tuple(sorted((a,b,c)))
        key=delim.join(map(str,values))
        if key in odds[pool]:tickets[key]=float(odds[pool][key])
    return tickets


def formation_from_seed(seed, pool_arms, local_probs, odds, entries, rng):
    """Discover optional marks from learned conditional marginals; no pre-defined shape."""
    pool=seed["market"]
    delimiter="-" if pool=="3連単" else "="
    seats=tuple(int(t) for t in seed["selection"].split(delimiter))
    cars=sorted(entries)
    sets=[{c} for c in seats]
    total=float(sum(local_probs))
    if not total>0:raise AssertionError("empty distribution")
    for slot in range(3):
        marginal={c:0.0 for c in cars}
        for arm,p in zip(pool_arms,local_probs):
            car=int(arm["selection"].split(delimiter)[slot])
            marginal[car]+=float(p)/total
        for car in cars:
            if car not in sets[slot] and rng.random() < min(1.0,marginal[car]):
                sets[slot].add(car)
    normalized=[sorted(s) for s in sets]
    tickets=form_expand(pool,normalized,odds)
    if seed["selection"] not in tickets:
        raise AssertionError("seed is missing from its own formation")
    return {"type":pool,"columns":normalized,"point_count":len(tickets),
            "picks":sorted(tickets),"yen_per_pick":COST_PER_MARK,
            "total_yen":len(tickets)*COST_PER_MARK}


def make_sheets(arms, probabilities, odds, entries, rng):
    """No configured max_pages or max_points; each page is a freely expanded formation."""
    remaining=set(range(len(arms)))
    pages=[]
    # A neutral stop action with equal prior weight to "continue": not a betting rule.
    # It is sampled afresh after every sheet; physical forms are not capped.
    while remaining:
        ids=np.fromiter(sorted(remaining),dtype=np.int32)
        p=probabilities[ids]
        p=p/p.sum()
        k=int(rng.choice(ids,p=p))
        seed=arms[k]
        pool_arms=[a for a in arms if a["market"]==seed["market"]]
        pool_p=np.array([float(probabilities[j]) for j,a in enumerate(arms)
                         if a["market"]==seed["market"]],dtype=float)
        page=formation_from_seed(seed,pool_arms,pool_p,odds,entries,rng)
        page["sheet_number"]=len(pages)+1
        page["seed_selection"]=seed["selection"]
        pages.append(page)
        # Only remove a used seed. Overlapping paper marks are valid and accounted for.
        remaining.remove(k)
        if rng.random()<0.5:
            break
    return pages


def train():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Week 2 already exists; do not overwrite a saved learning run")
    w1_checkpoint=json.loads((W1/"checkpoint.json").read_text(encoding="utf-8"))
    w1_history=[json.loads(line) for line in (W1/"learning_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert w1_checkpoint["after_step"]==len(w1_history)==547
    assert w1_checkpoint["weights_sha256"]==w1_history[-1]["checkpoint_after_sha256"]
    assert w1_checkpoint["training_span"][1]=="2023-01-07"
    weights=np.asarray(w1_checkpoint["weights"],dtype=np.float64)
    if weights.shape!=(D,) or hashlib.sha256(weights.tobytes()).hexdigest()!=w1_checkpoint["weights_sha256"]:
        raise RuntimeError("First-week checkpoint mismatch: refusing to reset learner")
    parent_hash=hashlib.sha256(weights.tobytes()).hexdigest()
    rng=np.random.default_rng(SEED)
    cards,_=load_cards()
    target={rid:item for rid,item in cards.items()
            if WEEK_FIRST.isoformat()<=item["race"].get("race_date","")<=WEEK_LAST.isoformat()}
    results=load_results()
    odds=load_odds(set(target).intersection(results))
    all_days=Counter(r["race"].get("race_date","") for r in target.values())
    excluded=Counter()
    rows=[]
    for rid,item in target.items():
        if rid not in results:excluded["missing_result"]+=1;continue
        if rid not in odds:excluded["missing_complete_odds"]+=1;continue
        row={**item,"odds":odds[rid],"outcome":results[rid],"race_id":rid}
        if not eligible(row):excluded["incomplete_or_inconsistent"]+=1;continue
        rows.append(row)
    rows.sort(key=lambda r:(r["race"]["race_date"],r["race"].get("start_time",""),
                            r["race"].get("track",""),int(r["race"].get("race_no","0") or 0),r["race_id"]))
    if not rows:raise RuntimeError("No eligible racecard+odds+outcome rows in week 2")
    OUT.mkdir(parents=True,exist_ok=True)
    WEB.mkdir(parents=True,exist_ok=True)
    history=[]
    marked_file=OUT/"formation_mark_sheets.jsonl"
    for n,row in enumerate(rows,1):
        before=hashlib.sha256(weights.tobytes()).hexdigest()
        arms=feature_matrix(row["entries"],row["odds"])
        X=np.stack([a["x"] for a in arms])
        prior=np.array([a["prior"] for a in arms])
        probs=softmax(X@weights+prior)
        pages=make_sheets(arms,probs,row["odds"],row["entries"],rng)
        if not pages:raise AssertionError("must produce at least one sheet")
        expanded=Counter()
        for page in pages:
            for ticket in page["picks"]:
                expanded[(page["type"],ticket)]+=1
        # Seal COMPLETE paper bundle before looking at the outcome.
        record={
            "step":w1_checkpoint["after_step"]+n,"week_step":n,
            "race_id":row["race_id"],"date":row["race"]["race_date"],
            "track":row["race"].get("track"),"race_no":row["race"].get("race_no"),
            "race_type":row["race"].get("race_type"),
            "checkpoint_before_sha256":before,
            "sheets":pages,"sheet_count":len(pages),
            "expanded_marks":sum(expanded.values()),
            "distinct_marks":len(expanded),
            "overlap_marks":sum(expanded.values())-len(expanded),
            "total_yen":COST_PER_MARK*sum(expanded.values())
        }
        with marked_file.open("a",encoding="utf-8") as f:
            f.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            f.flush()
            os.fsync(f.fileno())
        # ONLY NOW unseal the actual order and train on the two true marks.
        finish=row["outcome"]
        true={("3連単","-".join(map(str,finish))),
              ("3連複","=".join(map(str,sorted(finish))))}
        winning=[j for j,a in enumerate(arms) if (a["market"],a["selection"]) in true]
        if len(winning)!=2:raise RuntimeError("full book should have two correct marks")
        loss=float(-sum(math.log(max(float(probs[j]),1e-30)) for j in winning)/2)
        weights-=LR*(probs@X-X[winning].mean(axis=0))
        stakes_hit=sum(expanded[t] for t in true)
        refund=int(round(sum(COST_PER_MARK*expanded[t]*row["odds"][t[0]][t[1]]
                             for t in true if t in expanded)))
        history.append({
            "week_step":n,"step":record["step"],"race_id":record["race_id"],
            "date":record["date"],"track":record["track"],"race_no":record["race_no"],
            "race_type":record["race_type"],"sheet_count":len(pages),
            "expanded_marks":record["expanded_marks"],"distinct_marks":record["distinct_marks"],
            "overlap_marks":record["overlap_marks"],"total_yen":record["total_yen"],
            "finish_order":"-".join(map(str,finish)),
            "winning_marks_count":stakes_hit,"hit":bool(stakes_hit),
            "payout_proxy_yen":refund,"net_proxy_yen":refund-record["total_yen"],
            "loss_before_update":loss,"checkpoint_before_sha256":before,
            "checkpoint_after_sha256":hashlib.sha256(weights.tobytes()).hexdigest()
        })
        if n%50==0:
            print("FORMATION_WEEK2_PROGRESS",n,len(rows),"mark_sheets",sum(x["sheet_count"] for x in history),
                  "hits",sum(x["hit"] for x in history),flush=True)
    stake=sum(x["total_yen"] for x in history)
    pay=sum(x["payout_proxy_yen"] for x in history)
    daily=[]
    for i in range(7):
        day=(WEEK_FIRST+timedelta(days=i)).isoformat()
        rr=[r for r in history if r["date"]==day]
        daily.append({"date":day,"races":len(rr),
                      "sheets":sum(x["sheet_count"] for x in rr),
                      "marks":sum(x["expanded_marks"] for x in rr),
                      "hits":sum(x["hit"] for x in rr),
                      "paper_stake_yen":sum(x["total_yen"] for x in rr),
                      "payout_proxy_yen":sum(x["payout_proxy_yen"] for x in rr),
                      "mean_loss":float(np.mean([x["loss_before_update"] for x in rr])) if rr else None})
    summary={
        "id":"GEN-001","learning_week":2,
        "dates":[WEEK_FIRST.isoformat(),WEEK_LAST.isoformat()],
        "status":"WEEK2_FORMATION_LEARNING_COMPLETE",
        "parent_checkpoint_sha256":parent_hash,"first_week_steps":w1_checkpoint["after_step"],
        "total_steps":w1_checkpoint["after_step"]+len(history),"races":len(history),
        "available_racecards_in_week":sum(all_days.values()),
        "missing_races":dict(excluded),"race_types":dict(Counter(x["race_type"] for x in history)),
        "sheets":sum(x["sheet_count"] for x in history),
        "expanded_marks":sum(x["expanded_marks"] for x in history),
        "distinct_marks":sum(x["distinct_marks"] for x in history),
        "overlap_marks":sum(x["overlap_marks"] for x in history),
        "multiple_sheet_races":sum(x["sheet_count"]>1 for x in history),
        "multiple_point_races":sum(x["expanded_marks"]>1 for x in history),
        "max_sheets_observed":max(x["sheet_count"] for x in history),
        "max_points_observed":max(x["expanded_marks"] for x in history),
        "hit_races":sum(x["hit"] for x in history),
        "hit_rate_percent":100*sum(x["hit"] for x in history)/len(history),
        "stake_yen":stake,"payout_proxy_yen":pay,
        "roi_proxy_percent":100*pay/stake if stake else None,
        "mean_loss":float(np.mean([x["loss_before_update"] for x in history])),
        "first_30_loss":float(np.mean([x["loss_before_update"] for x in history[:30]])),
        "last_30_loss":float(np.mean([x["loss_before_update"] for x in history[-30:]])),
        "daily":daily,
        "information_given":"same racecards and odds as previous week; only added explanation of formation slots and as many physical sheets as needed",
        "formation_rules":{
            "3連単":"column 1 = possible 1st, column 2 = possible 2nd, column 3 = possible 3rd; each unique ordered triplet is a mark",
            "3連複":"three columns choose a rider each; unordered unique three-rider sets count once per sheet",
            "validity":"repeated riders in a mark are invalid; each expanded bet must exist in complete odds",
            "paper_supply":"no fixed max sheets or points; finite distinct options from available riders/markets",
            "stop_policy":"neutral stochastic stop after each page; no preselected formation structure or bet-count limit",
            "denomination_yen_per_expanded_ticket":COST_PER_MARK
        },
        "limitations":[
            "Historic FINAL odds are not confirmed achievable pre-deadline prices.",
            "Racecards are retrospectively archived. Sources cannot certify pre-deadline fidelity.",
            "Only 3連単 and 3連複 odds are public in the joined archive; no other type is invented.",
            "The chosen stop mechanism is a neutral stochastic rule, not an evolved learned paper quantity strategy.",
            "Large coverage can trivially increase hit frequency while worsening efficiency; stake/payout also shown.",
            "Payout estimate is final odds times 100 yen per expanded mark, not confirmed exact settled official payouts.",
            "This is sequential training on historic results, not unseen validation of predictive profitability."
        ]
    }
    checkpoint={
        "id":"GEN-001","learning_week":2,"after_step":summary["total_steps"],
        "weights":[float(x) for x in weights],"dimension":D,
        "weights_sha256":hashlib.sha256(weights.tobytes()).hexdigest(),
        "parent_weights_sha256":parent_hash,
        "random_seed":SEED,"training_span":[WEEK_FIRST.isoformat(),WEEK_LAST.isoformat()]
    }
    for path,data in ((OUT/"summary.json",summary),(OUT/"checkpoint.json",checkpoint),
                      (WEB/"nursery-week2-summary.json",summary),
                      (WEB/"nursery-week2-checkpoint.json",checkpoint),
                      (WEB/"nursery-week2-history.json",history)):
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2 if path.name=="summary.json" else None)+"\n",encoding="utf-8")
    with (OUT/"learning_history.jsonl").open("w",encoding="utf-8") as f:
        for record in history:
            f.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
    print("GEN001_FORMATION_WEEK2_COMPLETE",json.dumps({
        "races":summary["races"],"sheets":summary["sheets"],
        "marks":summary["expanded_marks"],"multi_sheet_races":summary["multiple_sheet_races"],
        "hit_races":summary["hit_races"],"roi_proxy_percent":summary["roi_proxy_percent"],
        "saved_weights":checkpoint["weights_sha256"][:16]},ensure_ascii=False),flush=True)


if __name__=="__main__":train()

#!/usr/bin/env python3
"""Frozen V6 + full historical FINAL odds: January 2023 formation-only experiment.

IMPORTANT: final odds retrospectively acquired in 2026, NOT a bettable
T-minus snapshot. Original V6 is fit on Jan-Oct 2023, so Jan is INSAMPLE.
No result is read by candidate generation; outcomes are removed from each
race before row_probs and selection. This is an exploratory retrospective
portfolio test, not proven 2023-forward tradable ROI.

One complete Cartesian three-slot 3ren-tan formation per selected race,
100 yen EVERY distinct valid combination (no dutching). All component
tickets need not have EV>=100%. Formation EV is the simple arithmetic
mean of individual p_i * O_i, not p_F times harmonic synthetic odds.
The 75% market price hypothesis implies E[p_i * O_i]~.75 if calibrated
and markets efficient; do not subtract takeout twice from posted odds.

The selection function and settings must be frozen BEFORE reading results.
"""
from __future__ import annotations
import gzip,hashlib,itertools,json,math,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as obs
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import v5_rider_score_role_size_interactions as v5
import v6_opponent_competition_structure as v6
import weak_rider_podium_zero_excluded as weak
from v8_targeted_weak_podium_calibration import row_probs

START="2023-01-01"
END="2023-01-31"
ODDS=ROOT/"data/odds_only/odds_2023.jsonl.gz"
DIR=ROOT/"data/audits/v6_formation_ev_2023_01"
OUT=DIR/"pre_result_sealed_decisions.jsonl"
PRE_REPORT=DIR/"pre_result_source_and_coverage.json"
MIN_POINTS=4
MAX_POINTS=12
MIN_COMPOSITE=3.5
MIN_FLAT_EV=1.20
MIN_PSET=.01
MIN_VALUE_TICKETS=2
MIN_VALUE_SHARE=.35
MAX_SEEDS=15
STAKE_PER_POINT_YEN=100

def parse_market(row,cars):
    raw=(row.get("odds") or {}).get("3連単",{})
    expected=len(cars)*(len(cars)-1)*(len(cars)-2)
    if len(raw)!=expected:return None,"not_full_odds_grid"
    try:
        prices={tuple(k.split("-")):float(v) for k,v in raw.items()}
    except (TypeError,ValueError,AttributeError):return None,"bad_odds"
    if set(prices)!=set(itertools.permutations(cars,3)):
        return None,"odds_car_mismatch"
    if any(not math.isfinite(v) or v<=1 for v in prices.values()):
        return None,"not_positive_finite_final_price"
    return prices,"ok"

def candidate_slots(seed,cars,dist,prices):
    a,b,c=seed
    def alternatives(slot,limit):
        ranked=[]
        for rider in cars:
            if rider==seed[slot]:continue
            t=list(seed);t[slot]=rider
            if len(set(t))!=3:continue
            tup=tuple(t)
            pp=dist[tup];ev=pp*prices[tup]
            # Do not remove imperfect adjacent tickets merely because ev<1.
            # Relevance balances independent edge signal and model coverage.
            merit=(ev-0.85)*math.sqrt(pp)
            ranked.append((merit,pp,rider))
        ranked.sort(key=lambda x:(-x[0],-x[1],int(x[2])))
        return [z[2] for z in ranked[:limit]]
    aa=(a,*alternatives(0,1))
    bb=(b,*alternatives(1,1))
    cc=(c,*alternatives(2,4))
    forms=[]
    for k in (2,3,4):
        third=cc[:k]
        forms.extend((
            ("fixed12",((a,),(b,),third)),
            ("cross12",((a,b),(a,b),third)),
            ("expand1",(aa,(b,),third)),
            ("expand2",((a,),bb,third)),
            ("expand12",(aa,bb,third)),
        ))
    return forms

def select(e,dist,prices):
    cars=sorted(e["scores"],key=int)
    if len(dist)!=len(prices):raise AssertionError("Different 3rentan universes")
    if abs(sum(dist.values())-1)>1e-8:raise AssertionError("V6 p not 100%")
    margins=[]
    for t,p in dist.items():
        edge=p*prices[t]
        if p>=.0005 and edge>=1.10:
            margins.append((p*max(0,edge-1),p,t))
    margins.sort(key=lambda z:(-z[0],-z[1],z[2]))
    seeds=[t for _,_,t in margins[:MAX_SEEDS]]
    candidates=set()
    examined=Counter()
    best=None
    for seed in seeds:
        for shape,slots in candidate_slots(seed,cars,dist,prices):
            tt=tuple(sorted(t for t in itertools.product(*slots) if len(set(t))==3))
            if tt in candidates:continue
            candidates.add(tt)
            n=len(tt)
            examined["enumerated"]+=1
            if n<MIN_POINTS or n>MAX_POINTS:
                examined["point_count"]+=1;continue
            inv=sum(1/prices[t] for t in tt)
            synthetic=1/inv
            if synthetic<MIN_COMPOSITE:
                examined["low_synthetic_odds"]+=1;continue
            pset=sum(dist[t] for t in tt)
            if pset<MIN_PSET:
                examined["low_total_probability"]+=1;continue
            edges=[dist[t]*prices[t] for t in tt]
            value_count=sum(z>=1 for z in edges)
            if value_count<MIN_VALUE_TICKETS or value_count/n<MIN_VALUE_SHARE:
                examined["too_few_value_components"]+=1;continue
            ev_flat=sum(edges)/n
            if ev_flat<MIN_FLAT_EV:
                examined["low_formation_ev"]+=1;continue
            # Prioritize a broad enough win scenario with repeated independent
            # mispriced tickets. No future result / payout involved.
            merit=(ev_flat-1.10)*math.sqrt(pset)*math.log1p(value_count)
            tie=(merit,pset,value_count,-n,tt)
            if best is None or tie>best["tie"]:
                best={"tie":tie,"shape":shape,"slots":slots,"tickets":tt,
                      "p_set":pset,"flat_ev":ev_flat,"syn":synthetic,
                      "positive":value_count,
                      "negative":sum(z<1 for z in edges),
                      "near":sum(.90<=z<1 for z in edges),
                      "market_inv_full":sum(1/x for x in prices.values())}
            examined["passed"]+=1
    if best is None:return None,dict(examined)
    record={
        "formation":"/".join("-".join(z) for z in best["slots"]),
        "shape":best["shape"],
        "first":list(best["slots"][0]),"second":list(best["slots"][1]),"third":list(best["slots"][2]),
        "tickets":["-".join(t) for t in best["tickets"]],
        "ticket_data":[{"ticket":"-".join(t),"v6_p":round(dist[t],10),
                        "final_odds":prices[t],"v6_flat_ev":round(dist[t]*prices[t],7),
                        "p_vs_75pct_market":round(dist[t]/(.75/prices[t]),6)}
                       for t in best["tickets"]],
        "points":len(best["tickets"]),
        "stake_yen":100*len(best["tickets"]),
        "v6_formation_probability_pct":round(100*best["p_set"],7),
        "v6_flat_stake_expected_roi_pct":round(100*best["flat_ev"],6),
        "synthetic_equal_payout_odds_not_real_stake":round(best["syn"],6),
        "market_full_inverse_odds_mass":round(best["market_inv_full"],7),
        "market_hypothesis_roi_pct":round(100/best["market_inv_full"],5),
        "components_expected_over_100pct":best["positive"],
        "components_expected_below_100pct":best["negative"],
        "components_expected_90_to_100pct":best["near"],
        "examined":dict(examined)}
    return record,dict(examined)

def main():
    assert START=="2023-01-01" and END=="2023-01-31"
    assert MIN_POINTS==4 and MAX_POINTS==12
    all_events,ingest_audit=obs.ingest()
    train={rid:e for rid,e in all_events.items()
           if e["date"]<="2023-10-31" and weak.score_gate(e)=="ELIGIBLE"}
    assert len(train)==19017,("Original V6 reference train drift",len(train))
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    model=(wg,wr,d5,wl,wc,pw)
    del train
    jan={rid:{k:v for k,v in e.items() if k!="order"}
         for rid,e in all_events.items() if START<=e["date"]<=END}
    assert jan, "No valid January 2023 cards"
    print("V6_JAN2023_FROZEN_MODEL_TRAINED_IN_SAMPLE_JAN",len(jan),flush=True)
    markets={}
    cnt=Counter()
    with gzip.open(ODDS,"rt",encoding="utf-8") as f:
        for line in f:
            row=json.loads(line)
            rid=str(row.get("race_id",""))
            if rid not in jan:continue
            if rid in markets:raise AssertionError(("Duplicated odds record",rid))
            markets[rid]=row
    DIR.mkdir(parents=True,exist_ok=True)
    month_days=Counter()
    with OUT.open("w",encoding="utf-8") as fp:
        for i,(rid,e) in enumerate(sorted(jan.items(),key=lambda x:(x[1]["date"],x[0])),1):
            cnt["jan_valid_v6_card"]+=1
            row=markets.get(rid)
            if row is None:
                cnt["missing_complete_historical_final_price_book"]+=1;continue
            prices,why=parse_market(row,sorted(e["scores"],key=int))
            if prices is None:
                cnt[why]+=1;continue
            cnt["complete_final_odds_and_v6"]+=1
            dist=row_probs(e,*model) # e contains NO "order" or payout
            bet,examined=select(e,dist,prices)
            if bet is None:
                cnt["no_formation_passed_frozen_rules"]+=1
                continue
            bet.update({"race_id":rid,"date":e["date"],
                        "race_type":row.get("race_type"),
                        "odds_phase":"historical_final_NOT_preclose",
                        "v6_training_overlap":True,
                        "decision_does_not_use_result":True})
            fp.write(json.dumps(bet,ensure_ascii=False,separators=(",",":"))+"\n")
            cnt["purchase_races"]+=1
            cnt["purchase_points"]+=bet["points"]
            cnt["has_under_100_component"]+=int(bet["components_expected_below_100pct"]>0)
            month_days[e["date"]]+=1
            if i%250==0:
                print("V6_FORMATION_JAN_PROGRESS",i,dict(cnt),flush=True)
    fingerprint=hashlib.sha256(OUT.read_bytes()).hexdigest()
    summary={
      "state":"SEALED_BEFORE_RESULT_ONLY_FORMATION_PURCHASES_2023_JAN_COMPLETE",
      "period":[START,END],"no_result_in_decision":True,
      "source_is_historical_final_prices_not_preclose":True,
      "source_is_training_overlap":True,
      "market_odds_never_used_to_train_v6":True,
      "formation_score":"(flat_EV - 1.10) * sqrt(V6 formation hit P) * log1p(number of p_i*odds_i >=1)",
      "flat_100_yen_each_point":True,
      "synthetic_odds_are_diagnostic_not_payout":True,
      "settings":{"min_points":MIN_POINTS,"max_points":MAX_POINTS,
                  "min_flat_ev":MIN_FLAT_EV,"min_synthetic":MIN_COMPOSITE,
                  "min_formation_p":MIN_PSET,"min_value_tickets":MIN_VALUE_TICKETS,
                  "min_value_share":MIN_VALUE_SHARE,"max_seeds":MAX_SEEDS},
      "coverage":dict(cnt),"races_per_day":dict(month_days),
      "decisions_sha256":fingerprint,"jan_ingested_races":len(jan),
      "ingest_note":ingest_audit}
    PRE_REPORT.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V6_JAN_2023_PRE_RESULT_SEALED",json.dumps({
        "sha256":fingerprint,"coverage":dict(cnt),"settings":summary["settings"]},
        ensure_ascii=False),flush=True)

if __name__=="__main__":main()

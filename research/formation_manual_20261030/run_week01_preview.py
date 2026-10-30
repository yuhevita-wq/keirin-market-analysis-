#!/usr/bin/env python3
"""Use real Jan 1-7 full odds with frozen V6-C1 probabilities and NEW shapes.

Only a pre-result structural candidate preview; NO auto bet, no ROI backtest.
All results, winning podiums and paid amounts stripped before candidate engine.
Encrypted original remains immutable; price-role 2/3 mapping corrected in
the previously audited Jan W01 loader, verified 367/367 vs official paid.
"""
from __future__ import annotations
import itertools,json,sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.v6_c1_weekly_2023 import simulate_week01 as previous
from research.formation_manual_20261030.adapter import generate_preview,authorize_purchase

OUT=ROOT/"data/audits/v6_c1_manual_week01_pre_result_proposals.json"
SAMPLES=("2820230104020012","4520230107010006","2320230101010009",
         "2320230101020012","2720230101010012","7120230103010008")


def main():
    cards,_=previous.obs.ingest()
    frozen_v6=previous.fit_v6(cards)
    c1=previous.read_model()
    tally=Counter()
    breakdown=Counter()
    examples={}
    checked={}
    for day,rid,market in previous.source_days(tally):
        event=cards.get(rid)
        if not event:
            tally["missing_card"]+=1
            continue
        if previous.weak.score_gate(event)!="ELIGIBLE":
            tally["score_invalid"]+=1
            continue
        # Drop actual finish order *before* a scenario or market evaluation.
        source={k:v for k,v in event.items() if k!="order"}
        if set(source).intersection(("order","outcome","payout","finish_order")):
            raise AssertionError("Result leaked into race input")
        cars=tuple(sorted(source["scores"],key=int))
        prices,trios=previous.normalize_record(market)
        prices,reason=previous.parse_market(
            {"odds":{"3連単":{"-".join(t):v for t,v in prices.items()}}},
            cars)
        if prices is None:
            tally["bad_price_"+str(reason)]+=1
            continue
        if set(prices)!=set(itertools.permutations(cars,3)):
            tally["bad_model_market_grid"]+=1
            continue
        d0=previous.row_probs(source,*frozen_v6)
        d1=previous.apply_calibration(source,d0,c1)
        if set(d1)!=set(prices) or abs(sum(d1.values())-1)>1e-8:
            raise AssertionError(("Frozen C1 grid differs",rid))
        result=generate_preview(source,d1,prices)
        assert authorize_purchase(result)["decision"]=="見送り"
        tally["verified_race_with_manual_proposal"]+=1
        tally["total_structural_hypotheses"]+=result["total_scenario_candidates_generated"]
        tally["total_research_price_compatible_candidates"]+=result["price_floor_compatible_candidates"]
        alternatives=result["alternatives"]
        tally["visible_distinct_scenario_types"]+=len(alternatives)
        if not alternatives:tally["no_price_compatible_scenarios"]+=1
        for a in alternatives:
            breakdown["family_"+a.label]+=1
            if a.points>12:tally["over_12_points_ever_considered"]+=1
            if len(a.forms)>1:tally["multiple_shape_alternative_present"]+=1
        if rid in SAMPLES:
            examples[rid]={
                "date":day,
                "purchase":"見送り（市場誤差の検証なし）",
                "model":"frozen V6-C1",
                "total_scenario_copies":result["total_scenario_candidates_generated"],
                "top_structural_alternatives":[
                    {"family":a.label,"reason":a.explanation,
                     "formations":list(a.forms),
                     "points":a.points,
                     "covered_race_probability_pct":round(100*a.estimated_hit_probability,5),
                     "unverified_model_roi_pct":round(a.estimated_roi_pct,5),
                     "synthetic_odds":round(a.synthetic_odds,4)}
                    for a in alternatives[:6]]
            }
    report={
       "status":"W01_NEW_MANUAL_ENGINE_RAN_ON_REAL_FROZEN_C1_AND_CORRECTED_ARCHIVE",
       "period":["2023-01-01","2023-01-07"],
       "old_select_5_shapes_used":False,
       "no_results_in_candidate_generation":True,
       "no_auto_purchases":True,
       "no_market_odds_as_training_labels":True,
       "original_models_unchanged":True,
       "candidate_generator_independent_of_outcomes":True,
       "source_odds_have_rank_23_corrected":True,
       "price_evaluation_only_after_formation_construction":True,
       "coverage":dict(tally),
       "scenario_family_counts":dict(breakdown),
       "observed_examples":examples,
       "warning":"Price differences are NOT proven bets. Manual teaches structure, not a profitable threshold. In-sample V6/C1, retrospective final market prices."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("W01_MANUAL_FORMATION_PREVIEW",json.dumps({
       "coverage":dict(tally),
       "family_counts":dict(breakdown),
       "examples":examples
    },ensure_ascii=False),flush=True)
if __name__=="__main__":main()

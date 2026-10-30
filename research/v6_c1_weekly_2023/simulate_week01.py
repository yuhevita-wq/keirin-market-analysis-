#!/usr/bin/env python3
"""JAN 2023 W01 MAIN RUNNER -- manual-based scenario formations (not legacy).

This file REPLACES the five-shape selector in the executable weekly entrypoint
on the standalone research branch. The original script is preserved verbatim
as simulate_week01_legacy.py, which provides original frozen models, authenticated
7-bet encrypted source, and independently archived official payout functions.

Rule: do NOT change V6 or V6-C1, never auto-buy all races. Formation sets are
generated from pre-race line positions and frozen C1 model rank probabilities,
then whole-set prices are checked. Shadow portfolios are settled ONLY AFTER
SHA-256 sealing pre-result candidates. NO actual purchases are claimed. No
odds grids or encryption secrets are exported.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.v6_c1_weekly_2023 import simulate_week01_legacy as old
from research.formation_manual_20261030.adapter import generate_preview,authorize_purchase

START="2023-01-01"
END="2023-01-07"
OUT=ROOT/"data/audits/v6_c1_weekly_2023w01_MANUAL_MAIN_report.json"
SEALED=ROOT/"data/audits/v6_c1_weekly_2023w01_MANUAL_MAIN_candidates_sealed.json"


def _simple_candidate(x):
    return {
        "scenario":x.label,
        "explanation":x.explanation,
        "forms":list(x.forms),
        "points":x.points,
        "overlap_points":x.overlap_points,
        "tickets":["-".join(t) for t in sorted(x.ticket_set)],
        "model_formation_hit_probability_pct":round(100*x.estimated_hit_probability,6),
        "model_formation_expected_roi_pct":round(x.estimated_roi_pct,6),
        "synthetic_composite_odds":round(x.synthetic_odds,6),
        "stake_yen":x.stake_yen,
        "single_ticket_est_ev_under_100_count":x.low_EV_components,
    }


def _forecast_rules(event,dist,prices):
    """Real weekly entrypoint -- OLD select() IS NOT CALLED."""
    if "order" in event or "payout" in event:
        raise AssertionError("Truth entered forecast phase")
    p=generate_preview(event,dist,prices)
    if authorize_purchase(p)["decision"]!="見送り":
        raise AssertionError("Unauthorized automatic purchase")
    alternatives=p["alternatives"]
    # A deterministic SHADOW selection for research, not a purchase:
    # the first package under the existing coverage-first ordering, rather
    # than finding a ticket with maximum individual market p*odds.
    shadow=alternatives[0] if alternatives else None
    return p,shadow


def main():
    assert START=="2023-01-01" and END=="2023-01-07"
    events,ingest_audit=old.obs.ingest()
    model=old.fit_v6(events)  # frozen reference, with its original training split
    calibrator=old.read_model()  # frozen full-period V6-C1 calibrator
    cnt=Counter()
    daily=defaultdict(Counter)
    decisions=[]
    shadow_quote={}
    for day,rid,book in old.source_days(cnt):
        cnt["source_complete_book"]+=1
        event=events.get(rid)
        if event is None:
            cnt["no_original_event_card"]+=1
            continue
        if old.weak.score_gate(event)!="ELIGIBLE":
            cnt["score_card_not_eligible"]+=1
            continue
        if event["date"]!=day:
            cnt["race_date_mismatch"]+=1
            continue
        source={k:v for k,v in event.items() if k!="order"}
        if "order" in source:raise AssertionError("Race result leakage")
        cars=sorted(source["scores"],key=int)
        ordered,trios=old.normalize_record(book)
        prices,why=old.parse_market({"odds":{"3連単":{
            "-".join(ticket):value for ticket,value in ordered.items()}}},cars)
        if prices is None:
            cnt["invalid_price_grid_"+str(why)]+=1
            continue
        if set(prices)!=set(itertools.permutations(cars,3)):
            raise AssertionError(("Trifecta universe incomplete",rid))
        p_v6=old.row_probs(source,*model)
        p_c1=old.apply_calibration(source,p_v6,calibrator)
        if set(p_c1)!=set(prices) or abs(sum(p_c1.values())-1)>1e-8:
            raise AssertionError(("Frozen model alignment failed",rid))
        preview,shadow=_forecast_rules(source,p_c1,prices)
        cnt["valid_weekly_model_and_price_race"]+=1
        daily[day]["valid_weekly_model_and_price_race"]+=1
        cnt["candidate_scenarios_generated"]+=preview["total_scenario_candidates_generated"]
        cnt["candidate_price_compatible"]+=preview["price_floor_compatible_candidates"]
        if not shadow:
            cnt["no_3p5_synthetic_price_compatible_candidate"]+=1
        if shadow and shadow.points>12:
            cnt["shadow_more_than_12_points"]+=1
        if shadow and len(shadow.forms)>1:
            cnt["shadow_composed_of_multiple_formation_shapes"]+=1
        # Do not turn a plausible looking projected return into an admission.
        admitted=authorize_purchase(preview)
        if admitted["decision"]!="見送り":raise AssertionError("all-buy regression")
        item={
            "race_id":rid,"date":day,
            "model":"FROZEN_V6_C1",
            "selection_engine":"LIBRARY_MANUAL_40P_FAMILY_ENGINE_V1",
            "decision":"見送り",
            "reason":"実測で固定条件検証された購入判定が存在しない",
            "candidate_count":preview["total_scenario_candidates_generated"],
            "eligible_candidate_count":preview["price_floor_compatible_candidates"],
            "shadow_scenario_for_research":_simple_candidate(shadow) if shadow else None,
            "alternative_shapes":[_simple_candidate(x) for x in preview["alternatives"][:8]],
            "no_result_in_selection":True,
            "manual_multi_shape_not_cross_product":True,
            "old_five_shape_ev_seeds_not_used":True,
        }
        decisions.append(item)
        # Hold only the winning price per hypothetical tickets IN MEMORY.
        if shadow:
            shadow_quote[rid]={
                "-".join(t):prices[t] for t in shadow.ticket_set
            }
    # A seal is recorded BEFORE any actual outcome or official payment is read.
    payload=json.dumps(decisions,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    seal=hashlib.sha256(payload.encode("utf-8")).hexdigest()
    SEALED.parent.mkdir(parents=True,exist_ok=True)
    SEALED.write_text(payload+"\n",encoding="utf-8")
    print("MANUAL_WEEK01_CANDIDATE_DECISIONS_PRE_RESULT_SEALED",
          json.dumps({"sha256":seal,"races":len(decisions),
                      "all_default_skip":all(x["decision"]=="見送り" for x in decisions),
                      "source_audit":dict(cnt)},ensure_ascii=False),flush=True)

    # Shadow assessment is DIAGNOSTIC, NOT a purchase/backtest profit claim.
    required={x["race_id"] for x in decisions if x["shadow_scenario_for_research"]}
    paid,conflicts,payout_sources=old.payout.load_settled_payments(required)
    shadow_stats=Counter()
    bytype=defaultdict(Counter)
    for item in decisions:
        cand=item["shadow_scenario_for_research"]
        if not cand:continue
        rid=item["race_id"]
        actual="-".join(events[rid]["order"][:3])
        hit=actual in cand["tickets"]
        q=shadow_quote[rid]
        quote_amount=round(100*q[actual]) if hit else 0
        official_amt=None
        if hit and rid not in conflicts and rid in paid and "-".join(paid[rid][0])==actual:
            official_amt=paid[rid][1]
            if abs(official_amt-quote_amount)>1:
                raise AssertionError(("Corrected market quote disagrees with official result",rid))
        shadow_stats["races"]+=1
        shadow_stats["points"]+=cand["points"]
        shadow_stats["hypothetical_stake_yen"]+=cand["stake_yen"]
        shadow_stats["hit"]+=int(hit)
        shadow_stats["estimated_hit_sum_x_100000"]+=round(100000*cand["model_formation_hit_probability_pct"]/100)
        shadow_stats["hypothetical_payout_from_verified_final_quote_yen"]+=quote_amount
        shadow_stats["verified_official_hits"]+=int(hit and official_amt is not None)
        shadow_stats["unverified_official_paid_hits"]+=int(hit and official_amt is None)
        bytype[cand["scenario"]]["races"]+=1
        bytype[cand["scenario"]]["hits"]+=int(hit)
        bytype[cand["scenario"]]["stake"]+=cand["stake_yen"]
        bytype[cand["scenario"]]["price_proxy_payout"]+=quote_amount
        item["shadow_research_actual_hit"]=hit
        item["shadow_research_official_payout_verified"]=official_amt if hit else None
        # No actual race bets are authorized.
    report={
        "status":"MANUAL_2023_W01_WEEKLY_MAIN_EXECUTED",
        "source_archive":"authenticated_seven_bet_2023_Jan1_7_corrected_rank23",
        "original_model_v6_unchanged":True,
        "original_model_v6_c1_unchanged":True,
        "old_select_function_called":False,
        "new_engine_is_main_weekly_program":True,
        "no_outcome_in_candidate_generator":True,
        "zero_auto_purchase":True,
        "real_money_bet_records":0,
        "all_prices_are_retrospective_final_quotes":True,
        "training_overlap_2023_Jan":True,
        "no_claim_of_market_profit":True,
        "period":[START,END],
        "sealed_prior_to_settlement_sha256":seal,
        "all_candidate_decisions_written_before_actual_result":True,
        "source_stats":dict(cnt),
        "daily_coverage":{d:dict(x) for d,x in sorted(daily.items())},
        "shadow_performance_NOT_actually_purchased":dict(shadow_stats),
        "shadow_by_scenario_NOT_actually_purchased":{k:dict(v) for k,v in sorted(bytype.items())},
        "official_payout_source_audit":payout_sources,
        "review_examples":[
            {k:v for k,v in row.items() if k not in ("alternative_shapes",)}
            for row in decisions if row["race_id"] in {
                "2820230104020012","4520230107010006","2320230101010009",
                "2320230101020012","2720230101010012","7120230103010008"}
        ],
        "limitations":[
            "Manual supplies valid strategic ticket structures, NOT empirically proven market arbitrage.",
            "All model probability predictions for Jan 2023 were learned with overlapping training.",
            "All historical odds are after-race FINAL; preclosing odds may differ.",
            "Shadow 'ROI' is diagnostic and no ticket was issued, approved or bought.",
            "No new validated selective buy/not-buy criterion has been fitted."
        ],
    }
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MANUAL_WEEK01_MAIN_REPLACEMENT_RESULT",json.dumps({
        "status":report["status"],"source":report["source_stats"],
        "shadow":report["shadow_performance_NOT_actually_purchased"],
        "family":report["shadow_by_scenario_NOT_actually_purchased"],
        "example_count":len(report["review_examples"]),
        "sha256":seal,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

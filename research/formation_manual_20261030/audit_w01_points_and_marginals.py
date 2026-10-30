#!/usr/bin/env python3
"""Jan W01 structural bet audit, frozen decisions, no backfit.

Two independent observations:
1. Predetermined 100JPY/combination shadow buys grouped by point size.
2. Structural contributions: for a multi-form package, first form (core)
   versus unique tickets introduced by second form; for a single-form
   package, largest strict nested subset of SAME scenario family if one
   was already generated before results. Also show below/above EV 1.0
   contributions as DIAGNOSTIC only (never cut tickets ad hoc).

Entire candidate list is rebuilt from ORIGINAL frozen model and audited
sealed encrypted odds. SHA256 verifies exact historical pre-result selection.
Settling is only after all decisions fixed. Original models untouched.
"""
from __future__ import annotations
import hashlib,itertools,json,sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.v6_c1_weekly_2023 import simulate_week01 as w
from research.formation_manual_20261030.formation import from_manual_notation
from research.formation_manual_20261030.scenarios import construct_candidates

PERIOD=("2023-01-01","2023-01-07")
EXPECTED_SEAL="e257f93b7d981d5a9df3bc73a416b40709c06a6f13a1cdd856c3e47c036bd497"
OUT=ROOT/"data/audits/v6_c1_w01_manual_point_vs_structure_audit.json"
DETAIL=ROOT/"data/audits/v6_c1_w01_manual_point_vs_structure_per_race.json"

def band(n):
    if n<=4:return "01-04"
    if n<=8:return "05-08"
    if n<=12:return "09-12"
    if n<=16:return "13-16"
    if n<=20:return "17-20"
    return "21_plus"

def add_layer(bucket,tickets,dist,odds,outcome,official,validated):
    if not tickets:return
    n=len(tickets)
    bucket["races_with_layer"]+=1
    bucket["points"]+=n
    bucket["stake_yen"]+=100*n
    bucket["model_expected_hits"]+=sum(dist[t] for t in tickets)
    bucket["model_expected_payout_yen"]+=100*sum(dist[t]*odds[t] for t in tickets)
    hit=outcome in tickets
    bucket["winning_tickets"]+=int(hit)
    pay=round(100*odds[outcome]) if hit else 0
    if hit and validated is not None:
        if abs(pay-validated)>1:raise AssertionError(("quote/official mismatch",pay,validated))
        bucket["wins_official_paid_verified"]+=1
    if hit and validated is None:bucket["wins_with_official_paid_unavailable"]+=1
    bucket["price_based_paid_yen"]+=pay
    bucket["price_based_profit_yen"]+=pay-100*n
    bucket["ticket_EV_ge_100"]+=sum(dist[t]*odds[t]>=1.0 for t in tickets)
    bucket["ticket_EV_lt_100"]+=sum(dist[t]*odds[t]<1.0 for t in tickets)

def fmt(group):
    obj={}
    for key,z in sorted(group.items()):
        z=dict(z)
        n=z.get("races_with_layer",0);stake=z.get("stake_yen",0)
        for name in ("model_expected_hits","model_expected_payout_yen"):
            if name in z:z[name]=round(z[name],6)
        z["price_proxy_roi_pct"]=round(z.get("price_based_paid_yen",0)*100/stake,5) if stake else None
        z["price_proxy_net_yen"]=z.get("price_based_paid_yen",0)-stake
        z["avg_points_per_race"]=round(z.get("points",0)/n,4) if n else None
        z["model_expected_roi_pct"]=round(z.get("model_expected_payout_yen",0)*100/stake,5) if stake else None
        obj[key]=z
    return obj

def main():
    events,_=w.old.obs.ingest()
    model=w.old.fit_v6(events)
    calibrator=w.old.read_model()
    audit=Counter()
    selection=[]
    cache={}
    for day,rid,book in w.old.source_days(audit):
        event=events.get(rid)
        if event is None or w.old.weak.score_gate(event)!="ELIGIBLE" or event["date"]!=day:
            continue
        source={k:v for k,v in event.items() if k!="order"}
        cars=tuple(sorted(source["scores"],key=int))
        ordered,_=w.old.normalize_record(book)
        odds,why=w.old.parse_market({"odds":{"3連単":{
             "-".join(t):v for t,v in ordered.items()}}},cars)
        if odds is None:raise AssertionError(("Full source failed",rid,why))
        p0=w.old.row_probs(source,*model)
        p=w.old.apply_calibration(source,p0,calibrator)
        preview,shadow=w._forecast_rules(source,p,odds)
        row={
            "race_id":rid,"date":day,
            "model":"FROZEN_V6_C1",
            "selection_engine":"LIBRARY_MANUAL_40P_FAMILY_ENGINE_V1",
            "decision":"見送り",
            "reason":"実測で固定条件検証された購入判定が存在しない",
            "candidate_count":preview["total_scenario_candidates_generated"],
            "eligible_candidate_count":preview["price_floor_compatible_candidates"],
            "shadow_scenario_for_research":w._simple_candidate(shadow) if shadow else None,
            "alternative_shapes":[w._simple_candidate(x) for x in preview["alternatives"][:8]],
            "no_result_in_selection":True,
            "manual_multi_shape_not_cross_product":True,
            "old_five_shape_ev_seeds_not_used":True,
        }
        selection.append(row)
        if shadow:
            cache[rid]=(source,cars,p,odds,shadow)
    payload=json.dumps(selection,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    seal=hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if seal!=EXPECTED_SEAL:
        raise AssertionError(("STOP: frozen pre-result selection SHA256 diverged",seal,EXPECTED_SEAL))
    print("W01_POINT_AUDIT_FIXED_PRE_RESULT_SELECTION_SHA256_OK",seal,flush=True)
    # ONLY NOW read the true result and official winning payout archive.
    wanted=set(cache)
    official,conflicts,sourceaudit=w.old.payout.load_settled_payments(wanted)

    bands=defaultdict(lambda:defaultdict(float))
    evgroups=defaultdict(lambda:defaultdict(float))
    mixed=defaultdict(lambda:defaultdict(float))
    nested=defaultdict(lambda:defaultdict(float))
    overall=defaultdict(lambda:defaultdict(float))
    race_rows=[]
    for rid,(source,cars,p,odds,shadow) in sorted(cache.items()):
        out=tuple(events[rid]["order"][:3])
        if out not in odds:raise AssertionError(("Actual finish outside ordered book",rid))
        winner="-".join(out)
        paid=official.get(rid)
        verified=None
        if paid and rid not in conflicts and tuple(map(str,paid[0]))==out:
            verified=paid[1]
        elif paid and tuple(map(str,paid[0]))!=out:
            raise AssertionError(("Official payout result disagrees with event",rid))
        all_t=frozenset(shadow.ticket_set)
        if len(all_t)!=shadow.points:raise AssertionError(("Selected points differ",rid))
        add_layer(overall["full"],all_t,p,odds,out,paid,verified)
        add_layer(bands[band(len(all_t))],all_t,p,odds,out,paid,verified)
        low=frozenset(t for t in all_t if p[t]*odds[t]<1)
        high=all_t-low
        add_layer(evgroups["EV_lt_100pct"],low,p,odds,out,paid,verified)
        add_layer(evgroups["EV_ge_100pct"],high,p,odds,out,paid,verified)
        core=None;addition=None;layer_kind=None
        if len(shadow.forms)>1:
            # Literal first full formation is the control. The next is an
            # additive second structural hypothesis, with exact overlaps
            # removed rather than an accidental Cartesian product.
            core=from_manual_notation(cars,shadow.forms[0]).tickets
            if not core.issubset(all_t):
                raise AssertionError(("multi component not subset",rid))
            addition=all_t-core
            layer_kind="second_structural_form_without_overlap"
            add_layer(mixed["base_first_structural_form"],core,p,odds,out,paid,verified)
            add_layer(mixed["additional_second_form_unique_tickets"],addition,p,odds,out,paid,verified)
            add_layer(mixed["full_both_forms_combined"],all_t,p,odds,out,paid,verified)
            if core & addition:raise AssertionError("overlap double counted")
        else:
            # For SINGLE-SHAPE selections, use only a strict subset that
            # the exact SAME pre-result generator created under SAME family.
            options=construct_candidates(cars,p,source["positions"],max_scenario_groups=160)
            same=[x.pack.tickets for x in options
                  if x.scenario==shadow.label and x.pack.tickets < all_t]
            if same:
                core=max(same,key=lambda x:(len(x),sum(p[t] for t in x),sorted(x)))
                addition=all_t-core
                layer_kind="same_family_nested_expansion"
                add_layer(nested["base_same_family_smaller_shape"],core,p,odds,out,paid,verified)
                add_layer(nested["added_tickets_same_family"],addition,p,odds,out,paid,verified)
                add_layer(nested["full_same_family_larger_shape"],all_t,p,odds,out,paid,verified)
        if core is not None and addition is not None and (core|addition)!=all_t:
            raise AssertionError(("non-partition",rid))
        actual_hit=out in all_t
        payment=round(100*odds[out]) if actual_hit else 0
        race_rows.append({
            "race_id":rid,"date":source["date"],"shape_type":shadow.label,
            "formations":list(shadow.forms),
            "points":shadow.points,
            "picks_hit":actual_hit,
            "actual_ordered_trifecta":"-".join(out),
            "official_paid_match_checked":actual_hit and verified is not None,
            "official_paid_missing_for_hit":actual_hit and verified is None,
            "model_set_hit_probability_pct":round(100*shadow.estimated_hit_probability,6),
            "model_set_roi_pct":round(shadow.estimated_roi_pct,6),
            "model_expected_ev_under100_tickets":len(low),
            "model_expected_ev_ge100_tickets":len(high),
            "ticket_ev_under100_stake_yen":100*len(low),
            "amount_spent_yen":100*shadow.points,
            "price_proxy_payout_yen":payment,
            "price_proxy_profit_yen":payment-100*shadow.points,
            "structural_expansion_kind":layer_kind,
            "structural_core_points":len(core) if core is not None else None,
            "structural_added_points":len(addition) if addition is not None else None,
            "structural_core_hit":out in core if core is not None else None,
            "structural_addition_hit":out in addition if addition is not None else None,
            "structural_added_predicted_hit_prob_pct":round(100*sum(p[t] for t in addition),6) if addition is not None else None,
            "structural_added_model_expected_return_pct":round(100*sum(p[t]*odds[t] for t in addition)/len(addition),6) if addition else None,
            "structural_added_price_payout_yen":round(100*odds[out]) if addition is not None and out in addition else 0
        })
    if len(race_rows)!=444:raise AssertionError(("Not original 444 selected cases",len(race_rows)))
    top=fmt(overall)["full"]
    if top["points"]!=5748 or top["winning_tickets"]!=81 or top["price_based_paid_yen"]!=330260:
        raise AssertionError(("original test numerical anchors mismatched",top))
    if sum(r["points"] for r in race_rows)!=top["points"]:
        raise AssertionError("point duplication")
    for key in ("points","stake_yen","price_based_paid_yen","price_based_profit_yen",
                "model_expected_payout_yen","model_expected_hits"):
        x=sum(b[key] for b in bands.values())
        if abs(x-overall["full"][key])>1e-3:
            raise AssertionError(("band decomposition invalid",key,x,overall["full"][key]))
        x=sum(e[key] for e in evgroups.values())
        if abs(x-overall["full"][key])>1e-3:
            raise AssertionError(("EV decomposition invalid",key,x,overall["full"][key]))
    result={
       "status":"W01_MANUAL_444_FIXED_FORMATIONS_MARGINAL_ACCOUNTING_FINISHED",
       "period":PERIOD,
       "selected_pre_result_sha256":seal,
       "original_model_and_selection_unchanged":True,
       "not_actual_purchases":True,
       "all_price_based_payouts_are_final_quote_proxies":True,
       "missing_official_payouts_are_separately_counted":True,
       "not_backtest_rule_optimization":True,
       "one_hundred_yen_per_unique_ticket":True,
       "point_bands":fmt(bands),
       "ticket_level_EV_diagnostic_only_not_a_purchase_rule":fmt(evgroups),
       "multi_form_second_shape_contribution":fmt(mixed),
       "same_scenario_type_nested_subshape_contribution":fmt(nested),
       "totals":fmt(overall),
       "source_audit":dict(audit),
       "official_source_audit":sourceaudit,
       "limitations":[
          "2023 W01 is in sample for V6 and V6-C1.",
          "Retrospective final odds are not pre-closing executable odds.",
          "16 of original 81 shadow wins lacked independently archived official payout; final corrected price used as a separately disclosed proxy.",
          "Ticket EV <100% vs >=100% is model-based and poorly calibrated, NOT an approved selective ticket deletion algorithm.",
          "First structural form is the multi-shape control, not uniquely objectively optimal; second form is incremental for paired selected package.",
          "Same-family nested comparisons are observational; they are paired subset contrasts, NOT a tuned purchase strategy.",
          "Fixed research shadow scenario selected by estimated mass; zero actual adopted bet records."
       ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    DETAIL.write_text(json.dumps(race_rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("W01_POINTS_AND_ADDED_TICKETS_FINAL",json.dumps({
        "total":result["totals"],
        "point_bands":result["point_bands"],
        "ticket_EV":result["ticket_level_EV_diagnostic_only_not_a_purchase_rule"],
        "second_form":result["multi_form_second_shape_contribution"],
        "nested":result["same_scenario_type_nested_subshape_contribution"],
        "n":len(race_rows),
        "sha256":seal},ensure_ascii=False),flush=True)

if __name__=="__main__":main()

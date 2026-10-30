#!/usr/bin/env python3
"""Audit Feb-Apr 2023 on VERIFIED encrypted 7-bet odds, compare old all-buy to new gate.

- NEVER writes or uploads raw decrypted odds. Uses repository Actions secret.
- All FULL race books from every verified ciphertext archive per day (retries
  unioned by exact race ID, contradictory quotes quarantined).
- Original V6 and saved V6-C1 trained models unchanged.
- Formation construction exactly original Jan code; candidate gate separate.
- All selections sealed BEFORE the payout source is opened.
- Actual result never passed to row_probs, select or gate.
- This 2023 period is IN SAMPLE for BOTH V6 and V6-C1, not future profit proof.
- Fixed rules, cannot optimize using same reported Feb-Apr historical outcomes.
"""
from __future__ import annotations
import hashlib,itertools,json,sys,tempfile
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
for suffix in ("","research/independent_trifecta_2023",
               "research/independent_trifecta_2024",
               "research/v6_january_2023_formation_ev",
               "research/v6_c1_full_data",
               "research/deletion_ai",
               "research/v6_c1_full_odds_gate"):
    sys.path.insert(0,str(ROOT/suffix))
from research.v6_january_2023_formation_ev.prepare import select,parse_market
from research.v6_c1_full_data.apply import apply_calibration,read_model
from research.v6_c1_full_odds_gate.admission import decide
import research.deletion_ai.secure_seven_bet_bridge as bridge
import observe as obs
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import v5_rider_score_role_size_interactions as v5
import v6_opponent_competition_structure as v6
import weak_rider_podium_zero_excluded as weak
from v8_targeted_weak_podium_calibration import row_probs
import v6_settled_payout_only_2023_2026 as payout

START="2023-02-01"
END="2023-04-30"
OUT=ROOT/"data/audits/v6_c1_full_vault_purchase_gate_feb_apr_2023.json"
MODELS=("old_v6_all_buy","old_c1_all_buy","c1_new_gate")
STAKE=100

def verify_sources():
    partial=json.loads((ROOT/"data/audits/odds_only_export.json").read_text())
    sealed=json.loads((ROOT/"data/audits/odds_seven_bet_archive_count.json").read_text())
    assert partial["years"]["2023"]["complete_races"]==6301
    assert sealed["year"]["2023"]["confirmed_distinct_race_lower_bound"]>=25085
    return {"odds_only_2023_complete":6301,
            "ciphertext_2023_confirmed_race_lower_bound":
              sealed["year"]["2023"]["confirmed_distinct_race_lower_bound"]}

def normalize_record(rec):
    # Raw encrypted markets validated by original bridge as full SEVEN-bet
    # books. Only 3連単/3連複 are needed in this specific trial.
    rows=rec["odds"]
    prices={tuple(k.split("-")):v for k,v in rows["3連単"].items()}
    trios={}
    for raw,price in rows["3連複"].items():
        key=tuple(sorted(raw.split("="),key=int))
        if key in trios:raise ValueError(("duplicate trio",key))
        trios[key]=price
    return prices,trios

def race_month(dt):return dt[:7]

def source_days(stats):
    # The secret is passed ONLY into the ephemeral runner, stripped from env,
    # and written to a chmod 0600 temporary file. Never printed or persisted.
    src,_=bridge.catalog(START,END)
    assert len(src)==89,("Sealed source calendar incomplete",len(src))
    tempkey,key=bridge.secure_key_temp()
    try:
        for day,files in src.items():
            union={};conflicted=set()
            with tempfile.TemporaryDirectory(prefix="keirin_v6_c1_private_day_") as work:
                for file,sha in files:
                    verified=bridge.decrypt_sealed(file,sha,key,work)
                    stats["sealed_files_verified"]+=1
                    stats["raw_books_including_retries"]+=len(verified)
                    for rid,items in verified.items():
                        if rid in conflicted:continue
                        val=sorted(items)
                        if rid in union and union[rid]!=val:
                            del union[rid]; conflicted.add(rid)
                        else:union[rid]=val
            stats["days_loaded"]+=1
            stats["conflicting_duplicate_odds_quarantined"]+=len(conflicted)
            stats["deduplicated_sealed_full_books"]+=len(union)
            for rid,rows in sorted(union.items()):
                normalized,reason=bridge.normalize_book(rid,day,rows)
                if reason:
                    stats["source_rejected_"+reason]+=1
                    continue
                stats["source_eligible_books"]+=1
                yield day,rid,normalized
            if stats["days_loaded"]%10==0:
                print("V6_C1_SECURE_ARCHIVE_PROGRESS",json.dumps({
                  "days":stats["days_loaded"],
                  "sealed_complete_books":stats["deduplicated_sealed_full_books"],
                  "source_eligible_books":stats["source_eligible_books"],
                  "candidate_races":stats["matched_full_V6_card"]},ensure_ascii=False),flush=True)
    finally:
        tempkey.cleanup()

def fit_v6(all_events):
    train={rid:e for rid,e in all_events.items()
           if e["date"]<="2023-10-31" and weak.score_gate(e)=="ELIGIBLE"}
    if len(train)!=19017:raise AssertionError(("Original V6 training drift",len(train)))
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    return (wg,wr,d5,wl,wc,pw)

def simplify(name,rid,day,entry,gate_report=None):
    if entry is None:return None
    # No decrypted source prices are saved in this decision record.
    res={"race_id":rid,"date":day,"model":name,
         "tickets":entry["tickets"],"formation":entry["formation"],
         "points":entry["points"],"stake_yen":entry["stake_yen"],
         "model_predicted_flat_roi_pct":entry["v6_flat_stake_expected_roi_pct"],
         "model_predicted_formation_hit_probability_pct":entry["v6_formation_probability_pct"]}
    if gate_report is not None:
        res["admission_reason"]=gate_report["reason"]
        res["supported_tickets"]=gate_report["supported_tickets"]
        res["distinct_supported_trios"]=gate_report["distinct_supported_trios"]
    return res

def outcome_audit(all_events,decisions,quoted):
    # This call is executed AFTER every candidate formation was independently
    # selected and a SHA256 seal recorded. Source itself requires verified
    # official payout match. No label enters selection/model.
    wanted={d["race_id"] for allrows in decisions.values() for d in allrows}
    official,conflicts,source=payout.load_settled_payments(wanted)
    summary={};month={};hits_by_model={}
    for name,rows in decisions.items():
        stats=Counter();bym=defaultdict(Counter)
        full_price_proxy=Counter();proxy_month=defaultdict(Counter)
        largest=[]
        for d in rows:
            rid=d["race_id"];e=all_events.get(rid)
            if e is None:raise AssertionError(("No result card",rid))
            true="-".join(e["order"][:3])
            winner=true in d["tickets"]
            quote=quoted[name][rid]
            proxy_paid=round(100*float(quote[true])) if winner else 0
            for proxy_stats in (full_price_proxy,proxy_month[race_month(d["date"])]):
                proxy_stats["races"]+=1
                proxy_stats["stake_yen"]+=d["stake_yen"]
                proxy_stats["payout_yen"]+=proxy_paid
                proxy_stats["hits"]+=int(winner)
                proxy_stats["ticket_points"]+=d["points"]
            status="verified_loss";paid=0
            if winner:
                if rid in conflicts:status="official_payout_conflict"
                elif rid not in official:status="official_payout_missing"
                elif "-".join(official[rid][0])!=true:status="official_podium_disagreement"
                else:
                    paid=official[rid][1];status="verified_win"
                    stats["verified_payout_comparison_count"]+=1
                    if abs(paid-proxy_paid)>1:
                        stats["verified_payout_does_not_match_final_quote"]+=1
            z=stats
            z["all_selected"]+=1
            z["settle_"+status]+=1
            if status.startswith("verified_"):
                for obj in (z,bym[race_month(d["date"])]):
                    obj["races"]+=1
                    obj["stake_yen"]+=d["stake_yen"]
                    obj["payout_yen"]+=paid
                    obj["hits"]+=int(paid>0)
                    obj["ticket_points"]+=d["points"]
                if paid>0:largest.append((paid,rid,d["date"],d["formation"]))
        def report(z):
            stake=z["stake_yen"]
            return {"races":z["races"],"hits":z["hits"],"points":z["ticket_points"],
                    "stake_yen":stake,"payout_yen":z["payout_yen"],
                    "profit_yen":z["payout_yen"]-stake,
                    "roi_pct":round(100*z["payout_yen"]/stake,5) if stake else None,
                    "hit_pct":round(100*z["hits"]/z["races"],5) if z["races"] else None}
        summary[name]={"overall":report(stats),
                       "historical_final_quote_proxy_all_selected":report(full_price_proxy),
                       "official_payout_matches_final_quote_pct":round(100*(stats["verified_payout_comparison_count"]-stats["verified_payout_does_not_match_final_quote"])/stats["verified_payout_comparison_count"],4) if stats["verified_payout_comparison_count"] else None,
                       "official_payout_quote_mismatches":stats["verified_payout_does_not_match_final_quote"],
                       "settlement_status":{k:v for k,v in stats.items() if k.startswith("settle_")},
                       "top5_paid":[{"payout_yen":v,"race_id":rid,"date":day,"formation":f}
                                    for v,rid,day,f in sorted(largest,reverse=True)[:5]]}
        month[name]={dt:report(v) for dt,v in sorted(bym.items())}
        summary[name]["historical_final_quote_proxy_by_month"]={dt:report(v) for dt,v in sorted(proxy_month.items())}
        hits_by_model[name]={d["race_id"] for d in rows if "-".join(all_events[d["race_id"]]["order"][:3]) in d["tickets"]}
    comparisons={
       "old_c1_vs_new_gate":{
           "purchase_races_eliminated":summary["old_c1_all_buy"]["overall"]["races"]-summary["c1_new_gate"]["overall"]["races"],
           "gate_successful_purchases":summary["c1_new_gate"]["overall"]["races"],
           "changed_selection_only":True
       },
       "C1_minus_original_V6_profit_yen":summary["old_c1_all_buy"]["overall"]["profit_yen"]-summary["old_v6_all_buy"]["overall"]["profit_yen"]}
    return summary,month,comparisons,source

def main():
    print("V6_C1_SOURCE_AUDIT",json.dumps(verify_sources()),flush=True)
    read_model()
    all_events,_=obs.ingest()
    original_v6=fit_v6(all_events)
    calibrator=read_model()
    counters=Counter();by_month=defaultdict(Counter)
    decisions={model:[] for model in MODELS}
    # Keep only bought ticket quotes in ephemeral RAM, never persist prices.
    quoted={model:{} for model in MODELS}
    seen_ids=set()
    for day,rid,book in source_days(counters):
        m=race_month(day)
        by_month[m]["sealed_verified_races"]+=1
        if rid in seen_ids:raise AssertionError(("Duplicate across days",rid))
        seen_ids.add(rid)
        event=all_events.get(rid)
        if not event:
            counters["source_no_matching_V6_card"]+=1
            by_month[m]["source_no_matching_V6_card"]+=1
            continue
        if weak.score_gate(event)!="ELIGIBLE":
            counters["source_invalid_score_card"]+=1
            by_month[m]["source_invalid_score_card"]+=1
            continue
        if day!=event["date"]:
            counters["source_race_date_conflict"]+=1
            continue
        card={k:v for k,v in event.items() if k!="order"}
        cars=sorted(card["scores"],key=int)
        prices,triop=normalize_record(book)
        # Never invent odds for a model's missing/extra rider or valid race with cancellations.
        if set(prices)!=set(itertools.permutations(cars,3)):
            counters["source_race_cards_different_rider_universe"]+=1
            continue
        # Original legacy market parser already enforces numeric odds and complete grid.
        prices,why=parse_market({"odds":{"3連単":{"-".join(k):v for k,v in prices.items()}}},cars)
        if prices is None:
            counters["source_invalid_legacy_trifecta_odds_"+why]+=1
            continue
        counters["matched_full_V6_card"]+=1
        by_month[m]["matched_full_V6_card"]+=1
        p0=row_probs(card,*original_v6)
        p1=apply_calibration(card,p0,calibrator)
        if set(p0)!=set(prices) or set(p1)!=set(prices):
            raise AssertionError(("Model prices mismatch",rid))
        b0,_=select(card,p0,prices)
        b1,_=select(card,p1,prices)
        if b0:
            decisions["old_v6_all_buy"].append(simplify("old_v6_all_buy",rid,day,b0))
            quoted["old_v6_all_buy"][rid]={t:prices[tuple(t.split("-"))] for t in b0["tickets"]}
            by_month[m]["old_v6_all_buy"]+=1
        if b1:
            decisions["old_c1_all_buy"].append(simplify("old_c1_all_buy",rid,day,b1))
            quoted["old_c1_all_buy"][rid]={t:prices[tuple(t.split("-"))] for t in b1["tickets"]}
            by_month[m]["old_c1_all_buy"]+=1
            ok,reason=decide(b1,p0,p1,prices,triop)
            counters["gate_"+reason["reason"]]+=1
            if ok:
                decisions["c1_new_gate"].append(simplify("c1_new_gate",rid,day,b1,reason))
                quoted["c1_new_gate"][rid]={t:prices[tuple(t.split("-"))] for t in b1["tickets"]}
                by_month[m]["c1_new_gate"]+=1
        else:
            counters["candidate_rejected_by_original_select"]+=1
    # PRE RESULT: seal the entire unmodified decision mapping, no odds/outcome
    # and no tuning or review of actual paid winnings.
    doc=json.dumps(decisions,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    seal=hashlib.sha256(doc.encode("utf-8")).hexdigest()
    print("V6_C1_FULL_VAULT_DECISIONS_SEALED",json.dumps({
        "sha256":seal,
        "coverage":dict(counters),
        "month_coverage":{k:dict(v) for k,v in by_month.items()},
        "selected":{k:len(v) for k,v in decisions.items()}},ensure_ascii=False),flush=True)
    totals,month,comparison,payout_source=outcome_audit(all_events,decisions,quoted)
    report={
        "status":"FULL_SEVEN_BET_VAULT_FEB_APR_REPLAY_COMPLETE",
        "period":[START,END],
        "source":"Sealed RSA CMS 7-market archive, verified by SHA256 and original complete_saved_races, no plaintext odds exported",
        "source_coverage_baselines":verify_sources(),
        "original_V6_unmodified":True,"original_V6_C1_unmodified":True,
        "original_formation_generator_unmodified":True,
        "new_admission_gate_is_hypothesis_only":True,
        "no_result_join_before_selection_seal":True,
        "not_future_holdout":True,
        "odds_phase":"historical final, not predeadline",
        "decision_sha256":seal,
        "source_audit":dict(counters),
        "month_source_coverage":{k:dict(v) for k,v in sorted(by_month.items())},
        "selection_settings":{"min_supported_tickets":2,"distinct_supported_trios":2,
                              "min_order_disagreement_in_both_V6_and_C1":1.5,
                              "min_trio_model_to_market_ratio":.75,
                              "max_trio_model_to_market_ratio":1.35,
                              "minimum_formation_flat_EV_in_both_models":1.2},
        "official_settlement":totals,"settlement_by_month":month,
        "final_quote_proxy_is_not_official_payout":True,
        "comparisons":comparison,
        "limitations":["Results are IN SAMPLE for V6 and V6-C1, so not evidence of prospective profitability.",
                       "Sealed odds are postrace FINAL and not actual preclose bettable prices.",
                       "Gate is a PREDECLARED hypothesis and must not be tuned using these returns.",
                       "The gate relies on related 3renpuku and 3rentan pools, not an independent prediction model.",
                       "Only male valid 5-9 rider events with positive complete source prices and matching racecards are evaluated.",
                       "Final quote proxy includes hits without available official payout and MUST NOT be misrepresented as verified official return.",
                       "No encrypted source plaintext or private key is exported."]}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V6_C1_FULL_VAULT_GATE_RESULT",json.dumps({
       "period":report["period"],"coverage":report["month_source_coverage"],
       "source":report["source_audit"],"results":report["official_settlement"],
       "months":report["settlement_by_month"],"comparison":comparison,
       "decision_sha256":seal},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

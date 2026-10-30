#!/usr/bin/env python3
"""Unseal original results after PRE-RESULT formation decisions were written.

Official paid 100-yen 3ren-tan winning payout is the settlement source.
Historical final odds are not substituted for an official missing payout
in the verified ROI; they are reported only as secondary price-based proxy.
A losing formation is safely assigned payout=0 from recorded finish alone.
"""
from __future__ import annotations
import hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
import observe as obs
import v6_settled_payout_only_2023_2026 as prior

DIR=ROOT/"data/audits/v6_formation_ev_2023_01"
PRE=DIR/"pre_result_sealed_decisions.jsonl"
METADATA=DIR/"pre_result_source_and_coverage.json"
OUT=DIR/"settled_formation_purchases.jsonl"
SUMMARY=DIR/"jan2023_formation_roi_report.json"

def newstat():
    return {"races":0,"hits":0,"stake_yen":0,"payout_yen":0,
            "expected_payout_yen":0.,"market_fair_payout_yen":0.,
            "positive_ticket_count":0,"negative_ticket_count":0,
            "positive_expected_underlying":0.}

def update(z,record,payout):
    z["races"]+=1
    z["hits"]+=int(payout>0)
    z["stake_yen"]+=record["stake_yen"]
    z["payout_yen"]+=payout
    z["expected_payout_yen"]+=record["stake_yen"]*record["v6_flat_stake_expected_roi_pct"]/100
    z["market_fair_payout_yen"]+=record["stake_yen"]/record["market_full_inverse_odds_mass"]
    z["positive_ticket_count"]+=record["components_expected_over_100pct"]
    z["negative_ticket_count"]+=record["components_expected_below_100pct"]
    z["positive_expected_underlying"]+=sum(
        float(t["v6_p"])*float(t["final_odds"])-1 for t in record["ticket_data"] if float(t["v6_flat_ev"])>=1)

def emit(z):
    d=dict(z)
    d.update({"actual_roi_pct":round(100*z["payout_yen"]/z["stake_yen"],5) if z["stake_yen"] else None,
              "v6_expected_roi_pct":round(100*z["expected_payout_yen"]/z["stake_yen"],5) if z["stake_yen"] else None,
              "market_normalized_hypothesis_roi_pct":round(100*z["market_fair_payout_yen"]/z["stake_yen"],5) if z["stake_yen"] else None,
              "profit_yen":z["payout_yen"]-z["stake_yen"],
              "hit_pct":round(100*z["hits"]/z["races"],5) if z["races"] else None,
              "mean_points":round(z["stake_yen"]/100/z["races"],4) if z["races"] else None})
    d["expected_payout_yen"]=round(d["expected_payout_yen"],4)
    d["market_fair_payout_yen"]=round(d["market_fair_payout_yen"],4)
    d["positive_expected_underlying"]=round(d["positive_expected_underlying"],5)
    return d

def main():
    meta=json.loads(METADATA.read_text(encoding="utf-8"))
    assert meta["state"]=="SEALED_BEFORE_RESULT_ONLY_FORMATION_PURCHASES_2023_JAN_COMPLETE"
    assert hashlib.sha256(PRE.read_bytes()).hexdigest()==meta["decisions_sha256"]
    decisions=[json.loads(z) for z in PRE.read_text(encoding="utf-8").splitlines() if z.strip()]
    assert len(decisions)==meta["coverage"].get("purchase_races",0)
    assert len({z["race_id"] for z in decisions})==len(decisions)
    for d in decisions:
        assert "order" not in d and "payout" not in d
        assert d["odds_phase"]=="historical_final_NOT_preclose"
        assert d["decision_does_not_use_result"]
        assert 4<=d["points"]<=12
        assert d["stake_yen"]==100*d["points"]
        assert d["v6_flat_stake_expected_roi_pct"]>=120-1e-6
        assert d["synthetic_equal_payout_odds_not_real_stake"]>=3.5-1e-6
        first,second,third=d["first"],d["second"],d["third"]
        import itertools
        assert sorted(d["tickets"])==sorted("-".join(t) for t in itertools.product(first,second,third)
                if len(set(t))==3),d["race_id"]
        assert d["points"]==len(set(d["tickets"]))
        assert sum(x["v6_p"] for x in d["ticket_data"])>0
    print("JAN2023_DECISIONS_SEALED_SHA256_VERIFIED",meta["decisions_sha256"],len(decisions),flush=True)

    # Only now load outcome labels and official winning payouts. Never re-pick.
    all_events,_=obs.ingest()
    official,conflicts,pay_audit=prior.load_settled_payments({x["race_id"] for x in decisions})
    total=newstat(); by_day=defaultdict(newstat); by_low={True:newstat(),False:newstat()}
    audited=Counter(); proxy=newstat()
    paid_odd_ratio=[]
    samples=[]
    with OUT.open("w",encoding="utf-8") as fp:
        for d in decisions:
            rid=d["race_id"]; e=all_events.get(rid)
            if e is None:raise AssertionError(("Selected race with no results",rid))
            true="-".join(e["order"][:3])
            hit=true in d["tickets"]
            payout=0;status="verified_lose"
            final_price_by_ticket={x["ticket"]:x["final_odds"] for x in d["ticket_data"]}
            if hit:
                if rid in conflicts:
                    status="official_payout_conflict"
                elif rid not in official:
                    status="official_payout_missing"
                elif "-".join(official[rid][0])!=true:
                    status="official_winner_disagrees_with_result"
                else:
                    status="verified_win"
                    payout=official[rid][1]
                    paid_odd_ratio.append(payout/(100*final_price_by_ticket[true]))
            audited[status]+=1
            proxy_payout=100*final_price_by_ticket[true] if hit else 0
            update(proxy,d,proxy_payout)
            rec=dict(d)
            rec.update({"actual_result":true,"hit":hit,"settlement_status":status,
                        "verified_payout_yen":payout if status.startswith("verified_") else None,
                        "proxy_payout_from_historical_final_odds_yen":round(proxy_payout,2),
                        "profit_yen":payout-d["stake_yen"] if status.startswith("verified_") else None})
            fp.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n")
            if status.startswith("verified_"):
                update(total,d,payout)
                update(by_day[d["date"]],d,payout)
                update(by_low[d["components_expected_below_100pct"]>0],d,payout)
                if len(samples)<25:
                    samples.append({"race_id":rid,"date":d["date"],
                                    "formation":d["formation"],"points":d["points"],
                                    "positive":d["components_expected_over_100pct"],
                                    "below100":d["components_expected_below_100pct"],
                                    "v6_expected_roi_pct":d["v6_flat_stake_expected_roi_pct"],
                                    "v6_total_probability_pct":d["v6_formation_probability_pct"],
                                    "synthetic_odds":d["synthetic_equal_payout_odds_not_real_stake"],
                                    "stake_yen":d["stake_yen"],"result":true,
                                    "hit":hit,"payout_yen":payout})
    if total["races"]+sum(audited[k] for k in ("official_payout_missing","official_payout_conflict","official_winner_disagrees_with_result"))!=len(decisions):
        raise AssertionError("Settlement partition failed")
    result={
      "state":"JAN2023_FROZEN_V6_FORMATION_100YEN_PER_POINT_ROI_SETTLED",
      "research_not_genuine_future_OOS":True,
      "v6_fitted_using_january_2023_outcomes":True,
      "historical_final_odds_only_not_bettable_Tminus":True,
      "all_selections_written_before_result_join":True,
      "no_formations_changed_during_settlement":True,
      "decision_hash_sha256":meta["decisions_sha256"],
      "period":["2023-01-01","2023-01-31"],
      "coverage":meta["coverage"],"settings":meta["settings"],
      "settlement_status_counts":dict(audited),
      "actual_verified_official_payouts":emit(total),
      "proxy_from_final_odds_all_decisions_not_official_settlement":emit(proxy),
      "by_day_verified":{day:emit(z) for day,z in sorted(by_day.items())},
      "by_contains_below_100_component_verified":{str(k):emit(v) for k,v in by_low.items()},
      "official_payout_to_final_odds_ratio":{
          "n":len(paid_odd_ratio),
          "mean":round(sum(paid_odd_ratio)/len(paid_odd_ratio),6) if paid_odd_ratio else None,
          "max_abs_disagreement":round(max(abs(x-1) for x in paid_odd_ratio),6) if paid_odd_ratio else None},
      "official_payout_source_audit":pay_audit,
      "first_purchase_samples":samples,
      "strict_warnings":[
          "January 2023 is IN SAMPLE for original frozen V6: fitting on 2023 Jan-Oct uses future labels. Cannot claim a 2023 live strategy.",
          "All price books are archived FINAL odds, acquired retrospectively; unavailable before deadline as recorded.",
          "A one-month backtest is extremely noisy, especially on rare high-payout results.",
          "Winning official payout missing or conflicting is excluded only from verified settlement, never imputed as zero.",
          "Selections are complete formations; 100 JPY per unique ticket, NOT fractional dutching.",
          "Under-100-EV individual components may be included if entire formation has expected ROI >=120%.",
          "Theoretical synthetic odds are not the actual payout multiplier for flat staking.",
          "Thresholds were set before inspecting January outcomes in this run; external past research may have explored these dates.",
          "Cash backtests are not proof of market mispricing; need frozen protocol on actual later period with preclose odds."
      ]}
    SUMMARY.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("JAN2023_FORMATION_SETTLEMENT_FINAL",json.dumps({
        "coverage":meta["coverage"],"verified":result["actual_verified_official_payouts"],
        "proxy":result["proxy_from_final_odds_all_decisions_not_official_settlement"],
        "settlement_status":dict(audited),
        "by_below100":result["by_contains_below_100_component_verified"],
        "samples":samples[:8]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()

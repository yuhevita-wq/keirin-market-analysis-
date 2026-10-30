#!/usr/bin/env python3
"""Settle exact sealed March 2023 original V6 vs calibrated V6-C1 formations.

Use official winning 3ren-tan paid amount; missing/conflicted paid records
never silently count as losses. Identical 100-yen flat ticket stakes.
"""
from __future__ import annotations
import hashlib,itertools,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/v6_january_2023_formation_ev"))
import observe as obs
import v6_settled_payout_only_2023_2026 as prior
from research.v6_january_2023_formation_ev.settle import newstat,update,emit

DIR=ROOT/"data/audits/v6_c1_formation_mar2023"
VERSIONS=("V6","V6-C1")
def pack(d):
    if not d["races"]:return None
    return emit(d)

def main():
    meta=json.loads((DIR/"sealed_decision_metadata.json").read_text())
    assert meta["state"]=="V6_C1_MAR2023_BOTH_MODELS_PRE_RESULT_SEALED"
    assert meta["period"]==["2023-03-01","2023-03-31"]
    decisions={}
    for name in VERSIONS:
        p=DIR/("sealed_"+name.replace("-","_")+"_formations.jsonl")
        assert p.exists()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==meta["sha256"][name]
        rows=[json.loads(z) for z in p.read_text(encoding="utf-8").splitlines() if z]
        assert len(rows)==meta["coverage"][name]["purchase_races"]
        assert len({r["race_id"] for r in rows})==len(rows)
        for d in rows:
            assert d["model_name"]==name
            assert "order" not in d and "payout" not in d and d["no_result_used_in_selection"]
            assert d["date"].startswith("2023-03-")
            assert 4<=d["points"]<=12 and d["stake_yen"]==100*d["points"]
            assert d["v6_flat_stake_expected_roi_pct"]>=120-1e-6
            assert d["synthetic_equal_payout_odds_not_real_stake"]>=3.5-1e-6
            assert len(d["tickets"])==len(set(d["tickets"]))==d["points"]
            expected={"-".join(t) for t in itertools.product(d["first"],d["second"],d["third"])
                      if len(set(t))==3}
            assert expected==set(d["tickets"]),("Incomplete formation",d["race_id"])
        decisions[name]={d["race_id"]:d for d in rows}
    print("MAR2023_HASH_SEAL_AND_FORMATIONS_VERIFIED",
          json.dumps({k:len(d) for k,d in decisions.items()}),flush=True)

    # OUTCOME JOIN happens strictly after both candidate and budget sets are sealed.
    events,_=obs.ingest()
    wanted=set().union(*(set(d) for d in decisions.values()))
    official,conflicts,payout_source=prior.load_settled_payments(wanted)
    audited={}
    by_day={}
    breakdown={}
    by_shape={}
    samples={}
    for name in VERSIONS:
        z=newstat();day=defaultdict(newstat);low={True:newstat(),False:newstat()}
        shape=defaultdict(newstat);states=Counter();wins=[]
        path=DIR/("settled_"+name.replace("-","_")+"_purchases.jsonl")
        with path.open("w",encoding="utf-8") as out:
            for rid,rec in decisions[name].items():
                if rid not in events:raise AssertionError(("Race missing",rid))
                true="-".join(events[rid]["order"][:3])
                hit=true in rec["tickets"]
                state="verified_lose";paid=0
                if hit:
                    if rid in conflicts:state="official_payout_conflict"
                    elif rid not in official:state="official_payout_missing"
                    elif "-".join(official[rid][0])!=true:state="official_winner_disagrees"
                    else:
                        paid=official[rid][1];state="verified_win"
                        p_by_ticket={row["ticket"]:row["final_odds"] for row in rec["ticket_data"]}
                        if abs(100*p_by_ticket[true]-paid)>.011:
                            raise AssertionError(("Real payout != archived final odds",rid,paid,p_by_ticket[true]))
                states[state]+=1
                row=dict(rec)
                row.update({"true_result":true,"hit":hit,
                            "official_payout_status":state,
                            "actual_official_payout_yen":paid if state.startswith("verified_") else None,
                            "realized_profit_yen":paid-rec["stake_yen"] if state.startswith("verified_") else None})
                out.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
                if state.startswith("verified_"):
                    update(z,rec,paid);update(day[rec["date"]],rec,paid)
                    update(shape[rec["shape"]],rec,paid)
                    update(low[rec["components_expected_below_100pct"]>0],rec,paid)
                if state=="verified_win":
                    wins.append({"date":rec["date"],"race_id":rid,
                                 "formation":rec["formation"],"points":rec["points"],
                                 "v6_or_c1_predicted_roi_pct":rec["v6_flat_stake_expected_roi_pct"],
                                 "actual_result":true,"official_payout_yen":paid})
        assert z["races"]+sum(states[x] for x in ("official_payout_conflict","official_payout_missing","official_winner_disagrees"))==len(decisions[name])
        audited[name]={"verified":emit(z),"settlement_status":dict(states),
                       "purchase_count":len(decisions[name]),"wins":len(wins)}
        by_day[name]={dt:emit(s) for dt,s in sorted(day.items())}
        breakdown[name]={str(k):pack(s) for k,s in low.items()}
        by_shape[name]={k:emit(s) for k,s in shape.items()}
        samples[name]={"first_15_wins":wins[:15],"total_wins":len(wins)}
    ids_a=set(decisions["V6"]);ids_b=set(decisions["V6-C1"])
    common=ids_a&ids_b
    change=[]
    same=0;different=0
    for rid in sorted(common):
        a=decisions["V6"][rid];b=decisions["V6-C1"][rid]
        if set(a["tickets"])==set(b["tickets"]):same+=1
        else:
            different+=1
            if len(change)<25:
                true="-".join(events[rid]["order"][:3])
                change.append({"date":a["date"],"race_id":rid,
                               "V6_formation":a["formation"],
                               "C1_formation":b["formation"],
                               "V6_hit":true in a["tickets"],"C1_hit":true in b["tickets"]})
    diff={
       "both_buy":len(common),"both_same_ticket_set":same,
       "both_different_ticket_set":different,
       "V6_only_buy":len(ids_a-ids_b),"C1_only_buy":len(ids_b-ids_a),
       "examples":change
    }
    p_a=audited["V6"]["verified"];p_b=audited["V6-C1"]["verified"]
    result={
       "state":"V6_C1_MAR2023_FORMATION_OFFICIAL_PAYOUT_COMPLETE",
       "period":["2023-03-01","2023-03-31"],
       "reused_exact_january_4_12_point_selector":True,
       "nonoptimal_point_range_is_original_research_constraint":True,
       "C1_fitted_on_Mar2023_labels":True,
       "V6_base_fitted_on_Mar2023_labels":True,
       "historical_final_odds_not_verified_bettable":True,
       "market_odds_not_used_to_fit_C1_or_V6":True,
       "selected_before_joining_results":True,
       "meta_sha256":meta["sha256"],
       "coverage":meta["coverage"],"official_sources":payout_source,
       "by_model":audited,"by_day":by_day,
       "by_contains_sub100_ticket":breakdown,
       "by_shape":by_shape,
       "both_models_purchases":diff,
       "model_comparison":{
          "C1_minus_V6_actual_roi_percentage_points":
             round(p_b["actual_roi_pct"]-p_a["actual_roi_pct"],5)
             if p_b["actual_roi_pct"] is not None and p_a["actual_roi_pct"] is not None else None,
          "C1_minus_V6_actual_profit_yen":p_b["profit_yen"]-p_a["profit_yen"],
          "C1_minus_V6_hits":p_b["hits"]-p_a["hits"],
          "C1_minus_V6_purchase_races":p_b["races"]-p_a["races"]
       },
       "wins":samples,
       "limitations":[
        "March 2023 is IN-SAMPLE for V6 base and V6-C1 calibrator, not evidence of prospective investment profit",
        "Historical final odds were collected after the races and not confirmed available pre-bet",
        "4-12 ticket limit is January research code legacy, not theoretical optimal formation range",
        "All official unsettled hits excluded from verified performance; reason counts displayed",
        "One-month realized return dominated by volatile payouts; no cutoff tune on results"
       ]
    }
    (DIR/"mar2023_V6_vs_C1_official_roi_report.json").write_text(
      json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MAR2023_V6_C1_SETTLEMENT_FINAL",json.dumps({
       "coverage":meta["coverage"],"V6":audited["V6"],"V6_C1":audited["V6-C1"],
       "comparison":result["model_comparison"],"purchase_change":{k:v for k,v in diff.items() if k!="examples"},
       "samples":{k:v["first_15_wins"][:5] for k,v in samples.items()}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()

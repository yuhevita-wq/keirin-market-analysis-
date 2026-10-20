"""Diagnostic: separate forecast skill, market-price edge and realized 2025 ROI.

No fitting or policy selection. Runs fixed 2024-end parents and deletion scores on
the same 2025 races used for the previously published frozen-deletion evaluation.
2025 is already examined research data, NOT a pristine independent test.

Measures:
- proper per-market multiclass log loss and Brier loss for normalized probabilities;
- market-implied normalized probabilities as a baseline;
- expected vs realized payouts for exactly the previously sealed deletion mask;
- retrospective diagnostic positive-p*odds subsets (NOT deployable policy);
- predeclared fixed-price-edge bins, with every expensive missed winner counted.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np

import chapter1_frozen_2025 as frozen
import chapter1_iterative_deletion as learner
import chapter1_accuracy_calibration_duel as duel
import chapter1_full_sheet_deletion as original

ROOT=Path("docs/paper-only-ai-2023/chapter1_deletion_continual/price_gap_2025")
ROLES=learner.ROLES
KINDS=learner.KINDS
PROBS=("parent_probability","deletion_normalized_score","market_normalized")
EDGES=("parent_probability","deletion_normalized_score")
# Fixed descriptive segmentation. Never choose a profitable bin and call it tested.
EDGE_LIMITS=(-float("inf"),-.5,-.2,0,.1,.25,.5,1,2,float("inf"))
EDGE_NAMES=("<-50%","-50..-20%","-20..0%","0..+10%","+10..+25%",
            "+25..+50%","+50..+100%","+100..+200%",">=+200%")


def empty_cash():
    return {"tickets":0,"hits":0,"stake_yen":0.0,"payout_yen":0.0,
            "predicted_payout_yen":0.0,"market_expected_payout_yen":0.0}

def empty_model():
    return {"races":0,"logloss_sum":0.0,"brier_sum":0.0,
            "top1_hits":0,"top1_confidence_sum":0.0}

def initialize():
    return {
        role: {kind: {
            "forecast":{s:empty_model() for s in PROBS},
            "selected_deletion":{s:empty_cash() for s in EDGES},
            "positive_ev":{s:empty_cash() for s in EDGES},
            "edge_bins":{s:{b:empty_cash() for b in EDGE_NAMES} for s in EDGES},
        } for kind in KINDS} for role in ROLES
    }

def cash_update(stats,keep,prob,market,odds,winner):
    n=int(np.count_nonzero(keep))
    stats["tickets"]+=n
    stats["hits"]+=int(keep[winner])
    stats["stake_yen"]+=100*n
    stats["payout_yen"]+=100*float(odds[winner])*bool(keep[winner])
    stats["predicted_payout_yen"]+=100*float(np.sum(prob[keep]*odds[keep]))
    stats["market_expected_payout_yen"]+=100*float(np.sum(market[keep]*odds[keep]))

def evaluate(stats,prob,winner):
    n=len(prob)
    if not np.all(np.isfinite(prob)) or np.any(prob<=0) or abs(float(prob.sum())-1)>1e-7:
        raise RuntimeError("Invalid normalized full-book probability")
    stats["races"]+=1
    stats["logloss_sum"]+=-math.log(float(prob[winner]))
    stats["brier_sum"]+=float(np.dot(prob,prob)+1-2*prob[winner])
    top=int(np.argmax(prob))
    stats["top1_hits"]+=int(winner==top)
    stats["top1_confidence_sum"]+=float(prob[top])

def score_quarter(q):
    target=ROOT/f"quarter_2025_q{q}.json"
    if target.exists():
        raise RuntimeError("Already examined quarter. Refusing rewrite.")
    theta,fingerprint=frozen.frozen_state()
    frozen.configure_year()
    complete,audit=frozen.load_2025_aligned(q)
    parents=learner.frozen_deletion_parents()
    asset=duel.fetch_immutable_weights()
    summary=initialize()
    total=0
    for off in range(0,len(complete),16):
        chunk=complete[off:off+16]
        for rid,day,groups in learner.records_for_batch(chunk,*asset[:2]):
            # All prices, cards, parents, and frozen weights are decided before labels.
            for row in groups:
                role,kind=row["role"],row["kind"]
                stats=summary[role][kind]
                odds=np.asarray(row["prices"],dtype=float)
                if not np.all(np.isfinite(odds)) or np.any(odds<=1):
                    raise RuntimeError("Invalid final odds")
                parent=np.asarray(row["model_p"],dtype=float)
                adjusted=learner.ticket_probabilities(row,parents[role][kind],theta[role][kind])
                market=(1/odds);market=market/market.sum()
                if not np.allclose(parent.sum(),1,atol=1e-7):
                    raise RuntimeError("Parent probability not normalized")
                keep=learner.remove_from_all(adjusted,odds,learner.COVERAGE[role])
                # Winner is accessed ONLY to grade sealed decisions below.
                win=row["winner_index"]
                for label,p in (("parent_probability",parent),
                                ("deletion_normalized_score",adjusted),
                                ("market_normalized",market)):
                    evaluate(stats["forecast"][label],p,win)
                for label,p in (("parent_probability",parent),
                                ("deletion_normalized_score",adjusted)):
                    cash_update(stats["selected_deletion"][label],keep,p,market,odds,win)
                    edge=p*odds-1
                    positive=edge>0
                    cash_update(stats["positive_ev"][label],positive,p,market,odds,win)
                    bins=np.searchsorted(EDGE_LIMITS[1:-1],edge,side="right")
                    for i,name in enumerate(EDGE_NAMES):
                        mask=bins==i
                        if np.any(mask):
                            cash_update(stats["edge_bins"][label][name],mask,p,market,odds,win)
            total+=1
        if off%256==0 or off+16>=len(complete):
            print("PRICE_GAP_2025_PROGRESS",q,total,len(complete),flush=True)
    if total!=len(complete):
        raise RuntimeError("Partial quarter")
    frozen.assert_unchanged(fingerprint)
    ROOT.mkdir(parents=True,exist_ok=True)
    report={
        "status":"FROZEN_2025_PRICE_GAP_QUARTER_DIAGNOSTIC",
        "quarter":q,"races":total,"frozen_checkpoint":fingerprint,
        "no_2025_fitting":True,"not_new_blind":True,
        "retrospective_ev_threshold_analysis_not_forward_policy":True,
        "deletion_normalized_score_not_proven_calibrated_probability":True,
        "data_audit":audit,"raw":summary,
    }
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PRICE_GAP_2025_QUARTER_SAVED",q,total,flush=True)


def merge_into(dest,src):
    for k,v in src.items():
        if isinstance(v,dict):
            merge_into(dest[k],v)
        else:
            dest[k]+=v

def cash_summary(a):
    stake=a["stake_yen"]
    def pct(num):
        return 100*num/stake if stake else None
    market=pct(a["market_expected_payout_yen"])
    projected=pct(a["predicted_payout_yen"])
    actual=pct(a["payout_yen"])
    return {**a,"roi_pct":actual,"model_predicted_roi_pct":projected,
            "market_normalized_expected_roi_pct":market,
            "model_predicted_edge_vs_market_pp":projected-market if projected is not None else None,
            "actual_minus_predicted_pp":actual-projected if actual is not None else None,
            "actual_edge_vs_market_pp":actual-market if actual is not None else None,
            "net_yen":a["payout_yen"]-stake}

def forecast_summary(a):
    n=a["races"]
    return {"races":n,"mean_nll":a["logloss_sum"]/n,
            "mean_multiclass_brier":a["brier_sum"]/n,
            "top1_accuracy_pct":100*a["top1_hits"]/n,
            "mean_top1_confidence_pct":100*a["top1_confidence_sum"]/n}

def annual():
    target=ROOT/"annual_2025.json"
    if target.exists():raise RuntimeError("Annual diagnostic already written")
    _, fingerprint=frozen.frozen_state()
    data=[json.loads((ROOT/f"quarter_2025_q{q}.json").read_text("utf-8"))
          for q in (1,2,3,4)]
    if any(d["quarter"]!=i or d["frozen_checkpoint"]!=fingerprint
           for i,d in enumerate(data,1)):
        raise RuntimeError("Mismatched historical frozen quarters")
    overall=initialize()
    for item in data:merge_into(overall,item["raw"])
    results={}
    for role in ROLES:
        results[role]={}
        for kind in KINDS:
            x=overall[role][kind]
            results[role][kind]={
                "forecast":{name:forecast_summary(s) for name,s in x["forecast"].items()},
                "selected_deletion":{name:cash_summary(s) for name,s in x["selected_deletion"].items()},
                "positive_ev":{name:cash_summary(s) for name,s in x["positive_ev"].items()},
                "edge_bins":{name:{bin_name:cash_summary(s) for bin_name,s in x["edge_bins"][name].items()}
                             for name in EDGES},
            }
    report={"status":"FROZEN_2025_FORECAST_VS_PRICE_GAP_DIAGNOSTIC_COMPLETE",
            "races":sum(d["races"] for d in data),
            "quarterly_races":[d["races"] for d in data],
            "checkpoint":fingerprint,
            "no_training_or_hyperparameter_selection_on_2025":True,
            "already_explored_2025_historical_data":True,
            "final_odds_only":True,
            "retrospective_positive_EV_not_a_preregistered_trading_strategy":True,
            "edge_bin_thresholds_fixed_not_selected_by_results":list(EDGE_NAMES),
            "results":results,
            "interpretation_limits":[
              "Parent distributions are normalized actual outcome probabilities, but their calibration must be empirically checked.",
              "Deletion model output is a normalized ticket-protection score; do NOT presume it is calibrated as a literal event probability.",
              "Positive p*final odds and EV bins are descriptive ex-post diagnostics; using the same 2025 results to select a rule forfeits independent-test status.",
              "Market probabilities normalize inverse final odds and assume an internally coherent mutually-exclusive exhaustive ticket book.",
              "2025 was previously used in other project studies, not a globally untouched holdout.",
              "All prices are historical FINAL odds; forward live purchaseability is not established.",
            ]}
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PRICE_GAP_2025_ANNUAL_COMPLETE",sum(d["races"] for d in data),flush=True)
    for role in ROLES:
        for kind in KINDS:
            s=results[role][kind]
            print("SUMMARY",role,kind,
                  "parent_nll",round(s["forecast"]["parent_probability"]["mean_nll"],5),
                  "market_nll",round(s["forecast"]["market_normalized"]["mean_nll"],5),
                  "deletion_roi",round(s["selected_deletion"]["deletion_normalized_score"]["roi_pct"],3),
                  "projected_roi",round(s["selected_deletion"]["deletion_normalized_score"]["model_predicted_roi_pct"],3),
                  flush=True)


def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--quarter",type=int)
    g.add_argument("--annual",action="store_true")
    args=p.parse_args()
    if args.annual:annual()
    else:score_quarter(args.quarter)

if __name__=="__main__":
    main()

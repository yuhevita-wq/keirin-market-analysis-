"""GEN-008 parallel nursery: WEEK 01, 2023-08-01..2023-08-07 only.

Runs the pre-existing old/control and new/shadow-learning Branch algorithms
with an identical frozen GEN-007-C2 parent and identical exact-ID racecards.
Seals decisions for both branches before reading the outcome for each race.
Saves one immutable week checkpoint for EACH branch, not the legacy parent.

NO use of 2023-10-01..14 forward exam during training. This first week's
training scores are NOT evidence of future predictive improvement.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter,defaultdict
from datetime import date,timedelta
from pathlib import Path

import numpy as np

from train_week import ROOT,D,feature_matrix,softmax
from source_bridge import joined_races_for_training
from parallel_gen008 import (
    Branch,load_both_parents,model_digest,read_json,seal,pure_pre_result,
    page_increment_nets,TRAIN_START as PLAN_FIRST,TRAIN_END as PLAN_LAST,
    TEST_START as EXAM_START,OUT as EXPERIMENT_ROOT,
)
from gen008_experiment import calibration_metrics

WEEK=1
FIRST=date(2023,8,1)
LAST=FIRST+timedelta(days=6)
OUT=EXPERIMENT_ROOT/"week-01"
BOTH=("old","new")
OBJECTIVE="real_paper_payout_minus_real_paper_stake"
MANIFEST=ROOT/"docs/ai-lab/gen008-week01.json"


def market_assessment(branch,payload,finish):
    """Comparable normalized probability scores BEFORE revealing labels to learner.

    The control samples across two markets, so renormalize its sampling
    distribution separately within each market for predictive scoring only.
    Candidate's ordered/unordered model is already coherent.
    """
    arms,probs,_,market_probs,_,_=payload
    if branch=="new":
        assert market_probs is not None
        return calibration_metrics(market_probs,finish)
    out={}
    for market,sep in (("3連単","-"),("3連複","=")):
        indexes=[i for i,a in enumerate(arms) if a["market"]==market]
        mass=sum(float(probs[i]) for i in indexes)
        if mass<=0:raise RuntimeError("Zero market probability")
        key=sep.join(map(str,finish if market=="3連単" else sorted(finish)))
        target=next((i for i in indexes if arms[i]["selection"]==key),None)
        if target is None:raise RuntimeError("No winning combination in complete market")
        prob=max(float(probs[target])/mass,1e-30)
        out["trifecta_logloss" if market=="3連単" else "trio_logloss"]=-math.log(prob)
        out["trifecta_winner_probability" if market=="3連単" else "trio_winner_probability"]=prob
        out["trifecta_mass" if market=="3連単" else "trio_mass"]=mass/mass
    return out


def summarize(rows):
    result={}
    for name in BOTH:
        subset=[x for x in rows if x["branch"]==name]
        assert subset
        stakes=sum(x["actual"]["stake_yen"] for x in subset)
        paid=sum(x["actual"]["payout_yen"] for x in subset)
        shadow_skips=[x for x in subset if x.get("shadow_skipped_net_yen") is not None]
        result[name]={
            "races":len(subset),
            "entered":sum(x["entered"] for x in subset),
            "skipped":sum(not x["entered"] for x in subset),
            "marks":sum(x["actual"]["marks"] for x in subset),
            "hit_races":sum(x["actual"]["hit"] for x in subset),
            "staked_yen":stakes,"payout_proxy_yen":paid,
            "pure_net_yen":paid-stakes,
            "roi_proxy_percent":round(100*paid/stakes,2) if stakes else None,
            "mean_pre_update_trifecta_logloss":round(
                sum(x["pre_update"]["trifecta_logloss"] for x in subset)/len(subset),6),
            "mean_pre_update_trio_logloss":round(
                sum(x["pre_update"]["trio_logloss"] for x in subset)/len(subset),6),
            "shadow_skip_races":len(shadow_skips),
            "shadow_profitable_skipped_races":sum(
                x["shadow_skipped_net_yen"]>0 for x in shadow_skips),
            "shadow_skipped_single_page_total_net_yen":sum(
                x["shadow_skipped_net_yen"] for x in shadow_skips),
        }
    return result


def run_week():
    if FIRST.isoformat()!=PLAN_FIRST:
        raise RuntimeError("Week must begin at pre-registered parallel experiment start")
    if LAST.isoformat()>PLAN_LAST or LAST.isoformat()>=EXAM_START:
        raise RuntimeError("Week overlaps reserved October exam")
    if OUT.exists():
        raise RuntimeError("GEN-008 week-01 immutable folder exists: refusing overwrite")
    # Verify exact-ID join, but no lookahead or data from held-out October.
    races,reports=joined_races_for_training(FIRST.isoformat(),LAST.isoformat())
    if not races:raise RuntimeError("No joined races for one week")
    if len({r["race_id"] for r in races})!=len(races):
        raise RuntimeError("Duplicate race IDs in week")
    assert all(FIRST.isoformat()<=r["race"]["race_date"]<=LAST.isoformat() for r in races)
    parent,weights,old_policy,new_policy=load_both_parents()
    twins={"old":Branch("old",weights.copy(),old_policy),
           "new":Branch("new",weights.copy(),new_policy)}
    assert model_digest(twins["old"].weights)==model_digest(twins["new"].weights)==parent["weights_sha256"]
    assert twins["old"].policy.digest()==twins["new"].policy.digest()==parent["behavior_policy_sha256"]
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[]
    day=defaultdict(list)
    with (OUT/"sealed_before_results.jsonl").open("x",encoding="utf-8") as paper,\
         (OUT/"after_results_learning.jsonl").open("x",encoding="utf-8") as ledger:
        for i,race in enumerate(races,1):
            # A strict row whitelist excludes the outcome from all precommit APIs.
            pre=pure_pre_result(race)
            sealed={}
            payloads={}
            for name in BOTH:
                decision,payload=twins[name].precommit(pre)
                sealed[name]=decision
                payloads[name]=payload
            seal(paper,{
                "sequence":i,"race_id":pre["race_id"],
                "date":pre["race"]["race_date"],
                "old":sealed["old"],"new":sealed["new"],
                "results_visible_before_seal":False,
                "actual_stakes_exclude_shadow":True,
            })
            # Only NOW access the known true finishing order.
            finish=race["outcome"]
            for name in BOTH:
                assessed=market_assessment(name,payloads[name],finish)
                settled=twins[name].settle_and_learn(
                    pre,finish,payloads[name],sealed[name])
                shadow=settled.get("shadow_missed_profit_if_skipped_yen")
                sample={
                    "sequence":i,"date":pre["race"]["race_date"],
                    "race_id":pre["race_id"],"branch":name,
                    "entered":settled["entered"],
                    "actual":settled["actual"],
                    "pre_update":assessed,
                    "post_update_predictor_sha256":settled["prediction_after_sha256"],
                    "post_update_policy_sha256":settled["policy_after_sha256"],
                    "shadow_skipped_net_yen":shadow,
                    "shadow_counted_as_actual":False,
                    "temperature_after":twins[name].temperature,
                    "policy_updates":len(settled["updates"]),
                }
                rows.append(sample);day[sample["date"]].append(sample)
                seal(ledger,sample)
    frozen={}
    for name in BOTH:
        branch=twins[name]
        folder=OUT/name
        folder.mkdir()
        saved={
            "branch":name,"stage":"GEN-008-PARALLEL-WEEK-01",
            "parent":"GEN-007-C2",
            "parent_weights_sha256":parent["weights_sha256"],
            "parent_policy_sha256":parent["behavior_policy_sha256"],
            "after_step":parent["after_step"]+len(races),
            "trained_dates":[FIRST.isoformat(),LAST.isoformat()],
            "trained_race_count":len(races),
            "weights":branch.weights.tolist(),
            "weights_sha256":model_digest(branch.weights),
            "temperature":branch.temperature,
        }
        (folder/"checkpoint.json").write_text(
            json.dumps(saved,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        policy=branch.policy.save(folder/"behavior_policy.json")
        assert policy["weights_sha256"]==branch.policy.digest()
        frozen[name]={"weights_sha256":saved["weights_sha256"],
                      "policy_sha256":policy["weights_sha256"],
                      "temperature":saved["temperature"],
                      "after_step":saved["after_step"]}
    scores=summarize(rows)
    daily=[]
    for i in range(7):
        d=(FIRST+timedelta(days=i)).isoformat()
        current=day[d]
        if current:
            daily.append({"date":d,"races":len(current)//2,
                          "metrics":summarize(current)})
        else:
            daily.append({"date":d,"races":0})
    report={
        "stage":"GEN-008-PARALLEL-WEEK-01",
        "status":"COMPLETE_TRAINING_ONLY",
        "original_GEN007_C2_untouched":True,
        "same_parent_predictor_sha256":parent["weights_sha256"],
        "same_parent_action_policy_sha256":parent["behavior_policy_sha256"],
        "identical_race_ids_and_order_for_both_branches":True,
        "races":len(races),
        "training_dates":[FIRST.isoformat(),LAST.isoformat()],
        "days":7,
        "later_frozen_exam_is_reserved":["2023-10-01","2023-10-14"],
        "later_frozen_exam_races_read_for_training":0,
        "old_vs_new_training_scores":scores,
        "daily":daily,
        "source_reports":reports,
        "checkpoints":frozen,
        "shadow_paper_real_spend_yen":0,
        "net_counts_only_actual_purchased_paper":True,
        "no_live_bets":True,
        "odds_are_historical_final_not_predeadline":True,
        "comparison_type":"within-training outcomes; NOT independent generalization proof",
        "status_note":"After week-01 the separate models can continue training from their own checkpoints. October forward exam stays untouched."
    }
    (OUT/"summary.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    MANIFEST.write_text(json.dumps({
        "id":"GEN-008-PARALLEL-WEEK-01",
        "status":"COMPLETE_TRAINING_ONLY",
        "source":"research/ai_nursery/outputs/GEN-008-PARALLEL/week-01/summary.json",
        "parent":"GEN-007-C2",
        "training_dates":report["training_dates"],
        "races":len(races),
        "old":scores["old"],"new":scores["new"],
        "forward_exam_status":"RESERVED_NOT_TOUCHED",
        "shadow_is_cash":False
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_WEEK01_PARALLEL_COMPLETE",json.dumps({
        "races":len(races),"scores":scores},ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-training",action="store_true")
    args=ap.parse_args()
    if args.run_training:
        run_week()
    else:
        print("GEN-008 week-01 configured; no training without --run-training.")

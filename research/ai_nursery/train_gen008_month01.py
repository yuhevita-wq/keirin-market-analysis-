"""GEN-008 first MONTH of parallel training, old and new, 31 active days.

Both children continue from THEIR OWN 2023-08-14 week-02 checkpoints.
2023-08-15 through 2023-09-14 are newly learned; the October 01-14
independent holdout is never queried. Old archives remain immutable.
A distinct month-01 checkpoint is saved for EACH branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date,timedelta
from pathlib import Path

import numpy as np

from train_week import ROOT,D
from behavior_policy import BehaviorPolicy
from profit_objective import ProfitPolicy
from gen008_experiment import (
    ShadowProfitPolicy, model_digest, inherited_shadow_policy
)
from parallel_gen008 import Branch,load_both_parents,pure_pre_result,seal
from source_bridge import joined_races_for_training

BASE=ROOT/"research/ai_nursery/outputs/GEN-008-PARALLEL"
FIRST=date(2023,8,15)
LAST_TRAIN=date(2023,9,14)
FUTURE_EXAM=date(2023,10,1)
PERIOD_DAYS=31
STAGE="GEN-008-PARALLEL-MONTH-01"

def paths(week):
    return BASE/"month-01" if week==3 else BASE/f"week-{week:02d}"

def interval(week):
    if week!=3:
        raise ValueError("This month continuation requires stage 3 after frozen week-02")
    start=FIRST
    end=LAST_TRAIN
    assert (end-start).days+1==31
    if start>LAST_TRAIN or not end<FUTURE_EXAM:
        raise ValueError("Outside reserved training period")
    return start.isoformat(),end.isoformat()

def load_inherited(week):
    previous=paths(week-1)
    if not (previous/"summary.json").is_file():
        raise RuntimeError(f"Week {week-1} checkpoint must complete first")
    meta=json.loads((previous/"summary.json").read_text(encoding="utf-8"))
    prior_status=meta.get("status")
    first_week_legacy=(week==2 and meta.get("stage")=="GEN-008-PARALLEL-WEEK-01"
                       and prior_status=="COMPLETE_TRAINING_ONLY")
    if not first_week_legacy and (
            prior_status!="COMPLETE" or meta.get("week")!=week-1):
        raise RuntimeError("Previous week is incomplete")
    branches=[]
    sha={}
    for name in ("old","new"):
        checkpoint=json.loads((previous/name/"checkpoint.json").read_text(encoding="utf-8"))
        policy_path=previous/name/"behavior_policy.json"
        cp_week=checkpoint.get("week")
        if checkpoint.get("branch")!=name or (
                not first_week_legacy and cp_week!=week-1):
            raise RuntimeError("Corrupted previous-branch lineage")
        w=np.asarray(checkpoint["weights"],dtype=np.float64)
        if w.shape!=(D,) or model_digest(w)!=checkpoint["weights_sha256"]:
            raise RuntimeError("Corrupt inherited prediction weights")
        policy=(ProfitPolicy.inherit(policy_path) if name=="old"
                else inherited_shadow_policy(policy_path))
        stored_policy_sha=(meta["checkpoints"][name]["policy_sha256"]
                           if first_week_legacy else checkpoint["policy_sha256"])
        if policy.digest()!=stored_policy_sha:
            raise RuntimeError("Corrupt inherited action weights")
        branches.append(Branch(name,w,policy,float(checkpoint["temperature"])))
        sha[name]={"prediction":model_digest(w),"behavior":policy.digest()}
    return branches,{"origin":"GEN-008-PARALLEL-week-02",
                     "inherited_hashes":sha,"parent_end_date":"2023-08-14"}

def forward_compatible_probabilities(arms,payload):
    """Put *both* branch types on the SAME 3連単 and 3連複 scales."""
    mix=np.asarray(payload[1],dtype=np.float64)
    if payload[3] is not None:
        return payload[3]
    result={}
    for market in ("3連単","3連複"):
        group={a["selection"]:float(mix[i]) for i,a in enumerate(arms)
               if a["market"]==market}
        denom=sum(group.values())
        if not denom>0:raise AssertionError("Empty legacy market")
        result[market]={k:v/denom for k,v in group.items()}
    return result

def revealed_metric(outcome,probabilities):
    t="-".join(map(str,outcome))
    c="=".join(map(str,sorted(outcome)))
    p=probabilities["3連単"][t]
    q=probabilities["3連複"][c]
    return {"trifecta_logloss":-math.log(max(float(p),1e-30)),
            "trio_logloss":-math.log(max(float(q),1e-30))}

def summarize(rows):
    summary={}
    for name in ("old","new"):
        r=[x for x in rows if x["branch"]==name]
        if not r:raise RuntimeError("Missing cohort")
        stakes=sum(x["actual"]["stake_yen"] for x in r)
        payout=sum(x["actual"]["payout_yen"] for x in r)
        success=sum(x["actual"]["hit"] for x in r)
        marks=sum(x["actual"]["marks"] for x in r)
        cohort={
            "races":len(r),"entered":sum(x["entered"] for x in r),
            "skipped":sum(not x["entered"] for x in r),
            "actual_marks":marks,"hit_races":success,
            "actual_staked_yen":stakes,
            "actual_payout_proxy_yen":payout,
            "actual_net_yen":payout-stakes,
            "paper_roi_percent":100*payout/stakes if stakes else None,
            "mean_trifecta_logloss":sum(x["trifecta_logloss"] for x in r)/len(r),
            "mean_trio_logloss":sum(x["trio_logloss"] for x in r)/len(r),
        }
        if name=="new":
            shadows=[x for x in r if x["skipped_shadow_net_yen"] is not None]
            cohort["skipped_with_presealed_shadow"]=len(shadows)
            cohort["profitable_skipped_shadows"]=sum(
                x["skipped_shadow_net_yen"]>0 for x in shadows)
            cohort["missed_profitable_shadow_yen"]=sum(
                max(x["skipped_shadow_net_yen"],0) for x in shadows)
            cohort["shadow_net_on_skips_yen"]=sum(
                x["skipped_shadow_net_yen"] for x in shadows)
            cohort["shadow_net_is_not_realized_payout"]=True
        summary[name]=cohort
    return summary

def train_month(week=3):
    start,end=interval(week)
    target=paths(week)
    if target.exists():
        raise RuntimeError("Never overwrite an already created monthly model: "+str(target))
    models,origin=load_inherited(week)
    if {m.name for m in models}!={"old","new"}:
        raise AssertionError("Missing one parallel clone")
    races,sources=joined_races_for_training(start,end)
    if not races:raise RuntimeError("No complete joined races in this training week")
    if any(not start<=r["race"]["race_date"]<=end for r in races):
        raise RuntimeError("Out-of-window race")
    assert end=="2023-09-14" and FUTURE_EXAM.isoformat()=="2023-10-01"
    race_ids=[r["race_id"] for r in races]
    if len(race_ids)!=len(set(race_ids)):
        raise RuntimeError("Duplicate IDs in week")
    target.mkdir(parents=True,exist_ok=False)
    results=[]
    with (target/"sealed_pre_result_choices.jsonl").open("x",encoding="utf-8") as selections,\
         (target/"unsealed_post_result_learning.jsonl").open("x",encoding="utf-8") as records:
        for i,row in enumerate(races,1):
            # Inputs passed to the predictors DO NOT contain outcomes.
            pre=pure_pre_result(row)
            old_record,old_payload=models[0].precommit(pre)
            new_record,new_payload=models[1].precommit(pre)
            assert new_record["shadow_page"] is not None
            assert new_record["shadow_is_not_spent"] is True
            # Both decisions and shadow one-page ticket are fsync'd first.
            seal(selections,{"stage":STAGE,"step":i,"race_id":pre["race_id"],
                             "race_date":pre["race"]["race_date"],
                             "old":old_record,"new":new_record})
            # Only now reveal the actual order, AFTER both sealed paper records.
            outcome=row["outcome"]
            for branch,record,payload in (
                    (models[0],old_record,old_payload),
                    (models[1],new_record,new_payload)):
                probabilities=forward_compatible_probabilities(payload[0],payload)
                loss=revealed_metric(outcome,probabilities)
                settled=branch.settle_and_learn(pre,outcome,payload,record)
                if settled["actual"]["net_yen"]!=(
                        settled["actual"]["payout_yen"]-settled["actual"]["stake_yen"]):
                    raise AssertionError("Invalid actual paper accounting")
                result={"stage":STAGE,"step":i,
                        "date":pre["race"]["race_date"],**settled,**loss,
                        "skipped_shadow_net_yen":(
                            settled.get("shadow_missed_profit_if_skipped_yen")
                            if branch.name=="new" else None)}
                seal(records,result)
                results.append(result)
            if i%30==0:
                print("GEN008_PARALLEL_MONTH",STAGE,"RACES",i,"OF",len(races),flush=True)
    summary=summarize(results)
    for branch in models:
        folder=target/branch.name
        folder.mkdir(exist_ok=False)
        cp={
            "branch":branch.name,"stage":STAGE,"week":week,
            "dates":[start,end],
            "parent_source":origin,
            "weights":branch.weights.tolist(),
            "weights_sha256":model_digest(branch.weights),
            "policy_sha256":branch.policy.digest(),
            "temperature":float(branch.temperature),
            "cumulative_races_from_GEN007":5328+sum(
                 json.loads((paths(n)/"summary.json").read_text(encoding="utf-8"))["races"]
                 for n in range(1,week) if (paths(n)/"summary.json").is_file()
            )+len(races)
        }
        (folder/"checkpoint.json").write_text(
            json.dumps(cp,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        branch.policy.save(folder/"behavior_policy.json")
        if cp["policy_sha256"]!=BehaviorPolicy.load(
               folder/"behavior_policy.json").digest():
            raise AssertionError("Policy failed round-trip")
    digest=hashlib.sha256("\n".join(race_ids).encode()).hexdigest()
    report={
        "id":STAGE,
        "status":"COMPLETE",
        "stage":STAGE,"week":week,"days":31,
        "dates":[start,end],"races":len(races),
        "source_reports":sources,"race_ids_sha256":digest,
        "parent":origin,
        "models":{"old":"unchanged GEN-007 training rule; independent cloned weights",
                  "new":"shadow-on-skip + coherent outcomes + online past-only calibration"},
        "comparison":summary,
        "training_only":True,
        "independent_held_out_exam_completed":False,
        "reserved_never_trained_exam_dates":["2023-10-01","2023-10-14"],
        "previously_reserved_exams_preserved":[
            ["2023-06-01","2023-06-07"],["2023-07-01","2023-07-07"]],
        "real_cash_excludes_shadow":True,
        "price_phase":"historical FINAL odds (not deadline quotes)",
        "paper_only":True,
        "old_model_files_modified":False,
        "next_stage_must_inherit_each_branch_separately":True,
        "note":"In-sample online paper profits are descriptive, NOT forward evidence."
    }
    (target/"summary.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_ONE_MONTH_BOTH_MODELS_COMPLETE",json.dumps(
        {"stage":STAGE,"races":len(races),"comparison":summary},
        ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-training",action="store_true")
    args=parser.parse_args()
    if args.run_training:
        train_month(3)
    else:
        print("GEN008 parallel month stage configured, no training without --run-training")

if __name__=="__main__":
    main()

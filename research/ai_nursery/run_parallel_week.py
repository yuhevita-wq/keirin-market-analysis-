"""Exactly ONE seven-day parallel training increment, old vs GEN-008.

Original GEN-007 checkpoints are immutable. No evaluation-period leakage.
Week 1: 2023-08-01..07, both clones from exact same parent weights.
Week N: both children inherit their OWN week N-1 frozen weights.
2023-10-01..14 is reserved for a later, single independent forward exam.

Invocation: python research/ai_nursery/run_parallel_week.py --week 1
No autonomous iteration or later weeks are triggered by this command.
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
FIRST=date(2023,8,1)
LAST_TRAIN=date(2023,9,30)
FUTURE_EXAM=date(2023,10,1)
PERIOD_DAYS=7

def paths(week):
    return BASE/f"week-{week:02d}"

def interval(week):
    if not isinstance(week,int) or not 1<=week<=9:
        raise ValueError("Week must be 1..9")
    start=FIRST+timedelta(days=(week-1)*PERIOD_DAYS)
    end=min(start+timedelta(days=6),LAST_TRAIN)
    if start>LAST_TRAIN or not end<FUTURE_EXAM:
        raise ValueError("Outside reserved training period")
    return start.isoformat(),end.isoformat()

def load_inherited(week):
    if week==1:
        parent,w,old,new=load_both_parents()
        parent_hash=parent["weights_sha256"]
        assert model_digest(w)==parent_hash
        assert old.digest()==new.digest()==parent["behavior_policy_sha256"]
        return [Branch("old",w.copy(),old),Branch("new",w.copy(),new)],{
            "origin":"GEN-007-C2","origin_weights_sha256":parent_hash,
            "origin_policy_sha256":parent["behavior_policy_sha256"]
        }
    previous=paths(week-1)
    if not (previous/"summary.json").is_file():
        raise RuntimeError(f"Week {week-1} checkpoint must complete first")
    meta=json.loads((previous/"summary.json").read_text(encoding="utf-8"))
    if meta.get("status")!="COMPLETE" or meta["week"]!=week-1:
        raise RuntimeError("Previous week is incomplete")
    branches=[]
    sha={}
    for name in ("old","new"):
        checkpoint=json.loads((previous/name/"checkpoint.json").read_text(encoding="utf-8"))
        policy_path=previous/name/"behavior_policy.json"
        if checkpoint["branch"]!=name or checkpoint["week"]!=week-1:
            raise RuntimeError("Corrupted previous-branch lineage")
        w=np.asarray(checkpoint["weights"],dtype=np.float64)
        if w.shape!=(D,) or model_digest(w)!=checkpoint["weights_sha256"]:
            raise RuntimeError("Corrupt inherited prediction weights")
        policy=(ProfitPolicy.inherit(policy_path) if name=="old"
                else inherited_shadow_policy(policy_path))
        if policy.digest()!=checkpoint["policy_sha256"]:
            raise RuntimeError("Corrupt inherited action weights")
        branches.append(Branch(name,w,policy,float(checkpoint["temperature"])))
        sha[name]={"prediction":model_digest(w),"behavior":policy.digest()}
    return branches,{"origin":f"GEN-008-PARALLEL-week-{week-1:02d}",
                     "inherited_hashes":sha}

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

def train_week(week):
    start,end=interval(week)
    target=paths(week)
    if target.exists():
        raise RuntimeError("Never overwrite an already created weekly model: "+str(target))
    models,origin=load_inherited(week)
    if {m.name for m in models}!={"old","new"}:
        raise AssertionError("Missing one parallel clone")
    races,sources=joined_races_for_training(start,end)
    if not races:raise RuntimeError("No complete joined races in this training week")
    if any(not start<=r["race"]["race_date"]<=end for r in races):
        raise RuntimeError("Out-of-window race")
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
            seal(selections,{"week":week,"step":i,"race_id":pre["race_id"],
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
                result={"week":week,"step":i,
                        "date":pre["race"]["race_date"],**settled,**loss,
                        "skipped_shadow_net_yen":(
                            settled.get("shadow_missed_profit_if_skipped_yen")
                            if branch.name=="new" else None)}
                seal(records,result)
                results.append(result)
            if i%30==0:
                print("GEN008_PARALLEL_WEEK",week,"RACES",i,"OF",len(races),flush=True)
    summary=summarize(results)
    for branch in models:
        folder=target/branch.name
        folder.mkdir(exist_ok=False)
        cp={
            "branch":branch.name,"week":week,
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
        "id":f"GEN-008-PARALLEL-WEEK-{week:02d}",
        "status":"COMPLETE",
        "week":week,"dates":[start,end],"races":len(races),
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
        "next_week_must_inherit_each_branch_separately":True,
        "note":"In-sample online paper profits are descriptive, NOT forward evidence."
    }
    (target/"summary.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_ONE_WEEK_BOTH_MODELS_COMPLETE",json.dumps(
        {"week":week,"races":len(races),"comparison":summary},
        ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--week",type=int,default=1)
    args=parser.parse_args()
    train_week(args.week)

if __name__=="__main__":
    main()

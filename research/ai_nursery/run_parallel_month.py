"""GEN-008 side-by-side additional 31-day learning, AFTER completed week-02.

Source: both week-02 branches (old and new) stay independent. Continue
chronologically on the SAME race IDs 2023-08-15..2023-09-14 inclusive.
Sealed-before-result files and post-result updates are never mixed.
No use of reserved future exam 2023-10-01..14. No original files overwritten.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import date,timedelta

import numpy as np

from train_week import ROOT,D
from behavior_policy import BehaviorPolicy
from gen008_experiment import model_digest
from parallel_gen008 import pure_pre_result,seal
from run_parallel_week import load_inherited,forward_compatible_probabilities,revealed_metric,summarize
from source_bridge import joined_races_for_training

BASE=ROOT/"research/ai_nursery/outputs/GEN-008-PARALLEL"
PARENT=BASE/"week-02"
OUT=BASE/"month-01-after-week02"
WEB=ROOT/"docs/ai-lab/gen008-month01.json"
FIRST=date(2023,8,15)
LAST=date(2023,9,14)
EXAM_FIRST=date(2023,10,1)
EXAM_LAST=date(2023,10,14)
DAYS=(LAST-FIRST).days+1
ID="GEN-008-PARALLEL-MONTH-01"
BRANCHES=("old","new")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_parent(branches,inheritance):
    s=_read(PARENT/"summary.json")
    if s["status"]!="COMPLETE" or s.get("week")!=2 or s["dates"]!=["2023-08-08","2023-08-14"]:
        raise RuntimeError("Previous frozen week-02 incomplete or incompatible")
    if inheritance.get("origin")!="GEN-008-PARALLEL-week-02":
        raise RuntimeError("Month does not inherit week-02")
    for b in branches:
        cp=_read(PARENT/b.name/"checkpoint.json")
        assert cp["dates"]==["2023-08-08","2023-08-14"]
        assert model_digest(b.weights)==cp["weights_sha256"]
        assert b.policy.digest()==cp["policy_sha256"]
        assert cp["cumulative_races_from_GEN007"]==5616
        assert math.isclose(b.temperature,float(cp["temperature"]))
    return s


def run():
    if OUT.exists():
        raise RuntimeError("Immutable GEN-008 monthly stage exists: refusing overwrite")
    if DAYS!=31 or FIRST<=date(2023,8,14) or LAST>=EXAM_FIRST:
        raise RuntimeError("Time guard violated")
    parents,inheritance=load_inherited(3)
    parent_summary=check_parent(parents,inheritance)
    all_rows,source_reports=joined_races_for_training(FIRST.isoformat(),LAST.isoformat())
    if not all_rows:raise RuntimeError("No joined data for next month")
    ids=[r["race_id"] for r in all_rows]
    if len(set(ids))!=len(ids):
        raise RuntimeError("Duplicate race ID in month")
    if any(not FIRST.isoformat()<=r["race"]["race_date"]<=LAST.isoformat() for r in all_rows):
        raise RuntimeError("Out-of-range race")
    if any(EXAM_FIRST.isoformat()<=r["race"]["race_date"]<=EXAM_LAST.isoformat()
           for r in all_rows):
        raise RuntimeError("Holdout leaked into training")
    models={b.name:b for b in parents}
    initial={
        name:{"prediction":model_digest(b.weights),"behavior":b.policy.digest(),
              "temperature":b.temperature}
        for name,b in models.items()
    }
    OUT.mkdir(parents=True,exist_ok=False)
    post=[]
    by_day=defaultdict(list)
    with (OUT/"sealed_pre_result_choices.jsonl").open("x",encoding="utf-8") as seals, \
         (OUT/"unsealed_post_result_learning.jsonl").open("x",encoding="utf-8") as history:
        for index,row in enumerate(all_rows,1):
            pre=pure_pre_result(row)
            choices={};payloads={}
            for name in BRANCHES:
                decisions,data=models[name].precommit(pre)
                choices[name]=decisions
                payloads[name]=data
            assert choices["new"]["shadow_page"] is not None
            assert choices["new"]["shadow_is_not_spent"] is True
            # Persist BOTH paper decisions + pre-result shadow BEFORE opening the result.
            seal(seals,{"stage":ID,"sequence":index,
                        "race_id":pre["race_id"],"date":pre["race"]["race_date"],
                        "old":choices["old"],"new":choices["new"],
                        "sealing_happened_before_result":True})
            outcome=row["outcome"]
            for name in BRANCHES:
                branch=models[name]
                probs=forward_compatible_probabilities(payloads[name][0],payloads[name])
                loss=revealed_metric(outcome,probs)
                learned=branch.settle_and_learn(
                    pre,outcome,payloads[name],choices[name])
                if learned["actual"]["net_yen"]!=learned["actual"]["payout_yen"]-learned["actual"]["stake_yen"]:
                    raise AssertionError("Impossible net ledger")
                record={
                    "stage":ID,"sequence":index,"date":pre["race"]["race_date"],
                    "race_id":pre["race_id"],
                    **learned,**loss,
                    "skipped_shadow_net_yen":(
                        learned.get("shadow_missed_profit_if_skipped_yen")
                        if name=="new" else None),
                    "shadow_counted_as_actual_cash":False,
                }
                seal(history,record)
                post.append(record)
                by_day[record["date"]].append(record)
            if index%100==0:
                print("GEN008_MONTH_PROGRESS",index,"/",len(all_rows),flush=True)
    score=summarize(post)
    closing={}
    for name in BRANCHES:
        branch=models[name]
        path=OUT/name
        path.mkdir()
        saved={
            "branch":name,"stage":ID,
            "dates":[FIRST.isoformat(),LAST.isoformat()],
            "parent_source":"GEN-008-PARALLEL/week-02",
            "parent_weights_sha256":initial[name]["prediction"],
            "parent_policy_sha256":initial[name]["behavior"],
            "training_races_this_stage":len(all_rows),
            "cumulative_races_from_GEN007":5616+len(all_rows),
            "weights":branch.weights.tolist(),
            "weights_sha256":model_digest(branch.weights),
            "policy_sha256":branch.policy.digest(),
            "temperature":float(branch.temperature),
        }
        (path/"checkpoint.json").write_text(
            json.dumps(saved,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        branch.policy.save(path/"behavior_policy.json")
        if BehaviorPolicy.load(path/"behavior_policy.json").digest()!=saved["policy_sha256"]:
            raise RuntimeError("Bad policy checkpoint")
        closing[name]={
            "prediction":saved["weights_sha256"],
            "behavior":saved["policy_sha256"],
            "temperature":saved["temperature"]
        }
    days=[]
    for offset in range(DAYS):
        day=(FIRST+timedelta(days=offset)).isoformat()
        rr=by_day[day]
        days.append({"date":day,"races":len(rr)//2,
                     "old_net_yen":sum(x["actual"]["net_yen"] for x in rr if x["branch"]=="old"),
                     "new_net_yen":sum(x["actual"]["net_yen"] for x in rr if x["branch"]=="new")})
    report={
        "id":ID,"status":"COMPLETE","generation":"GEN-008",
        "training_only":True,"paper_only":True,
        "days":DAYS,"dates":[FIRST.isoformat(),LAST.isoformat()],
        "races":len(all_rows),
        "parent":"GEN-008-PARALLEL/week-02",
        "inherited_checkpoints":initial,
        "previous_week02_races":parent_summary["races"],
        "same_race_ids_and_order_for_both":True,
        "race_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "comparison":score,"daily":days,
        "source_reports":source_reports,
        "closing_checkpoints":closing,
        "reserved_frozen_holdout":["2023-10-01","2023-10-14"],
        "other_protected_holdouts":[["2023-06-01","2023-06-07"],["2023-07-01","2023-07-07"]],
        "holdout_results_read_for_training":0,
        "all_shadow_returns_excluded_from_cash":True,
        "existing_GEN007_and_week01_week02_unchanged":True,
        "price_phase":"final historical odds, NOT 15-minute pre-close",
        "forward_evaluation":"NOT DONE: training returns are not unseen future results"
    }
    (OUT/"summary.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    WEB.parent.mkdir(parents=True,exist_ok=True)
    WEB.write_text(json.dumps({
        "id":ID,"status":report["status"],"dates":report["dates"],
        "days":DAYS,"races":len(all_rows),"comparison":score,
        "daily":days,"forward_evaluation":report["forward_evaluation"],
        "source":str((OUT/"summary.json").relative_to(ROOT))
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_ONE_MONTH_PARALLEL_COMPLETE",json.dumps({
        "races":len(all_rows),"comparison":score},ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()

"""GEN-008 V3: September-only calibration, untouched November forward exam.

Controls OLD and NEW are frozen as of 2023-09-14. V3 inherits frozen NEW
prediction weights, trains only a new scalar alpha on Sep 15-22,
audits Sep 23-30 without adjusting it and evaluates Nov 1-14 unchanged.

October 1-14 (previously observed) is NEVER used for V3 calibration.
Decisions of all 3 branches are sealed before reading each November result.
"""
from __future__ import annotations

import hashlib,json,math,os
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_week import ROOT
from source_bridge import joined_races
from gen008_frozen_exam import load_frozen as load_two_parents
from parallel_gen008 import pure_pre_result,seal
from gen008_experiment import score_papers,model_digest
from run_parallel_week import forward_compatible_probabilities,revealed_metric
from gen008_v3 import (TRAIN_FIRST,TRAIN_LAST,VALID_FIRST,VALID_LAST,TEST_FIRST,
    TEST_LAST,calibrate,matrix,blend,score,joint_portfolio,settle_v3,fit_signature)

BASE=ROOT/"research/ai_nursery/outputs/GEN-008-V3"
OUT=BASE/"first-independent-november-2023"
WEB=ROOT/"docs/ai-lab/gen008-v3-first-november.json"
STAGE="GEN-008-V3-FIRST-NOV-2023"


def load_model_settings():
    parents,snap=load_two_parents()
    new=parents["new"]
    meta=json.loads((ROOT/"research/ai_nursery/outputs/GEN-008-PARALLEL/month-01-after-week02/new/checkpoint.json").read_text())
    assert model_digest(new.weights)==meta["weights_sha256"]
    assert new.policy.digest()==meta["policy_sha256"]
    assert new.temperature==meta["temperature"]
    return parents,snap,meta


def recalc(pre,new,alpha):
    keys,m,q=matrix(pre,new.weights,new.temperature)
    return keys,blend(m,q,alpha)


def rows_in(first,last):
    rows,rep=joined_races(2023,first,last)
    assert all(first<=r["race"]["race_date"]<=last for r in rows)
    return rows,rep


def summarize(records):
    ans={}
    for name in ("old","new","v3"):
        cohort=[r for r in records if r["branch"]==name]
        st=sum(x["actual"]["stake_yen"] for x in cohort)
        paid=sum(x["actual"]["payout_yen"] for x in cohort)
        ans[name]={
          "races":len(cohort),"entered":sum(x["entered"] for x in cohort),
          "skipped":sum(not x["entered"] for x in cohort),
          "hit_races":sum(x["actual"]["hit"] for x in cohort),
          "marks":sum(x["actual"]["marks"] for x in cohort),
          "paper_staked_yen":st,"paper_payout_yen":paid,
          "paper_net_yen":paid-st,
          "paper_roi_percent":100*paid/st if st else None,
          "mean_trifecta_logloss":sum(x["trifecta_logloss"] for x in cohort)/len(cohort),
          "mean_trio_logloss":sum(x["trio_logloss"] for x in cohort)/len(cohort)
        }
    return ans


def check_before_exam(calib,train,val,exam_rows):
    assert len(train)>0 and len(val)>0 and len(exam_rows)>0
    assert set(r["race_id"] for r in train).isdisjoint(r["race_id"] for r in val)
    assert set(r["race_id"] for r in exam_rows).isdisjoint(
        {r["race_id"] for r in train+val})
    assert all(r["race"]["race_date"]<TEST_FIRST for r in train+val)
    assert calib["alpha"] in (i/10 for i in range(11))


def run():
    if OUT.exists() or WEB.exists():raise RuntimeError("V3 exam immutable archive exists")
    parents,start_hash,checkpoint=load_model_settings()
    train,train_report=rows_in(TRAIN_FIRST,TRAIN_LAST)
    valid,valid_report=rows_in(VALID_FIRST,VALID_LAST)
    exam,test_report=rows_in(TEST_FIRST,TEST_LAST)
    calibration=calibrate(train,valid,parents["new"].weights,parents["new"].temperature)
    check_before_exam(calibration,train,valid,exam)
    alpha=calibration["alpha"]
    signature=fit_signature(checkpoint,alpha)
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"v3_checkpoint.json").write_text(json.dumps({
        **signature,"train_report":train_report,"validation_report":valid_report,
        "calibration":calibration,"frozen_test_dates":[TEST_FIRST,TEST_LAST],
        "candidate_uses_october_results":False,"training_done_on_november":False
        },ensure_ascii=False,indent=2)+"\n")
    records=[]
    dates=defaultdict(list)
    with (OUT/"sealed_before_results.jsonl").open("x",encoding="utf-8") as decisions, \
         (OUT/"revealed_after_results.jsonl").open("x",encoding="utf-8") as settled:
        for i,row in enumerate(exam,1):
            pre=pure_pre_result(row)
            choices={}
            payloads={}
            for name in ("old","new"):
                choices[name],payloads[name]=parents[name].precommit(pre)
                assert choices[name]["prediction_before_sha256"]==start_hash[name]["prediction"]
                assert choices[name]["policy_before_sha256"]==start_hash[name]["behavior"]
            choices["v3"]=joint_portfolio(pre,parents["new"].weights,parents["new"].temperature,alpha)
            assert choices["v3"]["predictor_sha256"]==start_hash["new"]["prediction"]
            seal(decisions,{
              "stage":STAGE,"sequence":i,"date":pre["race"]["race_date"],
              "race_id":pre["race_id"],"predictor_start":start_hash,
              "v3_algorithm_digest":signature["algorithm_digest"],
              "decisions":{"old":choices["old"],"new":choices["new"],"v3":choices["v3"]},
              "before_result_revealed":True})
            outcome=row["outcome"]  # Result read ONLY after every choice is sealed.
            for name in ("old","new","v3"):
                if name=="v3":
                    result=settle_v3(choices["v3"],outcome)
                    keys,p=recalc(pre,parents["new"],alpha)
                    metrics=score(keys,p,outcome)
                else:
                    result=score_papers(choices[name]["pages"],pre["odds"],outcome)
                    arms=payloads[name][0]
                    q=forward_compatible_probabilities(arms,payloads[name])
                    metrics=revealed_metric(outcome,q)
                assert result["stake_yen"]==choices[name]["actual_paper_stake_yen"]
                obj={"stage":STAGE,"sequence":i,
                    "date":pre["race"]["race_date"],"race_id":pre["race_id"],
                    "branch":name,"entered":result["stake_yen"]>0,
                    "outcome_after_seal":list(outcome),"actual":result,**metrics,
                    "no_real_money":True}
                seal(settled,obj)
                records.append(obj)
                dates[obj["date"]].append(obj)
            if i%100==0:print("GEN008_V3_FORWARD_PROGRESS",i,"/",len(exam),flush=True)
    ending={name:{"prediction":model_digest(model.weights),
                   "behavior":model.policy.digest(),"temperature":model.temperature}
            for name,model in parents.items()}
    assert ending==start_hash,"Branch predictor or policy mutated in test"
    comparison=summarize(records)
    dsum=[{"date":d,"races":len(rr)//3,
           **{n+"_net_yen":sum(r["actual"]["net_yen"] for r in rr if r["branch"]==n)
               for n in ("old","new","v3")}}
           for d,rr in sorted(dates.items())]
    summary={
        "stage":STAGE,"status":"COMPLETE","kind":"FIRST_UNSEEN_FROZEN_COMPARISON",
        "branches":["old","new","v3"],
        "calibration_dates":[TRAIN_FIRST,TRAIN_LAST],
        "no_retune_validation_dates":[VALID_FIRST,VALID_LAST],
        "frozen_exam_dates":[TEST_FIRST,TEST_LAST],
        "november_exam_races":len(exam),
        "races_sha256":hashlib.sha256("\n".join(r["race_id"] for r in exam).encode()).hexdigest(),
        "v3_algorithm_digest":signature["algorithm_digest"],
        "v3_alpha":alpha,
        "v3_alpha_fit":calibration,
        "frozen_parent_start":start_hash,"frozen_parent_end":ending,
        "weights_policy_updated_during_test":False,
        "v3_predictor_weights_inherited_unchanged":True,
        "controls_do_not_use_v3_extra_calibration":True,
        "v3_extra_training_relative_to_controls":"Sep 15-22 market blending scalar ONLY",
        "october_test_data_not_used_for_v3_fit":True,
        "decision_before_outcome_seal":True,
        "historical_final_odds_only":True,
        "actual_real_bets":False,
        "source_report":test_report,"comparison":comparison,"daily":dsum,
        "interpretation":"NEW+OLD predictor frozen as of Sep 14, V3 alpha calibrated Sep 15-22 only; not a fully matched-training comparison."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    WEB.parent.mkdir(parents=True,exist_ok=True)
    WEB.write_text(json.dumps({
        "id":STAGE,"status":"COMPLETE","races":len(exam),"comparison":comparison,
        "daily":dsum,"source":str((OUT/"summary.json").relative_to(ROOT)),
        "market_blend_alpha":alpha,"historical_final_odds_only":True},
        ensure_ascii=False,indent=2)+"\n")
    print("GEN008_V3_FROZEN_EXAM_DONE",json.dumps({
        "races":len(exam),"alpha":alpha,"comparison":comparison},
        ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()

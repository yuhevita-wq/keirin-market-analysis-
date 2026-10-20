"""Continue FOUR deletion-first GEN-008 children throughout calendar 2024.

Parent weights and decision policies are sourced verbatim from 2023 reset
checkpoint. 2024-01-01..2024-12-31: racewise prequential shadow purchase,
then reveal winner and update the SAME four child weight arrays.
2025-01-01..2025-01-14: frozen out-of-sample exam, NO updating or retuning.

Historical 2024-01-01..01-14 was an earlier *frozen test*: it remains
immutable, but this NEW experiment explicitly promotes it to training input.
Do not represent any resulting 2024 score as an untouched evaluation.
Every book uses genuine FINAL 3連単/3連複 odds; no invented prices.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from deletion_core import (
    ACTIVE,BRANCHES,FEATURES,MARKETS,UNIT,LR,RIDGE_STEP,MAX_WEIGHT,
    books,deletion_plan,evaluate_plan,full_sheet_score,model_digest,
    probabilities,update
)
from run_reset_2023 import (
    OUT as PARENT_OUT,pre_result,all_book_digest,p_digest,seal
)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research/deletion_ai/outputs/GEN-008-DELETION-CONTINUE-2024"
WEB=ROOT/"docs/ai-lab/gen008-deletion-continue-2024.json"
STAGE="GEN008-DELETION-2024-CONTINUAL-LEARNING"
TRAIN_FIRST,TRAIN_LAST="2024-01-01","2024-12-31"
TEST_FIRST,TEST_LAST="2025-01-01","2025-01-14"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parent():
    previous=read(PARENT_OUT/"checkpoint.json")
    summary=read(PARENT_OUT/"summary.json")
    assert summary["status"]=="COMPLETE"
    assert previous["model_family"]=="GEN-008-DELETION-RESET"
    assert previous["training_dates"]==["2023-01-01","2023-09-30"]
    assert previous["validation_dates"]==["2023-10-01","2023-12-31"]
    assert previous["test_dates"]==["2024-01-01","2024-01-14"]
    assert previous["2024_test_did_not_train_or_tune"] is True
    assert previous["features"]==list(FEATURES)
    assert previous["feature_masks"]=={b:list(ACTIVE[b]) for b in BRANCHES}
    assert previous["learning_rate"]==LR
    assert previous["per_step_weight_shrink"]==RIDGE_STEP
    assert previous["max_absolute_coefficient"]==MAX_WEIGHT
    assert previous["policy_selected_only_from_2023_validation"]==summary["selected_probability_coverage"]
    weights={b:{m:np.asarray(previous["weights"][b][m],dtype=np.float64).copy()
                for m in MARKETS} for b in BRANCHES}
    assert {b:model_digest(weights,b) for b in BRANCHES}==previous["after_training_sha256"]
    assert previous["training_races"]==8088 and previous["validation_races"]==1908
    return weights,previous,summary


def load_data():
    sys.path.insert(0,str(ROOT/"research/ai_nursery"))
    from source_bridge import joined_races
    train,source_train=joined_races(2024,TRAIN_FIRST,TRAIN_LAST)
    test,source_test=joined_races(2025,TEST_FIRST,TEST_LAST)
    assert len(train)>5000 and len(test)>100,(len(train),len(test))
    assert train[0]["race"]["race_date"]>=TRAIN_FIRST
    assert train[-1]["race"]["race_date"]<=TRAIN_LAST
    assert test[0]["race"]["race_date"]>=TEST_FIRST
    assert test[-1]["race"]["race_date"]<=TEST_LAST
    assert len({r["race_id"] for r in train+test})==len(train)+len(test)
    return train,test,{"train":source_train,"test":source_test}


def add(target,record):
    target["races"]+=1
    for name in ("stake_yen","payout_yen","net_yen"):
        target[name]+=record[name]


def scored_summaries(st):
    ans={}
    for m,raw in st.items():
        item=dict(raw)
        item["roi_pct"]=100*item["payout_yen"]/item["stake_yen"] if item["stake_yen"] else None
        item["deleted_marks"]=int(item.get("deleted_marks",0))
        ans[m]=item
    return ans


def selected_plans(book,weights,covers):
    q={};plan={}
    for branch in BRANCHES:
        q[branch]={}
        plan[branch]={}
        for market in MARKETS:
            pred=probabilities(book[market],weights[branch][market],branch)
            q[branch][market]=pred
            plan[branch][market]=deletion_plan(book[market],pred,covers[branch])
    return q,plan


def controls_at_equal_count(pre,bk,plans,finish):
    """Negative control with exactly each branch's per-market retained counts.

    Uniformly deleted expected payout conditional on actual winning ticket,
    and realized cheapest-odds first tickets. Neither control is trained.
    """
    outcome={"3連単":"-".join(map(str,finish)),
             "3連複":"=".join(map(str,sorted(finish)))}
    controls={}
    for branch in BRANCHES:
        favorite_paid=0
        random_expected_paid=0.
        variance=0.
        saved_marks=0
        favorite_hit=0
        for market in MARKETS:
            book=bk[market]
            n=len(book["keys"])
            k=len(plans[branch][market]["kept"])
            assert 0<k<=n
            ticket=outcome[market]
            odds=float(pre["odds"][market][ticket])
            favorite=set(sorted(book["keys"],key=lambda t:(float(pre["odds"][market][t]),t))[:k])
            if ticket in favorite:
                favorite_paid+=int(round(UNIT*odds))
                favorite_hit+=1
            chance=k/n
            random_expected_paid+=chance*UNIT*odds
            variance+=chance*(1.-chance)*(UNIT*odds)**2
            saved_marks+=k
        controls[branch]={
            "same_stake_yen":saved_marks*UNIT,
            "favorite_payout_yen":favorite_paid,
            "favorite_winning_marks":favorite_hit,
            "random_expected_payout_yen":random_expected_paid,
            "random_payout_variance_yen2":variance,
            "expected_uniform_random":True,
        }
    return controls


def shadow(book,plans,finish):
    entries={}
    for branch in BRANCHES:
        entries[branch]=evaluate_plan(book,plans[branch],finish)
    return entries


def run():
    if OUT.exists() or WEB.exists():
        raise RuntimeError("Immutable 2024 continuation evidence already exists")
    weights,prior,prior_report=parent()
    original={b:model_digest(weights,b) for b in BRANCHES}
    original_policy=prior["policy_selected_only_from_2023_validation"]
    train,test,sources=load_data()
    OUT.mkdir(parents=True,exist_ok=False)
    training=defaultdict(lambda:defaultdict(float))
    training_monthly=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    frozen=defaultdict(lambda:defaultdict(float))
    daily_test=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    test_controls=defaultdict(lambda:defaultdict(float))
    stage_monthly_checkpoint=[]
    initial_test_hashes=None
    pre_path=OUT/"2024_pre_result_delete_sheets.jsonl.gz"
    result_path=OUT/"2024_outcome_and_updates.jsonl.gz"
    with gzip.open(pre_path,"wt",encoding="utf-8",compresslevel=6) as pre_file, \
         gzip.open(result_path,"wt",encoding="utf-8",compresslevel=6) as result_file:
        for index,row in enumerate(train,1):
            pre=pre_result(row)
            book=books(pre)
            q,plans=selected_plans(book,weights,original_policy)
            before={b:model_digest(weights,b) for b in BRANCHES}
            form={
                "stage":STAGE,"split":"2024_online_train",
                "sequence":index,"race_id":row["race_id"],
                "date":row["race"]["race_date"],
                "full_book_sha256":all_book_digest(book),
                "full_3rentan_mark_count":len(book["3連単"]["keys"]),
                "full_3renpuku_mark_count":len(book["3連複"]["keys"]),
                "full_book_paper_stake_yen":UNIT*(
                    len(book["3連単"]["keys"])+len(book["3連複"]["keys"])),
                "learners_2023_model_hash_before":before,
                "original_2023_coverage":original_policy,
                "probabilities_sha256":{b:{m:p_digest(q[b][m]) for m in MARKETS}
                                       for b in BRANCHES},
                "deleted":{b:{m:plans[b][m]["deleted"] for m in MARKETS}
                           for b in BRANCHES},
                "retained":{b:{m:plans[b][m]["kept"] for m in MARKETS}
                            for b in BRANCHES},
                "sealed_before_outcome":True,
                "no_real_wagers":True,
            }
            seal(pre_file,form)
            # Only now consume the stored outcome.
            finish=row["outcome"]
            full=full_sheet_score(book,finish)
            purchases=shadow(book,plans,finish)
            updated={}
            for b in BRANCHES:
                updated[b]=update(weights,book,b,finish)
            after={b:model_digest(weights,b) for b in BRANCHES}
            receipt={"stage":STAGE,"split":"2024_online_train",
                     "sequence":index,"race_id":row["race_id"],
                     "date":row["race"]["race_date"],
                     "result_after_seal":list(finish),
                     "all_marks_virtual":full,
                     "shadow_deletion_purchases":purchases,
                     "update_labels":updated,
                     "model_sha256_after":after}
            seal(result_file,receipt)
            month=row["race"]["race_date"][:7]
            add(training["ALL"],full)
            add(training_monthly[month]["ALL"],full)
            for b in BRANCHES:
                add(training[b],purchases[b])
                add(training_monthly[month][b],purchases[b])
                for key in (
                    "deleted_marks","saved_stake_yen",
                    "forfeited_payout_yen","deleted_winning_tickets",
                    "retained_winning_tickets","delta_net_vs_all_marks"):
                    training[b][key]+=purchases[b][key]
                    training_monthly[month][b][key]+=purchases[b][key]
            if index%650==0:
                print("GEN008_DELETION_2024_PROGRESS",index,len(train),month,flush=True)
            next_month=(train[index]["race"]["race_date"][:7]
                        if index<len(train) else None)
            if next_month!=month:
                stage_monthly_checkpoint.append({
                    "month":month,
                    "through_race_sequence":index,
                    "after_sha256":after,
                })
    completed={b:model_digest(weights,b) for b in BRANCHES}
    assert stage_monthly_checkpoint[-1]["month"]=="2024-12"
    assert stage_monthly_checkpoint[-1]["after_sha256"]==completed
    with gzip.open(OUT/"2025_frozen_pre_result_delete_sheets.jsonl.gz",
                   "wt",encoding="utf-8",compresslevel=6) as pre_file, \
         gzip.open(OUT/"2025_frozen_outcomes.jsonl.gz",
                   "wt",encoding="utf-8",compresslevel=6) as result_file:
        for index,row in enumerate(test,1):
            pre=pre_result(row)
            book=books(pre)
            q,plans=selected_plans(book,weights,original_policy)
            hashes={b:model_digest(weights,b) for b in BRANCHES}
            assert hashes==completed
            seal(pre_file,{
                "stage":STAGE,"split":"2025_frozen_out_of_sample",
                "sequence":index,"race_id":row["race_id"],
                "date":row["race"]["race_date"],
                "full_book_sha256":all_book_digest(book),
                "full_book_paper_stake_yen":UNIT*(
                    len(book["3連単"]["keys"])+len(book["3連複"]["keys"])),
                "weights_sha256":hashes,"coverage_targets_from_2023":original_policy,
                "probabilities_sha256":{b:{m:p_digest(q[b][m]) for m in MARKETS}
                                       for b in BRANCHES},
                "deleted":{b:{m:plans[b][m]["deleted"] for m in MARKETS}
                           for b in BRANCHES},
                "retained":{b:{m:plans[b][m]["kept"] for m in MARKETS}
                            for b in BRANCHES},
                "no_future_labels_or_updates":True,
            })
            finish=row["outcome"]
            full=full_sheet_score(book,finish)
            purchases=shadow(book,plans,finish)
            controls=controls_at_equal_count(pre,book,plans,finish)
            seal(result_file,{
                "stage":STAGE,"split":"2025_frozen_out_of_sample",
                "sequence":index,"race_id":row["race_id"],
                "date":row["race"]["race_date"],
                "result_after_seal":list(finish),
                "all_marks_virtual":full,
                "shadow_deletion_purchases":purchases,
                "same_mark_negative_controls":controls,
            })
            add(frozen["ALL"],full)
            add(daily_test[row["race"]["race_date"]]["ALL"],full)
            for b in BRANCHES:
                add(frozen[b],purchases[b])
                add(daily_test[row["race"]["race_date"]][b],purchases[b])
                for key in (
                    "deleted_marks","saved_stake_yen",
                    "forfeited_payout_yen","deleted_winning_tickets",
                    "retained_winning_tickets","delta_net_vs_all_marks"):
                    frozen[b][key]+=purchases[b][key]
                    daily_test[row["race"]["race_date"]][b][key]+=purchases[b][key]
                for key,value in controls[b].items():
                    if key in ("expected_uniform_random",):continue
                    test_controls[b][key]+=value
            if index%300==0:
                print("GEN008_DELETION_2025_FROZEN_PROGRESS",index,len(test),flush=True)
    assert completed=={b:model_digest(weights,b) for b in BRANCHES}
    checkpoint={
        "stage":STAGE,"model_family":"GEN-008-DELETION-2024-CONTINUATION",
        "parent_checkpoint_path":str((PARENT_OUT/"checkpoint.json").relative_to(ROOT)),
        "parent_2023_after_training_sha256":original,
        "2024_learned_sha256":completed,
        "branch_weights":{b:{m:weights[b][m].tolist() for m in MARKETS}
                          for b in BRANCHES},
        "branch_feature_masks":{b:list(ACTIVE[b]) for b in BRANCHES},
        "original_2023_validation_coverage_unchanged":original_policy,
        "learning_hyperparameters_unchanged":{"lr":LR,"ridge_step":RIDGE_STEP,
                                               "max_coefficient":MAX_WEIGHT},
        "2024_observed_old_january_2024_exam_now_promoted_to_development":True,
        "prior_2023_exam_artifacts_not_overwritten":True,
        "2025_outcomes_never_used_for_training_or_policy_tuning":True,
        "training_dates":[TRAIN_FIRST,TRAIN_LAST],
        "frozen_exam_dates":[TEST_FIRST,TEST_LAST],
        "monthly_weight_hashes":stage_monthly_checkpoint,
        "2024_training_races":len(train),"2025_test_races":len(test),
    }
    (OUT/"checkpoint.json").write_text(json.dumps(checkpoint,
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    train_s=scored_summaries(training)
    frozen_s=scored_summaries(frozen)
    same_mark={}
    for b in BRANCHES:
        v=test_controls[b]
        s=frozen_s[b]["stake_yen"]
        assert abs(v["same_stake_yen"]-s)<1e-6
        same_mark[b]={
            "learned_roi_pct":frozen_s[b]["roi_pct"],
            "favorite_same_count_roi_pct":100*v["favorite_payout_yen"]/s,
            "uniform_random_conditional_expected_roi_pct":100*v["random_expected_payout_yen"]/s,
            "learned_minus_favorite_pct_points":100*(
                frozen_s[b]["payout_yen"]-v["favorite_payout_yen"])/s,
            "learned_minus_random_expected_pct_points":100*(
                frozen_s[b]["payout_yen"]-v["random_expected_payout_yen"])/s,
            "favorite_winning_ticket_marks":int(v["favorite_winning_marks"]),
            "random_conditional_payout_sd_yen":math.sqrt(
                v["random_payout_variance_yen2"]),
            "same_stake_yen":int(s),
        }
    report={
        "stage":STAGE,"status":"COMPLETE",
        "original_2023_models_not_modified":True,
        "date_ranges":{"training":[TRAIN_FIRST,TRAIN_LAST],
                       "untouched_next_year":[TEST_FIRST,TEST_LAST]},
        "prior_january_2024_holdout_is_reused_as_NEW_training":True,
        "2024_training_is_prequential_not_blind_holdout":True,
        "2025_test_never_updates_model_or_policy":True,
        "original_2023_policy_and_learning_hyperparameters_frozen":True,
        "original_odds_and_podium_joined_exact_race_id":True,
        "both_3rentan_and_3renpuku_complete_only":True,
        "odds_phase":"FINAL (historical); NOT before closure",
        "paper_only_not_real_stakes":True,
        "train_races":len(train),"2025_test_races":len(test),
        "train":train_s,"test":frozen_s,
        "test_equal_mark_count_controls":same_mark,
        "train_by_month":{month:scored_summaries(v)
                          for month,v in sorted(training_monthly.items())},
        "test_by_day":{day:scored_summaries(v)
                       for day,v in sorted(daily_test.items())},
        "input_parent_sha256":original,
        "output_2024_sha256":completed,
        "source_reports":sources,
        "full_book_no_deletion_baseline_is_not_an_actual_profit_strategy":True,
        "interpretation":"2024 calendar year online learning is development. Claims of independently tested improvements apply only to frozen 2025 Jan 1-14."
    }
    (OUT/"summary.json").write_text(json.dumps(report,
          ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    WEB.parent.mkdir(parents=True,exist_ok=True)
    WEB.write_text(json.dumps({
        "stage":STAGE,"status":"COMPLETE",
        "train_races":len(train),"test_races":len(test),
        "train":train_s,"test":frozen_s,
        "matched_mark_controls":same_mark,
        "source":str((OUT/"summary.json").relative_to(ROOT)),
        "real_money":False,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_DELETION_CONTINUATION_2024_COMPLETE",json.dumps({
        "train_races":len(train),"test_races":len(test),
        "2025_test":frozen_s,"controls":same_mark},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.run:run()
    else:print("Frozen 2023 deletion checkpoint recognized. Run only with --run.")

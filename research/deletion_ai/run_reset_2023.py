"""Start FOUR deletion-first keirin AI children from ZERO on 2023-01-01.

Train: 2023-01-01..09-30, chronological pre-result full-book decisions.
Validate deletion policy: 2023-10-01..12-31 (weights frozen).
Test untouched by THESE fresh learners: 2024-01-01..01-14 (fully frozen).

A bet begins as the ENTIRE 3連単+3連複 book, then tickets are removed.
No ticket cap, box template, winner guess, or inherited 2023 Q3 knowledge.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deletion_core import (
    BRANCHES,MARKETS,COVERAGES,FEATURES,ACTIVE,LR,RIDGE_STEP,MAX_WEIGHT,
    UNIT,books,new_weights,probabilities,model_digest,
    full_sheet_score,deletion_plan,evaluate_plan,update,stable_hash,
)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research/deletion_ai/outputs/GEN-008-DELETION-RESET-2023"
WEB=ROOT/"docs/ai-lab/gen008-deletion-reset-2023.json"
STAGE="GEN008-2023-FROM-ZERO-DELETION-RESTART"
START="2023-01-01"
TRAIN_LAST="2023-09-30"
VAL_FIRST="2023-10-01"
VAL_LAST="2023-12-31"
TEST_FIRST="2024-01-01"
TEST_LAST="2024-01-14"
TRAIN_PROVISIONAL_COVERAGE=0.8


def seal(f,obj):
    f.write(json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    f.flush()


def pre_result(row):
    # Outcome deliberately omitted. No leaked rider post-result fields.
    return {"race_id":row["race_id"],"entries":row["entries"],
            "race":row["race"],"odds":row["odds"]}


def load_data():
    import sys
    sys.path.insert(0,str(ROOT/"research/ai_nursery"))
    from source_bridge import joined_races
    historical,hs=joined_races(2023,START,VAL_LAST)
    future,fs=joined_races(2024,TEST_FIRST,TEST_LAST)
    train=[r for r in historical if r["race"]["race_date"]<=TRAIN_LAST]
    valid=[r for r in historical if r["race"]["race_date"]>=VAL_FIRST]
    assert len(train)>=200 and len(valid)>=100 and len(future)>=100, (
        "insufficient genuine complete books",len(train),len(valid),len(future))
    assert train[0]["race"]["race_date"]>=START
    assert train[-1]["race"]["race_date"]<=TRAIN_LAST
    assert valid[0]["race"]["race_date"]>=VAL_FIRST
    assert valid[-1]["race"]["race_date"]<=VAL_LAST
    assert future[0]["race"]["race_date"]>=TEST_FIRST
    assert future[-1]["race"]["race_date"]<=TEST_LAST
    all_ids=[r["race_id"] for r in train+valid+future]
    assert len(set(all_ids))==len(all_ids)
    return train,valid,future,{"2023":hs,"2024":fs}


def p_digest(p):
    return hashlib.sha256(np.asarray(p,dtype="<f8").tobytes()).hexdigest()


def all_book_digest(book):
    return stable_hash({m:{"keys":list(book[m]["keys"]),
             "prices":list(map(float,book[m]["prices"]))} for m in MARKETS})


def learned_probability_cache(book,weights):
    return {b:{m:probabilities(book[m],weights[b][m],b)
               for m in MARKETS} for b in BRANCHES}


def plans_for(book,cache,coverage):
    return {b:{m:deletion_plan(book[m],cache[b][m],coverage)
               for m in MARKETS} for b in BRANCHES}


def train_online(rows,weights,folder):
    stats=defaultdict(lambda:defaultdict(float))
    months=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    paper=folder/"2023_training_pre_result.jsonl.gz"
    finishfile=folder/"2023_training_revealed.jsonl.gz"
    with gzip.open(paper,"wt",encoding="utf-8",compresslevel=6) as decision, \
         gzip.open(finishfile,"wt",encoding="utf-8",compresslevel=6) as after:
        for step,row in enumerate(rows,1):
            pre=pre_result(row)
            book=books(pre)
            qs=learned_probability_cache(book,weights)
            begin={b:model_digest(weights,b) for b in BRANCHES}
            full_marks=sum(len(book[m]["keys"]) for m in MARKETS)
            # Provisional deletion is NOT tuned on the current race's outcome.
            plans=plans_for(book,qs,TRAIN_PROVISIONAL_COVERAGE)
            sealed={"stage":STAGE,"step":step,"race_id":pre["race_id"],
                "date":pre["race"]["race_date"],"split":"train",
                "pre_result_only":True,"full_sheet_all_triples_and_trios":True,
                "full_book_sha256":all_book_digest(book),
                "full_mark_count":full_marks,"full_stake_yen":full_marks*UNIT,
                "branch_weights_before":begin,
                "probability_sha256":{b:{m:p_digest(qs[b][m]) for m in MARKETS}
                                      for b in BRANCHES},
                "provisional_coverage_target":TRAIN_PROVISIONAL_COVERAGE,
                "provisional_deletions":{b:{m:plans[b][m]["deleted"] for m in MARKETS}
                                          for b in BRANCHES}}
            seal(decision,sealed)
            # Read finish only after every branch decision record was flushed.
            finish=row["outcome"]
            actual=full_sheet_score(book,finish)
            updates={}
            for b in BRANCHES:
                entry=evaluate_plan(book,plans[b],finish)
                changes=update(weights,book,b,finish)
                updates[b]={"provisional_purchase":entry,
                            "feedback":changes,
                            "weights_after":model_digest(weights,b)}
                for metric in ("net_yen","stake_yen","payout_yen",
                               "saved_stake_yen","forfeited_payout_yen",
                               "deleted_winning_tickets","deleted_marks"):
                    stats[b][metric]+=entry[metric]
                    months[pre["race"]["race_date"][:7]][b][metric]+=entry[metric]
                stats[b]["races"]+=1
            stats["full"]["net_yen"]+=actual["net_yen"]
            stats["full"]["stake_yen"]+=actual["stake_yen"]
            stats["full"]["payout_yen"]+=actual["payout_yen"]
            stats["full"]["races"]+=1
            seal(after,{"stage":STAGE,"step":step,"race_id":pre["race_id"],
                 "after_pre_result_seal":True,"finish":list(finish),
                 "full_sheet":actual,"children":updates})
            if step%600==0:print("DELETION_TRAIN_2023",step,"/",len(rows),flush=True)
    for b in BRANCHES:
        assert stats[b]["races"]==len(rows)
    return ({"summary":{b:dict(v) for b,v in stats.items()},
             "monthly":{m:{b:dict(v) for b,v in branch.items()}
                        for m,branch in sorted(months.items())}},weights)


def frozen_evaluation(rows,weights,folder,split,selected=None):
    """Compute BOTH precommitted coverage frontier and realized deletion cost."""
    is_test=split=="test"
    if is_test:
        assert selected is not None and set(selected)==set(BRANCHES)
    recordfile=folder/(f"{split}_sealed_pre_result.jsonl.gz")
    resultfile=folder/(f"{split}_revealed.jsonl.gz")
    bymonth=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(float))))
    accum=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    bstart={b:model_digest(weights,b) for b in BRANCHES}
    with gzip.open(recordfile,"wt",encoding="utf-8",compresslevel=6) as prefile, \
         gzip.open(resultfile,"wt",encoding="utf-8",compresslevel=6) as postfile:
        for step,row in enumerate(rows,1):
            pre=pre_result(row)
            bk=books(pre)
            qp=learned_probability_cache(bk,weights)
            decisions={}
            frontier={}
            covlist={b:((selected[b],) if is_test else COVERAGES)
                     for b in BRANCHES}
            for b in BRANCHES:
                frontier[b]={}
                for cov in covlist[b]:
                    ps={m:deletion_plan(bk[m],qp[b][m],cov) for m in MARKETS}
                    frontier[b][str(cov)]=ps
                if is_test:
                    decisions[b]={"coverage_target":selected[b],
                       "kept":{m:frontier[b][str(selected[b])][m]["kept"]
                               for m in MARKETS},
                       "deleted":{m:frontier[b][str(selected[b])][m]["deleted"]
                                  for m in MARKETS},
                       "model_hash":bstart[b]}
                else:
                    decisions[b]={"candidate_coverages":
                       {str(c):{
                        m:{"deleted":frontier[b][str(c)][m]["deleted"],
                           "retained_marks":len(frontier[b][str(c)][m]["kept"])}
                        for m in MARKETS}
                        for c in covlist[b]},
                      "model_hash":bstart[b]}
            entry={"stage":STAGE,"step":step,"split":split,"race_id":pre["race_id"],
                   "date":pre["race"]["race_date"],"model_frozen":True,
                   "pre_result_only":True,"book_digest":all_book_digest(bk),
                   "all_marks_count":sum(len(bk[m]["keys"]) for m in MARKETS),
                   "all_full_stake_yen":UNIT*sum(len(bk[m]["keys"]) for m in MARKETS),
                   "probability_sha256":{b:{m:p_digest(qp[b][m]) for m in MARKETS}
                                         for b in BRANCHES},
                   "branches":decisions}
            seal(prefile,entry)
            # Outcomes are not touched before the above durable pre-result record.
            finish=row["outcome"]
            full=full_sheet_score(bk,finish)
            result={"stage":STAGE,"step":step,"race_id":pre["race_id"],
                    "split":split,"outcome_after_seal":list(finish),
                    "full_purchase":full,"branch_purchases":{}}
            month=pre["race"]["race_date"][:7]
            for b in BRANCHES:
                result["branch_purchases"][b]={}
                for cov in covlist[b]:
                    one=evaluate_plan(bk,frontier[b][str(cov)],finish)
                    result["branch_purchases"][b][str(cov)]=one
                    f=accum[b][str(cov)]
                    month_f=bymonth[month][b][str(cov)]
                    for dest in (f,month_f):
                        dest["races"]+=1
                        for metric in (
                          "stake_yen","payout_yen","net_yen","deleted_marks",
                          "saved_stake_yen","forfeited_payout_yen",
                          "deleted_winning_tickets","retained_winning_tickets",
                          "delta_net_vs_all_marks"):
                            dest[metric]+=one[metric]
            for dest in (accum["full"]["all"],bymonth[month]["full"]["all"]):
                dest["races"]+=1
                for metric in ("stake_yen","payout_yen","net_yen"):
                    dest[metric]+=full[metric]
            seal(postfile,result)
            if step%500==0:print("DELETION_"+split.upper()+"_2023_REPLAY",step,"/",len(rows),flush=True)
    assert bstart=={b:model_digest(weights,b) for b in BRANCHES}
    monthly={month:{b:{c:dict(v) for c,v in values.items()}
             for b,values in bs.items()} for month,bs in sorted(bymonth.items())}
    totals={b:{c:dict(v) for c,v in vs.items()} for b,vs in accum.items()}
    return {"monthly":monthly,"totals":totals,"model_start":bstart,
            "model_end":{b:model_digest(weights,b) for b in BRANCHES}}


def choose_coverage(valid):
    """Conservative 2023-only tuning: maximize worst-month ROI uplift.

    Control 1.0 = delete nothing, expected uplift 0.
    This discourages winning via one lucky payoff month or stopping all bets.
    """
    months=list(valid["monthly"])
    if len(months)<3:raise RuntimeError("All Oct/Nov/Dec required for 2023-only coverage calibration")
    policy={}
    table={}
    for b in BRANCHES:
        candidates=[]
        for c in COVERAGES:
            monthly_gains=[]
            for mon in months:
                original=valid["monthly"][mon]["full"]["all"]
                row=valid["monthly"][mon][b][str(c)]
                if original["stake_yen"]==0 or row["stake_yen"]==0:
                    raise AssertionError("Missing market bets")
                monthly_gains.append(
                    100*(row["payout_yen"]/row["stake_yen"]-
                         original["payout_yen"]/original["stake_yen"]))
            overall=valid["totals"][b][str(c)]
            full=valid["totals"]["full"]["all"]
            entry={"coverage":c,"monthly_roi_uplift_points":monthly_gains,
                "minimum_monthly_uplift_points":min(monthly_gains),
                "overall_roi_uplift_points":100*(
                    overall["payout_yen"]/overall["stake_yen"]-
                    full["payout_yen"]/full["stake_yen"]),
                "removed_marks":overall["deleted_marks"],
                "forfeited_payout_yen":overall["forfeited_payout_yen"],
                "saved_stake_yen":overall["saved_stake_yen"]}
            candidates.append(entry)
        best=max(candidates,key=lambda d:(
            round(d["minimum_monthly_uplift_points"],8),
            round(d["overall_roi_uplift_points"],8),
            d["coverage"]))
        policy[b]=best["coverage"]
        table[b]={"chosen":best,"candidate_table":candidates,
                  "no_selection_by_2024_outcomes":True}
    return policy,table


def run():
    if OUT.exists() or WEB.exists():
        raise RuntimeError("Immutable 2023 restart already exists; refusing overwrite")
    train,val,test,sources=load_data()
    weights=new_weights()
    initial={b:model_digest(weights,b) for b in BRANCHES}
    assert len(set(initial.values()))==1, "Scratch children must all begin at exact ZERO"
    OUT.mkdir(parents=True,exist_ok=False)
    train_result,weights=train_online(train,weights,OUT)
    after_train={b:model_digest(weights,b) for b in BRANCHES}
    validation=frozen_evaluation(val,weights,OUT,"validation")
    policies,policy_tables=choose_coverage(validation)
    # No 2024 result can change coefficients, masks, or retention target.
    test_result=frozen_evaluation(test,weights,OUT,"test",policies)
    assert after_train==validation["model_start"]==test_result["model_end"]
    assert validation["model_start"]==validation["model_end"]
    checkpoint={
        "model_family":"GEN-008-DELETION-RESET",
        "created_as_scratch_models":True,"initial_weights_all_zero":True,
        "inherited_old_new_v3_v3me_checkpoint_weights":False,
        "features":list(FEATURES),"feature_masks":{b:list(ACTIVE[b]) for b in BRANCHES},
        "initial_sha256":initial,"after_training_sha256":after_train,
        "weights":{b:{m:weights[b][m].tolist() for m in MARKETS} for b in BRANCHES},
        "training_dates":[START,TRAIN_LAST],
        "validation_dates":[VAL_FIRST,VAL_LAST],
        "test_dates":[TEST_FIRST,TEST_LAST],
        "training_races":len(train),"validation_races":len(val),"test_races":len(test),
        "trained_only_in_2023_01_to_09":True,
        "validation_2023_10_to_12_did_not_train":True,
        "2024_test_did_not_train_or_tune":True,
        "policy_selected_only_from_2023_validation":policies,
        "policy_selection_tables":policy_tables,
        "learning_rate":LR,"per_step_weight_shrink":RIDGE_STEP,
        "max_absolute_coefficient":MAX_WEIGHT,
    }
    (OUT/"checkpoint.json").write_text(
        json.dumps(checkpoint,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary={
        "stage":STAGE,"status":"COMPLETE",
        "2023_from_zero":True,"learning_goal":"Learn which tickets to DELETE from full books",
        "original_old_new_v3_v3me_untouched":True,
        "fresh_child_models":list(BRANCHES),
        "full_books_both_3rentan_and_3renpuku":True,
        "every_kept_market_has_at_least_one_real_ticket":True,
        "100_yen_per_mark":True,"no_real_money_bets":True,
        "historical_final_odds_not_pre_close":True,
        "train_period":[START,TRAIN_LAST],"validation_period":[VAL_FIRST,VAL_LAST],
        "test_period":[TEST_FIRST,TEST_LAST],
        "train_races":len(train),"validation_races":len(val),"test_races":len(test),
        "checkpoints":{b:{"before_training":initial[b],
                  "after_training":after_train[b]} for b in BRANCHES},
        "selected_probability_coverage":policies,
        "validation_tuning":policy_tables,
        "train_provisional_coverage":TRAIN_PROVISIONAL_COVERAGE,
        "train_prequential":train_result,
        "validation_scores":validation,
        "test_scores":test_result,
        "source":sources,
        "no_post_result_selection":True,
        "no_2024_outcome_impact_on_weights_or_policy":True,
        "notes":"In every race, deleting a losing ticket saves 100 yen, deleting the winning ticket forfeits its actual payout. Return percentages compare exactly the same races and both markets. The full-book baseline itself is not a profitable strategy."}
    (OUT/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    WEB.parent.mkdir(parents=True,exist_ok=True)
    WEB.write_text(json.dumps({
        "stage":STAGE,"status":"COMPLETE","train_races":len(train),
        "validation_races":len(val),"test_races":len(test),
        "selected_probability_coverage":policies,
        "test_comparison":test_result["totals"],
        "source":"research/deletion_ai/outputs/GEN-008-DELETION-RESET-2023/summary.json",
        "no_real_money_bets":True
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_DELETION_2023_RESTART_COMPLETE",json.dumps({
        "train_races":len(train),"validation_races":len(val),"test_races":len(test),
        "chosen_policy":policies,"test_comparison":test_result["totals"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if not args.run:
        print("2023-from-scratch deletion AI created. Explicit --run required.")
    else:
        run()

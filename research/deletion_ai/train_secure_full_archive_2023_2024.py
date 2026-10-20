"""Secure FULL-ARCHIVE GEN008 deletion re-training on 2023 + 2024 prices.

This is a REPLACEMENT experiment for the incorrectly called "full year"
learning from odds_only subsets. Source all authenticated encrypted 7-bet
odds 2023 / 2024, use price-validated 3連単 + 3連複 for today's deletion
models, and lock 2025 Jan 1..14 as independent exam.

No public plaintext odds, keys, individual market prices or per-ticket
picks. Training and label feedback run in an ephemeral trusted CI runner.
Sanitized aggregate scorecards + model weights only are Git-committed.

Period protocol:
    2023 Jan-Sep  / scratch training (no historical knowledge)
    2023 Oct-Dec / source-complete threshold validation, NO fitting
    2024 Jan-Dec / chronological per-race model updates
    2025 Jan1-14 / totally frozen, outcome revealed after decision
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from deletion_core import (
    ACTIVE,BRANCHES,COVERAGES,FEATURES,MARKETS,UNIT,
    LR,RIDGE_STEP,MAX_WEIGHT,books,deletion_plan,evaluate_plan,
    full_sheet_score,model_digest,new_weights,probabilities,update,
)
from secure_seven_bet_bridge import ROOT,iter_year,secure_key_temp
from run_reset_2023 import choose_coverage

STAGE="GEN008-SECURE-FULL-SEVEN-ARCHIVE-2023-2024"
OUT=ROOT/"research/deletion_ai/outputs/GEN-008-SECURE-FULL-ARCHIVE-2023-2024"
PUBLIC=ROOT/"docs/ai-lab/gen008-secure-full-archive.json"
TRAIN_2023=("2023-01-01","2023-09-30")
VALID_2023=("2023-10-01","2023-12-31")
TRAIN_2024=("2024-01-01","2024-12-31")
TEST_2025=("2025-01-01","2025-01-14")
PROVISIONAL_COVERAGE=.8
MIN_2023_TOTAL=20000
MIN_2024_TOTAL=19000


def counter():
    return defaultdict(float)


def get_prior_probabilities(b,weights):
    return {branch:{m:probabilities(b[m],weights[branch][m],branch)
             for m in MARKETS} for branch in BRANCHES}


def choices(book,weights,coverages):
    p=get_prior_probabilities(book,weights)
    return {branch:{m:deletion_plan(book[m],p[branch][m],coverages[branch])
             for m in MARKETS} for branch in BRANCHES}


def record(tgt,receipt):
    tgt["races"]+=1
    for k in ("stake_yen","payout_yen","net_yen"):
        tgt[k]+=receipt[k]
    for k in ("deleted_marks","saved_stake_yen","forfeited_payout_yen",
              "deleted_winning_tickets","retained_winning_tickets",
              "delta_net_vs_all_marks"):
        if k in receipt:tgt[k]+=receipt[k]


def public_scores(accounting):
    ans={}
    for name,raw in accounting.items():
        if not raw["races"]:continue
        x=dict(raw)
        x["races"]=int(raw["races"])
        x["roi_pct"]=100*raw["payout_yen"]/raw["stake_yen"] if raw["stake_yen"] else None
        for k in ("deleted_marks","deleted_winning_tickets","retained_winning_tickets"):
            if k in x:x[k]=int(x[k])
        ans[name]=x
    return ans


def favorite_vs_random(row,plans):
    """Use only already sealed market ticket counts, before revealing outcome."""
    predicted={}
    for branch in BRANCHES:
        predicted[branch]={}
        for market in MARKETS:
            k=len(plans[branch][market]["kept"])
            odds=row["odds"][market]
            assert 0<k<=len(odds)
            pred={"chosen_count":k,"total_count":len(odds),
                  "favorite_mark_set":set(sorted(odds,key=lambda t:(odds[t],t))[:k])}
            predicted[branch][market]=pred
    return predicted


def settle_controls(row,predicted,finish):
    wins={"3連単":"-".join(map(str,finish)),
          "3連複":"=".join(map(str,sorted(finish)))}
    totals={}
    for branch in BRANCHES:
        stake=0;favorite=0;random_mean=0;variance=0
        for market in MARKETS:
            p=predicted[branch][market]
            n=p["total_count"];k=p["chosen_count"]
            win=wins[market]
            cash=UNIT*row["odds"][market][win]
            chance=k/n
            stake+=UNIT*k
            favorite+=int(round(cash)) if win in p["favorite_mark_set"] else 0
            random_mean+=chance*cash
            variance+=chance*(1-chance)*cash*cash
        totals[branch]={"stake_yen":stake,"favorite_payout_yen":favorite,
                        "uniform_random_conditional_expected_payout_yen":random_mean,
                        "uniform_random_conditional_variance_yen2":variance}
    return totals


def private_receipt_digest(race_id,b,weights,plans):
    """Opaque public audit digest, never public selections or market prices."""
    obj={
      "race_id":race_id,
      "model_sha256":{branch:model_digest(weights,branch) for branch in BRANCHES},
      "price_hash":hashlib.sha256(json.dumps(
         {m:{"keys":list(b[m]["keys"]),
             "prices":[float(x) for x in b[m]["prices"]]}
          for m in MARKETS},ensure_ascii=False,separators=(",",":")).encode()).hexdigest(),
      "deletions_hash":hashlib.sha256(json.dumps(
         {name:{m:plans[name][m]["deleted"] for m in MARKETS}
          for name in BRANCHES},ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
    }
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def locked_month_counter(month_stats):
    return {m:public_scores(stats) for m,stats in sorted(month_stats.items())}


def run():
    if OUT.exists() or PUBLIC.exists():
        raise RuntimeError("Immutable full-archive training record already exists")
    weights=new_weights()
    first={b:model_digest(weights,b) for b in BRANCHES}
    assert len(set(first.values()))==1
    source_stats={name:defaultdict(int) for name in ("2023","2024","2025")}
    train23=defaultdict(counter)
    valid=defaultdict(lambda:defaultdict(counter))
    val_monthly=defaultdict(lambda:defaultdict(lambda:defaultdict(counter)))
    train24=defaultdict(counter)
    monthly24=defaultdict(lambda:defaultdict(counter))
    test25=defaultdict(counter)
    controls25=defaultdict(counter)
    monthly_model_sha=[]
    ledger=hashlib.sha256(b"GEN008_FULL_SEVEN_ARCHIVE_PRE_RESULT_LEDGER_V1")
    total_year={}
    # Keep plaintext odds and RSA key in only the runner's ephemeral temp tree.
    manager,key=secure_key_temp()
    try:
        for row in iter_year(2023,"2023-01-01","2023-12-31",key,source_stats["2023"]):
            day=row["race"]["race_date"]
            b=books(row)
            actual_full=full_sheet_score(b,row["outcome"])
            if day<=TRAIN_2023[1]:
                stage="2023_train"
                policy={branch:PROVISIONAL_COVERAGE for branch in BRANCHES}
            else:
                stage="2023_validation"
                policy=None
            if stage=="2023_train":
                plan=choices(b,weights,policy)
                ledger.update(private_receipt_digest(row["race_id"],b,weights,plan).encode())
                record(train23["ALL"],actual_full)
                for branch in BRANCHES:
                    ticket=evaluate_plan(b,plan[branch],row["outcome"])
                    record(train23[branch],ticket)
                    update(weights,b,branch,row["outcome"])
            else:
                # Candidate coverages are SEALED before using the known result;
                # training weights were frozen at the end of September.
                ledger.update(("VALIDATION_"+row["race_id"]).encode())
                record(valid["ALL"]["all"],actual_full)
                record(val_monthly[day[:7]]["full"]["all"],actual_full)
                p=get_prior_probabilities(b,weights)
                plans={
                  branch:{c:{m:deletion_plan(b[m],p[branch][m],c)
                             for m in MARKETS}
                          for c in COVERAGES} for branch in BRANCHES
                }
                for branch in BRANCHES:
                    for c in COVERAGES:
                        outcome=evaluate_plan(b,plans[branch][c],row["outcome"])
                        record(valid[branch][str(c)],outcome)
                        record(val_monthly[day[:7]][branch][str(c)],outcome)
        trained23={b:model_digest(weights,b) for b in BRANCHES}
        train23_races=int(train23["ALL"]["races"])
        valid23_races=int(valid["ALL"]["all"]["races"])
        if train23_races+valid23_races<MIN_2023_TOTAL:
            raise RuntimeError("INSUFFICIENT_2023_SEVEN_BET_SOURCE: refuse claiming full archive training")
        validation_report={
          "monthly":{mo:{k:{c:dict(t) for c,t in ss.items()}
                          for k,ss in books_.items()}
                     for mo,books_ in val_monthly.items()},
          "totals":{k:{c:dict(t) for c,t in ss.items()}
                    for k,ss in valid.items()},
        }
        policies,policy_tables=choose_coverage(validation_report)
        print("SECURE_SEVEN_BET_2023_PHASE_COMPLETE",json.dumps({
            "train_2023":train23_races,
            "validation_2023":valid23_races,
            "policy_2023_only":policies},ensure_ascii=False),flush=True)
        # 2024: ORIGINAL 2023 weights are continued. Never reset or refit.
        for row in iter_year(2024,"2024-01-01","2024-12-31",key,source_stats["2024"]):
            b=books(row)
            pre=choices(b,weights,policies)
            ledger.update(private_receipt_digest(row["race_id"],b,weights,pre).encode())
            full=full_sheet_score(b,row["outcome"])
            month=row["race"]["race_date"][:7]
            record(train24["ALL"],full)
            record(monthly24[month]["ALL"],full)
            for branch in BRANCHES:
                receipt=evaluate_plan(b,pre[branch],row["outcome"])
                record(train24[branch],receipt)
                record(monthly24[month][branch],receipt)
                update(weights,b,branch,row["outcome"])
            total_year[month]=int(monthly24[month]["ALL"]["races"])
            if total_year[month]%700==0:
                print("SECURE_SEVEN_BET_2024_MONTH_PROGRESS",json.dumps(
                    {"month":month,"races_this_month":total_year[month],
                     "year_total":int(train24["ALL"]["races"])},
                    ensure_ascii=False),flush=True)
        if train24["ALL"]["races"]<MIN_2024_TOTAL:
            raise RuntimeError("INSUFFICIENT_2024_SEVEN_BET_SOURCE: refuse claiming full archive training")
        final={b:model_digest(weights,b) for b in BRANCHES}
        # 2025 is held entirely frozen; identical code, no fitting or tuning.
        for row in iter_year(2025,"2025-01-01","2025-01-14",key,source_stats["2025"]):
            b=books(row)
            pre=choices(b,weights,policies)
            # A sealed decision digest is part of a running commitment, before
            # any settlement is calculated. Not a new fitted feature.
            ledger.update(private_receipt_digest(row["race_id"],b,weights,pre).encode())
            controls_pre=favorite_vs_random(row,pre)
            finish=row["outcome"]
            full=full_sheet_score(b,finish)
            record(test25["ALL"],full)
            for branch in BRANCHES:
                receipt=evaluate_plan(b,pre[branch],finish)
                record(test25[branch],receipt)
            controls=settle_controls(row,controls_pre,finish)
            for branch in BRANCHES:
                for k,v in controls[branch].items():
                    controls25[branch][k]+=v
        assert final=={b:model_digest(weights,b) for b in BRANCHES}
    finally:
        manager.cleanup()
    if test25["ALL"]["races"]<400:
        raise RuntimeError("Insufficient 2025 January independent exam")
    match={}
    for b in BRANCHES:
        kept=test25[b]
        v=controls25[b]
        stake=kept["stake_yen"]
        assert stake==v["stake_yen"]
        match[b]={
          "model_roi_pct":100*kept["payout_yen"]/stake,
          "same_count_favorite_roi_pct":100*v["favorite_payout_yen"]/stake,
          "same_count_uniform_random_expected_roi_pct":
              100*v["uniform_random_conditional_expected_payout_yen"]/stake,
          "model_minus_favorite_pct_points":
              100*(kept["payout_yen"]-v["favorite_payout_yen"])/stake,
          "model_minus_uniform_random_expected_pct_points":
              100*(kept["payout_yen"]-
              v["uniform_random_conditional_expected_payout_yen"])/stake,
          "exact_same_number_of_tickets":True,
        }
    out={
        "status":"COMPLETE","study":STAGE,
        "source":"data/odds_ciphertext YEAR/day/ sealed 7-bet odds",
        "original_7_bet_cms_ciphertexts_consumed":True,
        "all_original_seven_market_price_types_verified_per_race":True,
        "modeled_market_types":["3連単","3連複"],
        "other_five_price_types_not_yet_model_inputs":True,
        "not_all_seven_types_modeled":True,
        "private_key_or_plaintext_quotes_published":False,
        "only_aggregate_statistics_and_weights_published":True,
        "result_labels_source":"data/result_universe_v2/combined",
        "exact_race_id_result_join_without_racecard_dependency":True,
        "source_window_2023_training":list(TRAIN_2023),
        "source_window_2023_validation":list(VALID_2023),
        "source_window_2024_training":list(TRAIN_2024),
        "source_window_2025_frozen_exam":list(TEST_2025),
        "source_stats":{yr:dict(stats) for yr,stats in source_stats.items()},
        "counts":{"train23":train23_races,
                  "validation23":valid23_races,
                  "train24":int(train24["ALL"]["races"]),
                  "test25":int(test25["ALL"]["races"])},
        "baseline_all_ticket_100yen":True,"real_money_wagers":False,
        "2023_scratch_start_weights_sha256":first,
        "2023_after_training_weights_sha256":trained23,
        "2024_after_training_weights_sha256":final,
        "2023_validation_only_selected_coverage":policies,
        "2023_validation_policy_candidate_table":policy_tables,
        "parameters_constant_between_2023_and_2024":{
            "LR":LR,"RIDGE_STEP":RIDGE_STEP,"MAX_WEIGHT":MAX_WEIGHT,
            "FEATURES":list(FEATURES),"ACTIVE":{b:list(ACTIVE[b]) for b in BRANCHES}},
        "2023_train":public_scores(train23),
        "2023_validation":{b:{key:public_scores({"x":entry})["x"]
                    for key,entry in policies_.items()}
                   for b,policies_ in valid.items()},
        "2024_train_prequential":public_scores(train24),
        "2024_monthly":locked_month_counter(monthly24),
        "2025_frozen_exam":public_scores(test25),
        "2025_same_count_negative_controls":match,
        "pre_result_seal_digest_sha256":ledger.hexdigest(),
        "2025_outcomes_never_used_for_training_or_policy_tuning":True,
        "historical_prices_are_FINAL_not_T_minus_15":True,
        "no_new_collection_or_source_scraping":True,
        "warning":"This is a new true-full-archive experiment, not an overwrite or retroactive repair of previously published subset study results.",
    }
    checkpoint={
        "study":STAGE,"status":"COMPLETE",
        "weights":{b:{m:weights[b][m].tolist() for m in MARKETS} for b in BRANCHES},
        "sha256_start":first,
        "sha256_end_2023":trained23,"sha256_end_2024":final,
        "policy":policies,
        "parameter_masks":{b:list(ACTIVE[b]) for b in BRANCHES},
        "no_future_outcome_training":True,
        "source_race_count":out["counts"],
    }
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"checkpoint.json").write_text(json.dumps(checkpoint,ensure_ascii=False,
         indent=2)+"\n",encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(out,ensure_ascii=False,
         indent=2)+"\n",encoding="utf-8")
    PUBLIC.parent.mkdir(parents=True,exist_ok=True)
    PUBLIC.write_text(json.dumps({
        "study":STAGE,"status":"COMPLETE",
        "counts":out["counts"],"train_2024":out["2024_train_prequential"],
        "frozen_2025":out["2025_frozen_exam"],
        "negative_controls":match,
        "source":"research/deletion_ai/outputs/GEN-008-SECURE-FULL-ARCHIVE-2023-2024/summary.json",
        "plaintext_public":False,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_FULL_SEVEN_ARCHIVE_RUN_COMPLETE",json.dumps({
        "counts":out["counts"],
        "2025_frozen_roi":{b:x["roi_pct"] for b,x in out["2025_frozen_exam"].items()},
        "controls":match,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()

"""Frozen 2025 matched-race ablation: did 2024 updates improve deletion?

Re-score SAME 2025-01-01..14 races with 2023-end parent vs 2024-end
child weights, same frozen 2023 market coverage policies, unchanged odds.
2025 labels used ONLY to settle after each set of tickets is determined.
No updates, hyperparameter selection, or test retuning of either model.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from deletion_core import BRANCHES,MARKETS,books,probabilities,deletion_plan,evaluate_plan,full_sheet_score,model_digest
from continue_train_2024 import ROOT,OUT,load_data,parent,pre_result

TARGET=OUT/"2024_learning_gain_ablation_2025.json"


def weights_from_child(cp):
    result={b:{m:np.asarray(cp["branch_weights"][b][m],dtype=np.float64)
            for m in MARKETS} for b in BRANCHES}
    assert {b:model_digest(result,b) for b in BRANCHES}==cp["2024_learned_sha256"]
    return result


def decision(book,weight,branch,coverage):
    return {m:deletion_plan(book[m],probabilities(book[m],weight[branch][m],branch),coverage)
            for m in MARKETS}


def score(probs,book,finish):
    return -math.log(max(float(probs[book["keys"].index(finish)]),1e-50))


def evaluate():
    if TARGET.exists():
        raise RuntimeError("Immutable learning-gain report already exists")
    prior_weights,orig,_=parent()
    cp=json.loads((OUT/"checkpoint.json").read_text(encoding="utf-8"))
    prior_summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    child=weights_from_child(cp)
    assert prior_summary["status"]=="COMPLETE"
    assert cp["parent_2023_after_training_sha256"]=={
        b:model_digest(prior_weights,b) for b in BRANCHES}
    assert cp["2025_outcomes_never_used_for_training_or_policy_tuning"]
    coverage=cp["original_2023_validation_coverage_unchanged"]
    train,test,reports=load_data()
    assert len(test)==prior_summary["2025_test_races"]
    series={model:{"before_2024":defaultdict(float),"after_2024":defaultdict(float)}
            for model in BRANCHES}
    daily={model:defaultdict(lambda:{"before_net":0,"after_net":0})
           for model in BRANCHES}
    for index,row in enumerate(test,1):
        pre=pre_result(row)
        book=books(pre)
        # All predictions must be generated without the actual outcome.
        decided={}
        for branch in BRANCHES:
            prior_decision=decision(book,prior_weights,branch,coverage[branch])
            child_decision=decision(book,child,branch,coverage[branch])
            decided[branch]={
                "before_2024":prior_decision,
                "after_2024":child_decision,
                "prior_probs":{m:probabilities(book[m],prior_weights[branch][m],branch)
                               for m in MARKETS},
                "child_probs":{m:probabilities(book[m],child[branch][m],branch)
                               for m in MARKETS},
            }
        # Only now reveal the historical outcome, for fair paired settlement.
        finish=row["outcome"]
        winners={"3連単":"-".join(map(str,finish)),
                 "3連複":"=".join(map(str,sorted(finish)))}
        for branch in BRANCHES:
            obj=decided[branch]
            for stage,probkey in (("before_2024","prior_probs"),
                                  ("after_2024","child_probs")):
                entry=evaluate_plan(book,obj[stage],finish)
                totals=series[branch][stage]
                totals["races"]+=1
                for k in ("stake_yen","payout_yen","net_yen",
                          "deleted_marks","saved_stake_yen",
                          "forfeited_payout_yen","deleted_winning_tickets",
                          "retained_winning_tickets","delta_net_vs_all_marks"):
                    totals[k]+=entry[k]
                for m in MARKETS:
                    totals[m+"_logloss_sum"]+=score(obj[probkey][m],book[m],winners[m])
                daily[branch][row["race"]["race_date"]][
                    "before_net" if stage=="before_2024" else "after_net"
                ]+=entry["net_yen"]
        if index%350==0:print("GEN008_DELETION_ABLATION_2025",index,len(test),flush=True)
    comparison={}
    for branch in BRANCHES:
        a=series[branch]["before_2024"]
        b=series[branch]["after_2024"]
        comparison[branch]={
            "races":len(test),
            "baseline_2023_policy_coverage":coverage[branch],
            "before_2024":{
                **{k:v for k,v in a.items() if not k.endswith("_sum")},
                "roi_pct":100*a["payout_yen"]/a["stake_yen"],
                "trifecta_logloss":a["3連単_logloss_sum"]/len(test),
                "trio_logloss":a["3連複_logloss_sum"]/len(test),
            },
            "after_2024":{
                **{k:v for k,v in b.items() if not k.endswith("_sum")},
                "roi_pct":100*b["payout_yen"]/b["stake_yen"],
                "trifecta_logloss":b["3連単_logloss_sum"]/len(test),
                "trio_logloss":b["3連複_logloss_sum"]/len(test),
            },
            "delta_roi_pct_points":100*(
                b["payout_yen"]/b["stake_yen"]-a["payout_yen"]/a["stake_yen"]),
            "delta_trifecta_logloss":(
                b["3連単_logloss_sum"]-a["3連単_logloss_sum"])/len(test),
            "delta_trio_logloss":(
                b["3連複_logloss_sum"]-a["3連複_logloss_sum"])/len(test),
            "months_trained_between_checkpoints":12,
            "daily_net_yen":[{"date":d,**v} for d,v in sorted(daily[branch].items())],
        }
        for k in ("stake_yen","payout_yen","net_yen","deleted_marks",
                  "saved_stake_yen","forfeited_payout_yen",
                  "deleted_winning_tickets","retained_winning_tickets"):
            assert b[k]==prior_summary["test"][branch][k]
        assert math.isclose(comparison[branch]["after_2024"]["roi_pct"],
                            prior_summary["test"][branch]["roi_pct"],
                            rel_tol=1e-12)
    assert {b:model_digest(child,b) for b in BRANCHES}==cp["2024_learned_sha256"]
    assert {b:model_digest(prior_weights,b) for b in BRANCHES}==cp["parent_2023_after_training_sha256"]
    output={
        "study":"GEN-008-DELETION-2024-CONTINUATION-MATCHED-ABALATION",
        "status":"COMPLETE",
        "period":["2025-01-01","2025-01-14"],
        "races":len(test),
        "holdout_results_used_for_retraining":False,
        "market_coverage_and_code_same_both_checkpoints":True,
        "prior_2023_weights_and_2024_weights_unchanged":True,
        "prior_2023_frozen_eval_not_overwritten":True,
        "same_race_counterfactual_2023_parent_vs_2024_child":True,
        "not_a_new_independent_2025_holdout":True,
        "historical_final_odds_only":True,
        "real_money":False,
        "comparison":comparison,
        "method_note":"Compare precommitted deletions at same 2023-fixed probability mass target but potentially differing numbers of retained marks; equal-count popular/random controls are separately archived in the 2024 continuation.",
    }
    TARGET.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_DELETION_2024_GAIN_ABLATION_COMPLETE",json.dumps({
        b:{"roi_before":comparison[b]["before_2024"]["roi_pct"],
           "roi_after":comparison[b]["after_2024"]["roi_pct"],
           "improvement_pct_points":comparison[b]["delta_roi_pct_points"],
           "trifecta_loss_delta":comparison[b]["delta_trifecta_logloss"],
           "trio_loss_delta":comparison[b]["delta_trio_logloss"]}
        for b in BRANCHES},ensure_ascii=False),flush=True)


if __name__=="__main__":
    evaluate()

"""Post-sealed negative controls for 2023-from-zero deletion learners.

Frozen 2024 test tickets are NOT retuned. For each race and each model,
compare original deletion portfolio to:
1) Uniform random deletion with EXACTLY the same number of tickets per
   market and race (analytical expected return, no luck or RNG).
2) Keep the cheapest-odds tickets with those exact same per-market counts
   (simple favorite-ranking counterfactual).

This isolates whether deletion *which tickets* beats mere stake reduction.
Controls are diagnostics only, never used to fit or select either model.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict

from deletion_core import BRANCHES,MARKETS,UNIT
from run_reset_2023 import OUT,pre_result,load_data


def records(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)


def main():
    saved=OUT/"deletion_negative_controls.json"
    if saved.exists():
        raise RuntimeError("Immutable negative controls already exist")
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    chk=json.loads((OUT/"checkpoint.json").read_text(encoding="utf-8"))
    train,valid,test,_=load_data()
    assert summary["status"]=="COMPLETE"
    assert len(test)==summary["test_races"]
    assert summary["no_2024_outcome_impact_on_weights_or_policy"]
    assert chk["2024_test_did_not_train_or_tune"]
    choices=records(OUT/"test_sealed_pre_result.jsonl.gz")
    outcomes=records(OUT/"test_revealed.jsonl.gz")
    aggregation=defaultdict(lambda:defaultdict(float))
    day=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    for i,row in enumerate(test,1):
        sealed=next(choices)
        score=next(outcomes)
        assert sealed["step"]==score["step"]==i
        assert sealed["race_id"]==score["race_id"]==row["race_id"]
        assert sealed["pre_result_only"] and sealed["model_frozen"]
        assert score["outcome_after_seal"]==list(row["outcome"])
        day_id=row["race"]["race_date"]
        truth={
            "3連単":"-".join(map(str,row["outcome"])),
            "3連複":"=".join(map(str,sorted(row["outcome"])))
        }
        for branch in BRANCHES:
            purchased=score["branch_purchases"][branch][str(chk[
                "policy_selected_only_from_2023_validation"][branch])]
            marks=sealed["branches"][branch]["kept"]
            real_stake=0
            favorite_pay=0.
            random_expected_pay=0.
            random_var=0.
            favorite_winning_tickets=0
            for market in MARKETS:
                odds=row["odds"][market]
                k=len(marks[market])
                n=len(odds)
                assert 1<=k<=n
                assert set(marks[market]).issubset(odds)
                win=truth[market]
                assert win in odds
                real_stake+=k*UNIT
                win_pay=UNIT*float(odds[win])
                # Probability that a uniformly chosen k-mark subset covers
                # this one winning ticket, conditional on the observed result.
                pr=k/n
                random_expected_pay+=pr*win_pay
                random_var+=pr*(1.-pr)*win_pay**2
                favorite=set(sorted(odds,key=lambda t:(float(odds[t]),t))[:k])
                if win in favorite:
                    favorite_pay+=win_pay
                    favorite_winning_tickets+=1
            assert purchased["stake_yen"]==real_stake
            a=aggregation[branch]
            d=day[day_id][branch]
            for v in (a,d):
                v["races"]+=1
                v["stake_yen"]+=real_stake
                v["learned_payout_yen"]+=purchased["payout_yen"]
                v["random_expected_payout_yen"]+=random_expected_pay
                v["random_expected_payout_variance"]+=random_var
                v["favorite_only_payout_yen"]+=favorite_pay
                v["favorite_only_hit_marks"]+=favorite_winning_tickets
                v["learned_retained_winning_tickets"]+=purchased["retained_winning_tickets"]
                v["deleted_marks"]+=purchased["deleted_marks"]
        if i%400==0:print("DELETION_CONTROL_REPLAY",i,"/",len(test),flush=True)
    assert next(choices,None) is None and next(outcomes,None) is None
    comparison={}
    for branch in BRANCHES:
        a=aggregation[branch]
        stake=a["stake_yen"]
        assert stake==summary["test_scores"]["totals"][branch][
            str(chk["policy_selected_only_from_2023_validation"][branch])]["stake_yen"]
        assert a["learned_payout_yen"]==summary["test_scores"]["totals"][branch][
            str(chk["policy_selected_only_from_2023_validation"][branch])]["payout_yen"]
        compare={
            "races":int(a["races"]),
            "same_stake_yen":int(stake),
            "learned_payout_yen":int(a["learned_payout_yen"]),
            "learned_roi_pct":100*a["learned_payout_yen"]/stake,
            "uniform_random_expected_payout_yen":a["random_expected_payout_yen"],
            "uniform_random_expected_roi_pct":100*a["random_expected_payout_yen"]/stake,
            "learned_roi_uplift_over_random_expected_points":100*(
                a["learned_payout_yen"]-a["random_expected_payout_yen"])/stake,
            "random_payout_sd_yen":math.sqrt(a["random_expected_payout_variance"]),
            "learned_minus_random_sd_units":(
                (a["learned_payout_yen"]-a["random_expected_payout_yen"])/
                math.sqrt(a["random_expected_payout_variance"])
                if a["random_expected_payout_variance"]>0 else None),
            "favorite_only_payout_yen":int(round(a["favorite_only_payout_yen"])),
            "favorite_only_roi_pct":100*a["favorite_only_payout_yen"]/stake,
            "learned_roi_uplift_over_favorites_points":100*(
                a["learned_payout_yen"]-a["favorite_only_payout_yen"])/stake,
            "learned_retained_winning_tickets":int(a["learned_retained_winning_tickets"]),
            "favorite_retained_winning_tickets":int(a["favorite_only_hit_marks"]),
            "total_deleted_marks":int(a["deleted_marks"]),
        }
        comparison[branch]=compare
    out={
        "kind":"POST_SEALED_NEGATIVE_CONTROLS",
        "study":"GEN008-2023-FROM-ZERO-DELETION-RESTART",
        "test_period":summary["test_period"],
        "test_races":len(test),
        "child_weights_unchanged":True,
        "2023_policy_selection_unchanged":True,
        "no_model_training_or_retuning_using_2024_outcomes":True,
        "uniform_random_is_exact_conditional_expectation_not_a_simulated_winning_sample":True,
        "favorites_also_use_same_original_model_per_market_mark_count":True,
        "learned_vs_random_variance_is_conditional_on_observed_futures_not_full_model_uncertainty":True,
        "no_profit_or_statistical_significance_claimed":True,
        "comparison":comparison,
        "daily":[{"date":d,"branches":{b:dict(v) for b,v in bs.items()}}
                 for d,bs in sorted(day.items())],
    }
    saved.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_DELETION_NEGATIVE_CONTROLS_COMPLETE",json.dumps({
        "test_races":len(test),"comparison":comparison},ensure_ascii=False),flush=True)


if __name__=="__main__":main()

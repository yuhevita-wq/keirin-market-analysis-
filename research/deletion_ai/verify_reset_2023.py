"""Independent read-only replay: 2023 scratch deletion learning + 2024 frozen test.

All 2023 weight changes recomputed from chronological results only.
Every 2024 kept/deleted ticket and saved/forfeited yen is re-evaluated.
Reject unknown race IDs, mutable checkpoints, post-result policy tuning.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np

from deletion_core import (
    BRANCHES,MARKETS,FEATURES,ACTIVE,COVERAGES,UNIT,
    new_weights,books,probabilities,model_digest,deletion_plan,
    evaluate_plan,full_sheet_score,update
)
from run_reset_2023 import (
    OUT,WEB,STAGE,TRAIN_PROVISIONAL_COVERAGE,load_data,pre_result,
    all_book_digest,p_digest,learned_probability_cache,choose_coverage
)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def file_rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)


def eq_stat(actual,reported):
    assert set(actual)==set(reported),(set(actual)^set(reported))
    for k in actual:
        a,b=actual[k],reported[k]
        if isinstance(a,float) or isinstance(b,float):
            assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),(k,a,b)
        else:assert a==b,(k,a,b)


def main():
    report=read(OUT/"summary.json")
    chk=read(OUT/"checkpoint.json")
    web=read(WEB)
    assert report["status"]==web["status"]=="COMPLETE"
    assert report["stage"]==web["stage"]==STAGE
    assert report["2023_from_zero"] and report["original_old_new_v3_v3me_untouched"]
    assert report["no_real_money_bets"] and report["no_post_result_selection"]
    assert report["no_2024_outcome_impact_on_weights_or_policy"]
    assert chk["initial_weights_all_zero"]
    assert not chk["inherited_old_new_v3_v3me_checkpoint_weights"]
    assert chk["validation_2023_10_to_12_did_not_train"]
    assert chk["2024_test_did_not_train_or_tune"]
    assert report["selected_probability_coverage"]==chk["policy_selected_only_from_2023_validation"]
    assert report["selected_probability_coverage"]==web["selected_probability_coverage"]
    assert chk["features"]==list(FEATURES)
    assert chk["feature_masks"]=={b:list(ACTIVE[b]) for b in BRANCHES}
    train,val,test,sources=load_data()
    assert (len(train),len(val),len(test))==(
        report["train_races"],report["validation_races"],report["test_races"])
    current=new_weights()
    assert chk["initial_sha256"]=={b:model_digest(current,b) for b in BRANCHES}
    assert len(set(chk["initial_sha256"].values()))==1
    train_pre=file_rows(OUT/"2023_training_pre_result.jsonl.gz")
    train_after=file_rows(OUT/"2023_training_revealed.jsonl.gz")
    for step,row in enumerate(train,1):
        p=next(train_pre)
        a=next(train_after)
        pre=pre_result(row)
        bk=books(pre)
        assert p["step"]==a["step"]==step
        assert p["stage"]==a["stage"]==STAGE
        assert p["race_id"]==a["race_id"]==pre["race_id"]
        assert p["pre_result_only"] and a["after_pre_result_seal"]
        assert p["full_sheet_all_triples_and_trios"]
        assert p["provisional_coverage_target"]==TRAIN_PROVISIONAL_COVERAGE
        assert p["full_book_sha256"]==all_book_digest(bk)
        assert p["full_mark_count"]==sum(len(bk[m]["keys"]) for m in MARKETS)
        assert p["full_stake_yen"]==p["full_mark_count"]*UNIT
        assert list(row["outcome"])==a["finish"]
        assert a["full_sheet"]==full_sheet_score(bk,row["outcome"])
        for branch in BRANCHES:
            assert p["branch_weights_before"][branch]==model_digest(current,branch)
            plans={}
            for market in MARKETS:
                prob=probabilities(bk[market],current[branch][market],branch)
                assert p["probability_sha256"][branch][market]==p_digest(prob)
                plan=deletion_plan(bk[market],prob,TRAIN_PROVISIONAL_COVERAGE)
                assert p["provisional_deletions"][branch][market]==plan["deleted"]
                plans[market]=plan
            assert a["children"][branch]["provisional_purchase"]==evaluate_plan(
                bk,plans,row["outcome"])
            u=update(current,bk,branch,row["outcome"])
            observed=a["children"][branch]["feedback"]
            for market in MARKETS:
                eq_stat(u[market],observed[market])
            assert a["children"][branch]["weights_after"]==model_digest(current,branch)
        if step%1500==0:print("DELETION_AUDIT_TRAIN",step,len(train),flush=True)
    assert next(train_pre,None) is None and next(train_after,None) is None
    final={b:model_digest(current,b) for b in BRANCHES}
    assert final==chk["after_training_sha256"]
    for branch in BRANCHES:
        for market in MARKETS:
            assert np.allclose(current[branch][market],
                chk["weights"][branch][market],atol=1e-14)
    validation_months={}

    def verify_frozen(rows,split):
        pres=file_rows(OUT/f"{split}_sealed_pre_result.jsonl.gz")
        after=file_rows(OUT/f"{split}_revealed.jsonl.gz")
        for index,row in enumerate(rows,1):
            p=next(pres)
            a=next(after)
            pre=pre_result(row)
            bk=books(pre)
            assert p["step"]==a["step"]==index
            assert p["stage"]==a["stage"]==STAGE
            assert p["split"]==a["split"]==split
            assert p["race_id"]==a["race_id"]==pre["race_id"]
            assert p["pre_result_only"] and p["model_frozen"]
            assert p["book_digest"]==all_book_digest(bk)
            assert p["all_marks_count"]==sum(len(bk[m]["keys"]) for m in MARKETS)
            assert p["all_full_stake_yen"]==p["all_marks_count"]*UNIT
            assert a["outcome_after_seal"]==list(row["outcome"])
            assert a["full_purchase"]==full_sheet_score(bk,row["outcome"])
            assert set(p["branches"])==set(a["branch_purchases"])==set(BRANCHES)
            for branch in BRANCHES:
                assert p["branches"][branch]["model_hash"]==final[branch]
                coverages=(COVERAGES if split=="validation" else (
                    chk["policy_selected_only_from_2023_validation"][branch],))
                assert set(a["branch_purchases"][branch])==set(str(x) for x in coverages)
                probs={}
                for market in MARKETS:
                    probs[market]=probabilities(bk[market],current[branch][market],branch)
                    assert p["probability_sha256"][branch][market]==p_digest(probs[market])
                for coverage in coverages:
                    ps={market:deletion_plan(bk[market],probs[market],coverage)
                        for market in MARKETS}
                    if split=="validation":
                        saved=p["branches"][branch]["candidate_coverages"][str(coverage)]
                        for market in MARKETS:
                            assert saved[market]["deleted"]==ps[market]["deleted"]
                            assert saved[market]["retained_marks"]==len(ps[market]["kept"])
                    else:
                        assert p["branches"][branch]["coverage_target"]==coverage
                        for market in MARKETS:
                            assert p["branches"][branch]["kept"][market]==ps[market]["kept"]
                            assert p["branches"][branch]["deleted"][market]==ps[market]["deleted"]
                    actual=evaluate_plan(bk,ps,row["outcome"])
                    assert a["branch_purchases"][branch][str(coverage)]==actual
            if index%1200==0:
                print("DELETION_AUDIT_"+split.upper(),index,len(rows),flush=True)
        assert next(pres,None) is None and next(after,None) is None
        assert {b:model_digest(current,b) for b in BRANCHES}==final

    verify_frozen(val,"validation")
    verify_frozen(test,"test")
    # Selected targets must correspond to 2023 validation only; reselect independently.
    policies,table=choose_coverage(report["validation_scores"])
    assert policies==chk["policy_selected_only_from_2023_validation"]
    assert table==chk["policy_selection_tables"]
    assert policies==report["selected_probability_coverage"]
    assert web["test_comparison"]==report["test_scores"]["totals"]
    assert report["test_scores"]["model_start"]==final
    assert report["test_scores"]["model_end"]==final
    assert report["validation_scores"]["model_start"]==final
    assert report["validation_scores"]["model_end"]==final
    print("GEN008_DELETION_2023_READONLY_AUDIT_PASS",
        json.dumps({"train":len(train),"validation":len(val),"holdout":len(test),
            "selected_coverage":policies,
            "test_allmarks":report["test_scores"]["totals"]["full"]["all"],
            "test_children":{b:report["test_scores"]["totals"][b][str(policies[b])]
                             for b in BRANCHES},
            "no_future_learning":True},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

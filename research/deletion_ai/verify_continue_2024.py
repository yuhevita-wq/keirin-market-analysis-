"""Audit chronological 2024 continuation and immutable 2025 deletion holdout.

Replay every 2024 pre-result deletion and subsequent update from original
2023 weights. Independently reproduce 2025 frozen prices, model hashes,
missing winning-ticket penalties, random/favorite controls and aggregates.
"""
from __future__ import annotations
import gzip
import json
import math
from collections import defaultdict

import numpy as np

from deletion_core import (
    BRANCHES,MARKETS,books,probabilities,deletion_plan,
    evaluate_plan,full_sheet_score,model_digest,update
)
from continue_train_2024 import (
    OUT,WEB,STAGE,TRAIN_FIRST,TRAIN_LAST,TEST_FIRST,TEST_LAST,
    parent,load_data,pre_result,all_book_digest,p_digest,
    selected_plans,controls_at_equal_count,
)


def iter_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)


def check_equal(a,b,label):
    """Exact JSON replay: no post-hoc choice or payout changes."""
    if isinstance(a,float) or isinstance(b,float):
        assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(label,a,b)
    else:
        assert a==b,(label,str(a)[:120],str(b)[:120])


def audit():
    report=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    cp=json.loads((OUT/"checkpoint.json").read_text(encoding="utf-8"))
    public=json.loads(WEB.read_text(encoding="utf-8"))
    assert report["status"]==public["status"]=="COMPLETE"
    assert report["stage"]==cp["stage"]==public["stage"]==STAGE
    assert report["original_2023_models_not_modified"] is True
    assert report["prior_january_2024_holdout_is_reused_as_NEW_training"] is True
    assert report["2024_training_is_prequential_not_blind_holdout"]
    assert report["2025_test_never_updates_model_or_policy"]
    assert cp["prior_2023_exam_artifacts_not_overwritten"]
    assert cp["2025_outcomes_never_used_for_training_or_policy_tuning"]
    assert cp["training_dates"]==[TRAIN_FIRST,TRAIN_LAST]
    assert cp["frozen_exam_dates"]==[TEST_FIRST,TEST_LAST]
    assert cp["2024_observed_old_january_2024_exam_now_promoted_to_development"]
    weights,older,older_summary=parent()
    selected=older["policy_selected_only_from_2023_validation"]
    assert cp["original_2023_validation_coverage_unchanged"]==selected
    before={b:model_digest(weights,b) for b in BRANCHES}
    assert before==cp["parent_2023_after_training_sha256"]==report["input_parent_sha256"]
    train,test,origins=load_data()
    assert len(train)==report["train_races"]==cp["2024_training_races"]
    assert len(test)==report["2025_test_races"]==cp["2025_test_races"]
    pre=iter_gz(OUT/"2024_pre_result_delete_sheets.jsonl.gz")
    after=iter_gz(OUT/"2024_outcome_and_updates.jsonl.gz")
    aggregates=defaultdict(lambda:defaultdict(float))
    checkpoints=[]
    for i,row in enumerate(train,1):
        p=next(pre)
        s=next(after)
        inp=pre_result(row)
        b=books(inp)
        assert p["stage"]==s["stage"]==STAGE
        assert p["split"]==s["split"]=="2024_online_train"
        assert p["sequence"]==s["sequence"]==i
        assert p["race_id"]==s["race_id"]==row["race_id"]
        assert p["date"]==s["date"]==row["race"]["race_date"]
        assert p["sealed_before_outcome"] and p["no_real_wagers"]
        assert p["full_book_sha256"]==all_book_digest(b)
        assert p["full_3rentan_mark_count"]==len(b["3連単"]["keys"])
        assert p["full_3renpuku_mark_count"]==len(b["3連複"]["keys"])
        assert p["full_book_paper_stake_yen"]==100*(
            len(b["3連単"]["keys"])+len(b["3連複"]["keys"]))
        assert p["original_2023_coverage"]==selected
        prob,plans=selected_plans(b,weights,selected)
        original={branch:model_digest(weights,branch) for branch in BRANCHES}
        assert p["learners_2023_model_hash_before"]==original
        for branch in BRANCHES:
            for market in MARKETS:
                assert p["probabilities_sha256"][branch][market]==p_digest(
                    prob[branch][market])
                assert p["deleted"][branch][market]==plans[branch][market]["deleted"]
                assert p["retained"][branch][market]==plans[branch][market]["kept"]
        outcome=row["outcome"]
        assert s["result_after_seal"]==list(outcome)
        all_marks=full_sheet_score(b,outcome)
        assert s["all_marks_virtual"]==all_marks
        aggregates["ALL"]["races"]+=1
        for k in ("stake_yen","payout_yen","net_yen"):
            aggregates["ALL"][k]+=all_marks[k]
        for branch in BRANCHES:
            receipt=evaluate_plan(b,plans[branch],outcome)
            assert s["shadow_deletion_purchases"][branch]==receipt
            aggregates[branch]["races"]+=1
            for k in ("stake_yen","payout_yen","net_yen",
                      "deleted_marks","saved_stake_yen","forfeited_payout_yen",
                      "deleted_winning_tickets","retained_winning_tickets","delta_net_vs_all_marks"):
                aggregates[branch][k]+=receipt[k]
            actual_update=update(weights,b,branch,outcome)
            for market in MARKETS:
                check_equal(actual_update[market]["observed_logloss"],
                   s["update_labels"][branch][market]["observed_logloss"],(i,branch,market))
                assert actual_update[market]["outcome"]==s["update_labels"][branch][market]["outcome"]
        hashes={branch:model_digest(weights,branch) for branch in BRANCHES}
        assert s["model_sha256_after"]==hashes
        month=row["race"]["race_date"][:7]
        nextmonth=train[i]["race"]["race_date"][:7] if i<len(train) else None
        if nextmonth!=month:checkpoints.append({"month":month,
              "through_race_sequence":i,"after_sha256":hashes})
        if i%1000==0:
            print("GEN008_DELETION_2024_REPLAY_AUDIT",i,len(train),flush=True)
    assert next(pre,None) is None and next(after,None) is None
    assert checkpoints==cp["monthly_weight_hashes"]
    final={branch:model_digest(weights,branch) for branch in BRANCHES}
    assert final==cp["2024_learned_sha256"]==report["output_2024_sha256"]
    for branch in BRANCHES:
        for market in MARKETS:
            assert np.allclose(weights[branch][market],cp["branch_weights"][branch][market],atol=1e-14)
    for model in ("ALL",)+BRANCHES:
        recorded=report["train"][model]
        for k,a in aggregates[model].items():
            check_equal(a,recorded[k],("2024train",model,k))
    scores=defaultdict(lambda:defaultdict(float))
    ctl=defaultdict(lambda:defaultdict(float))
    frozen_pre=iter_gz(OUT/"2025_frozen_pre_result_delete_sheets.jsonl.gz")
    frozen_after=iter_gz(OUT/"2025_frozen_outcomes.jsonl.gz")
    for i,row in enumerate(test,1):
        p=next(frozen_pre)
        s=next(frozen_after)
        inp=pre_result(row)
        b=books(inp)
        assert p["stage"]==s["stage"]==STAGE
        assert p["split"]==s["split"]=="2025_frozen_out_of_sample"
        assert p["sequence"]==s["sequence"]==i
        assert p["race_id"]==s["race_id"]==row["race_id"]
        assert p["date"]==s["date"]==row["race"]["race_date"]
        assert p["no_future_labels_or_updates"] and p["weights_sha256"]==final
        assert p["coverage_targets_from_2023"]==selected
        assert p["full_book_sha256"]==all_book_digest(b)
        assert p["full_book_paper_stake_yen"]==100*(
            len(b["3連単"]["keys"])+len(b["3連複"]["keys"]))
        prob,plans=selected_plans(b,weights,selected)
        for branch in BRANCHES:
            for market in MARKETS:
                assert p["probabilities_sha256"][branch][market]==p_digest(prob[branch][market])
                assert p["deleted"][branch][market]==plans[branch][market]["deleted"]
                assert p["retained"][branch][market]==plans[branch][market]["kept"]
        outcome=row["outcome"]
        assert s["result_after_seal"]==list(outcome)
        actual=full_sheet_score(b,outcome)
        assert s["all_marks_virtual"]==actual
        scores["ALL"]["races"]+=1
        for k in ("stake_yen","payout_yen","net_yen"):
            scores["ALL"][k]+=actual[k]
        neg=controls_at_equal_count(inp,b,plans,outcome)
        for branch in BRANCHES:
            ticket=evaluate_plan(b,plans[branch],outcome)
            assert s["shadow_deletion_purchases"][branch]==ticket
            assert s["same_mark_negative_controls"][branch]==neg[branch]
            scores[branch]["races"]+=1
            for k in ("stake_yen","payout_yen","net_yen",
                      "deleted_marks","saved_stake_yen","forfeited_payout_yen",
                      "deleted_winning_tickets","retained_winning_tickets",
                      "delta_net_vs_all_marks"):
                scores[branch][k]+=ticket[k]
            for k,v in neg[branch].items():
                if k=="expected_uniform_random":continue
                ctl[branch][k]+=v
        assert {branch:model_digest(weights,branch) for branch in BRANCHES}==final
    assert next(frozen_pre,None) is None and next(frozen_after,None) is None
    assert report["test"]["ALL"]["races"]==len(test)
    for model in ("ALL",)+BRANCHES:
        r=report["test"][model]
        for k,v in scores[model].items():
            check_equal(v,r[k],("2025test",model,k))
        assert math.isclose(100*scores[model]["payout_yen"]/
           scores[model]["stake_yen"],r["roi_pct"],rel_tol=1e-12)
    for model in BRANCHES:
        row=report["test_equal_mark_count_controls"][model]
        total=ctl[model]
        st=scores[model]["stake_yen"]
        assert st==total["same_stake_yen"]
        check_equal(row["favorite_same_count_roi_pct"],
            100*total["favorite_payout_yen"]/st,("favorite",model))
        check_equal(row["uniform_random_conditional_expected_roi_pct"],
            100*total["random_expected_payout_yen"]/st,("random",model))
        check_equal(row["learned_minus_favorite_pct_points"],
            100*(scores[model]["payout_yen"]-total["favorite_payout_yen"])/st,
            ("uplift_favorites",model))
        check_equal(row["learned_minus_random_expected_pct_points"],
            100*(scores[model]["payout_yen"]-total["random_expected_payout_yen"])/st,
            ("uplift_random",model))
    assert public["train"]==report["train"]
    assert public["test"]==report["test"]
    assert public["matched_mark_controls"]==report["test_equal_mark_count_controls"]
    print("GEN008_DELETION_2024_TO_2025_AUDIT_PASS",json.dumps({
        "training_races":len(train),"independent_2025_races":len(test),
        "parents_unchanged":True,"original_2023_policy_unchanged":True,
        "frozen_2025_outcomes_not_used_for_training":True,
        "test":{b:{"roi_pct":report["test"][b]["roi_pct"],
                     "net_yen":report["test"][b]["net_yen"]}
                for b in ("ALL",)+BRANCHES},
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    audit()

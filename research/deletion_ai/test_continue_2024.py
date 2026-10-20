"""Regression tests for continuing deletion-first children without 2023 reinitialization."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import math

import numpy as np

from deletion_core import (
    BRANCHES,MARKETS,UNIT,books,deletion_plan,probabilities,
    evaluate_plan,full_sheet_score,model_digest,update,
)
from continue_train_2024 import (
    parent,controls_at_equal_count,selected_plans,
)
from test_deletion_2023 import fixture


def run_tests():
    weights,checkpoint,summary=parent()
    assert checkpoint["training_dates"]==["2023-01-01","2023-09-30"]
    assert checkpoint["2024_test_did_not_train_or_tune"]
    policies=checkpoint["policy_selected_only_from_2023_validation"]
    assert set(policies)==set(BRANCHES)
    before={b:model_digest(weights,b) for b in BRANCHES}
    assert before==checkpoint["after_training_sha256"]
    expected_initial="38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca"
    assert any(x!=expected_initial for x in before.values())
    p=fixture()
    book=books(p)
    q,plans=selected_plans(book,weights,policies)
    actual=full_sheet_score(book,(1,2,3))
    controls=controls_at_equal_count(p,book,plans,(1,2,3))
    for b in BRANCHES:
        assert set(plans[b])==set(MARKETS)
        settled=evaluate_plan(book,plans[b],(1,2,3))
        assert settled["net_yen"]-actual["net_yen"]==(
            settled["saved_stake_yen"]-settled["forfeited_payout_yen"])
        assert settled["deleted_marks"]+settled["retained_marks"]==245
        assert controls[b]["same_stake_yen"]==settled["stake_yen"]
        assert 0<=controls[b]["favorite_winning_marks"]<=2
        assert 0<=controls[b]["random_expected_payout_yen"]<=actual["payout_yen"]
        assert controls[b]["random_payout_variance_yen2"]>=0
    for b in BRANCHES:
        old={k:model_digest(weights,k) for k in BRANCHES}
        updated=update(weights,book,b,(1,2,3))
        assert model_digest(weights,b)!=old[b]
        assert all(model_digest(weights,k)==old[k] for k in BRANCHES if k!=b)
        assert set(updated)==set(MARKETS)
    print("GEN008_DELETION_2024_CONTINUATION_TESTS_PASS",policies)


if __name__=="__main__":
    run_tests()

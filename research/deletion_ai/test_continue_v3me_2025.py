"""Small deterministic regression test; no historical results are used here."""
from __future__ import annotations

import numpy as np

from deletion_core import (
    ACTIVE, FEATURES, MARKETS, books, evaluate_plan, full_sheet_score,
    model_digest, update,
)
from continue_v3me_2025 import (
    MODEL, TRAIN, TEST, snapshot, decide, controls_equal_count, winnings,
)
from test_deletion_2023 import fixture


def test():
    assert MODEL == "V3-ME-R"
    assert ACTIVE[MODEL] == tuple(range(len(FEATURES)))
    assert TRAIN == ("2025-01-01", "2025-12-31")
    assert TEST == ("2026-01-01", "2026-06-30")
    model, coverage, checkpoint = snapshot()
    assert checkpoint["2024_training_races"] == 9594
    before = model_digest(model, MODEL)
    assert before == checkpoint["2024_learned_sha256"][MODEL]
    prior = {m: model[MODEL][m].copy() for m in MARKETS}
    book = books(fixture())
    q, plans = decide(book, model, coverage)
    for m in MARKETS:
        assert np.isclose(sum(q[m]), 1.)
        assert len(plans[m]["kept"]) + len(plans[m]["deleted"]) == len(book[m]["keys"])
    target = (1, 2, 3)
    virtual = evaluate_plan(book, plans, target)
    controls = controls_equal_count(book, plans, target)
    losses = winnings(book, q, target)
    assert virtual["stake_yen"] == controls["same_stake_yen"]
    assert controls["uniform_random_conditional_expected_payout_yen"] >= 0
    assert all(np.isfinite(x) and x > 0 for x in losses.values())
    assert virtual["net_yen"] - full_sheet_score(book, target)["net_yen"] == (
        virtual["saved_stake_yen"] - virtual["forfeited_payout_yen"])
    update(model, book, MODEL, target)
    assert model_digest(model, MODEL) != before
    assert all(np.array_equal(prior[m], checkpoint["branch_weights"][MODEL][m])
               for m in MARKETS)
    print("V3ME_2025_REGRESSION_PASS")


if __name__ == "__main__":
    test()

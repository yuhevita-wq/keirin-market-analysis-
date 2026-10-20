"""Continue the original V3-ME-R deletion learner on all usable 2025 books.

The pre-existing 2023 and 2024 experiments/checkpoints are immutable.
2025-01-01..12-31 is NEW development; the old 2025 Jan frozen exam is
explicitly promoted to training, not represented as a new untouched test.
2026-01-01..06-30 is frozen, paired, same-race evaluation: 2024-end
weights against 2025-end weights, without tuning or fitting on 2026.

Complete historical FINAL trifecta/trio odds only. This does NOT decrypt
the separately stored seven-bet encrypted archive or infer missing odds.
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
    ACTIVE, FEATURES, MARKETS, LR, RIDGE_STEP, MAX_WEIGHT, UNIT,
    books, deletion_plan, evaluate_plan, full_sheet_score, model_digest,
    probabilities, update,
)
from run_reset_2023 import pre_result, all_book_digest, p_digest, seal
from continue_train_2024 import OUT as PARENT_OUT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/deletion_ai/outputs/GEN-008-V3-ME-R-CONTINUE-2025"
WEB = ROOT / "docs/ai-lab/gen008-v3me-continue-2025.json"
STAGE = "GEN008-V3-ME-R-2025-CONTINUATION"
MODEL = "V3-ME-R"
TRAIN = ("2025-01-01", "2025-12-31")
TEST = ("2026-01-01", "2026-06-30")
COUNTER_KEYS = (
    "stake_yen", "payout_yen", "net_yen", "deleted_marks", "saved_stake_yen",
    "forfeited_payout_yen", "deleted_winning_tickets",
    "retained_winning_tickets", "delta_net_vs_all_marks",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot():
    cp = read(PARENT_OUT / "checkpoint.json")
    sm = read(PARENT_OUT / "summary.json")
    assert sm["status"] == "COMPLETE"
    assert cp["2024_training_races"] == 9594
    assert cp["2025_test_races"] == 1061
    assert cp["model_family"] == "GEN-008-DELETION-2024-CONTINUATION"
    assert cp["branch_feature_masks"][MODEL] == list(ACTIVE[MODEL])
    assert cp["learning_hyperparameters_unchanged"] == {
        "lr": LR, "ridge_step": RIDGE_STEP, "max_coefficient": MAX_WEIGHT,
    }
    parent = {MODEL: {
        market: np.asarray(cp["branch_weights"][MODEL][market],
                           dtype=np.float64).copy()
        for market in MARKETS
    }}
    assert model_digest(parent, MODEL) == cp["2024_learned_sha256"][MODEL]
    coverage = cp["original_2023_validation_coverage_unchanged"][MODEL]
    assert 0 < coverage <= 1
    return parent, coverage, cp


def load_data():
    sys.path.insert(0, str(ROOT / "research/ai_nursery"))
    from source_bridge import joined_races
    train, tr_report = joined_races(2025, *TRAIN)
    test, te_report = joined_races(2026, *TEST)
    assert len(train) > 5000, ("not enough verified 2025 data", len(train))
    assert len(test) > 100, ("not enough verified 2026 data", len(test))
    assert train[0]["race"]["race_date"] >= TRAIN[0]
    assert train[-1]["race"]["race_date"] <= TRAIN[1]
    assert test[0]["race"]["race_date"] >= TEST[0]
    assert test[-1]["race"]["race_date"] <= TEST[1]
    assert len({x["race_id"] for x in train + test}) == len(train) + len(test)
    return train, test, {"2025": tr_report, "2026_h1": te_report}


def decide(book, weights, coverage):
    probs = {
        m: probabilities(book[m], weights[MODEL][m], MODEL)
        for m in MARKETS
    }
    plans = {
        m: deletion_plan(book[m], probs[m], coverage)
        for m in MARKETS
    }
    return probs, plans


def winnings(book, probs, finish):
    real = {
        "3連単": "-".join(map(str, finish)),
        "3連複": "=".join(map(str, sorted(finish))),
    }
    return {
        m: -math.log(max(float(probs[m][book[m]["keys"].index(real[m])]), 1e-50))
        for m in MARKETS
    }


def controls_equal_count(book, plans, finish):
    real = {
        "3連単": "-".join(map(str, finish)),
        "3連複": "=".join(map(str, sorted(finish))),
    }
    stake, fav_paid, rand_exp, rand_var = 0, 0, 0., 0.
    for m in MARKETS:
        b = book[m]
        n, k = len(b["keys"]), len(plans[m]["kept"])
        assert 0 < k <= n
        sorted_by_price = sorted(
            range(n), key=lambda i: (float(b["prices"][i]), b["keys"][i]))
        fav = {b["keys"][i] for i in sorted_by_price[:k]}
        win = real[m]
        price = float(b["prices"][b["keys"].index(win)])
        cash = int(round(UNIT * price))
        stake += k * UNIT
        if win in fav:
            fav_paid += cash
        q = k / n
        rand_exp += q * cash
        rand_var += q * (1. - q) * cash ** 2
    return {
        "same_stake_yen": stake,
        "favorite_payout_yen": fav_paid,
        "uniform_random_conditional_expected_payout_yen": rand_exp,
        "uniform_random_conditional_variance_yen2": rand_var,
    }


def add(target, ticket):
    target["races"] += 1
    for field in COUNTER_KEYS:
        if field in ticket:
            target[field] += ticket[field]


def add_controls(target, control):
    for k, v in control.items():
        target[k] += v


def summarize(stats):
    out = {}
    for name, r in stats.items():
        row = dict(r)
        if "races" in row:
            row["races"] = int(row["races"])
            for f in ("deleted_marks", "deleted_winning_tickets",
                      "retained_winning_tickets"):
                if f in row:
                    row[f] = int(row[f])
            row["roi_pct"] = (100. * row["payout_yen"] / row["stake_yen"]
                              if row["stake_yen"] else None)
        out[name] = row
    return out


def iterator(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run():
    if OUT.exists() or WEB.exists():
        raise RuntimeError("Immutable V3-ME-R 2025 evidence already exists")
    base, coverage, cp_2024 = snapshot()
    weights = {MODEL: {m: base[MODEL][m].copy() for m in MARKETS}}
    first_hash = model_digest(weights, MODEL)
    train, test, sources = load_data()
    OUT.mkdir(parents=True, exist_ok=False)
    training = defaultdict(lambda: defaultdict(float))
    monthly = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    monthly_hashes = []
    with gzip.open(OUT / "2025_pre_result.jsonl.gz", "wt", encoding="utf-8",
                   compresslevel=6) as sealed, gzip.open(
            OUT / "2025_after_result.jsonl.gz", "wt",
            encoding="utf-8", compresslevel=6) as revealed:
        for i, row in enumerate(train, 1):
            pre = pre_result(row)
            book = books(pre)
            probs, plans = decide(book, weights, coverage)
            before = model_digest(weights, MODEL)
            seal(sealed, {
                "stage": STAGE, "split": "2025_online_training",
                "sequence": i, "race_id": row["race_id"],
                "date": row["race"]["race_date"],
                "complete_book_sha256": all_book_digest(book),
                "weights_sha256_before": before,
                "probabilities_sha256": {m: p_digest(probs[m]) for m in MARKETS},
                "retained": {m: plans[m]["kept"] for m in MARKETS},
                "deleted": {m: plans[m]["deleted"] for m in MARKETS},
                "coverage_frozen_from_2023": coverage,
                "decision_sealed_before_label": True, "real_money": False,
            })
            # Training sees the outcome ONLY after sealing its decision.
            finish = row["outcome"]
            full = full_sheet_score(book, finish)
            chosen = evaluate_plan(book, plans, finish)
            observed_loss = update(weights, book, MODEL, finish)
            after = model_digest(weights, MODEL)
            seal(revealed, {
                "stage": STAGE, "split": "2025_online_training",
                "sequence": i, "race_id": row["race_id"],
                "date": row["race"]["race_date"],
                "finish_after_seal": list(finish),
                "full_sheet": full, "V3-ME-R": chosen,
                "learning": observed_loss,
                "weights_sha256_after": after,
            })
            month = row["race"]["race_date"][:7]
            add(training["ALL"], full)
            add(training[MODEL], chosen)
            add(monthly[month]["ALL"], full)
            add(monthly[month][MODEL], chosen)
            next_month = train[i]["race"]["race_date"][:7] if i < len(train) else None
            if month != next_month:
                monthly_hashes.append({"month": month, "through_race": i,
                                       "weights_sha256": after})
            if i % 1000 == 0:
                print("V3ME_2025_TRAIN", i, len(train), month, flush=True)
    last_hash = model_digest(weights, MODEL)
    assert first_hash != last_hash
    assert monthly_hashes[-1]["weights_sha256"] == last_hash
    assert monthly_hashes[-1]["month"] == "2025-12"

    frozen = defaultdict(lambda: defaultdict(float))
    controls = defaultdict(lambda: defaultdict(float))
    loss = defaultdict(lambda: defaultdict(float))
    test_month = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with gzip.open(OUT / "2026_h1_frozen_pre_result.jsonl.gz", "wt",
                   encoding="utf-8", compresslevel=6) as sealed, gzip.open(
            OUT / "2026_h1_frozen_after_result.jsonl.gz", "wt",
            encoding="utf-8", compresslevel=6) as revealed:
        for i, row in enumerate(test, 1):
            pre = pre_result(row)
            book = books(pre)
            versions = {"2024_end": base, "2025_end": weights}
            decisions = {tag: decide(book, model, coverage)
                         for tag, model in versions.items()}
            assert model_digest(weights, MODEL) == last_hash
            seal(sealed, {
                "stage": STAGE, "split": "2026_h1_frozen_holdout",
                "sequence": i, "race_id": row["race_id"],
                "date": row["race"]["race_date"],
                "complete_book_sha256": all_book_digest(book),
                "weights_sha256": {
                    "2024_end": first_hash, "2025_end": last_hash},
                "probabilities_sha256": {
                    tag: {m: p_digest(q[m]) for m in MARKETS}
                    for tag, (q, _) in decisions.items()},
                "retained": {tag: {m: plans[m]["kept"] for m in MARKETS}
                             for tag, (_, plans) in decisions.items()},
                "deleted": {tag: {m: plans[m]["deleted"] for m in MARKETS}
                            for tag, (_, plans) in decisions.items()},
                "coverage_frozen_from_2023": coverage,
                "future_outcome_not_used_for_decision": True,
            })
            finish = row["outcome"]
            full = full_sheet_score(book, finish)
            settled, negatives, logloss = {}, {}, {}
            add(frozen["ALL"], full)
            for tag, (q, plan) in decisions.items():
                settled[tag] = evaluate_plan(book, plan, finish)
                negatives[tag] = controls_equal_count(book, plan, finish)
                logloss[tag] = winnings(book, q, finish)
                add(frozen[tag], settled[tag])
                add_controls(controls[tag], negatives[tag])
                for m in MARKETS:
                    loss[tag][m] += logloss[tag][m]
                add(test_month[row["race"]["race_date"][:7]][tag],
                    settled[tag])
            seal(revealed, {
                "stage": STAGE, "split": "2026_h1_frozen_holdout",
                "sequence": i, "race_id": row["race_id"],
                "date": row["race"]["race_date"],
                "finish_after_seal": list(finish), "full_sheet": full,
                "purchases": settled,
                "same_count_negative_controls": negatives,
                "winners_logloss": logloss,
            })
            if i % 1000 == 0:
                print("V3ME_2026_FROZEN", i, len(test), flush=True)
    assert model_digest(weights, MODEL) == last_hash

    score_test = summarize(frozen)
    score_train = summarize(training)
    controls_report = {
        tag: {
            "same_stake_yen": int(controls[tag]["same_stake_yen"]),
            "favorite_same_count_roi_pct": (
                100. * controls[tag]["favorite_payout_yen"] /
                frozen[tag]["stake_yen"]),
            "uniform_random_conditional_expected_roi_pct": (
                100. * controls[tag]["uniform_random_conditional_expected_payout_yen"] /
                frozen[tag]["stake_yen"]),
        }
        for tag in ("2024_end", "2025_end")
    }
    eval_losses = {
        tag: {m: loss[tag][m] / len(test) for m in MARKETS}
        for tag in ("2024_end", "2025_end")
    }
    summary = {
        "stage": STAGE, "status": "COMPLETE",
        "model": MODEL, "markets": list(MARKETS),
        "odds_are_historical_final_not_close_15_minutes_before": True,
        "seven_bet_encrypted_archive_was_not_decrypted": True,
        "unavailable_pricing_never_fabricated": True,
        "prior_2025_january_holdout_promoted_to_new_training": True,
        "previous_experiment_artifacts_immutable": True,
        "2025_is_prequential_development_not_new_holdout": True,
        "2026_h1_test_does_not_train_or_select_hyperparameters": True,
        "2023_validation_coverage_frozen": coverage,
        "2024_parent_sha256": first_hash, "2025_child_sha256": last_hash,
        "2025_train_dates": list(TRAIN), "2026_test_dates": list(TEST),
        "2025_training_races": len(train), "2026_h1_test_races": len(test),
        "training": score_train,
        "train_by_month": {month: summarize(d)
                           for month, d in sorted(monthly.items())},
        "2026_h1_frozen": score_test,
        "2026_h1_by_month": {month: summarize(d)
                             for month, d in sorted(test_month.items())},
        "2026_h1_market_logloss": eval_losses,
        "2026_h1_same_mark_count_controls": controls_report,
        "2026_h1_roi_delta_pct_points_2025_minus_2024": (
            score_test["2025_end"]["roi_pct"] -
            score_test["2024_end"]["roi_pct"]),
        "2026_h1_net_delta_yen_2025_minus_2024": (
            score_test["2025_end"]["net_yen"] -
            score_test["2024_end"]["net_yen"]),
        "source_reports": sources, "real_stakes": False,
        "not_a_full_national_archive_training": True,
    }
    checkpoint = {
        "stage": STAGE, "model": MODEL,
        "parent_checkpoint_path": str(
            (PARENT_OUT / "checkpoint.json").relative_to(ROOT)),
        "parent_model_sha256": first_hash,
        "after_2025_sha256": last_hash,
        "weights": {m: weights[MODEL][m].tolist() for m in MARKETS},
        "active_features": list(ACTIVE[MODEL]),
        "feature_names": list(FEATURES),
        "training_lr": LR, "ridge_step": RIDGE_STEP,
        "max_coefficient": MAX_WEIGHT,
        "coverage_frozen_from_2023": coverage,
        "2025_training_races": len(train),
        "2026_h1_frozen_races": len(test),
        "monthly_sha256": monthly_hashes,
        "old_2025_january_exam_reused_as_training": True,
        "2026_h1_not_used_as_training": True,
    }
    (OUT / "checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    WEB.parent.mkdir(parents=True, exist_ok=True)
    WEB.write_text(json.dumps({
        "stage": STAGE, "status": "COMPLETE", "model": MODEL,
        "trained_races_2025": len(train), "frozen_races_2026_h1": len(test),
        "frozen_scores": score_test, "delta_roi_pct_points":
        summary["2026_h1_roi_delta_pct_points_2025_minus_2024"],
        "source": str((OUT / "summary.json").relative_to(ROOT)),
        "real_money": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("V3ME_2025_COMPLETE", json.dumps({
        "trained": len(train), "tested": len(test),
        "roi_before": score_test["2024_end"]["roi_pct"],
        "roi_after": score_test["2025_end"]["roi_pct"],
        "net_after": score_test["2025_end"]["net_yen"],
    }, ensure_ascii=False), flush=True)


def audit():
    cp = read(OUT / "checkpoint.json")
    report = read(OUT / "summary.json")
    web = read(WEB)
    assert report["status"] == web["status"] == "COMPLETE"
    assert cp["stage"] == report["stage"] == web["stage"] == STAGE
    assert report["2026_h1_test_does_not_train_or_select_hyperparameters"]
    assert report["prior_2025_january_holdout_promoted_to_new_training"]
    assert cp["old_2025_january_exam_reused_as_training"]
    assert cp["2026_h1_not_used_as_training"]
    weights, coverage, _ = snapshot()
    old = {MODEL: {m: weights[MODEL][m].copy() for m in MARKETS}}
    assert model_digest(weights, MODEL) == cp["parent_model_sha256"]
    assert coverage == cp["coverage_frozen_from_2023"]
    train, test, sources = load_data()
    assert report["source_reports"] == sources
    assert len(train) == cp["2025_training_races"]
    assert len(test) == cp["2026_h1_frozen_races"]
    before = iterator(OUT / "2025_pre_result.jsonl.gz")
    after = iterator(OUT / "2025_after_result.jsonl.gz")
    month_hash = []
    for i, row in enumerate(train, 1):
        p, r = next(before), next(after)
        pre = pre_result(row)
        book = books(pre)
        probs, plans = decide(book, weights, coverage)
        assert p["sequence"] == r["sequence"] == i
        assert p["race_id"] == r["race_id"] == row["race_id"]
        assert p["decision_sealed_before_label"]
        assert p["complete_book_sha256"] == all_book_digest(book)
        assert p["weights_sha256_before"] == model_digest(weights, MODEL)
        assert p["coverage_frozen_from_2023"] == coverage
        assert p["probabilities_sha256"] == {
            m: p_digest(probs[m]) for m in MARKETS}
        assert p["retained"] == {m: plans[m]["kept"] for m in MARKETS}
        assert p["deleted"] == {m: plans[m]["deleted"] for m in MARKETS}
        outcome = row["outcome"]
        assert r["finish_after_seal"] == list(outcome)
        assert r["full_sheet"] == full_sheet_score(book, outcome)
        assert r["V3-ME-R"] == evaluate_plan(book, plans, outcome)
        updated = update(weights, book, MODEL, outcome)
        for m in MARKETS:
            assert math.isclose(
                updated[m]["observed_logloss"],
                r["learning"][m]["observed_logloss"],
                rel_tol=1e-12, abs_tol=1e-10)
            assert updated[m]["outcome"] == r["learning"][m]["outcome"]
        assert r["weights_sha256_after"] == model_digest(weights, MODEL)
        month = row["race"]["race_date"][:7]
        next_month = train[i]["race"]["race_date"][:7] if i < len(train) else None
        if next_month != month:
            month_hash.append({"month": month, "through_race": i,
                               "weights_sha256": model_digest(weights, MODEL)})
        if i % 1000 == 0:
            print("V3ME_2025_REPLAY", i, len(train), flush=True)
    assert next(before, None) is None and next(after, None) is None
    assert month_hash == cp["monthly_sha256"]
    assert model_digest(weights, MODEL) == cp["after_2025_sha256"]
    for m in MARKETS:
        assert np.allclose(weights[MODEL][m], cp["weights"][m],
                           rtol=0, atol=1e-14)

    before = iterator(OUT / "2026_h1_frozen_pre_result.jsonl.gz")
    after = iterator(OUT / "2026_h1_frozen_after_result.jsonl.gz")
    audit_totals = defaultdict(lambda: defaultdict(float))
    audit_losses = defaultdict(lambda: defaultdict(float))
    for i, row in enumerate(test, 1):
        p, r = next(before), next(after)
        book = books(pre_result(row))
        versions = {"2024_end": old, "2025_end": weights}
        decisions = {tag: decide(book, w, coverage)
                     for tag, w in versions.items()}
        assert p["sequence"] == r["sequence"] == i
        assert p["race_id"] == r["race_id"] == row["race_id"]
        assert p["future_outcome_not_used_for_decision"]
        assert p["complete_book_sha256"] == all_book_digest(book)
        assert p["weights_sha256"] == {
            "2024_end": cp["parent_model_sha256"],
            "2025_end": cp["after_2025_sha256"]}
        assert p["probabilities_sha256"] == {
            tag: {m: p_digest(q[m]) for m in MARKETS}
            for tag, (q, _) in decisions.items()}
        assert p["retained"] == {
            tag: {m: plans[m]["kept"] for m in MARKETS}
            for tag, (_, plans) in decisions.items()}
        assert p["deleted"] == {
            tag: {m: plans[m]["deleted"] for m in MARKETS}
            for tag, (_, plans) in decisions.items()}
        finish = row["outcome"]
        assert r["finish_after_seal"] == list(finish)
        full = full_sheet_score(book, finish)
        assert r["full_sheet"] == full
        add(audit_totals["ALL"], full)
        for tag, (q, plans) in decisions.items():
            receipt = evaluate_plan(book, plans, finish)
            assert r["purchases"][tag] == receipt
            assert r["same_count_negative_controls"][tag] == (
                controls_equal_count(book, plans, finish))
            for m, v in winnings(book, q, finish).items():
                assert math.isclose(v, r["winners_logloss"][tag][m],
                                    rel_tol=1e-12)
                audit_losses[tag][m] += v
            add(audit_totals[tag], receipt)
        assert model_digest(weights, MODEL) == cp["after_2025_sha256"]
        if i % 1000 == 0:
            print("V3ME_2026_REPLAY", i, len(test), flush=True)
    assert next(before, None) is None and next(after, None) is None
    for tag in ("ALL", "2024_end", "2025_end"):
        calc = summarize({tag: audit_totals[tag]})[tag]
        recorded = report["2026_h1_frozen"][tag]
        for key, value in calc.items():
            assert math.isclose(value, recorded[key], rel_tol=1e-12,
                                abs_tol=1e-8), (tag, key)
    for tag in ("2024_end", "2025_end"):
        for m in MARKETS:
            assert math.isclose(
                audit_losses[tag][m] / len(test),
                report["2026_h1_market_logloss"][tag][m],
                rel_tol=1e-12)
    assert web["frozen_scores"] == report["2026_h1_frozen"]
    print("V3ME_2025_AND_2026_H1_FULL_REPLAY_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    elif args.audit:
        audit()
    else:
        parser.error("choose --run or --audit")

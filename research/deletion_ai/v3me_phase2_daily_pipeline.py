"""GEN008 V3-ME-R Phase 2: immutable, restartable, day-frozen odds-only training.

Do not pass the outcome into the prediction engine. All decisions for all
races on the same day use one frozen opening weight snapshot, regardless of
race-ID sorting or unavailable actual start-time information. The model may
only fit outcomes after the ENTIRE day's decisions are hashed and sealed.

One YEAR per invocation (2021..2026), persist a small SHA-256-verified
year-end checkpoint. Start later years from prior year's exact checkpoint;
2026 H1 is a NO-FIT evaluation. Results and odds never stored or printed
at individual race level; encrypted odds never exported to public GitHub.

Input: existing authenticated seven-pool bridge (with private key in CI)
or real checksum-verified two-market subset. Never claim the latter is full.
Historical prices are FINAL, not T-minus-15.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from deletion_core import (
    ACTIVE, FEATURES, LR, MARKETS, MAX_WEIGHT, RIDGE_STEP, UNIT,
    books, deletion_plan, evaluate_plan, full_sheet_score, model_digest,
    new_weights, probabilities, stable_hash, update,
)
from secure_seven_bet_bridge import ROOT, iter_year, secure_key_temp
from v3me_archive_scale_2021_2026 import (
    baselines, stream_plain, controls_equal_count, true_logloss, add,
    finish_scores, public_count_2026,
)

MODEL = "V3-ME-R"
COVERAGE = .25
STAGE = "GEN008-V3ME-PHASE2-DAY-FROZEN"
BASE = ROOT / "research/deletion_ai/outputs"
DATA = ROOT / "data"
SEVEN_CENSUS = DATA / "audits/odds_seven_bet_archive_count.json"
TWO_CENSUS = DATA / "audits/odds_only_export.json"
FIRST_YEAR = 2021
LAST_YEAR = 2026
LAST_TRAIN_YEAR = 2025
EXACT_SETTINGS = {
    "branch": MODEL,
    "markets": list(MARKETS),
    "features": list(FEATURES),
    "active_indices": list(ACTIVE[MODEL]),
    "lr": LR,
    "ridge_step": RIDGE_STEP,
    "max_weight": MAX_WEIGHT,
    "probability_coverage": COVERAGE,
    "train_years": [2021, 2022, 2023, 2024, 2025],
    "frozen_test": ["2026-01-01", "2026-06-30"],
    "same_day_outcomes_used_only_after_all_day_decisions": True,
}
SETTINGS_HASH = stable_hash(EXACT_SETTINGS)


def weights_from_raw(raw):
    keys = set(raw)
    if keys != set(MARKETS):
        raise ValueError("Markets mismatch in checkpoint")
    w = {MODEL: {m: np.asarray(raw[m], dtype=np.float64).copy()
                 for m in MARKETS}}
    for m in MARKETS:
        theta = w[MODEL][m]
        if theta.shape != (len(FEATURES),) or not np.isfinite(theta).all():
            raise ValueError("Invalid checkpoint weights")
    return w


def weight_dump(weights):
    return {m: weights[MODEL][m].tolist() for m in MARKETS}


def output_directory(source):
    if source not in ("plain", "sealed"):
        raise ValueError("Unknown source")
    return BASE / f"GEN-008-V3-ME-R-PHASE2-DAY-FROZEN-{source.upper()}-2021-2026"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def begin_year(source, year):
    """Hard link to the preceding YEAR's immutable checkpoint."""
    if year == FIRST_YEAR:
        weights = {MODEL: {m: new_weights()[MODEL][m].copy() for m in MARKETS}}
        return weights, "ZERO_INITIALIZATION"
    folder = output_directory(source) / str(year - 1)
    cp = read(folder / "checkpoint.json")
    summary = read(folder / "summary.json")
    if cp["year"] != year - 1 or cp["source"] != source:
        raise RuntimeError("WRONG_PARENT_YEAR_SOURCE")
    if cp["settings_sha256"] != SETTINGS_HASH:
        raise RuntimeError("PARENT_LEARNING_RULES_CHANGED")
    if summary["status"] != "COMPLETE" or summary["weight_end_sha256"] != cp["weight_sha256"]:
        raise RuntimeError("PARENT_NOT_VERIFIED")
    weights = weights_from_raw(cp["weights"])
    if model_digest(weights, MODEL) != cp["weight_sha256"]:
        raise RuntimeError("PARENT_WEIGHT_DIGEST_MISMATCH")
    return weights, cp["weight_sha256"]


def _book_pre_result(row):
    """No race outcome, result payout, finish rank, or known next-race label."""
    return books({
        "race_id": row["race_id"],
        "race": {"race_date": row["race"]["race_date"]},
        "entries": row["entries"],
        "odds": row["odds"],
    })


def _decide(book, weights):
    ps = {m: probabilities(book[m], weights[MODEL][m], MODEL) for m in MARKETS}
    selected = {m: deletion_plan(book[m], ps[m], COVERAGE) for m in MARKETS}
    return ps, selected


def day_predictions(rows, weights, branches=None):
    """Pure prediction stage. The labels are deliberately unread.

    All models have the same STARTING weights for every race in this calendar
    day. Within-day outcome updates are forbidden in this function.
    """
    if not rows:
        raise RuntimeError("EMPTY_DATE")
    days = {row["race"]["race_date"] for row in rows}
    if len(days) != 1:
        raise RuntimeError("CROSS_DAY_PREDICTION_BATCH")
    active = branches if branches is not None else {"phase2": weights}
    snapshot = {name: model_digest(w, MODEL) for name, w in active.items()}
    decisions = []
    for row in sorted(rows, key=lambda v: v["race_id"]):
        book = _book_pre_result(row)
        choices = {name: _decide(book, w) for name, w in active.items()}
        # The hash of the *price and pre-outcome selections* is immutable.
        # No per-race receipt or price is written to the public record.
        proof = stable_hash({
            "day": row["race"]["race_date"], "race_id": row["race_id"],
            "models_at_day_open": snapshot,
            "books": {m: {"keys": list(book[m]["keys"]),
                           "prices": [float(n) for n in book[m]["prices"]]}
                      for m in MARKETS},
            "selected": {name: {m: plan[m]["kept"]
                                 for m in MARKETS}
                         for name, (_, plan) in choices.items()},
        })
        decisions.append((row, book, choices, proof))
    return decisions, snapshot


def settle_and_fit_day(decisions, weights, training, scored, losses,
                       negatives, month_scored):
    """Outcome may first be read here, AFTER ALL day decisions were sealed."""
    today = decisions[0][0]["race"]["race_date"]
    month = today[:7]
    for row, book, choices, _ in decisions:
        result = tuple(int(x) for x in row["outcome"])
        if len(result) != 3 or len(set(result)) != 3:
            raise RuntimeError("MISSING_OR_CONFLICTING_PODIUM")
        add(scored, "full_sheet", full_sheet_score(book, result))
        for label, (ps, plans) in choices.items():
            receipt = evaluate_plan(book, plans, result)
            add(scored, label, receipt)
            add(month_scored[month], label, receipt)
            for m in MARKETS:
                losses[label][m] += true_logloss(book, ps, result)[m]
            neg = controls_equal_count(book, plans, result)
            for k, v in neg.items():
                negatives[label][k] += v
        # Only AFTER entire day's decisions are in memory may weights update.
        if training:
            update(weights, book, MODEL, result)
    return len(decisions)


def chronological_days(source, year, key, stats):
    """Yield complete day batches without silently skipping a source year."""
    if source == "plain":
        # Export is race-ID sorted (track first), NOT chronological.
        rows = list(stream_plain(year, stats))
        rows.sort(key=lambda r: (r["race"]["race_date"], r["race_id"]))
    else:
        assert key is not None
        stop = f"{year}-06-30" if year == 2026 else f"{year}-12-31"
        rows = iter_year(year, f"{year}-01-01", stop, key, stats)
    seen = set()
    last_day = ""
    for day, group in itertools.groupby(rows, key=lambda x: x["race"]["race_date"]):
        if day <= last_day:
            raise RuntimeError("NONCHRONOLOGICAL_SOURCE_DATES")
        if not day.startswith(str(year)) or (year == 2026 and day > "2026-06-30"):
            raise RuntimeError("RACE_DATE_OUTSIDE_YEAR")
        last_day = day
        batch = list(group)
        for row in batch:
            rid = row["race_id"]
            if rid in seen:
                raise RuntimeError("DUPLICATE_EXACT_RACE_ID")
            seen.add(rid)
        yield day, batch


def run(source, year, verify=False):
    if year < FIRST_YEAR or year > LAST_YEAR:
        raise ValueError("Unsupported year")
    out = output_directory(source) / str(year)
    if verify and not (out / "checkpoint.json").exists():
        raise RuntimeError("NO_SAVED_CHECKPOINT_TO_VERIFY")
    if not verify and out.exists():
        raise RuntimeError("IMMUTABLE_YEAR_ALREADY_SAVED")
    weights, parent_hash = begin_year(source, year)
    start = model_digest(weights, MODEL)
    old, recent = baselines() if year == 2026 else (None, None)
    if year == 2026:
        frozen_weights = model_digest(weights, MODEL)
    stats = defaultdict(int)
    scores = defaultdict(lambda: defaultdict(float))
    monthly = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    losses = defaultdict(lambda: defaultdict(float))
    controls = defaultdict(lambda: defaultdict(float))
    day_proofs = []
    total = 0
    group_days = 0
    previous_day = ""
    key_manager = key_path = None
    if source == "sealed":
        phase1 = DATA / "audits/v3me_full_sealed_source_integrity_2021_2026.json"
        if not phase1.exists():
            raise RuntimeError("PHASE1_SEALED_INTEGRITY_AUDIT_NOT_COMPLETE")
        audited = read(phase1)
        if (audited["status"] != "COMPLETE" or
                not audited["all_six_years_sealed_source_decrypted"] or
                not audited["all_source_cipher_sha256_verified"]):
            raise RuntimeError("PHASE1_SEALED_AUDIT_NOT_VALID")
        key_manager, key_path = secure_key_temp()
    try:
        for day, rows in chronological_days(source, year, key_path, stats):
            if day <= previous_day:
                raise RuntimeError("DATE_ORDER_REGRESSED")
            previous_day = day
            train = year <= LAST_TRAIN_YEAR
            active = ({"phase2": weights} if train
                      else {"phase2": weights,
                            "2024_end": old, "2025_end": recent})
            # Seal entire day's outcomes-independent decisions first.
            decisions, before_hashes = day_predictions(rows, weights, active)
            pre_hash = hashlib.sha256("".join(proof for _, _, _, proof in decisions).encode()).hexdigest()
            assert all(model_digest(w, MODEL) == before_hashes[name]
                       for name, w in active.items())
            # Reveal results and train only once all forecasts are pre-committed.
            count = settle_and_fit_day(
                decisions, weights, train, scores, losses, controls, monthly)
            total += count
            group_days += 1
            after_hash = model_digest(weights, MODEL)
            if not train and after_hash != frozen_weights:
                raise RuntimeError("2026_HOLDOUT_UPDATED_MODEL")
            day_proofs.append({
                "day": day, "races": count,
                "start_sha256": before_hashes["phase2"],
                "seal_digest_sha256": pre_hash,
                "end_sha256": after_hash,
            })
            if group_days % 31 == 0:
                print("PHASE2_DAY_FROZEN_PROGRESS", json.dumps({
                    "source": source, "year": year, "days": group_days,
                    "used": total, "weight_sha256": after_hash,
                }, ensure_ascii=False), flush=True)
    finally:
        if key_manager is not None:
            key_manager.cleanup()
    if total < 100:
        raise RuntimeError("EMPTY_OR_TRUNCATED_SUPERVISED_SOURCE")
    year_declared = (read(TWO_CENSUS)["years"][str(year)]["complete_races"]
                     if source == "plain" else
                     read(SEVEN_CENSUS)["year"][str(year)]["confirmed_distinct_race_lower_bound"])
    if source == "plain":
        if total > year_declared:
            raise RuntimeError("BAD_PLAIN_DENOMINATOR")
        if total / year_declared < .95:
            raise RuntimeError("PLAIN_MISSING_MORE_THAN_5_PERCENT_LABELS")
    # Seven-bet census is a LOWER BOUND on raw full books, not a known result denominator.
    if year == 2026:
        public_count_2026(year_declared if source == "plain" else 0, total)
        assert model_digest(weights, MODEL) == frozen_weights
    end = model_digest(weights, MODEL)
    if sum(d["races"] for d in day_proofs) != total:
        raise RuntimeError("DAY_LEDGER_NOT_COMPLETE")
    totals = finish_scores(scores)
    cleaned = {}
    for label in totals:
        if label == "full_sheet":
            continue
        stake = totals[label]["stake_yen"]
        c = controls[label]
        if stake != c["same_stake_yen"]:
            raise RuntimeError("CONTROL_STAKE_MISMATCH")
        cleaned[label] = {
            "same_count_favorite_roi_pct": 100*c["favorite_payout_yen"]/stake,
            "same_count_uniform_random_expected_roi_pct":
                100*c["uniform_random_conditional_expected_payout_yen"]/stake,
            "mean_3rentan_logloss": losses[label]["3連単"]/total,
            "mean_3renpuku_logloss": losses[label]["3連複"]/total,
        }
    # This cryptographic ledger represents ALL races in exact time order
    # with no plaintext source prices or per-race selection published.
    daily_chain = stable_hash(day_proofs)
    final_cp = {
        "stage": STAGE, "source": source, "year": year,
        "start_parent_sha256": parent_hash, "start_weights_sha256": start,
        "weight_sha256": end,
        "weights": weight_dump(weights),
        "settings_sha256": SETTINGS_HASH,
        "training": year <= LAST_TRAIN_YEAR,
        "race_count": total,
        "daily_ledger_sha256": daily_chain,
    }
    summary = {
        "stage": STAGE, "source": source, "status": "COMPLETE",
        "year": year, "learning_mode": "DAY_FROZEN_PRE_RESULT_PREQUENTIAL"
                              if year <= LAST_TRAIN_YEAR else "NO_FIT_2026_H1",
        "parameters_frozen_for_full_year": True,
        "same_day_future_outcomes_cannot_affect_day_predictions": True,
        "parent_model_sha256": parent_hash,
        "weight_start_sha256": start, "weight_end_sha256": end,
        "settings_sha256": SETTINGS_HASH,
        "source_7_market_encrypted_full_original": source == "sealed",
        "source_plain_only_two_price_markets": source == "plain",
        "original_complete_books_or_lower_bound": year_declared,
        "actual_supervised_usable_races": total,
        "days_with_eligible_books": group_days,
        "source_rejection_counts": dict(stats),
        "day_level_pre_result_ledger": day_proofs,
        "daily_ledger_sha256": daily_chain,
        "accounting": totals,
        "same_ticket_count_controls_and_logloss": cleaned,
        "monthly": {month: finish_scores(v) for month, v in sorted(monthly.items())},
        "historical_final_odds_not_15_minutes_preclose": True,
        "same_model_original_eight_features": True,
        "original_baselines_not_overwritten": True,
        "candidate_not_automatically_deployed": True,
        "no_racecards_required": True,
        "2026_results_not_used_for_training": year == 2026,
        "results_per_race_and_plaintext_odds_not_published": True,
        "2026_h1_not_a_pristine_holdout_across_prior_experiments": year == 2026,
        "real_money_purchases": False,
    }
    if verify:
        saved = read(out / "checkpoint.json")
        old_summary = read(out / "summary.json")
        if final_cp != saved or summary != old_summary:
            raise RuntimeError("NONREPRODUCIBLE_YEARLY_CHECKPOINT_OR_LEDGER")
        print("PHASE2_YEAR_REPLAY_VERIFIED", json.dumps({
            "source": source, "year": year, "count": total,
            "ending_sha256": end}, ensure_ascii=False), flush=True)
        return summary
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoint.json").write_text(
        json.dumps(final_cp, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PHASE2_YEAR_COMPLETE", json.dumps({
        "source": source, "year": year, "actual_usable": total,
        "declared_raw": year_declared, "sha": end,
        "roi": {label: obj["roi_pct"] for label, obj in totals.items()},
    }, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("plain", "sealed"), required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    run(args.source, args.year, verify=args.verify)

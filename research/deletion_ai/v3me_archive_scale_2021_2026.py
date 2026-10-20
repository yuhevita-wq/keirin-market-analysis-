"""V3-ME-R archive-scale training without the artificial racecard bottleneck.

Two evidence tiers, deliberately NEVER conflated:
  --source plain: ALL checksum-verified 3連単+3連複 source records, 2021..2026H1;
                  join verified outcomes by exact race ID, no racecard needed.
  --source sealed: ALL checksum-verified encrypted 7-market files, 2021..2026H1;
                   ephemeral private CMS decryption requires Actions secret.

Either tier: scratch-train the SAME eight-feature V3-ME-R 3連単/3連複 engine
chronologically on 2021-2025, keep 2026H1 completely frozen, and compare
on identical races against immutable 2024-end and 2025-end V3-ME-R parents.

The complete 2023 and 2024 data count, not a convenient subset, determines
the true source denominator. Price data is FINAL historical, not T-minus-15.
No other wager types are modeled; seven source types are only authenticated.
Never publish secret, decrypted prices, individual tickets, or race-ID ledgers.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from deletion_core import (
    ACTIVE, FEATURES, MARKETS, LR, RIDGE_STEP, MAX_WEIGHT, UNIT,
    books, deletion_plan, evaluate_plan, full_sheet_score,
    model_digest, new_weights, probabilities, update,
)
from secure_seven_bet_bridge import (
    ROOT, iter_year, secure_key_temp, verified_results, race_day,
)
from continue_v3me_2025 import controls_equal_count

DATA = ROOT / "data"
MODEL = "V3-ME-R"
TRAIN_FIRST, TRAIN_END = 2021, 2025
TEST_YEAR = 2026
BASELINE_2024 = (ROOT / "research/deletion_ai/outputs/"
                 "GEN-008-DELETION-CONTINUE-2024/checkpoint.json")
BASELINE_2025 = (ROOT / "research/deletion_ai/outputs/"
                 "GEN-008-V3-ME-R-CONTINUE-2025/checkpoint.json")
ODDS_AUDIT = DATA / "audits/odds_only_export.json"
SEVEN_CENSUS = DATA / "audits/odds_seven_bet_archive_count.json"
GIRLS = re.compile(r"ガールズ|女子|L級|ガ予|ガ決|ガ一般|ガ選", re.I)
STAGE = "GEN008-V3-ME-R-ARCHIVE-SCALE-2021-2026H1"
COVERAGE = 0.25  # Fixed from original 2023 validation, never optimized on 2026
PROVISIONAL_COVERAGE = COVERAGE
METRICS = (
    "stake_yen", "payout_yen", "net_yen", "deleted_marks", "saved_stake_yen",
    "forfeited_payout_yen", "deleted_winning_tickets",
    "retained_winning_tickets", "delta_net_vs_all_marks",
)


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def hashfile(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_weights(model):
    return {MODEL: {
        m: np.asarray(model[m], dtype=np.float64).copy()
        for m in MARKETS
    }}


def baselines():
    cp24 = load_json(BASELINE_2024)
    cp25 = load_json(BASELINE_2025)
    assert cp25["parent_model_sha256"] == cp24["2024_learned_sha256"][MODEL]
    assert cp25["coverage_frozen_from_2023"] == COVERAGE
    assert cp24["original_2023_validation_coverage_unchanged"][MODEL] == COVERAGE
    old = clone_weights(cp24["branch_weights"][MODEL])
    recent = clone_weights(cp25["weights"])
    assert model_digest(old, MODEL) == cp25["parent_model_sha256"]
    assert model_digest(recent, MODEL) == cp25["after_2025_sha256"]
    assert cp25["2026_h1_not_used_as_training"]
    return old, recent


def stream_plain(year, stats):
    """Use complete genuine market books even where racecard files are missing."""
    path = DATA / f"odds_only/odds_{year}.jsonl.gz"
    declared = load_json(ODDS_AUDIT)["years"][str(year)]
    if hashfile(path) != declared["sha256"]:
        raise RuntimeError(f"CHECKSUM_ERROR: plaintext odds year {year}")
    known, label_stats = verified_results(year)
    stats["verified_result_ids"] = label_stats["verified_distinct_results"]
    stats["declared_full_two_market_races"] = declared["complete_races"]
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n += 1
            raw = json.loads(line)
            rid = str(raw.get("race_id", ""))
            day = race_day(rid)
            if day is None or not day.startswith(str(year)):
                stats["invalid_meeting_day_id"] += 1
                continue
            if GIRLS.search(str(raw.get("race_type", ""))):
                stats["girls_excluded"] += 1
                continue
            if not isinstance(raw.get("odds"), dict):
                stats["bad_market_record"] += 1
                continue
            finish = known.get(rid)
            if finish is None:
                stats["missing_official_result"] += 1
                continue
            # Exact ticket grid is validated in the official V3-ME-R books().
            # No rider/line/entry feature enters this odds-only model.
            row = {"race_id": rid, "race": {"race_date": day},
                   "entries": {}, "odds": raw["odds"], "outcome": finish}
            try:
                t = raw["odds"]["3連単"]
                if not isinstance(t, dict) or not t:
                    raise ValueError("empty odds")
                cars = {int(k) for ticket in t for k in ticket.split("-")}
                if cars != set(range(1, len(cars) + 1)):
                    raise ValueError("invalid complete car grid")
                row["entries"] = {car: {} for car in sorted(cars)}
                book = books(row)
                wins = {
                    "3連単": "-".join(map(str, finish)),
                    "3連複": "=".join(map(str, sorted(finish))),
                }
                if any(wins[m] not in book[m]["keys"] for m in MARKETS):
                    raise ValueError("winner not in market")
            except (KeyError, ValueError, TypeError, IndexError, OverflowError):
                stats["bad_full_grid_or_winner"] += 1
                continue
            stats["usable_exact_odds_and_outcome"] += 1
            yield row
    assert n == declared["complete_races"], (year, n, declared["complete_races"])


def iter_source(source, year, keypath, stats):
    if source == "plain":
        yield from stream_plain(year, stats)
    else:
        assert source == "sealed" and keypath is not None
        last = f"{year}-06-30" if year == 2026 else f"{year}-12-31"
        yield from iter_year(year, f"{year}-01-01", last, keypath, stats)


def decide(book, weight, coverage=COVERAGE):
    q = {m: probabilities(book[m], weight[MODEL][m], MODEL) for m in MARKETS}
    plan = {m: deletion_plan(book[m], q[m], coverage) for m in MARKETS}
    return q, plan


def add(scores, label, receipt):
    v = scores[label]
    v["races"] += 1
    for k in METRICS:
        if k in receipt:
            v[k] += receipt[k]


def finish_scores(scores):
    out = {}
    for label, d in scores.items():
        o = dict(d)
        o["races"] = int(o["races"])
        for k in ("deleted_marks", "deleted_winning_tickets",
                  "retained_winning_tickets"):
            if k in o:
                o[k] = int(o[k])
        o["roi_pct"] = 100 * o["payout_yen"] / o["stake_yen"] if o["stake_yen"] else None
        out[label] = o
    return out


def true_logloss(book, probs, finish):
    keys = {
        "3連単": "-".join(map(str, finish)),
        "3連複": "=".join(map(str, sorted(finish))),
    }
    return {m: -math.log(max(float(probs[m][book[m]["keys"].index(keys[m])]), 1e-50))
            for m in MARKETS}


def public_count_2026(expected, count):
    # Refuse accidental empty/severely partial holdout.
    assert count > 1500, ("2026_H1_TOO_SMALL", count)
    if expected > 0 and count / expected < 0.45:
        raise RuntimeError(f"2026_H1_COVERAGE_TOO_LOW: {count}/{expected}")


def run(source):
    if source not in ("sealed", "plain"):
        raise ValueError(source)
    dirname = ("GEN-008-V3-ME-R-FULL-SEVEN-ARCHIVE-2021-2026"
               if source == "sealed" else
               "GEN-008-V3-ME-R-ALL-PLAIN-2021-2026-CHRONO")
    out = ROOT / "research/deletion_ai/outputs" / dirname
    web = ROOT / "docs/ai-lab" / (
        "gen008-v3me-archive-scale-plain-chrono.json" if source == "plain"
        else "gen008-v3me-archive-scale-sealed.json")
    if out.exists() or web.exists():
        raise RuntimeError("Immutable archive-scale run already exists: " + source)
    old, recent = baselines()
    model = new_weights()
    assert set(ACTIVE[MODEL]) == set(range(len(FEATURES)))
    first = model_digest(model, MODEL)
    assert all(abs(x) < 1e-16 for m in MARKETS for x in model[MODEL][m])
    census = load_json(SEVEN_CENSUS)
    population = load_json(ODDS_AUDIT)
    keymanager = keypath = None
    if source == "sealed":
        keymanager, keypath = secure_key_temp()  # immediately abort if no secret
    training = defaultdict(lambda: defaultdict(float))
    holdout = defaultdict(lambda: defaultdict(float))
    monthly = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    source_stats = {}
    logloss = defaultdict(lambda: defaultdict(float))
    controls = defaultdict(lambda: defaultdict(float))
    per_year = {}
    last_fingerprint = first
    ledger = hashlib.sha256(b"V3ME_ALL_YEARS_DECIDE_BEFORE_RESULT_V1")
    try:
        for year in range(2021, 2027):
            stats = defaultdict(int)
            source_stats[str(year)] = stats
            seen = set()
            n = 0
            yearscores = defaultdict(lambda: defaultdict(float))
            dates = Counter()
            # Plain export is sorted by numeric race_id (track first), NOT date.
            # A prequential model must train on actual chronological dates.
            # Sealed iter_year already streams dates in ascending calendar order.
            if source == "plain":
                rows = list(iter_source(source, year, keypath, stats))
                rows.sort(key=lambda x: (x["race"]["race_date"], x["race_id"]))
            else:
                rows = iter_source(source, year, keypath, stats)
            for row in rows:
                rid = row["race_id"]
                if rid in seen:
                    raise RuntimeError(f"DUPLICATE_YEAR_RACE_ID: {year}")
                seen.add(rid)
                day = row["race"]["race_date"]
                if not day.startswith(str(year)) or (year == 2026 and day > "2026-06-30"):
                    raise RuntimeError("INCORRECT_YEAR_DATE_IN_LOADED_BOOK")
                n += 1
                dates[day[:7]] += 1
                book = books(row)
                finish = row["outcome"]
                if year <= TRAIN_END:
                    # Chronological same-date rows: only data known before each
                    # step update the weights for the next step.
                    q, plans = decide(book, model)
                    sealed = {
                        "race_id": rid, "date": day,
                        "pre_model_sha256": model_digest(model, MODEL),
                        "book_digest": hashlib.sha256(b"".join(
                            book[m]["prices"].astype("<f8").tobytes()
                            for m in MARKETS)).hexdigest(),
                        "retained_digest": hashlib.sha256(json.dumps(
                            {m: plans[m]["kept"] for m in MARKETS},
                            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                    }
                    # Digest pre-result decisions BEFORE consuming winner.
                    ledger.update(json.dumps(sealed, sort_keys=True).encode("utf-8"))
                    full = full_sheet_score(book, finish)
                    paid = evaluate_plan(book, plans, finish)
                    add(training, "full_sheet", full)
                    add(training, MODEL, paid)
                    add(yearscores, "full_sheet", full)
                    add(yearscores, MODEL, paid)
                    add(monthly[day[:7]], MODEL, paid)
                    update(model, book, MODEL, finish)
                    last_fingerprint = model_digest(model, MODEL)
                else:
                    assert year == TEST_YEAR
                    if n == 1:
                        assert model_digest(model, MODEL) == last_fingerprint
                    # Freeze all 3 models on exactly the same 2026 races.
                    variants = {"2024_end": old, "2025_end": recent,
                                "all_archive": model}
                    choices = {label: decide(book, w) for label, w in variants.items()}
                    ledger.update(json.dumps({
                        "race_id": rid, "date": day,
                        "model_sha": {name: model_digest(w, MODEL)
                                      for name, w in variants.items()},
                        "retained_sha": {label: hashlib.sha256(json.dumps(
                            {m: plans[m]["kept"] for m in MARKETS},
                            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                            for label, (_, plans) in choices.items()},
                    }, sort_keys=True).encode("utf-8"))
                    add(holdout, "full_sheet", full_sheet_score(book, finish))
                    for label, (q, plan) in choices.items():
                        result = evaluate_plan(book, plan, finish)
                        add(holdout, label, result)
                        add(yearscores, label, result)
                        for m, ll in true_logloss(book, q, finish).items():
                            logloss[label][m] += ll
                        control = controls_equal_count(book, plan, finish)
                        for k, v in control.items():
                            controls[label][k] += v
                    assert model_digest(model, MODEL) == last_fingerprint
                if n % 1000 == 0:
                    print("V3ME_ARCHIVE_SCALE", json.dumps({
                        "source": source, "year": year, "processed": n,
                        "train": year <= TRAIN_END
                    }, ensure_ascii=False), flush=True)
            exp = (population["years"][str(year)]["complete_races"]
                   if source == "plain" else
                   census["year"][str(year)]["confirmed_distinct_race_lower_bound"])
            stats["declared_archive_race_count_or_lower_bound"] = exp
            stats["actually_used_labeled_full_books"] = n
            stats["covered_months"] = sorted(dates)
            # A sealed manifest count is a lower bound on unique *raw* odds;
            # the supervised count can be lower due to invalid/missing labels.
            stats["manifest_count_is_not_labeled_denominator"] = (source == "sealed")
            if year == TEST_YEAR:
                public_count_2026(exp, n)
            if n < 100:
                raise RuntimeError(f"ALMOST_NO_LABELLED_BOOKS {year}: {n}")
            per_year[str(year)] = {
                "races": n, "scores": finish_scores(yearscores),
                "source_expected_count_or_lower_bound": exp,
            }
            print("V3ME_ARCHIVE_SCALE_YEAR_COMPLETE", json.dumps({
                "source": source, "year": year, "used": n,
                "expected_source_count": exp,
                "end_weight_sha256": model_digest(model, MODEL),
            }, ensure_ascii=False), flush=True)
    finally:
        if keymanager is not None:
            keymanager.cleanup()

    assert model_digest(model, MODEL) == last_fingerprint
    assert len(per_year) == 6
    frozen = finish_scores(holdout)
    train_summary = finish_scores(training)
    ntest = frozen["all_archive"]["races"]
    control_results = {}
    for label in ("2024_end", "2025_end", "all_archive"):
        stake = frozen[label]["stake_yen"]
        c = controls[label]
        assert stake == c["same_stake_yen"]
        control_results[label] = {
            "favorite_same_ticket_count_roi_pct": 100 * c["favorite_payout_yen"] / stake,
            "uniform_random_conditional_expected_roi_pct": (
                100 * c["uniform_random_conditional_expected_payout_yen"] / stake),
        }
    metadata = {
        "study": STAGE, "status": "COMPLETE", "mode": source,
        "validation": "2026-H1-frozen-parent-vs-archive-scale-child",
        "trained_years": [2021, 2022, 2023, 2024, 2025],
        "holdout_year": "2026-H1",
        "same_original_V3_ME_R_eight_features_and_gradient": True,
        "train_from_zero_to_rebuild_consistent_historical_timeline": True,
        "immutable_existing_V3_ME_R_checkpoints_preserved": True,
        "new_candidate_not_deployed_automatically": True,
        "coverage_fixed_from_original_2023_validation": COVERAGE,
        "all_usable_full_3rentan_3renpuku_books_no_racecard_join": True,
        "all_seven_stored_price_types_cryptographically_verified": source == "sealed",
        "non_modeled_five_bet_markets_not_used_as_predictors": True,
        "odds_final_not_15_minutes_pre_close": True,
        "original_inputs_never_modified": True,
        "unavailable_prices_never_guessed": True,
        "plain_mode_is_not_sealed_mode": True,
        "dataset_counts_not_confused_with_successfully_labeled_races": True,
        "no_future_test_outcomes_used_for_fit_or_tuning": True,
        "test_is_not_completely_new_since_prior_experiments_tested_2026h1": True,
        "real_money_wagers": False,
        "source_years": {y: dict(v) for y, v in source_stats.items()},
        "source_denominators": {y: per_year[y]["source_expected_count_or_lower_bound"]
                                for y in per_year},
        "yearly_train_and_test": per_year,
        "original_baseline_2024_h1_model_hash": model_digest(old, MODEL),
        "original_baseline_2025_model_hash": model_digest(recent, MODEL),
        "initial_new_model_sha256": first,
        "final_new_model_sha256": last_fingerprint,
        "training": train_summary,
        "frozen_2026_h1": frozen,
        "frozen_2026_h1_logloss": {
            label: {m: logloss[label][m] / ntest for m in MARKETS}
            for label in ("2024_end", "2025_end", "all_archive")},
        "same_mark_count_controls": control_results,
        "holdout_roi_delta_vs_2024_end": (
            frozen["all_archive"]["roi_pct"] - frozen["2024_end"]["roi_pct"]),
        "holdout_roi_delta_vs_2025_end": (
            frozen["all_archive"]["roi_pct"] - frozen["2025_end"]["roi_pct"]),
        "monthly_2021_to_2025": {
            month: finish_scores(v) for month, v in sorted(monthly.items())},
        "sealed_before_outcome_cumulative_digest_sha256": ledger.hexdigest(),
    }
    checkpoint = {
        "stage": STAGE, "status": "COMPLETE", "mode": source,
        "model": MODEL, "eight_feature_names": list(FEATURES),
        "active_feature_indices": list(ACTIVE[MODEL]),
        "source": "original_seven_bet_verified_ciphertext" if source == "sealed"
                  else "verified_plain_two_market",
        "weights": {m: model[MODEL][m].tolist() for m in MARKETS},
        "sha256": last_fingerprint, "parent_baselines_immutable": True,
        "hyperparameters": {"LR": LR, "RIDGE_STEP": RIDGE_STEP,
                            "MAX_WEIGHT": MAX_WEIGHT},
        "final_policy_coverage": COVERAGE,
        "trained_races": sum(per_year[str(y)]["races"] for y in range(2021, 2026)),
        "frozen_races": ntest,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    web.parent.mkdir(parents=True, exist_ok=True)
    web.write_text(json.dumps({
        "stage": STAGE, "status": "COMPLETE", "mode": source,
        "training_races": checkpoint["trained_races"],
        "test_races": ntest, "source_years": metadata["source_years"],
        "scores_2026_h1": frozen,
        "roi_delta_vs_2024_end": metadata["holdout_roi_delta_vs_2024_end"],
        "same_count_controls": control_results,
        "source": str((out / "summary.json").relative_to(ROOT)),
        "real_money": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("V3ME_ARCHIVE_SCALE_FINISHED", json.dumps({
        "source": source, "train": checkpoint["trained_races"], "test": ntest,
        "frozen_roi": {k: v["roi_pct"] for k, v in frozen.items()},
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("sealed", "plain"), required=True)
    args = p.parse_args()
    run(args.source)

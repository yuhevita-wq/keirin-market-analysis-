"""2025 frozen evaluation of the completed 2024 chapter-1 deletion AIs.

The 2024 year-end coefficients, frozen 2023 parent models, selected coverage,
ticket universe, and JPY100 flat stake are immutable. No 2025 training, model
selection, threshold tuning, or post-hoc filtering is permitted.

This is a historical, final-odds out-of-time evaluation relative to 2024
training, NOT a project-wide never-before-seen blind holdout. Other project
studies have previously examined 2025 outcomes.

Usage:
  python paper_only_ai/chapter1_frozen_2025.py --check
  python paper_only_ai/chapter1_frozen_2025.py --quarter 1
  ...
  python paper_only_ai/chapter1_frozen_2025.py --annual
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import chapter1_iterative_deletion as learner
import chapter1_blind_exam as exam
import chapter1_accuracy_calibration_duel as duel
import chapter1_full_sheet_deletion as original

YEAR = 2025
ROOT = Path("docs/paper-only-ai-2023/chapter1_deletion_continual/frozen_2025")
Q4 = Path("docs/paper-only-ai-2023/chapter1_deletion_continual/quarter_2024_q4.json")
SNAPSHOT = Path("docs/paper-only-ai-2023/chapter1_deletion_continual/checkpoint.json")
KINDS = learner.KINDS
ROLES = learner.ROLES


def frozen_state():
    before = SNAPSHOT.read_bytes()
    state = learner.load_state()
    last = json.loads(Q4.read_text("utf-8"))
    if state["processed_quarters"] != [1, 2, 3, 4] or state["status"] != "ALL_2024_QUARTERS_TRAINED":
        raise RuntimeError("2024 four-quarter learning is not frozen")
    if state["last_completed_race_date"] != "2024-12-31":
        raise RuntimeError("Wrong 2024 training cutoff")
    if state["parent_ids"] != {"accuracy": "acc-g08-c01", "calibration": "g01-c01"}:
        raise RuntimeError("Not the original duel models")
    if state["frozen_2023_coverage"] != {"accuracy": 0.35, "calibration": 0.5}:
        raise RuntimeError("Changed 2023 coverage policy")
    theta = {r: {k: np.asarray(state["weight_corrections"][r][k], dtype=float)
                 for k in KINDS} for r in ROLES}
    digest = learner.model_digest(theta)
    if digest != last["new_weight_sha256"]:
        raise RuntimeError("2024 frozen weights do not match Q4 learning report")
    return theta, {
        "checkpoint_sha256": hashlib.sha256(before).hexdigest(),
        "weight_model_sha256": digest,
        "2024_end": state["last_completed_race_date"],
        "parent_ids": state["parent_ids"],
        "coverage": state["frozen_2023_coverage"],
    }


def configure_year():
    learner.YEAR = YEAR
    learner.source_entries = lambda quarter: (
        exam.quarter_sources(YEAR, quarter)
        + [f"data/{YEAR}/s_class_yosen/entries.csv"]
    )


def load_2025_aligned(quarter):
    """Separate evaluator: old 2024 loader hardcodes its labels and odds."""
    start = f"{YEAR}-{(quarter - 1) * 3 + 1:02d}-01"
    end = f"{YEAR}-{quarter * 3:02d}-{31 if quarter in (1, 4) else 30:02d}"
    learner.bt.LABELS = f"data/result_universe_v2/combined/results_{YEAR}.jsonl.gz"
    learner.bt.START, learner.bt.END = start, end
    learner.bt.sources = lambda: learner.source_entries(quarter)
    learner.prices.ODDS = f"data/odds_only/odds_{YEAR}.jsonl.gz"
    cards, card_audit = learner.bt.load_2024_line_cards()
    odds, odds_audit = learner.prices.collect_2024_books()
    failures = Counter()
    matched = []
    for rid, race in cards.items():
        if rid not in odds:
            failures["missing_odds"] += 1
            continue
        try:
            book = learner.prices.decode_odds(
                odds[rid], rid, set(race["cars"])
            )
        except (TypeError, ValueError, KeyError) as exc:
            failures[type(exc).__name__ + ":" + str(exc)[:55]] += 1
            continue
        matched.append((rid, race, book))
    matched.sort(key=lambda entry: (entry[1]["date"], entry[0]))
    if len(matched) < 100:
        raise RuntimeError("Insufficient aligned 2025 races; no score will be claimed")
    return matched, {
        "from": start, "to": end,
        "cards": len(cards), "aligned_races": len(matched),
        "cards_audit": card_audit, "odds_audit": odds_audit,
        "dropped": dict(failures),
        "year_specific_result_path": learner.bt.LABELS,
        "year_specific_odds_path": learner.prices.ODDS,
    }


def score_quarter(quarter):
    if quarter not in (1, 2, 3, 4):
        raise ValueError("quarter must be 1, 2, 3, or 4")
    target = ROOT / f"quarter_2025_q{quarter}.json"
    if target.exists():
        raise RuntimeError("Already scored: refusing to overwrite the frozen exam")
    theta, fingerprint = frozen_state()
    configure_year()
    aligned, audit = load_2025_aligned(quarter)
    parents = learner.frozen_deletion_parents()
    original_weights = duel.fetch_immutable_weights()
    if not aligned:
        raise RuntimeError("No complete 2025 races")
    totals = {r: {k: learner.empty_stats() for k in KINDS} for r in ROLES}
    by_month = defaultdict(lambda: {r: {k: learner.empty_stats() for k in KINDS}
                                    for r in ROLES})
    by_class = defaultdict(lambda: {r: {k: learner.empty_stats() for k in KINDS}
                                    for r in ROLES})
    progress = 0
    for offset in range(0, len(aligned), 16):
        chunk = aligned[offset:offset + 16]
        for rid, date, groups in learner.records_for_batch(chunk, *original_weights[:2]):
            source = next(race["source"] for race_id, race, _ in chunk if race_id == rid)
            decisions = []
            for row in groups:
                role, kind = row["role"], row["kind"]
                probabilities = learner.ticket_probabilities(
                    row, parents[role][kind], theta[role][kind]
                )
                keep = learner.remove_from_all(
                    probabilities, row["prices"], learner.COVERAGE[role]
                )
                previous = original.delete_from_full(
                    parents[role][kind], row, learner.COVERAGE[role]
                )
                decisions.append((row, keep, previous))
            # All four decisions are sealed before any outcome is scored.
            for row, keep, previous in decisions:
                role, kind = row["role"], row["kind"]
                win = row["winner_index"]
                if not 0 <= win < len(row["prices"]):
                    raise RuntimeError("Winner absent from a complete odds book")
                for destination in (
                    totals[role][kind],
                    by_month[date[:7]][role][kind],
                    by_class[source][role][kind],
                ):
                    learner.mark_score(
                        destination, row["prices"], win, keep, previous
                    )
            # There is intentionally NO update_after_result call here.
            progress += 1
        if offset % 256 == 0 or offset + 16 >= len(aligned):
            print("FROZEN_2025_PROGRESS", quarter, progress, len(aligned), flush=True)
    for role in ROLES:
        for kind in KINDS:
            if totals[role][kind]["races"] != len(aligned):
                raise RuntimeError("One or more race/market predictions missing")
    assert_unchanged(fingerprint)
    report = {
        "status": "FROZEN_2025_QUARTER_SCORED_ONCE",
        "year": YEAR,
        "quarter": quarter,
        "evaluation": "untuned_out_of_time_historical_final_odds",
        "not_project_wide_pristine_blind": True,
        "historically_other_2025_studies_exist": True,
        "fit_on_2025_races": False,
        "selection_or_threshold_tuning_on_2025": False,
        "only_final_odds_not_purchaseable_preclose_quotes": True,
        "bet_size_per_mark_yen": 100,
        "full_ticket_universe_3rentan_3renpuku": True,
        "checkpoint": fingerprint,
        "frozen_parent_artifact_sha256": original_weights[2],
        "races": len(aligned),
        "data_audit": audit,
        "raw": totals,
        "by_model_and_market": {
            r: {k: learner.summarize(totals[r][k]) for k in KINDS} for r in ROLES
        },
        "by_month": {
            m: {r: {k: learner.summarize(s[r][k]) for k in KINDS}
                for r in ROLES} for m, s in sorted(by_month.items())
        },
        "by_class": {
            cls: {r: {k: learner.summarize(s[r][k]) for k in KINDS}
                  for r in ROLES} for cls, s in sorted(by_class.items())
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print("FROZEN_2025_QUARTER_COMPLETE", quarter, progress, target, flush=True)


def assert_unchanged(before):
    _, now = frozen_state()
    if now != before:
        raise RuntimeError("Frozen 2024 model checkpoint was changed")


def combine(stats):
    combined = learner.empty_stats()
    for s in stats:
        for name, v in s.items():
            combined[name] += v
    return learner.summarize(combined)


def annual():
    target = ROOT / "annual_2025.json"
    if target.exists():
        raise RuntimeError("Annual result already published: no re-selection")
    _, model = frozen_state()
    quarters = [
        json.loads((ROOT / f"quarter_2025_q{i}.json").read_text("utf-8"))
        for i in range(1, 5)
    ]
    for i, q in enumerate(quarters, 1):
        if (q["quarter"] != i or q["checkpoint"] != model
                or q["fit_on_2025_races"]
                or q["selection_or_threshold_tuning_on_2025"]):
            raise RuntimeError("Quarter cannot be combined as a frozen test")
    result = {}
    for role in ROLES:
        result[role] = {}
        for kind in KINDS:
            result[role][kind] = combine(
                [q["raw"][role][kind] for q in quarters]
            )
        result[role]["combined"] = combine(
            [q["raw"][role][kind] for q in quarters for kind in KINDS]
        )
    report = {
        "status": "2025_FROZEN_ANNUAL_EVALUATION_COMPLETE",
        "evaluation": "untuned_out_of_time_historical_final_odds",
        "historical_not_project_wide_pristine": True,
        "all_results_published_regardless_of_profit": True,
        "only_final_odds": True,
        "frozen_2024_checkpoint": model,
        "races": sum(q["races"] for q in quarters),
        "quarterly_races": [q["races"] for q in quarters],
        "by_role_market_and_combined": result,
        "passes_profit_and_simple_baselines": {
            r: {
                k: bool(
                    result[r][k]["roi_percent"] > 100
                    and result[r][k]["roi_percent"]
                    > result[r][k]["cheapest_odds_same_count_roi_percent"]
                    and result[r][k]["roi_percent"]
                    > result[r][k]["original_frozen_delete_roi_percent"]
                )
                for k in (*KINDS, "combined")
            }
            for r in ROLES
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print("FROZEN_2025_ANNUAL_COMPLETE", json.dumps({
        r: {
            k: result[r][k]["roi_percent"]
            for k in (*KINDS, "combined")
        } for r in ROLES
    }, ensure_ascii=False), flush=True)


def check():
    _, frozen = frozen_state()
    class Dummy:
        def predict_proba(self, x):
            vals = np.array([0.4, 0.6, 0.7], dtype=float)
            return np.column_stack((1 - vals, vals))
    x = np.zeros((3, learner.N_FEATURES), dtype=float)
    row = {"x": x, "prices": np.array([2.5, 4.5, 8.0]),
           "winner_index": 1}
    weights = np.zeros(learner.N_FEATURES, dtype=float)
    q1 = learner.ticket_probabilities(row, Dummy(), weights)
    m1 = learner.remove_from_all(q1, row["prices"], 0.35)
    q2 = learner.ticket_probabilities({**row, "winner_index": 2}, Dummy(), weights)
    m2 = learner.remove_from_all(q2, row["prices"], 0.35)
    if not np.array_equal(m1, m2):
        raise RuntimeError("Outcome leakage in the pre-result deletion decision")
    if not np.allclose(q1, q2):
        raise RuntimeError("Outcome leakage in predictions")
    print("FROZEN_2025_CHECK_PASSED", frozen["weight_model_sha256"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--quarter", type=int)
    group.add_argument("--annual", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    elif args.annual:
        annual()
    else:
        score_quarter(args.quarter)


if __name__ == "__main__":
    main()

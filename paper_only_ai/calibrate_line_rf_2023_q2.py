"""2023-only out-of-time temperature calibration, then untouched 2024 Q2 test.

Walsh & Joshi select models by calibration rather than accuracy; they did NOT
propose temperature scaling. This is a clearly labelled, one-parameter-per-
conditional-stage extension to the already-frozen 2023 LINE RF architecture.

No 2024 labels or ROI enter fitting, calibration, model/threshold selection.
2024 Q1 is deliberately excluded: its 54.89% return is already known.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from compare_lines_2023 import load, matrix, classifier, feature
import backtest_line_rf_2024_q1 as bt
from backtest_roi_2024_q1 import collect_2024_books, decode_odds

OUT = Path("docs/paper-only-ai-2023/line_calibration_2023_q2_2024")
EPS = 1e-12
SEED = 230306021
T_MIN, T_MAX = 0.5, 2.5
DATE_FIT = "2023-11"
DATE_VALID = "2023-12"
Q2_START, Q2_END = "2024-04-01", "2024-06-30"
BATCH = 24
MARKETS = ("3連単", "3連複")


def conditional_groups(events, model, stage):
    """One actual-prefix choice set per race, from chronologically held-out races.

    Actual first/second finishers appear ONLY to define the conditional target
    during calibration. There is no use of result data during test inference.
    """
    groups = []
    vectors = []
    for rid, race in events:
        cars = sorted(race["cars"], key=int)
        finish = race["order"]
        prefix = finish[:stage - 1]
        eligible = [c for c in cars if c not in prefix]
        truth = eligible.index(finish[stage - 1])
        start = len(vectors)
        for c in eligible:
            vectors.append(feature(race, c,
                                   prefix[0] if stage >= 2 else None,
                                   prefix[1] if stage >= 3 else None,
                                   use_line=True))
        groups.append((rid, race["date"], start, len(vectors), truth))
    raw = np.asarray(model.predict_proba(np.asarray(vectors, dtype=np.float32))[:, 1],
                     dtype=float)
    logs = np.log(np.clip(raw, EPS, 1.0))
    return [(rid, day, logs[start:end], truth)
            for rid, day, start, end, truth in groups]


def distribution(logits, temperature):
    z = logits / float(temperature)
    return np.exp(z - logsumexp(z))


def mean_nll(rows, temperature):
    if not rows:
        raise ValueError("Empty calibration date split")
    return float(np.mean([logsumexp(x / temperature) -
                          x[target] / temperature
                          for _, _, x, target in rows]))


def bin_diagnostic(rows, temperature):
    """Positive-class pooled 20-bin diagnostic; not literal NBA classwise ECE."""
    preds = []
    labels = []
    for _, _, log_score, target in rows:
        p = distribution(log_score, temperature)
        preds.extend(p)
        labels.extend(int(i == target) for i in range(len(p)))
    p = np.asarray(preds, dtype=float)
    y = np.asarray(labels, dtype=float)
    which = np.minimum((p * 20).astype(int), 19)
    occupied = 0
    ece = 0.0
    for b in range(20):
        take = which == b
        if not np.any(take):
            continue
        occupied += 1
        ece += float(np.sum(take) / len(p) *
                     abs(np.mean(y[take]) - np.mean(p[take])))
    return {"positive_class_pooled_ece20": ece,
            "occupied_bins": occupied,
            "total_bins": 20,
            "paper_80pct_bin_occupancy_gate_passed_for_this_diagnostic": occupied >= 16,
            "not_full_ticket_classwise_ece": True}


def train_2023_and_calibrate():
    training, holdout, audit = load()
    ordered = sorted(holdout.items(), key=lambda item: (item[1]["date"], item[0]))
    fit = [(rid, race) for rid, race in ordered
           if race["date"].startswith(DATE_FIT)]
    val = [(rid, race) for rid, race in ordered
           if race["date"].startswith(DATE_VALID)]
    if len(fit) < 100 or len(val) < 100:
        raise RuntimeError("2023 calendar calibration split is incomplete")
    summary = {"model_fit_date_range": ["2023-01-01", "2023-10-31"],
               "calibration_fit_month": DATE_FIT,
               "diagnostic_validation_month": DATE_VALID,
               "training_races": len(training),
               "fit_races": len(fit),
               "validation_races": len(val),
               "stage_metrics": {},
               "no_2024_labels_or_odds_used": True,
               "full_year_frozen_model_refitted": False}
    temps = {}
    for stage in (1, 2, 3):
        X, y, _ = matrix(training, stage, use_line=True)
        model = classifier("RF")
        model.fit(X, y)
        nov = conditional_groups(fit, model, stage)
        dec = conditional_groups(val, model, stage)
        opt = minimize_scalar(lambda t: mean_nll(nov, t),
                              method="bounded",
                              bounds=(T_MIN, T_MAX),
                              options={"xatol": 1e-4})
        if not opt.success:
            raise RuntimeError("Temperature optimizer failed")
        temp = float(opt.x)
        temps[str(stage)] = temp
        summary["stage_metrics"][str(stage)] = {
            "temperature_fitted_november_only": temp,
            "november_base_nll": mean_nll(nov, 1.0),
            "november_calibrated_nll": mean_nll(nov, temp),
            "december_base_nll": mean_nll(dec, 1.0),
            "december_calibrated_nll": mean_nll(dec, temp),
            "december_base_calibration": bin_diagnostic(dec, 1.0),
            "december_calibrated_calibration": bin_diagnostic(dec, temp)}
        print("STAGE", stage, "T", temp, "DEC_NLL",
              summary["stage_metrics"][str(stage)]["december_calibrated_nll"],
              flush=True)
    summary["december_trifecta_nll_via_conditional_chain_base"] = sum(
        summary["stage_metrics"][str(s)]["december_base_nll"]
        for s in (1, 2, 3))
    summary["december_trifecta_nll_via_conditional_chain_calibrated"] = sum(
        summary["stage_metrics"][str(s)]["december_calibrated_nll"]
        for s in (1, 2, 3))
    summary["audit_race_split"] = {
        "model_train_end": "2023-10-31",
        "calibrator_train_end": "2023-11-30",
        "first_calibrator_check": "2023-12-01",
        "2023_nov_dec_already_used_for_prior_line_model_selection": True,
        "2024_q1_known_result_never_used_in_calibrator": True}
    return temps, summary


def q2_sources_2024():
    return [
        "data/a_class_12/2024_03_04/entries.csv",
        "data/a_class_12/2024_05_06/entries.csv",
        "data/a3_challenge/2024_03_04/entries.csv",
        "data/a3_challenge/2024_05_06/entries.csv",
        "data/2024/s_class_f1_all_parts/2024_q2/entries.csv",
        "data/2024/s_class_yosen/entries.csv",
    ]


def score_q2(temperatures):
    # Use existing strict cards/results/odds joins; the date range is Q2 only.
    bt.START, bt.END = Q2_START, Q2_END
    bt.sources = q2_sources_2024
    cards, card_audit = bt.load_2024_line_cards()
    odds, odds_audit = collect_2024_books()
    intersect = sorted(set(cards) & set(odds),
                       key=lambda rid: (cards[rid]["date"], rid))
    models, manifest, hashes = bt.load_line_models()
    ledgers = {variant: {kind: bt.Ledger() for kind in MARKETS}
               for variant in ("unchanged_full_year_2023", "calibrated_from_2023")}
    audit = Counter()
    accepted = 0
    for off in range(0, len(intersect), BATCH):
        chunk = []
        books = {}
        for rid in intersect[off:off + BATCH]:
            race = cards[rid]
            try:
                books[rid] = decode_odds(odds[rid], rid, set(race["cars"]))
            except ValueError as exc:
                audit["excluded_odds_" + str(exc)] += 1
                continue
            chunk.append((rid, race))
        if not chunk:
            continue
        initial = bt.infer_new_chunk(chunk, models)
        changed = bt.infer_new_chunk(chunk, models, temperatures=temperatures)
        for rid, race in chunk:
            accepted += 1
            for name, dist_all in (("unchanged_full_year_2023", initial),
                                   ("calibrated_from_2023", changed)):
                dist = dist_all[rid]
                bt.safe_records(dist, books[rid])
                actual = bt.actual_tickets(race)
                for kind in MARKETS:
                    prices = books[rid][kind]
                    p_book = dist[kind]
                    # Predefined value-bet rule. No threshold tuning or point cap.
                    bets = {ticket: (kind, p, prices[ticket])
                            for ticket, p in p_book.items()
                            if p * prices[ticket] > 1.0}
                    ledgers[name][kind].update(bets, actual[kind],
                                               prices, p_book)
        print("Q2_PROGRESS", min(off + BATCH, len(intersect)),
              len(intersect), flush=True)
    if accepted == 0:
        raise RuntimeError("No valid 2024 Q2 odds/card/result joins")
    result = {}
    for name, by_kind in ledgers.items():
        result[name] = {kind: by_kind[kind].summary()
                        for kind in MARKETS}
        invested = sum(by_kind[k].invest for k in MARKETS)
        paid = sum(by_kind[k].paid for k in MARKETS)
        result[name]["combined"] = {
            "eligible_races": accepted,
            "purchased_tickets": sum(by_kind[k].tickets for k in MARKETS),
            "investment_yen": round(invested, 2),
            "payout_yen": round(paid, 2),
            "net_profit_yen": round(paid - invested, 2),
            "return_rate_percent": 100.0 * paid / invested if invested else None}
    return {"test_dates": [Q2_START, Q2_END],
            "joined_before_odds_validation": len(intersect),
            "accepted_complete_races": accepted,
            "excluded_card_without_complete_odds": len(cards) - len(intersect),
            "excluded_odds": dict(audit),
            "card_audit": card_audit,
            "odds_source_audit": odds_audit,
            "model_sha256": hashes,
            "model_training_dates": manifest["training_date_range"],
            "2024_fit_calls": 0,
            "2024_threshold_search_calls": 0,
            "odds_timing": "historical final (not 15 minutes before close)",
            "flat_stake_yen_per_ticket": 100,
            "result": result}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    (OUT / "progress.json").write_text(json.dumps({
        "status": "RUNNING", "started_at": started}, indent=2) + "\n")
    temps, cal_report = train_2023_and_calibrate()
    artifact = {"schema": "three_stage_temperature_2023_v1",
                "temperatures": temps,
                "method": "conditional-score temperature scaling",
                "origin": "posthoc engineering extension; NOT Walsh-Joshi specified procedure",
                "fit_only": "November 2023 from Jan-Oct 2023 RF",
                "december_validation_unchanged": True,
                "q1_2024_tuning": False,
                "min_temperature": T_MIN,
                "max_temperature": T_MAX}
    artifact_bytes = (json.dumps(artifact, ensure_ascii=False, indent=2) +
                      "\n").encode()
    (OUT / "temperatures.json").write_bytes(artifact_bytes)
    report = {"status": "COMPLETE_2023_ONLY_CALIBRATION_AND_2024_Q2_OOS",
              "started_at": started,
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "source_paper": "Walsh & Joshi, arXiv:2303.06021v4",
              "paper_method": "model selection based on calibration, using 20-bin classwise ECE and 80pct occupied-bin gate",
              "engineering_extension": "three conditional RF temperature parameters; not in paper",
              "temperature_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
              "2023_calibration": cal_report,
              "2024_q2_frozen_comparison": score_q2(temps),
              "selection_rule": "all tickets with P_model * historical final odds > 1",
              "limitations": [
                  "No 2024 label is used to adjust any temperature or threshold.",
                  "2023 Nov-Dec were previously consulted when choosing LINE-aware RF; Dec is only a diagnostic for this calibration fit, not a fully independent model-family holdout.",
                  "20-bin occupied-bin and positive-class diagnostics are not literal Walsh-Joshi multiclass ticket ECE certification.",
                  "Calibrator is learned using Jan-Oct model and transferred to a different full-year 2023 refit; transferability is tested, not assumed.",
                  "No 2024 Q1 outcomes or ROI are used here. Q2 labels are evaluation only.",
                  "Historical final odds are unavailable in actual pre-close form, so this does not claim live tradability.",
                  "Fixed 100-yen positive-EV ticket rule can buy many correlated outcomes; PnL measures a hypothetical unlimited-capital ledger.",
              ]}
    (OUT / "report.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "progress.json").write_text(json.dumps({
        "status": report["status"], "finished_at": report["finished_at"],
        "accepted_q2_races": report["2024_q2_frozen_comparison"]["accepted_complete_races"]},
        ensure_ascii=False, indent=2) + "\n")
    print("CALIBRATION_COMPLETE", json.dumps({
        "temperatures": temps,
        "q2": {k: v["combined"]
               for k, v in report["2024_q2_frozen_comparison"]["result"].items()}
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

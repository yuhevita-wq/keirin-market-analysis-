"""Learn market-aware ticket probabilities, then TEST actual historical payout.

NOT an automatic claim of profit: 2023 Sep learns the market/AI fusion; Oct
chooses the profitable decision policy; Nov and Dec are scored ONCE as separate
time-forward historical checks. 2024 Q1/Q2 known PnL and all 2025 data are never
consulted. The learned artifact is only eligible for promotion if its Nov-Dec
results support it. Its historical test is NOT a fully untouched academic exam
because prior 2023 model-family work consulted Nov-Dec.

This is an additional probabilistic model, not changing frozen LINE RF weights.
No arbitrary preselected ticket set: inspect every available trifecta and trio.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import backtest_line_rf_2024_q1 as bt
import backtest_roi_2024_q1 as odds_mod
import chapter1_blind_exam as exam

ROOT = Path("docs/paper-only-ai-2023/chapter1_profit_lab")
OUT = ROOT / "2023_market_fusion.json"
SEED = 230306021
KINDS = ("3連単", "3連複")
MIN_TICKETS = 300
MIN_WINNING_TICKETS = 20
BUFFERS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)
MIN_QUALIFIED_PER_MONTH = 150


def coherent_probability(kind, ai, market, params):
    """All ticket probabilities sum to one; one truth per complete market book."""
    ai = np.asarray(ai, dtype=np.float64)
    market = np.asarray(market, dtype=np.float64)
    if ai.shape != market.shape or len(ai) < 3:
        raise ValueError("Bad ticket market/model shape")
    if not np.all(np.isfinite(ai)) or not np.all(np.isfinite(market)):
        raise ValueError("Non-finite ticket probabilities")
    if np.min(ai) <= 0 or np.min(market) <= 0:
        raise ValueError("Nonpositive probability")
    m = market / market.sum()
    a = ai / ai.sum()
    alpha, gamma = params["alpha"], params["gamma"]
    if params["family"] == "linear":
        q = np.power(m, gamma)
        q = q / q.sum()
        v = (1 - alpha) * q + alpha * a
    elif params["family"] == "log":
        z = gamma * (1 - alpha) * np.log(m) + alpha * np.log(a)
        z -= np.max(z)
        v = np.exp(z)
        v = v / v.sum()
    else:
        raise ValueError("Unknown probability family")
    if not np.isfinite(v).all() or abs(float(v.sum())-1) > 1e-9:
        raise RuntimeError("Ticket distribution normalization failed")
    return v


def extract_book_rows(year, candidate):
    """Load only 2023 Sep-Dec. The model never sees 2024+ training labels."""
    if year != 2023:
        raise RuntimeError("Profit lab cannot read another result year")
    paths = list(dict.fromkeys(exam.quarter_sources(year, 3) +
                               exam.quarter_sources(year, 4)))
    bt.LABELS = "data/result_universe_v2/combined/results_2023.jsonl.gz"
    bt.START, bt.END = "2023-09-01", "2023-12-31"
    bt.sources = lambda: paths
    odds_mod.ODDS = "data/odds_only/odds_2023.jsonl.gz"
    cards, card_audit = bt.load_2024_line_cards()
    market_raw, odds_audit = odds_mod.collect_2024_books()
    books, failures = {}, Counter()
    matched = []
    for rid, race in cards.items():
        if rid not in market_raw:
            failures["missing_market"] += 1
            continue
        try:
            book = odds_mod.decode_odds(
                market_raw[rid], rid, set(race["cars"]))
        except (ValueError, KeyError, TypeError) as exc:
            failures[type(exc).__name__ + ":" + str(exc)[:80]] += 1
            continue
        books[rid] = book
        matched.append((rid, race))
    if len(matched) < 500:
        raise RuntimeError("Too few complete, odds-matched historical cards")
    with tempfile.TemporaryDirectory(prefix="chapter1_historical_") as td:
        model, sha256 = exam.load_frozen_champion(candidate, td)
        records = []
        for start in range(0, len(matched), 20):
            chunk = matched[start:start+20]
            predictions = bt.infer_new_chunk(chunk, model)
            for rid, race in chunk:
                actual = bt.actual_tickets(race)
                book = books[rid]
                for kind in KINDS:
                    b = book[kind]
                    combinations = list(b)
                    p = np.asarray([predictions[rid][kind][ticket]
                                    for ticket in combinations], dtype=np.float64)
                    prices = np.asarray([b[ticket] for ticket in combinations],
                                        dtype=np.float64)
                    implied = 1.0 / prices
                    fair = implied / implied.sum()
                    truth = combinations.index(actual[kind])
                    records.append({
                        "date": race["date"], "race": rid,
                        "kind": kind, "ai": p, "market": fair,
                        "prices": prices, "actual_index": truth,
                    })
            if (start // 20) % 30 == 0:
                print("LAB_DATA_PROCESS", start + len(chunk), len(matched),
                      flush=True)
    months = Counter(row["date"][:7] for row in records if row["kind"] == "3連単")
    for month in ("2023-09", "2023-10", "2023-11", "2023-12"):
        if months[month] < MIN_QUALIFIED_PER_MONTH:
            raise RuntimeError(f"Insufficient market-matched cohort in {month}: {months[month]}")
    return records, {"matched_races": len(matched), "month_counts": dict(months),
                     "model_id": candidate["id"], "model_archive_sha256": sha256,
                     "cards": card_audit, "odds": odds_audit,
                     "missing_book": dict(failures),
                     "source_dates": [bt.START, bt.END]}


def nll_on_rows(rows, parameters):
    loss = 0.0
    for row in rows:
        p = coherent_probability(row["kind"], row["ai"], row["market"],
                                 parameters)
        loss -= math.log(max(float(p[row["actual_index"]]), 1e-14))
    return loss / len(rows)


def learn_fusions(september):
    """No October/November/December outcomes in parameter fitting."""
    results = {}
    for kind in KINDS:
        rows = [r for r in september if r["kind"] == kind]
        if len(rows) < MIN_QUALIFIED_PER_MONTH:
            raise RuntimeError("Cannot train one of the odds/fusion markets")
        options = {
            "inherited_model": {"family": "linear", "alpha": 1.0, "gamma": 1.0},
            "normalized_market": {"family": "linear", "alpha": 0.0, "gamma": 1.0},
        }
        for family in ("linear", "log"):
            def loss(x):
                return nll_on_rows(rows, {
                    "family": family, "alpha": float(x[0]), "gamma": float(x[1])
                })
            solutions = []
            for init in ((0.05, 1), (0.35, 1), (0.7, 1)):
                opt = minimize(loss, x0=init, method="L-BFGS-B",
                               bounds=((0, 1), (0.65, 1.65)),
                               options={"maxiter": 90, "ftol": 1e-9})
                if opt.success and np.isfinite(opt.fun):
                    solutions.append(opt)
            if not solutions:
                raise RuntimeError("Historical market-price probability fit failed")
            opt = min(solutions, key=lambda s: s.fun)
            options["fitted_" + family] = {
                "family": family, "alpha": float(opt.x[0]),
                "gamma": float(opt.x[1]),
            }
        results[kind] = {
            "parameters_by_family": options,
            "september_training_nll": {
                family: nll_on_rows(rows, p) for family, p in options.items()
            },
            "september_races": len(rows),
        }
    return results


def realized_book_profit(rows, fused, buffer):
    stake = payout = expected = 0.0
    hits = tickets = races = bet_races = 0
    nll = market_nll = 0.0
    per_month = defaultdict(lambda: {"investment_yen": 0.0,
                                    "payout_yen": 0.0,
                                    "bets": 0, "winning_tickets": 0})
    for row in rows:
        prices = row["prices"]
        predicted = coherent_probability(
            row["kind"], row["ai"], row["market"],
            fused[row["kind"]])
        risk = predicted * prices
        selected = risk > (1.0 + buffer)
        count = int(selected.sum())
        winner = row["actual_index"]
        win = bool(selected[winner])
        cost = 100.0 * count
        paid = 100.0 * float(prices[winner]) if win else 0.0
        stake += cost
        payout += paid
        expected += float(100.0 * np.dot(selected, risk))
        hits += int(win)
        tickets += count
        races += 1
        bet_races += int(count > 0)
        nll -= math.log(max(float(predicted[winner]), 1e-14))
        market_nll -= math.log(max(float(row["market"][winner]), 1e-14))
        month = per_month[row["date"][:7]]
        month["investment_yen"] += cost
        month["payout_yen"] += paid
        month["bets"] += count
        month["winning_tickets"] += int(win)
    return {
        "eligible_market_race_observations": races,
        "purchased_tickets": tickets,
        "winning_tickets": hits,
        "betting_market_race_observations": bet_races,
        "investment_yen": round(stake, 2),
        "payout_yen": round(payout, 2),
        "profit_yen": round(payout - stake, 2),
        "return_percent": 100.0 * payout / stake if stake else None,
        "model_expected_return_percent": 100.0 * expected / stake if stake else None,
        "nll": nll / races if races else None,
        "market_nll": market_nll / races if races else None,
        "per_month": {k: {**v,
            "return_percent": 100 * v["payout_yen"] / v["investment_yen"]
                              if v["investment_yen"] else None}
                      for k, v in sorted(per_month.items())}
    }


def policies(models):
    names = sorted(set.intersection(
        *(set(models[k]["parameters_by_family"]) for k in KINDS)))
    return {name: {kind: models[kind]["parameters_by_family"][name]
                   for kind in KINDS} for name in names}


def choose_development_policy(october, candidate_policies):
    """October is development data. Do not use Nov-Dec to choose the policy."""
    rankings = []
    for label, fusion in candidate_policies.items():
        for buffer in BUFFERS:
            stats = realized_book_profit(october, fusion, buffer)
            hits = stats["winning_tickets"]
            size = stats["purchased_tickets"]
            passes_sample = size >= MIN_TICKETS and hits >= MIN_WINNING_TICKETS
            rankings.append({
                "label": label, "buffer": buffer, "october": stats,
                "sample_gate": passes_sample,
            })
    eligible = [r for r in rankings if r["sample_gate"]]
    if not eligible:
        return None, rankings
    # This is explicitly a development selection maximizing realized Oct return,
    # NOT a claim of independent profitability.
    best = max(eligible, key=lambda r: (
        r["october"]["return_percent"],
        r["october"]["profit_yen"],
        r["october"]["winning_tickets"]))
    return {"label": best["label"], "buffer": best["buffer"]}, rankings


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print("HISTORICAL_LAB_ALREADY_EXECUTED_ONCE__NO_RETROACTIVE_RESELECTION",
              flush=True)
        return
    state = json.loads(Path(
        "docs/paper-only-ai-2023/autonomous_chapter1/state.json").read_text("utf-8"))
    if state["generation"] < 6:
        raise RuntimeError("Minimum self-training generations not completed")
    candidate = state["champions"]["calibration"]
    rows, audit = extract_book_rows(2023, candidate)
    september = [r for r in rows if r["date"][:7] == "2023-09"]
    october = [r for r in rows if r["date"][:7] == "2023-10"]
    future = [r for r in rows if r["date"][:7] in ("2023-11", "2023-12")]
    learned = learn_fusions(september)
    candidate_policies = policies(learned)
    choice, oct_rankings = choose_development_policy(october, candidate_policies)
    outcomes = {}
    for label, fused in candidate_policies.items():
        outcomes[label] = {
            "september_train": realized_book_profit(september, fused, 0),
            "october_dev": realized_book_profit(october, fused, 0),
        }
    decision = None
    if choice is not None:
        policy = candidate_policies[choice["label"]]
        buffer = choice["buffer"]
        outcome = realized_book_profit(future, policy, buffer)
        nov = outcome["per_month"].get("2023-11")
        dec = outcome["per_month"].get("2023-12")
        # An honest improvement must be profitable over BOTH separate months
        # AND over the total, with at least 300 total tickets.
        qualifies = (
            outcome["return_percent"] is not None
            and outcome["return_percent"] > 100
            and outcome["purchased_tickets"] >= MIN_TICKETS
            and nov is not None and dec is not None
            and all(v["return_percent"] is not None and
                    v["return_percent"] > 100 for v in (nov, dec))
        )
        decision = {
            "october_selected": choice,
            "frozen_policy_before_november": policy,
            "october_result": next(r["october"] for r in oct_rankings
                                   if r["label"] == choice["label"]
                                   and r["buffer"] == buffer),
            "held_out_november_december": outcome,
            "historical_profit_qualifies_two_months": bool(qualifies),
            "status": ("PASSED_HISTORICAL_TWO_MONTHS" if qualifies
                       else "FAILED_HISTORICAL_TWO_MONTHS"),
        }
    report = {
        "schema": "chapter1-market-fusion-profit-lab-2023-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": audit,
        "objective": "Improve actual ticket PnL without leaking later race results into model decisions",
        "learned_parameters_using_september_only": learned,
        "october_policy_candidates": [
            {"label": r["label"], "buffer": r["buffer"],
             "sample_gate": r["sample_gate"],
             "tickets": r["october"]["purchased_tickets"],
             "hits": r["october"]["winning_tickets"],
             "roi": r["october"]["return_percent"],
             "net_yen": r["october"]["profit_yen"]}
            for r in oct_rankings],
        "baseline_and_nll_comparisons": outcomes,
        "decision_and_historical_check": decision,
        "2024plus_results_used_in_training": False,
        "2023_nov_dec_used_in_selection": False,
        "passed_new_unknown_year_test": False,
        "may_claim_true_profit_edge": False,
        "cautions": [
            "2023 Nov-Dec had already been consulted in prior LINE model family development, so not untouched.",
            "Sep-trained frozen candidate temperature was already fitted on Sep labels; historical profit lab only adds odds-informed calibration.",
            "Oct selection is a search on realized historical ROI; it can overfit October. Never select on November/December outcomes.",
            "Final odds were used, not live 15-min preclose betting odds.",
            "Market-weighted ticket probability can converge to market prices, yielding no EV opportunities; abstention is valid.",
            "Every high-odds ticket is included in the full distribution but bought only if predicted positive expected value after required buffer.",
            "This is a model calibration/market-fusion search, not arbitrary retrospective filtering of winning selections.",
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("PROFIT_LAB_FINISHED", json.dumps({
        "candidate": candidate["id"], "races": audit["month_counts"],
        "selection": choice, "historic": (
            None if decision is None else {
                "roi": decision["held_out_november_december"]["return_percent"],
                "profitable_both_months": decision["historical_profit_qualifies_two_months"],
                "nov": decision["held_out_november_december"]["per_month"].get("2023-11", {}).get("return_percent"),
                "dec": decision["held_out_november_december"]["per_month"].get("2023-12", {}).get("return_percent"),
            })
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

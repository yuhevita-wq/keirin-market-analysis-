"""Racewise multinomial model of historical keirin market-pricing errors.

Train an odds-anchored correction from 2023 JAN-AUG completed outcomes, pick
regularization using SEPT forecast NLL, use OCT for historical entry policy
development, and score NOV and DEC exactly once as ordered prospective
historical checks. Repeated checks on the same months must not be re-optimized.

This model builds actual race-level ticket distributions, and its learned weights
are saved so AI can predict future races; it is not a hand-selected formation.
"""
from __future__ import annotations
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import backtest_line_rf_2024_q1 as bt
import backtest_roi_2024_q1 as odds_mod
from train_2023 import sources

ROOT = Path("docs/paper-only-ai-2023/chapter1_market_error")
REPORT = ROOT / "market_error_2023.json"
EPS = 1e-14
KINDS = ("3連単", "3連複")
REGULARIZATION = (0.05, 0.5, 5.0)
ENTRY_MARGIN = (0., 0.05, 0.10, 0.20)
N_FEATURES = 17

FEATURES = [
 "price_shape_log_odds_center", "price_rank_percentile",
 "market_probability_concentration",
 "three_same_line", "same_line_pair_count",
 "first_second_same_line", "first_third_same_line",
 "second_third_same_line", "first_is_line_leader",
 "second_is_line_leader", "third_is_line_leader",
 "sum_relative_score", "first_minus_second_score",
 "first_minus_third_score", "mean_age_diff_race",
 "three_line_sizes", "first_second_position_relation"
]
assert len(FEATURES) == N_FEATURES


def _feature_matrix(race, tickets, market, kind):
    pos = race["positions"]
    cards = race["cars"]
    car_numbers = sorted(cards, key=int)
    avg_score = np.mean([float(cards[k][4]) for k in car_numbers])
    avg_age = np.mean([float(cards[k][1]) for k in car_numbers])
    order_idx = np.argsort(np.argsort(-market))
    concentration = float(np.max(market)) * len(market)
    concentration = np.log1p(concentration) / 4
    x = np.zeros((len(tickets), N_FEATURES), dtype=np.float32)
    for idx, (a, b, c) in enumerate(tickets):
        ga, ra, na = pos[a]
        gb, rb, nb = pos[b]
        gc, rc, nc = pos[c]
        ab, ac, bc = int(ga == gb), int(ga == gc), int(gb == gc)
        pairs = ab + ac + bc
        sa, sb, sc = (float(cards[k][4]) for k in (a,b,c))
        aa, bb, cc = (float(cards[k][1]) for k in (a,b,c))
        x[idx] = [
          float(np.clip(np.log(max(market[idx] * len(tickets), EPS))/3, -3, 3)),
          float(order_idx[idx]) / max(len(tickets)-1, 1),
          concentration,
          float(pairs == 3), float(pairs)/3,
          float(ab) if kind == "3連単" else 0,
          float(ac) if kind == "3連単" else 0,
          float(bc) if kind == "3連単" else 0,
          float(ra == 1) if kind == "3連単" else 0,
          float(rb == 1) if kind == "3連単" else 0,
          float(rc == 1) if kind == "3連単" else 0,
          float(np.clip(((sa+sb+sc)/3-avg_score)/20, -2, 2)),
          float(np.clip((sa-sb)/20, -2, 2)) if kind == "3連単" else 0,
          float(np.clip((sa-sc)/20, -2, 2)) if kind == "3連単" else 0,
          float(np.clip(((aa+bb+cc)/3-avg_age)/20, -2, 2)),
          float((na+nb+nc)/27),
          float((rb-ra)/8) if kind == "3連単" else 0,
        ]
    assert np.isfinite(x).all()
    return x


def _load():
    bt.LABELS = "data/result_universe_v2/combined/results_2023.jsonl.gz"
    bt.START, bt.END = "2023-01-01", "2023-12-31"
    bt.sources = sources
    odds_mod.ODDS = "data/odds_only/odds_2023.jsonl.gz"
    cards, card_audit = bt.load_2024_line_cards()
    all_books, odds_audit = odds_mod.collect_2024_books()
    matched = Counter()
    data = {"3連単": [], "3連複": []}
    for rid, race in cards.items():
        raw_book = all_books.get(rid)
        if raw_book is None:
            matched["missing_book"] += 1
            continue
        try:
            book = odds_mod.decode_odds(raw_book, rid, set(race["cars"]))
        except (ValueError, KeyError, TypeError):
            matched["invalid_book"] += 1
            continue
        actual = bt.actual_tickets(race)
        matched["joined_races"] += 1
        matched["month_"+race["date"][:7]] += 1
        for kind in KINDS:
            tickets = list(book[kind])
            price = np.array([book[kind][k] for k in tickets], dtype=np.float64)
            m = 1.0/price
            m /= m.sum()
            if abs(float(m.sum())-1) > 1e-8:
                raise RuntimeError("Broken market implied probability")
            features = _feature_matrix(race, tickets, m, kind)
            truth = tickets.index(actual[kind])
            data[kind].append((race["date"], rid, features, np.log(m),
                               price, truth))
    if matched["joined_races"] < 2500:
        raise RuntimeError("Not enough complete market/race pairs")
    ranges = {}
    for key, rows in data.items():
        ranges[key] = {
            "train": [r for r in rows if r[0] <= "2023-08-31"],
            "validation": [r for r in rows if "2023-09-01"<=r[0]<="2023-09-30"],
            "develop": [r for r in rows if "2023-10-01"<=r[0]<="2023-10-31"],
            "test": [r for r in rows if "2023-11-01"<=r[0]<="2023-12-31"],
        }
        if min(len(v) for v in ranges[key].values()) < 200:
            raise RuntimeError("Missing chronological market cohort in "+key)
    return ranges, {"cards":card_audit, "odds":odds_audit,
                    "alignment":dict(matched),
                    "record_counts":{k:{s:len(v) for s,v in z.items()}
                                     for k,z in ranges.items()}}


def probability_from_weights(record, weights):
    _,_,x, log_market,_,_=record
    z = log_market + x @ weights
    z -= np.max(z)
    probs = np.exp(z)
    probs /= probs.sum()
    if not np.all(np.isfinite(probs)) or abs(float(probs.sum())-1)>1e-8:
        raise RuntimeError("Invalid learned full ticket distribution")
    return probs


def fit_one_kind(train, reg):
    """L2-regularized conditional multinomial maximum likelihood on race wins."""
    if not train:
        raise RuntimeError("Empty market residual training cohort")
    def objective(w):
        grad = np.zeros(N_FEATURES, dtype=np.float64)
        loss = 0.0
        for _,_,x,log_m,_,truth in train:
            z = log_m + x @ w
            top = np.max(z)
            expv = np.exp(z-top)
            total = np.sum(expv)
            probs = expv/total
            loss += math.log(total)+top-z[truth]
            grad += np.dot(probs,x)-x[truth]
        n = len(train)
        return (float(loss/n + 0.5*reg*np.dot(w,w)),
                grad/n + reg*w)
    opt = minimize(objective, np.zeros(N_FEATURES), method="L-BFGS-B",
                   jac=True, options={"maxiter":65, "ftol":1e-8, "gtol":1e-5})
    if not opt.success and opt.nit < 60:
        raise RuntimeError("Market error model optimizer diverged: "+str(opt.message))
    return np.asarray(opt.x,dtype=np.float64)


def likelihood(rows,w):
    return float(np.mean([-np.log(max(probability_from_weights(r,w)[r[-1]],EPS))
                          for r in rows]))


def select_weight_model(splits):
    train, val = splits["train"],splits["validation"]
    scores=[]
    for reg in REGULARIZATION:
        weight = fit_one_kind(train,reg)
        nll = likelihood(val,weight)
        scores.append({"reg":reg,"validation_nll":nll,"weights":weight.tolist()})
        print("MODEL_TRAINED",reg,"nll",nll,flush=True)
    best = min(scores,key=lambda d:d["validation_nll"])
    return np.array(best["weights"]), scores, best["reg"]


def combined_backtest(data, weights, margin):
    invested = returned = expected = 0.0
    n_bets = n_hits = n_races = 0
    details=defaultdict(lambda:{"investment_yen":0.0,"payout_yen":0.0,
                                 "tickets":0,"hits":0})
    for kind in KINDS:
        for record in data[kind]:
            date, rid, _, _, odds, target = record
            prob = probability_from_weights(record, weights[kind])
            positive = prob*odds > 1.0+margin
            tickets=int(positive.sum())
            hit=bool(positive[target])
            investment=100*tickets
            payout=100*float(odds[target]) if hit else 0.
            invested += investment
            returned += payout
            expected += 100*float(np.sum((prob*odds)[positive]))
            n_bets += tickets
            n_hits += int(hit)
            n_races += 1
            d=details[date[:7]]
            d["investment_yen"]+=investment
            d["payout_yen"]+=payout
            d["tickets"]+=tickets
            d["hits"]+=int(hit)
    return {
        "qualified_market_observations":n_races,
        "purchased_tickets":n_bets,"winning_tickets":n_hits,
        "investment_yen":round(invested,2),"payout_yen":round(returned,2),
        "net_profit_yen":round(returned-invested,2),
        "return_percent":100*returned/invested if invested else None,
        "model_expected_return_percent":100*expected/invested if invested else None,
        "months":{k:{**v,"return_percent":100*v["payout_yen"]/v["investment_yen"]
                       if v["investment_yen"] else None}
                  for k,v in sorted(details.items())}
    }


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if REPORT.exists():
        print("STOP__HISTORICAL_TAIL_ALREADY_SCORED_NO_REPEAT_TUNING",flush=True)
        return
    splits, audit = _load()
    weights, training = {},{}
    for kind in KINDS:
        w, trials, lam = select_weight_model(splits[kind])
        weights[kind] = w
        training[kind] = {"selected_lambda_on_september":lam,
                          "september_options":trials,
                          "features":FEATURES,
                          "coefficient":w.tolist()}
    october = {kind: splits[kind]["develop"] for kind in KINDS}
    margins=[]
    for margin in ENTRY_MARGIN:
        outcome=combined_backtest(october,weights,margin)
        allowed=(outcome["purchased_tickets"]>=300
                 and outcome["winning_tickets"]>=20)
        margins.append({"margin":margin,"passes_sample_rule":allowed,
                        "october_metrics":outcome})
    eligible=[r for r in margins if r["passes_sample_rule"]]
    winner=max(eligible,key=lambda r:r["october_metrics"]["return_percent"]) if eligible else None
    hist=None
    if winner:
        test={kind:splits[kind]["test"] for kind in KINDS}
        scored=combined_backtest(test,weights,winner["margin"])
        nov=scored["months"].get("2023-11",{})
        dec=scored["months"].get("2023-12",{})
        good=(scored["return_percent"] is not None
              and scored["return_percent"]>100 and
              all(x.get("return_percent") is not None and
                  x["return_percent"]>100 for x in (nov,dec))
              and scored["purchased_tickets"]>=300)
        hist={"selected_margin_october_only":winner["margin"],
              "november_december_metrics":scored,
              "both_months_profitable":bool(good)}
    report={
        "schema":"market-residual-learner-2023-v1",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "training":training,"candidate_selection_october":margins,
        "frozen_historical_test":hist,"audit":audit,
        "weights_learned_from":"2023-01-01 through 2023-08-31",
        "hyperparameters_selected_using":"2023-09-01 through 2023-09-30",
        "purchase_policy_selected_using":"2023-10-01 through 2023-10-31",
        "prospective_hist_check":"2023-11-01 through 2023-12-31",
        "no_2024plus_results_or_odds_used":True,
        "won_real_money":False,
        "proves_forward_profit":False,
        "limitations":[
            "All prices are final odds; this is a retrospective simulation.",
            "2023 Nov-Dec has already been used by earlier LINE model selection, so it is not a pristine holdout.",
            "Optimization maximizes conditional likelihood, not backfitted realized ROI; October is used only to choose a pre-listed entry margin.",
            "Prior market-predicted probability is the normalized inverse of odds; residual weights learn market errors from historical outcomes.",
            "No fixed ticket count and no predetermined formation.",
            "A profitable 2023 test cannot establish repeatable profitability.",
        ],
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MARKET_ERROR_LEARNING_DONE",json.dumps({
        "counts":audit["record_counts"],
        "selected_margin":winner["margin"] if winner else None,
        "october_return":winner["october_metrics"]["return_percent"] if winner else None,
        "historical_return":hist["november_december_metrics"]["return_percent"] if hist else None,
        "nov_dec_profitable":hist["both_months_profitable"] if hist else False
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

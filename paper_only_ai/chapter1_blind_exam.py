"""One-way historical blind-exam of frozen accuracy-vs-calibration champions.

The evaluation is an independent program; its numerical outcome is never read
by the candidate generator. 2024 Q1/Q2 remain excluded because already seen.
Each 2025 quarter is scored AT MOST ONCE, in chronological order, and never
revisited to retune the same models. All purchases are simulated at final odds.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import joblib
import numpy as np

import backtest_line_rf_2024_q1 as bt
import backtest_roi_2024_q1 as market

ROOT = Path("docs/paper-only-ai-2023/autonomous_chapter1")
STATE = ROOT / "state.json"
BASE_YEAR = 2025
MARKETS = ("3連単", "3連複")
MIN_RACES = 100
EXAM_EVERY_GENERATIONS = 6
MAX_EXAMS = 4


class StageAdapter:
    def __init__(self, bundle):
        self.model = bundle["estimator"]
        self.columns = np.asarray(bundle["columns"], dtype=int)
        self.t = float(bundle["temperature"])
        if not 0.5 <= self.t <= 2.5:
            raise ValueError("Unapproved temperature artifact")

    def predict_proba(self, raw_features):
        X = np.asarray(raw_features, dtype=np.float32)
        scores = self.model.predict_proba(X[:, self.columns])[:, 1]
        scores = np.power(np.clip(scores, 1e-12, 1.0), 1.0 / self.t)
        return np.column_stack((1.0 - scores, scores))


def load_frozen_champion(candidate, working_dir):
    cid = candidate["id"]
    if not re.fullmatch(r"g\d{2}-c\d{2}", cid):
        raise RuntimeError("Model ID not in sealed generation format")
    generation = int(cid[1:3])
    tag = f"chapter1-autonomous-generation-{generation:02d}"
    where = Path(working_dir) / cid
    where.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "release", "download", tag,
                    "--pattern", cid + ".tar.gz", "--dir", str(where)],
                   check=True, timeout=120)
    archive = where / (cid + ".tar.gz")
    if not archive.is_file():
        raise RuntimeError("Missing champion Release payload")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    unpack = where / "unpacked"
    unpack.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        files = tar.getmembers()
        allowed = {"manifest.json", "stage_1.joblib", "stage_2.joblib", "stage_3.joblib"}
        if {f.name for f in files} != allowed or any(not f.isfile() for f in files):
            raise RuntimeError("Unexpected content in champion Release")
        tar.extractall(unpack, filter="data")
    m = json.loads((unpack / "manifest.json").read_text("utf-8"))
    if m["candidate_id"] != cid or m["trained_through"] != "2023-08-31":
        raise RuntimeError("Frozen championship artifact does not match ledger")
    if m["config"] != candidate["config"]:
        raise RuntimeError("Champion configuration mismatch")
    return ({i: StageAdapter(joblib.load(unpack / f"stage_{i}.joblib"))
             for i in (1, 2, 3)}, digest)


def quarter_sources(year, quarter):
    intervals = {
        1: ("01_02", "03_04"),
        2: ("03_04", "05_06"),
        3: ("07_08", "09_10"),
        4: ("09_10", "11_12"),
    }[quarter]
    return ([f"data/a_class_12/{year}_{it}/entries.csv" for it in intervals]
            + [f"data/a3_challenge/{year}_{it}/entries.csv" for it in intervals]
            + [f"data/{year}/s_class_f1_all_parts/{year}_q{quarter}/entries.csv"])


def score_one_quarter(year, quarter, models, window=None):
    start = f"{year}-{(quarter - 1) * 3 + 1:02d}-01"
    end_month = quarter * 3
    end_day = 31 if end_month in (3, 12) else 30
    end = f"{year}-{end_month:02d}-{end_day:02d}"
    if window is not None:
        start, end = window
        if not (f"{year}-{(quarter-1)*3+1:02d}-01" <= start <= end <= f"{year}-{quarter*3:02d}-31"):
            raise ValueError("Profitability preflight window must remain inside fixed quarter")
    bt.LABELS = f"data/result_universe_v2/combined/results_{year}.jsonl.gz"
    bt.START, bt.END = start, end
    bt.sources = lambda: quarter_sources(year, quarter)
    market.ODDS = f"data/odds_only/odds_{year}.jsonl.gz"
    cards, card_audit = bt.load_2024_line_cards()
    odds, _ = market.collect_2024_books()
    shared = sorted(set(cards) & set(odds), key=lambda rid: (cards[rid]["date"], rid))
    books = {}
    valid = []
    excluded = {}
    for rid in shared:
        try:
            books[rid] = market.decode_odds(odds[rid], rid, set(cards[rid]["cars"]))
        except ValueError as e:
            key = str(e)
            excluded[key] = excluded.get(key, 0) + 1
            continue
        valid.append((rid, cards[rid]))
    if len(valid) < MIN_RACES:
        raise RuntimeError(f"Insufficient aligned race/odds pairs: {year} Q{quarter}: {len(valid)}")
    ledger = {r: {kind: bt.Ledger() for kind in MARKETS} for r in models}
    for off in range(0, len(valid), 20):
        chunk = valid[off:off + 20]
        probabilities = {role: bt.infer_new_chunk(chunk, branch)
                         for role, branch in models.items()}
        for rid, race in chunk:
            book = books[rid]
            actual = bt.actual_tickets(race)
            for role in models:
                pred = probabilities[role][rid]
                bt.safe_records(pred, book)
                for kind in MARKETS:
                    choices = {ticket: (kind, p, book[kind][ticket])
                               for ticket, p in pred[kind].items()
                               if p * book[kind][ticket] > 1.0}
                    ledger[role][kind].update(
                        choices, actual[kind], book[kind], pred[kind])
        print("HISTORICAL_SCORING_PROGRESS", year, quarter, off + len(chunk), len(valid), flush=True)
    totals = {}
    for role, markets in ledger.items():
        s = {k: l.summary() for k, l in markets.items()}
        invest = sum(markets[k].invest for k in MARKETS)
        pay = sum(markets[k].paid for k in MARKETS)
        s["combined"] = {
            "investment_yen": round(invest, 2),
            "payout_yen": round(pay, 2),
            "net_profit_yen": round(pay - invest, 2),
            "return_rate_percent": (100 * pay / invest if invest else None),
            "eligible_races": len(valid),
            "purchased_tickets": sum(markets[k].tickets for k in MARKETS),
        }
        totals[role] = s
    return {
        "window": [start, end], "joined_before_odds_validation": len(shared),
        "qualified_races": len(valid),
        "eligible_cards_without_complete_odds": len(cards) - len(shared),
        "price_exclusions": excluded,
        "card_audit": card_audit,
        "odds_are_final_not_preclose": True, "payout_simulation_only": True,
        "flat_stake_yen_per_positive_EV_ticket": 100,
        "by_role": totals,
    }


def past_profit_gate(scored):
    """Predeclared historic-return filter, NOT proof of repeatable edge.

    The current model search must not consume these historical gate results.
    2023 Nov-Dec was studied in the earlier LINE-model family, so it is an
    honest development diagnostic, never a pristine academic holdout.
    """
    combined = scored["by_role"]["calibration"]["combined"]
    roi = combined.get("return_rate_percent")
    return (scored["qualified_races"] >= 200
            and combined.get("investment_yen", 0) > 0
            and roi is not None and np.isfinite(roi) and roi > 100)


def historic_preflight(state, champions):
    """Check a champion on completed 2023 races before burning 2025 tests.

    Each champion is checked only once; negative outcomes remain visible in
    the ledger and cannot be overwritten to pretend success.
    """
    candidate_id = champions["calibration"]["id"]
    reports = state.setdefault("historical_roi_checks", {})
    if candidate_id in reports:
        return bool(reports[candidate_id]["passed"])
    with tempfile.TemporaryDirectory(prefix="chapter1_past_") as td:
        models = {}
        digests = {}
        seen = {}
        for role in ("accuracy", "calibration"):
            cid = champions[role]["id"]
            if cid not in seen:
                seen[cid] = load_frozen_champion(champions[role], td)
            models[role], digests[role] = seen[cid]
        # The original 2023 full-year model is deliberately EXCLUDED here:
        # Nov-Dec 2023 is inside its training sample.
        scored = score_one_quarter(
            2023, 4, models, window=("2023-11-01", "2023-12-31"))
    passed = past_profit_gate(scored)
    report = {
        "evaluation_type": "development_historical_profit_gate",
        "not_a_new_unknown_test": True,
        "past_data_previously_used_for_model_family_study": True,
        "candidate_model_trained_through": "2023-08-31",
        "candidate_calibration_fit_only": "2023-09",
        "candidate_selected_with_accuracy_or_calibration_only": "2023-10",
        "result_period": ["2023-11-01", "2023-12-31"],
        "fixed_rule": "Every 3連単 and 3連複 ticket p*final_odds > 1, JPY100 each",
        "passes_only_if_calibration_champion_combined_roi_above_100_percent": True,
        "passes_if_at_least_200_matched_races": True,
        "passed": bool(passed),
        "champions": {role: c["id"] for role, c in champions.items()},
        "model_archive_sha256": digests,
        "metrics": scored,
        "not_used_by_candidate_generator": True,
        "future_year_exam_allowed": bool(passed),
    }
    path = ROOT / ("historical_gate_" + candidate_id + ".json")
    if path.exists():
        raise RuntimeError("Historic gate output already exists without ledger; unsafe overwrite")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    reports[candidate_id] = {
        "passed": bool(passed), "report": str(path),
        "qualified_races": scored["qualified_races"],
        "calibration_combined_return_percent": scored["by_role"]["calibration"]["combined"]["return_rate_percent"],
        "frozen_champions": report["champions"],
    }
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    print("HISTORICAL_GATE_COMPLETE", json.dumps({
        "candidate": candidate_id, "passed": passed,
        "roi": reports[candidate_id]["calibration_combined_return_percent"],
        "races": scored["qualified_races"],
    }), flush=True)
    return bool(passed)


def readiness(state):
    if int(state["generation"]) != len(state["attempts"]):
        raise RuntimeError("Tampered candidate generation ledger")
    prior = state.get("external_exams", [])
    if len(prior) >= MAX_EXAMS:
        return None
    if state["generation"] < EXAM_EVERY_GENERATIONS * (len(prior) + 1):
        return None
    champions = state["champions"]
    if not champions.get("accuracy") or not champions.get("calibration"):
        return None
    if champions["accuracy"]["id"] == champions["calibration"]["id"]:
        print("BLIND_EXAM_WAITING_FOR_DISTINCT_MODEL_SELECTION_BRANCHES", flush=True)
        return None
    return len(prior) + 1


def main():
    if not STATE.exists():
        print("BLIND_EXAM_WAITING_FOR_GENERATION_STATE")
        return
    state = json.loads(STATE.read_text("utf-8"))
    # A completed race profitability diagnostic must come BEFORE any 2025
    # evaluation. This is a fixed policy gate, not a search signal.
    if state["generation"] < EXAM_EVERY_GENERATIONS:
        print("HISTORICAL_GATE_WAITING_FOR_INITIAL_RESEARCH", flush=True)
        return
    if len(state.get("external_exams", [])) >= MAX_EXAMS:
        print("ALL_2025_QUARTER_EXAMS_USED", flush=True)
        return
    champions = state["champions"]
    if not all(champions.get(role) for role in ("accuracy", "calibration")):
        print("HISTORICAL_GATE_WAITING_FOR_TWO_CHAMPIONS", flush=True)
        return
    if not historic_preflight(state, champions):
        print("HISTORICAL_PROFIT_NEGATIVE__DO_NOT_EVALUATE_2025", flush=True)
        return
    q = readiness(state)
    if q is None:
        print("BLIND_EXAM_NOT_YET_DUE", flush=True)
        return
    fixed_ids = {role: candidate["id"] for role, candidate in champions.items()}
    with tempfile.TemporaryDirectory(prefix="chapter1_exam_") as td:
        models, checksums = {}, {}
        for role in ("accuracy", "calibration"):
            models[role], checksums[role] = load_frozen_champion(
                champions[role], td)
        original, _, original_sha = bt.load_line_models()
        models["inherited_full_year_2023_line_rf"] = original
        checksums["inherited_full_year_2023_line_rf"] = original_sha
        scored = score_one_quarter(BASE_YEAR, q, models)
    a = scored["by_role"]["accuracy"]["combined"]["return_rate_percent"]
    c = scored["by_role"]["calibration"]["combined"]["return_rate_percent"]
    # Do not call the entire paper replicated: stagewise selection is a proxy.
    supports_hypothesis_on_this_period = (
        a is not None and c is not None and c > a and c > 100
    )
    report = {
        "status": "FROZEN_EXTERNAL_EVALUATION_COMPLETED",
        "historical_evaluation_only": True,
        "quarter": f"{BASE_YEAR}_q{q}", "exam_index": q,
        "frozen_champions": fixed_ids, "release_sha256": checksums,
        "selection_based_on_2025_results": False,
        "scores": scored,
        "calibration_selected_model_higher_roi": c is not None and a is not None and c > a,
        "calibration_selected_model_profitable": c is not None and c > 100,
        "hypothesis_supported_in_this_one_period": supports_hypothesis_on_this_period,
        "full_paper_compliant_certificate": False,
        "do_not_tune_on_this_exam": True,
    }
    out_file = ROOT / f"blind_exam_{BASE_YEAR}_q{q}.json"
    if out_file.exists():
        raise RuntimeError("Blind exam was already performed; no overwriting")
    out_file.write_text(json.dumps(
        report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state.setdefault("external_exams", []).append({
        "quarter": f"{BASE_YEAR}_q{q}",
        "frozen_champions": fixed_ids,
        "report": str(out_file),
        "completed": True,
        "do_not_use_to_retrain_same_period": True,
    })
    state["locked_external_test_executed"] = True
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    print("BLIND_EXAM_SCORED_ONCE", json.dumps({
        "quarter": f"{BASE_YEAR}_q{q}", "accuracy_return": a,
        "calibration_return": c,
        "supports_research_hypothesis": supports_hypothesis_on_this_period,
    }), flush=True)


if __name__ == "__main__":
    main()

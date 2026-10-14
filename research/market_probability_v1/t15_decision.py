"""T-15 decision gate: consumes observed live snapshots, never final odds.

Outputs immutable PAPER decisions by default. No API-based stake execution.
A snapshot must precede the cutoff and include source freshness evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from train import FEATURES, lineup, normalized, rider_stats

JST = ZoneInfo("Asia/Tokyo")
FORBIDDEN = {"results", "result", "payouts", "payout", "finish_position", "winning_method",
             "final_odds", "actual_finish", "winner", "winning_order"}
MAX_AGE_SECONDS = 300
MODEL_DEFAULT = Path(__file__).resolve().parents[2] / "data/models/market_probability_v1_s_f1_2025.json"


def timestamp(raw):
    if not isinstance(raw, str):
        raise ValueError("must provide a timezone-aware timestamp")
    d = datetime.fromisoformat(raw)
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError("timestamps must include a timezone offset")
    return d.astimezone(timezone.utc)


def forbidden_fields(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if k in FORBIDDEN or str(k).lower().startswith("post_race"):
                raise ValueError(f"outcome contamination: {k}")
            forbidden_fields(v)
    elif isinstance(data, list):
        for x in data:
            forbidden_fields(x)


def checked_live(snapshot, age_seconds=MAX_AGE_SECONDS):
    forbidden_fields(snapshot)
    if snapshot.get("odds_phase") != "live":
        raise ValueError("REFUSED: only odds_phase=live is accepted, not final odds")
    race = snapshot["race"]
    deadline = timestamp(race["deadline_jst"])
    cutoff = deadline - timedelta(minutes=15)
    captured = timestamp(snapshot["captured_at_jst"])
    market_times = snapshot.get("market_as_of_jst")
    if not isinstance(market_times, dict) or set(market_times) != {"3連単", "3連複"}:
        raise ValueError("separate market_as_of_jst timestamps are required for both markets")
    for market, raw in market_times.items():
        asof = timestamp(raw)
        if asof > captured + timedelta(seconds=1):
            raise ValueError(f"{market}: market update timestamp is in the future")
        if asof > cutoff:
            raise ValueError(f"{market}: updated after the T-15 decision cutoff")
        if (cutoff - asof).total_seconds() > age_seconds:
            raise ValueError(f"{market}: odds are stale at T-15")
    if captured > cutoff:
        raise ValueError("captured after the T-15 decision cutoff")
    if (cutoff - captured).total_seconds() > age_seconds:
        raise ValueError("captured too early for a T-15 decision; stale quote")
    if not snapshot.get("source_url", "").startswith("https://"):
        raise ValueError("live odds source URL is required")
    entries = snapshot.get("entries", [])
    cars = sorted(int(e["car_no"]) for e in entries)
    if len(cars) != len(set(cars)) or len(cars) < 5 or len(cars) > 7:
        raise ValueError("require 5–7 unique riders")
    odds = snapshot.get("odds", {})
    orders = list(itertools.permutations(cars, 3))
    sets = list(itertools.combinations(cars, 3))
    if set(odds) != {"3連単", "3連複"}:
        raise ValueError("full trifecta and trio live markets required")
    if set(odds["3連単"]) != {"-".join(map(str, o)) for o in orders}:
        raise ValueError("incomplete trifecta market")
    if set(odds["3連複"]) != {"=".join(map(str, s)) for s in sets}:
        raise ValueError("incomplete trio market")
    for pool in odds.values():
        for key, value in pool.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 1:
                raise ValueError(f"invalid live price {key}")
    # Optional enforcement of a live-captured race card timestamp.
    card_time = race.get("card_as_of_jst")
    if card_time is None:
        raise ValueError("race.card_as_of_jst required to verify the card predates the cutoff")
    if timestamp(card_time) > cutoff:
        raise ValueError("race card was captured after the decision cutoff")
    return cutoff, cars, orders, sets


def calculate(snapshot, model, min_edge=0.15, odds_haircut=0.15, max_candidates=4):
    cutoff, cars, orders, sets = checked_live(snapshot)
    if model.get("features") != list(FEATURES) or len(model.get("coefficients", [])) != len(FEATURES):
        raise ValueError("model feature schema mismatch")
    if model.get("status") != "ARCHIVAL_FINAL_ODDS_MODEL_RESEARCH_NOT_EXECUTABLE":
        raise ValueError("model version not recognized: cannot assert T-15 calibration")
    coeff = np.asarray(model["coefficients"], dtype=float)
    entries = {int(e["car_no"]): e for e in snapshot["entries"]}
    stats = rider_stats(entries, cars)
    race = snapshot["race"]
    lines = lineup(race.get("predicted_line_formation", ""), set(cars))
    oo = {o: float(snapshot["odds"]["3連単"]["-".join(map(str, o))]) for o in orders}
    tt = {s: float(snapshot["odds"]["3連複"]["=".join(map(str, s))]) for s in sets}
    qt = normalized(tt)
    qo = normalized(oo)
    set_prob = {s: 0.0 for s in sets}
    for order, v in qo.items():
        set_prob[tuple(sorted(order))] += v
    x = np.zeros((len(orders), len(FEATURES)), dtype=float)
    lq = np.array([math.log(qo[o]) for o in orders], dtype=float)
    for row, order in enumerate(orders):
        a, b, c = order
        x[row, 0] = lq[row] / 5.0
        group = tuple(sorted(order))
        x[row, 1] = np.clip(math.log(set_prob[group] / qt[group]), -2, 2)
        for place, car in enumerate(order):
            start = 2 + place * len(stats[car])
            x[row, start:start + len(stats[car])] = stats[car]
        la, lb, lc = (lines.get(car, (-1, -1, 0)) for car in order)
        ab = la[0] >= 0 and la[0] == lb[0]
        bc = lb[0] >= 0 and lb[0] == lc[0]
        ac = la[0] >= 0 and la[0] == lc[0]
        x[row, -10:] = (ab, bc, ac, ab and bc,
                        la[1] == 0, lb[1] == 0, lc[1] == 0,
                        la[2]/4, lb[2]/4, lc[2]/4)
    logits = lq + x @ coeff
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if not np.isclose(probs.sum(), 1.0):
        raise AssertionError("probability mass is not 1")
    # Explicit odds-movement sensitivity; a haircut is an assumption,
    # NOT a learned or guaranteed future payout.
    candidates = []
    for order, odds, p in zip(orders, (oo[o] for o in orders), probs):
        ev = float(p * odds)
        conservative = ev * (1 - odds_haircut)
        if conservative < 1 + min_edge:
            continue
        candidates.append({
            "combination": "-".join(map(str, order)),
            "estimated_probability": float(p),
            "t15_display_odds": odds,
            "estimated_gross_ev_t15": ev,
            "assumed_haircut": odds_haircut,
            "sensitivity_ev_after_haircut": conservative,
            "paper_stake_yen": 0,
        })
    candidates.sort(key=lambda a: (-a["sensitivity_ev_after_haircut"], a["combination"]))
    payload = {
        "status": "PAPER_ONLY_UNVALIDATED_T15_MODEL",
        "race_id": race.get("race_id"),
        "track": race.get("track"),
        "decision_cutoff_jst": cutoff.astimezone(JST).isoformat(),
        "odds_as_of_jst": snapshot["market_as_of_jst"],
        "captured_at_jst": snapshot["captured_at_jst"],
        "market_sample_size": len(orders),
        "sum_probabilities": float(probs.sum()),
        "paper_selection_limit": max_candidates,
        "candidate_count_above_sensitivity_threshold": len(candidates),
        "paper_decisions": candidates[:max_candidates],
        "paper_pass": not candidates,
        "real_money_automated_purchases": False,
        "limitations": [
            "Final-odds trained model, never validated on historical T-15 market snapshots.",
            "T-15 quotes are not guaranteed execution odds in pari-mutuel betting.",
            "15 percent haircut is a user-configurable scenario, not a fitted movement distribution.",
        ],
    }
    digest = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["snapshot_sha256"] = hashlib.sha256(digest.encode()).hexdigest()
    return payload


def self_test():
    now = datetime(2026, 10, 14, 18, 0, tzinfo=JST)
    deadline = now + timedelta(minutes=30)
    cutoff = deadline - timedelta(minutes=15)
    cards = [
        {"car_no": i, "score": 100 + i/2, "win_rate": 20, "top3_rate": 45,
         "style": "逃" if i % 2 else "両", "age": 31,
         "nige_count": 2, "makuri_count": 3, "sashi_count": 4, "mark_count": 5,
         "b_count": 3} for i in range(1, 8)
    ]
    o3 = {"-".join(map(str, o)): 130.0 for o in itertools.permutations(range(1, 8), 3)}
    ot = {"=".join(map(str, s)): 28.0 for s in itertools.combinations(range(1, 8), 3)}
    stamp = (cutoff - timedelta(seconds=30)).isoformat()
    obj = {
        "race": {"race_id": "synthetic", "track": "TEST", "deadline_jst": deadline.isoformat(),
                 "card_as_of_jst": (cutoff - timedelta(minutes=4)).isoformat(),
                 "predicted_line_formation": "1-2-3/4-5/6-7"},
        "entries": cards, "odds_phase": "live",
        "captured_at_jst": stamp, "market_as_of_jst": {"3連単": stamp, "3連複": stamp},
        "source_url": "https://example.org/odds",
        "odds": {"3連単": o3, "3連複": ot},
    }
    model = {"status": "ARCHIVAL_FINAL_ODDS_MODEL_RESEARCH_NOT_EXECUTABLE",
             "features": list(FEATURES), "coefficients": [0.0] * len(FEATURES)}
    result = calculate(obj, model)
    assert result["paper_pass"] and result["market_sample_size"] == 210
    assert math.isclose(result["sum_probabilities"], 1)
    obj["odds_phase"] = "final"
    try:
        calculate(obj, model)
        raise AssertionError("final odds must be rejected")
    except ValueError:
        pass
    obj["odds_phase"] = "live"
    obj["captured_at_jst"] = (cutoff + timedelta(seconds=1)).isoformat()
    try:
        calculate(obj, model)
        raise AssertionError("late capture must be rejected")
    except ValueError:
        pass
    print("T15 SELF TEST PASS")


def main():
    ap = argparse.ArgumentParser(description="T-15 strictly time-gated 3連単 paper decisions")
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--min-edge", type=float, default=0.15)
    ap.add_argument("--odds-haircut", type=float, default=0.15)
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.snapshot is None or a.out is None:
        ap.error("--snapshot and --out are required")
    if not (0 <= a.odds_haircut < 1 and 0 <= a.min_edge <= 10 and 0 <= a.max_candidates <= 210):
        ap.error("invalid risk/sensitivity parameters")
    obj = json.loads(a.snapshot.read_text(encoding="utf-8"))
    model = json.loads(a.model.read_text(encoding="utf-8"))
    result = calculate(obj, model, a.min_edge, a.odds_haircut, a.max_candidates)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # Immutable: new file only; preserve decisions recorded before the race.
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"status": result["status"], "race": result["race_id"],
                      "candidates": len(result["paper_decisions"]), "file": str(a.out)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

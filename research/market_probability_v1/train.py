"""Market-anchored, race-normalised 3連単 probability correction.

This is an archival final-odds research model, NOT a tradable pre-deadline model.
No race outcome, payout, post-race finishing position or result-comment is a feature.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/audits/market_probability_v1_2023_2026.json"
MODEL = ROOT / "data/models/market_probability_v1_s_f1_2025.json"
RIDER = ("score", "win_rate", "top3_rate", "b_count", "nige_count",
         "makuri_count", "sashi_count", "mark_count", "age", "style_nige", "style_ryo")
REL = ("same_12", "same_23", "same_13", "all_same", "leader_1",
       "leader_2", "leader_3", "line_size_1", "line_size_2", "line_size_3")
FEATURES = ("market_logprob_div5", "trifecta_vs_trio_logratio") + tuple(
    f"{name}_place{place}" for place in (1, 2, 3) for name in RIDER
) + REL
F = len(FEATURES)
MAX_COMBOS = 210  # 7 riders; races with more entrants are excluded deliberately
LAMBDAS = (0.005, 0.02, 0.08, 0.3)


def csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def number(raw, default=0.0):
    try:
        v = float(str(raw).strip().replace("%", "").replace(",", ""))
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def normalized(odds):
    inverse = {k: 1.0 / v for k, v in odds.items() if v > 0 and math.isfinite(v)}
    z = sum(inverse.values())
    return {k: v / z for k, v in inverse.items()} if z > 0 else {}


def sources(root: Path):
    """F1 all-parts prioritized; independent S-yosen files fill uncovered races."""
    dirs = []
    for year in (2023, 2024, 2025):
        dirs.extend(sorted((root / f"data/{year}/s_class_f1_all_parts").glob("*")))
    dirs.append(root / "data/2026_h1/s_class_f1_all")
    for year in (2023, 2024, 2025):
        dirs.append(root / f"data/{year}/s_class_yosen")
    dirs.append(root / "data/2026_h1/s_class_yosen")
    return [d for d in dirs if d.is_dir() and all((d / name).exists() for name in ("races.csv", "entries.csv", "results.csv")) and bool(odds_path(d))]


def odds_path(folder):
    path = folder / "trifecta_final_odds.csv"
    if path.exists():
        return [path]
    return sorted(folder.glob("trifecta_final_odds_part*.csv"))


def outcome_index(folder):
    result = defaultdict(list)
    for r in csv_rows(folder / "results.csv"):
        try:
            pos, car = int(r["finish_position"]), int(r["car_no"])
        except (ValueError, KeyError):
            continue
        if pos in (1, 2, 3):
            result[r["race_id"]].append((pos, car))
    return {k: tuple(c for _, c in sorted(v)) for k, v in result.items()
            if len(v) == 3 and {p for p, _ in v} == {1, 2, 3}}


def lineup(text, cars):
    info = {}
    for line_id, part in enumerate(str(text).split("/")):
        members = []
        for token in part.split("-"):
            try:
                num = int(token.strip())
            except ValueError:
                continue
            if num in cars:
                members.append(num)
        for rank, car in enumerate(members):
            info[car] = (line_id, rank, len(members))
    return info


def rider_stats(entries, cars):
    fields = RIDER[:9]
    scales = (5.0, 20.0, 25.0, 6.0, 6.0, 6.0, 6.0, 6.0, 12.0)
    raw = np.array([[number(entries[c].get(f)) for f in fields] for c in cars],
                   dtype=np.float32)
    valid = np.array([[bool(str(entries[c].get(f, "")).strip()) for f in fields]
                      for c in cars], dtype=bool)
    for j in range(raw.shape[1]):
        raw[~valid[:, j], j] = float(np.mean(raw[valid[:, j], j])) if valid[:, j].any() else 0.0
    relative = np.clip((raw - raw.mean(axis=0)) / np.array(scales), -3, 3)
    stats = {}
    for ix, car in enumerate(cars):
        style = entries[car].get("style", "")
        stats[car] = np.r_[relative[ix], float("逃" in style), float("両" in style)]
    return stats


@dataclass
class Record:
    year: int
    date: str
    rid: str
    race_type: str
    x: np.ndarray
    logq: np.ndarray
    odds: np.ndarray
    winner: int
    cross_market: bool


def make_record(rid, odds, races, entries, results, trio):
    race, ent, winner = races.get(rid), entries.get(rid), results.get(rid)
    if not race or not ent or not winner:
        return None
    cars = sorted(ent)
    n = len(cars)
    if n < 5 or n > 7 or len(odds) != n * (n - 1) * (n - 2):
        return None
    if len(set(winner)) != 3 or any(c not in ent for c in winner):
        return None
    ordered = sorted(odds)
    if winner not in odds or len(ordered) > MAX_COMBOS:
        return None
    q = normalized(odds)
    if len(q) != len(odds):
        return None
    triples = defaultdict(float)
    for key, val in q.items():
        triples[tuple(sorted(key))] += val
    tq = normalized(trio.get(rid, {}))
    has_cross = (len(tq) == math.comb(n, 3))
    stats = rider_stats(ent, cars)
    line = lineup(race.get("predicted_line_formation", ""), set(cars))
    x = np.zeros((len(ordered), F), dtype=np.float32)
    lq = np.array([math.log(q[o]) for o in ordered], dtype=np.float32)
    for row, order in enumerate(ordered):
        a, b, c = order
        x[row, 0] = lq[row] / 5.0
        grouping = tuple(sorted(order))
        if has_cross and tq.get(grouping, 0) > 0:
            x[row, 1] = np.clip(math.log(triples[grouping] / tq[grouping]), -2, 2)
        for place, car in enumerate(order):
            start = 2 + place * len(RIDER)
            x[row, start:start + len(RIDER)] = stats[car]
        la, lb, lc = (line.get(car, (-1, -1, 0)) for car in order)
        eq_ab = la[0] >= 0 and la[0] == lb[0]
        eq_bc = lb[0] >= 0 and lb[0] == lc[0]
        eq_ac = la[0] >= 0 and la[0] == lc[0]
        x[row, -10:] = (
            eq_ab, eq_bc, eq_ac, eq_ab and eq_bc,
            la[1] == 0, lb[1] == 0, lc[1] == 0,
            la[2] / 4, lb[2] / 4, lc[2] / 4,
        )
    return Record(int(race["race_date"][:4]), race["race_date"], rid,
                  race.get("race_type", ""), x, lq,
                  np.array([odds[o] for o in ordered], dtype=np.float32),
                  ordered.index(winner), has_cross)


def load_directory(folder, seen, records, counts):
    race = {r["race_id"]: r for r in csv_rows(folder / "races.csv")}
    entries = defaultdict(dict)
    for r in csv_rows(folder / "entries.csv"):
        try:
            c = int(r["car_no"])
        except (KeyError, ValueError):
            continue
        entries[r["race_id"]][c] = r
    results = outcome_index(folder)
    trio = defaultdict(dict)
    if (folder / "trio_final_odds.csv").exists():
        for r in csv_rows(folder / "trio_final_odds.csv"):
            if r.get("odds_status") != "available":
                continue
            try:
                key = tuple(sorted(int(v) for v in r["combination"].split("=")))
                if len(set(key)) != 3:
                    continue
                value = float(r["odds"])
                if value > 0 and math.isfinite(value):
                    trio[r["race_id"]][key] = value
            except (KeyError, ValueError):
                pass
    for path in odds_path(folder):
        prev, odds = None, {}
        def finish():
            if prev is None:
                return
            if prev in seen:
                counts["duplicate_skipped"] += 1
                return
            item = make_record(prev, odds, race, entries, results, trio)
            if item is None:
                counts["incomplete_skipped"] += 1
                return
            seen.add(prev)
            records.append(item)
            counts["loaded"] += 1
        for row in csv_rows(path):
            rid = row["race_id"]
            if prev is not None and rid != prev:
                finish()
                odds = {}
            prev = rid
            if row.get("odds_status") != "available":
                continue
            try:
                combo = tuple(int(x) for x in row["combination"].split("-"))
                value = float(row["odds"])
                if len(combo) == 3 and len(set(combo)) == 3 and value > 0 and math.isfinite(value):
                    odds[combo] = value
            except (KeyError, ValueError):
                continue
        finish()


def get_records(root):
    records, seen = [], set()
    counts = {"loaded": 0, "duplicate_skipped": 0, "incomplete_skipped": 0}
    folders = sources(root)
    if not folders:
        raise RuntimeError("No source directories; use the full checked-out repository")
    for folder in folders:
        print(f"reading {folder.relative_to(root)}", flush=True)
        load_directory(folder, seen, records, counts)
    years = defaultdict(list)
    for r in sorted(records, key=lambda v: (v.date, v.rid)):
        years[r.year].append(r)
    return years, counts, [str(x.relative_to(root)) for x in folders]


def pack(records):
    if not records:
        raise RuntimeError("No complete joined races available")
    n = len(records)
    x = np.zeros((n, MAX_COMBOS, F), dtype=np.float32)
    logq = np.full((n, MAX_COMBOS), -10000.0, dtype=np.float32)
    odds = np.zeros((n, MAX_COMBOS), dtype=np.float32)
    winner = np.zeros(n, dtype=np.int32)
    length = np.zeros(n, dtype=np.int16)
    for i, r in enumerate(records):
        k = len(r.logq)
        x[i, :k] = r.x
        logq[i, :k] = r.logq
        odds[i, :k] = r.odds
        winner[i], length[i] = r.winner, k
    return (x, logq, odds, winner, length)


def join(packs):
    return tuple(np.concatenate([p[i] for p in packs], axis=0) for i in range(5))


def probabilities(pack, theta):
    x, logq, _, _, _ = pack
    logits = logq + np.einsum("nkd,d->nk", x, np.asarray(theta, dtype=np.float32),
                              optimize=True)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def optimize(pack, lam, power_only=False):
    x, logq, _, true, _ = pack
    n, d = len(true), x.shape[2]
    idx = np.arange(n)
    if power_only:
        def objective(u):
            theta = np.zeros(d, dtype=np.float32)
            theta[0] = float(u)
            p = probabilities(pack, theta)
            return float(-np.log(np.maximum(p[idx, true], 1e-30)).mean() + lam * float(u)**2 / 2)
        ans = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded",
                              options={"xatol": 0.002})
        weights = np.zeros(d, dtype=np.float32)
        weights[0] = float(ans.x)
        return weights
    def objective(w):
        w32 = np.asarray(w, dtype=np.float32)
        p = probabilities(pack, w32)
        true_logprob = -np.log(np.maximum(p[idx, true], 1e-30))
        residual = p.copy()
        residual[idx, true] -= 1.0
        grad = np.einsum("nk,nkd->d", residual, x, optimize=True) / n
        return float(np.mean(true_logprob) + lam * np.dot(w, w) / 2), (
            grad.astype(np.float64) + lam * w)
    ans = minimize(objective, np.zeros(d), jac=True, method="L-BFGS-B",
                   bounds=[(-2, 2)] + [(None, None)] * (d - 1),
                   options={"maxiter": 70, "ftol": 1e-9})
    if not np.isfinite(ans.fun):
        raise RuntimeError(f"Optimization failed: {ans.message}")
    return np.asarray(ans.x, dtype=np.float32)


def metrics(pack, theta):
    x, logq, odds, actual, length = pack
    q = np.exp(logq)
    if theta is None:
        p = q
    else:
        p = probabilities(pack, theta)
    n = len(actual)
    idx = np.arange(n)
    winner_probs = np.maximum(p[idx, actual], 1e-30)
    loss_by_race = -np.log(winner_probs)
    top1 = np.argmax(p, axis=1) == actual
    # Multi-class Brier: sum((p_k - y_k)^2)
    brier = np.sum(p * p, axis=1) - 2 * p[idx, actual] + 1
    result = {"races": n, "logloss": float(loss_by_race.mean()),
              "brier": float(brier.mean()), "top1_pct": float(np.mean(top1) * 100)}
    # Descriptive selection only: one max-xEV ticket per race, archived final odds.
    if theta is not None:
        ev = p * odds
        ev[np.arange(MAX_COMBOS)[None, :] >= length[:, None]] = -1
        pick = np.argmax(ev, axis=1)
        for t in (1.05, 1.10, 1.20):
            chosen = ev[idx, pick] >= t
            num = int(chosen.sum())
            hit = (pick == actual) & chosen
            returned = float(odds[idx[hit], pick[hit]].sum())
            result[f"max_ev_{t:.2f}"] = {
                "races_bet": num, "wins": int(hit.sum()),
                "roi_pct_final_odds_only": returned / num * 100 if num else None,
            }
    return result, loss_by_race


def compare(pack, market, power, model):
    base, base_loss = metrics(pack, None)
    calibr, calibr_loss = metrics(pack, power)
    corrected, corrected_loss = metrics(pack, model)
    def confidence(other):
        diff = other - corrected_loss
        se = float(np.std(diff, ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else None
        return {"mean_logloss_gain": float(diff.mean()),
                "standard_error": se,
                "approx_95pct_ci": [float(diff.mean() - 1.96 * se),
                                     float(diff.mean() + 1.96 * se)] if se else None}
    return {"market": base, "power_only": calibr, "corrected": corrected,
            "corrected_vs_market": confidence(base_loss),
            "corrected_vs_power_only": confidence(calibr_loss),
            "beats_both_by_logloss": corrected["logloss"] < min(base["logloss"], calibr["logloss"])}


def train(root: Path, out: Path, model_path: Path):
    years, counts, folders = get_records(root)
    for yr in (2023, 2024, 2025, 2026):
        if not years[yr]:
            raise RuntimeError(f"No eligible races in {yr}")
    data = {y: pack(years[y]) for y in (2023, 2024, 2025, 2026)}
    del years
    tuning = []
    for lam in LAMBDAS:
        weights = optimize(data[2023], lam)
        val, _ = metrics(data[2024], weights)
        tuning.append({"lambda": lam, "2024_validation_logloss": val["logloss"]})
        print("tune", tuning[-1], flush=True)
    chosen = min(tuning, key=lambda r: r["2024_validation_logloss"])["lambda"]
    m24 = optimize(data[2023], chosen)
    p24 = optimize(data[2023], chosen, power_only=True)
    validate24 = compare(data[2024], None, p24, m24)
    train_23_24 = join([data[2023], data[2024]])
    m25 = optimize(train_23_24, chosen)
    p25 = optimize(train_23_24, chosen, power_only=True)
    validate25 = compare(data[2025], None, p25, m25)
    del train_23_24
    train_23_25 = join([data[2023], data[2024], data[2025]])
    m26 = optimize(train_23_25, chosen)
    p26 = optimize(train_23_25, chosen, power_only=True)
    validate26 = compare(data[2026], None, p26, m26)
    report = {
        "status": "ARCHIVAL_FINAL_ODDS_MODEL_RESEARCH_NOT_EXECUTABLE",
        "scope": "S-class races with full trifecta markets and archived race cards",
        "odds_phase": "final",
        "source_folders": folders, "join_counts": counts,
        "valid_races_by_year": {str(y): int(len(data[y][3])) for y in data},
        "training": {"2023": "fit and lambda selection via 2024 validation",
                     "2024": "validation used to select regularization",
                     "2025": "out-of-time evaluation, trained 2023-2024",
                     "2026_h1": "out-of-time evaluation, trained 2023-2025"},
        "lambda_tuning_2024": tuning, "selected_lambda": chosen,
        "evaluations": {"2024_tuning_not_independent": validate24,
                        "2025_out_of_time": validate25,
                        "2026_h1_out_of_time_not_untouched": validate26},
        "limitations": [
            "Final odds are only known after betting closes: EV figures are not executable backtests.",
            "Archived race cards were collected retrospectively; source-time field fidelity is not independently verified.",
            "2024 is model-selection data; 2025/2026 analyses of this project have been seen previously.",
            "Payout impact, bet-size price impact, real-time betting odds and uncertainty in odds movement are missing.",
            "A-grade, G1/G2/G3 outcomes without full trifecta odds cannot be used for this market-anchored fit.",
            "A positive logloss gain does not establish profitable live betting.",
        ],
        "model": "p(order) = normalized_inverse_final_trifecta_odds(order) * exp(X_order * theta) / race_normalizer",
        "features": list(FEATURES),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_path.write_text(json.dumps({
        "status": report["status"], "train_through": "2025-12-31",
        "features": list(FEATURES), "lambda": chosen,
        "coefficients": [float(v) for v in m26],
        "baseline": "within-race normalized inverse trifecta final odds",
        "caution": report["limitations"][0]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def self_test():
    # Correct probability normalization and conditional-multinomial fitting.
    rng = np.random.default_rng(13)
    x = rng.normal(0, 0.3, size=(120, MAX_COMBOS, F)).astype(np.float32)
    x[:, 4:, :] = 0
    logq = np.full((120, MAX_COMBOS), -10000, dtype=np.float32)
    logq[:, :4] = -math.log(4)
    x[:, :4, 0] = logq[:, :4] / 5
    winner = np.array([0, 1, 2, 3] * 30)
    packed = (x, logq, np.ones((120, MAX_COMBOS), np.float32),
              winner.astype(np.int32), np.full(120, 4, np.int16))
    p = probabilities(packed, np.zeros(F))
    assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
    assert np.allclose(p[:, :4], 0.25, atol=1e-6)
    weights = optimize(packed, 0.3)
    assert np.all(np.isfinite(weights))
    base, _ = metrics(packed, None)
    fit, _ = metrics(packed, weights)
    assert math.isfinite(fit["logloss"]) and math.isfinite(base["logloss"])
    assert abs(base["logloss"] - math.log(4)) < 1e-5
    print("SELF TEST PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        report = train(args.root, args.out, args.model)
        print(json.dumps({"races": report["valid_races_by_year"],
                          "2025": report["evaluations"]["2025_out_of_time"],
                          "2026": report["evaluations"]["2026_h1_out_of_time_not_untouched"]},
                          ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

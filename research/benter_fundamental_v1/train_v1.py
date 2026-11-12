#!/usr/bin/env python3
"""Independent Benter-style fundamental win model; NO odds/features from result pages.

Exploratory retrospective evaluation only. No historical entry snapshot has been
certified as the exact information published before each race.
"""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

FEATURES = (
    "score", "age", "term", "s_count", "b_count", "nige_count",
    "makuri_count", "sashi_count", "mark_count", "first_count",
    "second_count", "third_count", "outside_count", "win_rate",
    "top2_rate", "top3_rate", "gear",
)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "benter_fundamental_v1" / "output"


def paths():
    result = []
    for year in (2023, 2024, 2025):
        for q in range(1, 5):
            d = ROOT / "data" / str(year) / "s_class_f1_all_parts" / f"{year}_q{q}"
            if (d / "entries.csv").exists() and (d / "results.csv").exists():
                result.append(d)
    d = ROOT / "data/2026_h1/s_class_f1_all"
    if (d / "entries.csv").exists() and (d / "results.csv").exists():
        result.append(d)
    return result


def as_number(value):
    value = str(value or "").replace(",", "").replace("%", "").strip()
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except (ValueError, TypeError):
        return np.nan


def read_data():
    audits = Counter()
    races = []
    unique = set()
    per_dir = {}
    for directory in paths():
        outcomes = defaultdict(list)
        with (directory / "results.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                outcomes[row["race_id"]].append(row)
        entrants = defaultdict(list)
        with (directory / "entries.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                entrants[row["race_id"]].append(row)
        good = 0
        for rid, es in entrants.items():
            audits["races_discovered"] += 1
            if not rid or rid in unique:
                audits["duplicate_or_missing_id"] += 1
                continue
            unique.add(rid)
            if any("ガールズ" in e.get("race_type", "") or "L級" in e.get("race_type", "") for e in es):
                audits["girls_excluded"] += 1
                continue
            rs = outcomes.get(rid, [])
            if not rs:
                audits["missing_result"] += 1
                continue
            try:
                es.sort(key=lambda e: int(e["car_no"]))
                ec = [int(e["car_no"]) for e in es]
                rc = [int(r["car_no"]) for r in rs]
            except (ValueError, KeyError):
                audits["bad_car_number"] += 1
                continue
            if not 5 <= len(ec) <= 9 or len(set(ec)) != len(ec) or len(set(rc)) != len(rc) or set(ec) != set(rc):
                audits["incomplete_or_irregular_race"] += 1
                continue
            win = [int(r["car_no"]) for r in rs if str(r.get("finish_position", "")).strip() == "1"]
            if len(win) != 1:
                audits["missing_or_ambiguous_winner"] += 1
                continue
            dates = {e.get("race_date", "") for e in es}
            if len(dates) != 1:
                audits["date_mismatch"] += 1
                continue
            race_date = next(iter(dates))
            if not (len(race_date) == 10 and race_date[4] == "-" and race_date[7] == "-"):
                audits["bad_date"] += 1
                continue
            x = np.array([[as_number(e.get(k)) for k in FEATURES] for e in es], dtype=float)
            if not np.isfinite(x).any():
                audits["all_features_missing"] += 1
                continue
            captures = [e.get("captured_at_utc", "") for e in es]
            retro = any(not ts or ts[:10] >= race_date for ts in captures)
            races.append({"id": rid, "date": race_date, "cars": ec, "winner": ec.index(win[0]),
                          "x": x, "retro": retro})
            audits["retrospective_snapshot_races" if retro else "apparently_pre_race_snapshot_races"] += 1
            good += 1
        per_dir[str(directory.relative_to(ROOT))] = {
            "races_in_entries": len(entrants), "races_valid": good,
            "rows_in_results": sum(map(len, outcomes.values()))
        }
    races.sort(key=lambda r: (r["date"], r["id"]))
    return races, dict(audits), per_dir


def partitions(races):
    train = [r for r in races if r["date"][:4] in ("2023", "2024")]
    validation = [r for r in races if r["date"][:4] == "2025"]
    test = [r for r in races if r["date"][:4] == "2026"]
    if min(len(train), len(validation), len(test)) < 200:
        raise RuntimeError(f"Insufficient temporal partitions: {len(train)}, {len(validation)}, {len(test)}")
    return train, validation, test


class Transform:
    def __init__(self, train):
        data = np.concatenate([r["x"] for r in train], axis=0)
        self.median = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in data.T])
        filled = np.where(np.isfinite(data), data, self.median)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std[self.std < 1e-6] = 1.0

    def apply(self, data):
        missing = ~np.isfinite(data)
        filled = np.where(missing, self.median, data)
        return np.column_stack(((filled - self.mean) / self.std, missing.astype(float)))

    def export(self):
        return {k: getattr(self, k).tolist() for k in ("median", "mean", "std")}


def matrix(races, scaler):
    n = len(races)
    x = np.zeros((n, 9, 2 * len(FEATURES)), dtype=np.float64)
    mask = np.zeros((n, 9), dtype=bool)
    winners = np.zeros(n, dtype=int)
    for i, r in enumerate(races):
        k = len(r["cars"])
        x[i, :k] = scaler.apply(r["x"])
        mask[i, :k] = True
        winners[i] = r["winner"]
    return x, mask, winners


def probabilities(x, mask, beta):
    z = np.einsum("rnf,f->rn", x, beta, optimize=True)
    z = np.where(mask, z, -1e10)
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def fit(train, scaler, alpha):
    x, mask, winner = matrix(train, scaler)
    hot = np.zeros(mask.shape, dtype=float)
    hot[np.arange(len(winner)), winner] = 1

    def fn(beta):
        p = probabilities(x, mask, beta)
        loss = -np.log(np.clip(p[np.arange(len(winner)), winner], 1e-300, 1)).mean()
        grad = np.einsum("rnf,rn->f", x, p - hot, optimize=True) / len(winner)
        return float(loss + 0.5 * alpha * (beta @ beta)), grad + alpha * beta

    sol = minimize(fn, np.zeros(x.shape[-1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 400, "ftol": 1e-10})
    if not sol.success:
        raise RuntimeError("Optimizer failed: " + str(sol.message))
    return sol.x


def evaluate(races, scaler, beta):
    x, mask, winner = matrix(races, scaler)
    p = probabilities(x, mask, beta)
    winner_p = np.clip(p[np.arange(len(races)), winner], 1e-300, 1)
    logloss = float(-np.log(winner_p).mean())
    baseline_ll = float(np.mean([math.log(len(r["cars"])) for r in races]))
    brier = float(np.mean([np.square(p[i, :len(r["cars"])] - 
                       (np.arange(len(r["cars"])) == r["winner"])).sum()
                       for i, r in enumerate(races)]))
    top1 = float(np.mean(np.argmax(p, axis=1) == winner))
    calibration = []
    prob_all = []
    won_all = []
    for i, r in enumerate(races):
        prob_all.extend(p[i, :len(r["cars"])].tolist())
        won_all.extend((np.arange(len(r["cars"])) == r["winner"]).tolist())
    pp = np.array(prob_all)
    yy = np.array(won_all, dtype=float)
    for lo, hi in zip(np.arange(0, 1, .1), np.arange(.1, 1.01, .1)):
        v = (pp >= lo) & ((pp < hi) if hi < 1 else (pp <= hi))
        if v.any():
            calibration.append({"lower": round(float(lo), 2), "upper": round(float(hi), 2),
                                "count": int(v.sum()), "mean_predicted": float(pp[v].mean()),
                                "actual_frequency": float(yy[v].mean())})
    return {"race_count": len(races), "from": races[0]["date"], "to": races[-1]["date"],
            "log_loss": logloss, "uniform_log_loss": baseline_ll,
            "gain_vs_uniform_log_loss": baseline_ll - logloss,
            "multiclass_brier": brier, "top1_hit_rate": top1, "calibration": calibration}


def save_probabilities(file, races, scaler, beta):
    with file.open("w", encoding="utf-8", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(("race_id","race_date","car_no","fundamental_win_probability","observed_win"))
        for r in races:
            x = scaler.apply(r["x"])
            z = x @ beta
            pr = np.exp(z - logsumexp(z))
            for j, car in enumerate(r["cars"]):
                writer.writerow((r["id"],r["date"],car,format(pr[j],".10f"),int(r["winner"]==j)))


def main():
    races, audit, per_dir = read_data()
    train, val, test = partitions(races)
    base_scaler = Transform(train)
    candidates = []
    for alpha in (0.0001, 0.001, 0.01, 0.1):
        beta = fit(train, base_scaler, alpha)
        scores = evaluate(val, base_scaler, beta)
        candidates.append({"ridge_alpha": alpha, "validation_log_loss": scores["log_loss"]})
    best = min(candidates, key=lambda a: a["validation_log_loss"])
    val_beta = fit(train, base_scaler, best["ridge_alpha"])
    val_metrics = evaluate(val, base_scaler, val_beta)
    fitted = train + val
    final_scaler = Transform(fitted)
    final_beta = fit(fitted, final_scaler, best["ridge_alpha"])
    test_metrics = evaluate(test, final_scaler, final_beta)
    OUT.mkdir(parents=True, exist_ok=True)
    save_probabilities(OUT / "validation_predictions.csv", val, base_scaler, val_beta)
    save_probabilities(OUT / "test_predictions.csv", test, final_scaler, final_beta)
    model = {"type": "multinomial_logit_fundamental_only", "features": FEATURES,
             "ridge_alpha": best["ridge_alpha"], "coefficients": final_beta.tolist(),
             "transform": final_scaler.export(), "market_odds_used": False}
    report = {"status": "EXPLORATORY_RETROSPECTIVE_NOT_PROVEN_AS_OF",
              "scope": "F1 S-class races in available paired entrant/result files only",
              "data_quality_note": "Historic entries were obtained retrospectively; no proof that stored factors equal the original pre-race snapshots.",
              "audit": audit, "by_directory": per_dir,
              "partition_counts": {"train": len(train), "validation": len(val), "test": len(test)},
              "hyperparameter_search": candidates,
              "validation": val_metrics, "test": test_metrics,
              "comparison_with_market": "NOT_YET_PERFORMED"}
    (OUT / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"partition_counts": report["partition_counts"], "validation": {k:v for k,v in val_metrics.items() if k != "calibration"}, "test": {k:v for k,v in test_metrics.items() if k != "calibration"}, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

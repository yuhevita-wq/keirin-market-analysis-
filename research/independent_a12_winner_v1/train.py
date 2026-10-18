"""A1/A2 independent first-place probability model.

Inputs: historical pre-race entry/racecard + published predicted line only.
Label: actual finish_position == 1.  No odds, tips, or result-side features.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

FEATURE_NAMES = [
    "score_relative", "win_rate_relative", "top2_rate_relative",
    "top3_rate_relative", "log_b_relative", "log_s_relative",
    "log_nige_relative", "log_makuri_relative", "log_sashi_relative",
    "log_mark_relative", "age_relative", "a1_relative",
    "style_escape_relative", "style_chase_relative", "style_allround_relative",
    "line_leader_relative", "line_second_relative", "line_third_relative",
    "line_single_relative", "line_size_relative", "partner_max_score_relative",
]
LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1.0]


def number(value: object) -> float:
    s = unicodedata.normalize("NFKC", str(value or "")).strip()
    s = s.replace(",", "").replace("%", "").replace("点", "")
    try:
        v = float(s)
        return v if math.isfinite(v) else float("nan")
    except ValueError:
        return float("nan")


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def line_map(formation: str) -> dict[int, tuple[int, list[int]]]:
    mapping = {}
    for segment in str(formation or "").replace("／", "/").split("/"):
        cars = [int(x) for x in re.findall(r"(?<!\d)[1-9](?!\d)", segment)]
        for pos, car in enumerate(cars):
            if car in mapping:
                return {}
            mapping[car] = (pos, cars)
    return mapping


def make_features(entries: list[dict], formation: str) -> np.ndarray | None:
    entries = sorted(entries, key=lambda row: int(row["car_no"]))
    if len(entries) < 5 or len(entries) > 9:
        return None
    if len({row["car_no"] for row in entries}) != len(entries):
        return None
    if any(unicodedata.normalize("NFKC", row.get("class", "")).strip() not in ("A1", "A2") for row in entries):
        return None
    lm = line_map(formation)
    scores = {int(row["car_no"]): number(row.get("score")) for row in entries}
    if not all(math.isfinite(x) for x in scores.values()):
        return None
    data = []
    for entry in entries:
        car = int(entry["car_no"])
        s = scores[car]
        style = unicodedata.normalize("NFKC", entry.get("style", "") or "")
        def num(key: str, default=0.0):
            value = number(entry.get(key))
            return value if math.isfinite(value) else default
        position, members = lm.get(car, (-1, []))
        others = [scores[c] for c in members if c != car and c in scores]
        partner_max = max(others) if others else s
        data.append([
            s / 10,
            num("win_rate") / 30, num("top2_rate") / 30,
            num("top3_rate") / 30,
            math.log1p(max(0, num("b_count"))) / 2,
            math.log1p(max(0, num("s_count"))) / 2,
            math.log1p(max(0, num("nige_count"))) / 2,
            math.log1p(max(0, num("makuri_count"))) / 2,
            math.log1p(max(0, num("sashi_count"))) / 2,
            math.log1p(max(0, num("mark_count"))) / 2,
            num("age", 35) / 20,
            float(entry.get("class") == "A1"),
            float("逃" in style), float("追" in style), float("両" in style),
            float(position == 0 and len(members) > 1),
            float(position == 1), float(position >= 2),
            float(position == 0 and len(members) == 1),
            len(members) / 3,
            (partner_max - s) / 10,
        ])
    x = np.asarray(data, dtype=float)
    if not np.all(np.isfinite(x)):
        return None
    return x - x.mean(axis=0, keepdims=True)


def load_history(root: Path):
    """Return nonoverlapping, date-sorted races.  Excludes later 2026 races."""
    all_races = {}
    audit = defaultdict(int)
    folders = sorted(p for p in root.iterdir()
                     if p.is_dir() and re.fullmatch(r"202[3456]_\d{2}_\d{2}", p.name))
    if not folders:
        raise ValueError(f"No A1/A2 two-month archives under {root}")
    for folder in folders:
        if not all((folder / fn).exists() for fn in ("entries.csv", "races.csv", "results.csv")):
            audit["missing_archive_files"] += 1
            continue
        meta = {r["race_id"]: r for r in rows(folder / "races.csv")}
        entrants = defaultdict(list)
        for r in rows(folder / "entries.csv"):
            entrants[r["race_id"]].append(r)
        finishes = defaultdict(list)
        for r in rows(folder / "results.csv"):
            finishes[r["race_id"]].append(r)
        for rid, race in meta.items():
            date = race.get("race_date", "")
            if not ("2023-01-01" <= date <= "2026-06-30"):
                continue
            if rid in all_races:
                raise ValueError(f"Duplicate race ID across archives: {rid}")
            ent = entrants.get(rid, [])
            result = finishes.get(rid, [])
            if len(ent) != len(result) or not ent:
                audit["incomplete_results"] += 1
                continue
            cars = sorted(int(e["car_no"]) for e in ent)
            res_cars = sorted(int(e["car_no"]) for e in result)
            if cars != res_cars:
                audit["entrant_result_mismatch"] += 1
                continue
            winners = [int(r["car_no"]) for r in result
                       if unicodedata.normalize("NFKC", r.get("finish_position", "")).strip() in ("1", "1着")]
            if len(winners) != 1:
                audit["no_unique_first_place"] += 1
                continue
            x = make_features(ent, race.get("predicted_line_formation", ""))
            if x is None:
                audit["invalid_racecard"] += 1
                continue
            y = np.array([int(c == winners[0]) for c in cars], dtype=float)
            if int(y.sum()) != 1:
                audit["winner_not_in_entries"] += 1
                continue
            if not race.get("predicted_line_formation"):
                audit["missing_line"] += 1
            all_races[rid] = (date, rid, x, y, cars)
            audit["accepted"] += 1
    return sorted(all_races.values(), key=lambda r: (r[0], r[1])), dict(audit)


def pack(races, indexes=None):
    if not races:
        raise ValueError("Empty race group")
    X = np.concatenate([r[2] if indexes is None else r[2][:, indexes] for r in races])
    y = np.concatenate([r[3] for r in races])
    n = np.asarray([len(r[3]) for r in races])
    start = np.cumsum(np.r_[0, n[:-1]])
    return X, y, start, n


def probabilities(X, start, beta):
    z = X @ beta
    z -= np.repeat(np.maximum.reduceat(z, start), np.diff(np.r_[start, len(z)]))
    p = np.exp(z)
    p /= np.repeat(np.add.reduceat(p, start), np.diff(np.r_[start, len(p)]))
    return p


def fit(races, columns, ridge):
    X, y, starts, _ = pack(races, columns)
    N = len(starts)
    def loss_grad(beta):
        p = probabilities(X, starts, beta)
        loss = float(-(np.log(np.maximum(p[y == 1], 1e-15))).sum() / N
                     + 0.5 * ridge * (beta @ beta))
        grad = X.T @ (p-y) / N + ridge * beta
        return loss, grad
    opt = minimize(loss_grad, np.zeros(X.shape[1]), method="L-BFGS-B",
                   jac=True, options={"maxiter": 250, "ftol": 1e-10})
    if not opt.success and np.linalg.norm(opt.jac) > 1e-4:
        raise RuntimeError(f"Optimization failed: {opt.message}")
    return opt.x


def metrics(races, columns, beta):
    X, y, starts, counts = pack(races, columns)
    p = probabilities(X, starts, beta)
    likelihood = -(np.log(np.maximum(p[y == 1], 1e-15))).mean()
    brier = np.add.reduceat((p-y)**2, starts).mean()
    top1 = 0
    score_top1 = 0
    observed = np.zeros(10)
    predicted = np.zeros(10)
    bucket_count = np.zeros(10)
    for j, start in enumerate(starts):
        end = start + counts[j]
        top1 += y[start + np.argmax(p[start:end])] == 1
        score_top1 += y[start + np.argmax(races[j][2][:, 0])] == 1
        for k in range(start, end):
            b = min(9, int(p[k]*10))
            observed[b] += y[k]
            predicted[b] += p[k]
            bucket_count[b] += 1
    total = int(counts.sum())
    ece = float(sum(abs(observed[i]-predicted[i]) for i in range(10)) / total)
    return {
        "races": len(races), "entrants": total,
        "log_loss": round(float(likelihood), 6),
        "race_brier": round(float(brier), 6),
        "top1_accuracy": round(float(top1/len(races)), 6),
        "score_rank1_accuracy": round(float(score_top1/len(races)), 6),
        "entrant_ece": round(ece, 6),
    }


def select(races_train, races_valid, columns):
    choices = []
    for lam in LAMBDAS:
        beta = fit(races_train, columns, lam)
        val = metrics(races_valid, columns, beta)
        choices.append((val["log_loss"], lam, beta, val))
    return min(choices, key=lambda r: (r[0], r[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/a_class_12")
    ap.add_argument("--output", default="research/independent_a12_winner_v1/report.json")
    args = ap.parse_args()
    races, audit = load_history(Path(args.data_root))
    train = [r for r in races if r[0] < "2025-01-01"]
    valid = [r for r in races if "2025-01-01" <= r[0] < "2026-01-01"]
    test = [r for r in races if "2026-01-01" <= r[0] <= "2026-06-30"]
    if min(len(train), len(valid), len(test)) < 100:
        raise ValueError(f"Insufficient year-separated data: {len(train)}, {len(valid)}, {len(test)}")
    base_cols = [0]
    full_cols = list(range(len(FEATURE_NAMES)))
    base = select(train, valid, base_cols)
    full = select(train, valid, full_cols)
    both = train + valid
    beta_base = fit(both, base_cols, base[1])
    beta_full = fit(both, full_cols, full[1])
    test_base = metrics(test, base_cols, beta_base)
    test_full = metrics(test, full_cols, beta_full)
    report = {
        "name": "A1/A2 independent winner v1 (odds excluded)",
        "status": "research; not approved for EV deployment",
        "train_period": "2023-01-01..2024-12-31",
        "validation_period": "2025-01-01..2025-12-31",
        "test_period": "2026-01-01..2026-06-30 (untouched for tuning)",
        "train_races": len(train), "validation_races": len(valid),
        "audit": audit,
        "selection": {
            "baseline_score_only": {"ridge": base[1], "validation": base[3]},
            "full_model": {"ridge": full[1], "validation": full[3]},
        },
        "test": {"baseline_score_only": test_base, "full_model": test_full},
        "production_fit_end": "2025-12-31",
        "production_coefficients": dict(zip(FEATURE_NAMES, [round(float(v), 10) for v in beta_full])),
        "baseline_coefficient": float(beta_base[0]),
        "policy": "Reject positive performance claim unless test log-loss, top1, and calibration support it. No odds consulted.",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Independent locked-performance audit of the A1/A2 first-place model.

Runs the predeclared 2023-2025 fits, evaluates January-June 2026 and
an extra post-June 2026 temporal holdout without parameter retuning.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from train import (
    FEATURE_NAMES, fit, load_history, make_features, pack,
    probabilities, rows, select,
)


def load_h2(root):
    seen = set()
    selected = []
    audit = defaultdict(int)
    for label in ("2026_07_08", "2026_09_1002"):
        p = root / label
        if not all((p / f).exists() for f in ("races.csv", "entries.csv", "results.csv")):
            audit["missing_folder"] += 1
            continue
        entrants, results = defaultdict(list), defaultdict(list)
        for r in rows(p / "entries.csv"):
            entrants[r["race_id"]].append(r)
        for r in rows(p / "results.csv"):
            results[r["race_id"]].append(r)
        for meta in rows(p / "races.csv"):
            date, rid = meta.get("race_date", ""), meta["race_id"]
            if not ("2026-07-01" <= date <= "2026-10-02"):
                continue
            if rid in seen:
                raise ValueError(f"Duplicated H2 race {rid}")
            seen.add(rid)
            entry = entrants.get(rid, [])
            result = results.get(rid, [])
            if not entry or len(entry) != len(result):
                audit["incomplete_results"] += 1
                continue
            ent_cars = sorted(int(r["car_no"]) for r in entry)
            res_cars = sorted(int(r["car_no"]) for r in result)
            if ent_cars != res_cars:
                audit["car_mismatch"] += 1
                continue
            winners = [int(r["car_no"]) for r in result
                       if unicodedata.normalize("NFKC", r.get("finish_position", "")).strip() in ("1", "1着")]
            if len(winners) != 1:
                audit["invalid_winner"] += 1
                continue
            x = make_features(entry, meta.get("predicted_line_formation", ""))
            if x is None:
                audit["invalid_card"] += 1
                continue
            y = np.array([int(c == winners[0]) for c in ent_cars], dtype=float)
            if int(y.sum()) != 1:
                audit["winner_not_entrant"] += 1
                continue
            selected.append((date, rid, x, y, ent_cars))
            audit["accepted"] += 1
    return sorted(selected, key=lambda r: (r[0], r[1])), dict(audit)


def race_vectors(races, columns, beta):
    X, y, starts, counts = pack(races, columns)
    p = probabilities(X, starts, beta)
    output = []
    for n, race in enumerate(races):
        a = int(starts[n]); b = a + int(counts[n])
        pr = p[a:b]
        actual = int(np.argmax(y[a:b]))
        chosen = int(np.argmax(pr))
        score_chosen = int(np.argmax(race[2][:, 0]))
        output.append({
            "date": race[0], "id": race[1], "cars": len(race[4]),
            "actual": race[4][actual], "chosen": race[4][chosen],
            "p_actual": float(pr[actual]),
            "top1": int(chosen == actual),
            "score_top1": int(score_chosen == actual),
            "neg_log_likelihood": -math.log(max(1e-15, float(pr[actual]))),
            "brier": float(((pr - y[a:b]) ** 2).sum()),
            "probabilities": [float(x) for x in pr], "outcomes": [int(x) for x in y[a:b]],
        })
    return output


def report_metrics(records):
    if not records:
        return {"races": 0}
    entrants = sum(r["cars"] for r in records)
    return {
        "races": len(records),
        "first_place_hits": sum(r["top1"] for r in records),
        "first_place_accuracy": round(np.mean([r["top1"] for r in records]), 6),
        "log_loss": round(np.mean([r["neg_log_likelihood"] for r in records]), 6),
        "race_brier": round(np.mean([r["brier"] for r in records]), 6),
        "entrants": entrants,
    }


def paired_ci(races, base, full, *, seed=20261018, nboot=3000):
    """Day-clustered bootstrap for correlated races on the same date."""
    if not races:
        return {}
    byday = defaultdict(list)
    for i, race in enumerate(races):
        byday[race[0]].append(i)
    groups = list(byday.values())
    a = np.asarray([x["top1"] - y["top1"] for x, y in zip(full, base)], dtype=float)
    b = np.asarray([y["neg_log_likelihood"] - x["neg_log_likelihood"] for x, y in zip(full, base)], dtype=float)
    daily_n = np.array([len(g) for g in groups])
    daily_top = np.array([a[g].sum() for g in groups])
    daily_log = np.array([b[g].sum() for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), (nboot, len(groups)))
    counts = daily_n[idx].sum(axis=1)
    top = daily_top[idx].sum(axis=1) / counts
    logs = daily_log[idx].sum(axis=1) / counts
    return {
        "days": len(groups), "cluster_unit": "race_date",
        "confidence": 0.95, "bootstrap_repetitions": nboot,
        "top1_improvement_pp": round(float(a.mean()*100), 3),
        "top1_improvement_pp_ci95": [round(float(x*100), 3) for x in np.quantile(top, [0.025, 0.975])],
        "logloss_reduction": round(float(b.mean()), 6),
        "logloss_reduction_ci95": [round(float(x), 6) for x in np.quantile(logs, [0.025, 0.975])],
    }


def ece_bins(records):
    result = []
    for i in range(10):
        probs, observed = [], []
        for row in records:
            for p, actual in zip(row["probabilities"], row["outcomes"]):
                if i/10 <= p < (i+1)/10 or (i == 9 and p == 1):
                    probs.append(p); observed.append(actual)
        if probs:
            result.append({
                "range": f"{i/10:.1f}-{(i+1)/10:.1f}",
                "count": len(probs),
                "mean_predicted": round(float(np.mean(probs)), 5),
                "observed_wins": int(sum(observed)),
                "observed_rate": round(float(np.mean(observed)), 5),
            })
    return result


def feature_completeness(root, date_start, date_end):
    required = ("score", "win_rate", "top2_rate", "top3_rate", "age",
                "s_count", "b_count", "nige_count", "makuri_count", "sashi_count", "mark_count")
    total, missing = 0, defaultdict(int)
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not re.fullmatch(r"202[3456]_\d{2}_\d{2}", p.name):
            continue
        if not (p / "entries.csv").exists():
            continue
        for entry in rows(p / "entries.csv"):
            if not (date_start <= entry.get("race_date", "") <= date_end):
                continue
            total += 1
            for f in required:
                raw = entry.get(f, "").replace("%", "").strip()
                if not raw or raw in ("-", "--"):
                    missing[f] += 1
    return {"rows": total, "missing_counts": dict(missing)}


def split_stats(races, records, metas):
    bucket = defaultdict(list)
    for race, item in zip(races, records):
        meta = metas.get(race[1], {})
        bucket[("month", race[0][:7])].append(item)
        bucket[("entry_count", str(len(race[4])))].append(item)
        bucket[("race_type", meta.get("race_type", "(missing)"))].append(item)
        bucket[("meeting_grade", meta.get("meeting_grade", "(missing)"))].append(item)
    result = defaultdict(dict)
    for (kind, label), group in sorted(bucket.items()):
        if len(group) >= 20:
            result[kind][label] = report_metrics(group)
    return dict(result)


def read_meta(root):
    metas = {}
    for folder in sorted(root.iterdir()):
        if not re.fullmatch(r"202[3456]_\d{2}_\d{2}", folder.name):
            continue
        if not (folder / "races.csv").exists():
            continue
        for row in rows(folder / "races.csv"):
            metas[row["race_id"]] = {
                "race_type": row.get("race_type", ""),
                "meeting_grade": row.get("meeting_grade", ""),
            }
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/a_class_12")
    ap.add_argument("--output", default="research/independent_a12_winner_v1/audit.json")
    args = ap.parse_args()
    root = Path(args.root)
    races, train_audit = load_history(root)
    train = [r for r in races if r[0] < "2025-01-01"]
    val = [r for r in races if "2025-01-01" <= r[0] < "2026-01-01"]
    h1 = [r for r in races if "2026-01-01" <= r[0] <= "2026-06-30"]
    h2, h2_audit = load_h2(root)
    assert train and val and h1 and h2, (len(train), len(val), len(h1), len(h2))
    full_cols = list(range(len(FEATURE_NAMES)))
    score_cols = [0]
    no_line_cols = list(range(15))
    selections = {
        "score_only": select(train, val, score_cols),
        "no_line": select(train, val, no_line_cols),
        "full": select(train, val, full_cols),
    }
    coef = {key: fit(train + val, cols, selections[key][1])
            for key, cols in (("score_only", score_cols),
                              ("no_line", no_line_cols), ("full", full_cols))}
    metas = read_meta(root)
    output = {
        "research_stage": "A1/A2 first place locked coefficient audit; odds excluded",
        "as_of": "2026-10-18",
        "training": {"races": len(train), "validation_races": len(val),
                     "ridge_selected_in_2025": {k: float(v[1]) for k, v in selections.items()}},
        "audits": {"h1": train_audit, "h2": h2_audit,
                   "field_completeness": feature_completeness(root, "2023-01-01", "2026-10-02")},
        "holdouts": {},
    }
    for name, part in (("2026_h1_primary", h1), ("2026_h2_extra", h2)):
        predictions = {
            "score_only": race_vectors(part, score_cols, coef["score_only"]),
            "no_line": race_vectors(part, no_line_cols, coef["no_line"]),
            "full": race_vectors(part, full_cols, coef["full"]),
        }
        output["holdouts"][name] = {
            "range": [part[0][0], part[-1][0]],
            "metrics": {key: report_metrics(items) for key, items in predictions.items()},
            "full_vs_score": paired_ci(part, predictions["score_only"], predictions["full"]),
            "full_vs_no_line": paired_ci(part, predictions["no_line"], predictions["full"], seed=20261019),
            "calibration": {
                "score_only": ece_bins(predictions["score_only"]),
                "full": ece_bins(predictions["full"]),
            },
            "full_strata": split_stats(part, predictions["full"], metas),
            "score_strata": split_stats(part, predictions["score_only"], metas),
        }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "training": output["training"],
        "holdouts": {k: {"metrics": v["metrics"],
                         "full_vs_score": v["full_vs_score"],
                         "full_vs_no_line": v["full_vs_no_line"]}
                     for k, v in output["holdouts"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

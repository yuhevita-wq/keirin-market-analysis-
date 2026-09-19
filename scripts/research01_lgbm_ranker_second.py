#!/usr/bin/env python3
from __future__ import annotations

"""
Research 01: current v31 second-place ranking vs LightGBM LGBMRanker.

Scope is intentionally narrow:
- seven-rider races only
- v21 participation is frozen
- row 1 and row 3 are frozen to the existing v37 evaluation log
- only the row-2 ranking engine is changed
- odds, popularity and payout data are never used as features
- train/calibration/test periods are identical to the frozen v31 study

The Ranker uses only the pre-race rider/line features already produced by
v25.race_features(). It does NOT use v21 first-place probabilities or any
predicted first-place score.
"""

import importlib.util
import json
from pathlib import Path
from statistics import mean

import numpy as np
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/keirin_shogi/research01_lgbm_ranker_second"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = "2025-07-01", "2025-10-26"
CAL_START, CAL_END = "2025-10-27", "2025-12-28"
TEST_START, TEST_END = "2025-12-29", "2026-06-28"

V37_LOG = ROOT / "results/keirin_shogi/v37_shrunk_board_third/race_log.json"

FORBIDDEN_FEATURE_TOKENS = (
    "odds",
    "popular",
    "payout",
    "finish",
    "result",
    "actual",
    "target",
)


def load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v31 = load_module("research01_v31", "scripts/keirin_shogi_v31_top2_membership_second.py")
v25 = v31.v25


VARIANTS = [
    {
        "name": "rank_d2a",
        "params": {
            "max_depth": 2,
            "num_leaves": 4,
            "min_child_samples": 30,
            "learning_rate": 0.040,
            "n_estimators": 180,
            "reg_lambda": 1.5,
        },
    },
    {
        "name": "rank_d3a",
        "params": {
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 24,
            "learning_rate": 0.035,
            "n_estimators": 220,
            "reg_lambda": 2.0,
        },
    },
    {
        "name": "rank_d3b",
        "params": {
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 40,
            "learning_rate": 0.025,
            "n_estimators": 280,
            "reg_lambda": 3.0,
        },
    },
    {
        "name": "rank_d4a",
        "params": {
            "max_depth": 4,
            "num_leaves": 12,
            "min_child_samples": 30,
            "learning_rate": 0.030,
            "n_estimators": 240,
            "reg_lambda": 3.0,
        },
    },
]


def participant_rows(ds):
    return [row for row in ds if row[4].get("v21_participate")]


def feature_names_from_dataset(ds):
    for _rid, _dt, base, _order, _vr in ds:
        if not base:
            continue
        first_no = sorted(base)[0]
        names = sorted(base[first_no]["x"])
        bad = [f for f in names if any(tok in f.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
        if bad:
            raise RuntimeError(f"forbidden feature(s) detected: {bad}")
        return names
    raise RuntimeError("no feature rows available")


def build_rank_matrix(ds, feature_names):
    X = []
    y = []
    group = []
    race_order = []

    # Keep every race contiguous. Never shuffle rows across race groups.
    ordered = sorted(ds, key=lambda row: (row[1], row[0]))
    for rid, dt, base, order, _vr in ordered:
        nos = sorted(base)
        if len(nos) != 7:
            raise RuntimeError(f"non-seven-rider race leaked into Research01: {rid} ({len(nos)})")
        actual_second = int(order[1])
        labels = []
        for no in nos:
            x = base[no]["x"]
            X.append([float(x[name]) for name in feature_names])
            label = 1 if int(no) == actual_second else 0
            y.append(label)
            labels.append(label)
        if sum(labels) != 1:
            raise RuntimeError(f"invalid relevance labels for race {rid}: {labels}")
        group.append(7)
        race_order.append((rid, dt))

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), group, race_order


def fit_ranker(ds, feature_names, params):
    X, y, group, _ = build_rank_matrix(ds, feature_names)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        random_state=101,
        n_jobs=-1,
        verbosity=-1,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.0,
        **params,
    )
    model.fit(X, y, group=group)
    return model


def rank_one(model, feature_names, base):
    nos = sorted(base)
    X = np.asarray(
        [[float(base[no]["x"][name]) for name in feature_names] for no in nos],
        dtype=float,
    )
    scores = model.predict(X)
    ranked = sorted(
        [(int(no), float(score)) for no, score in zip(nos, scores)],
        key=lambda item: (-item[1], item[0]),
    )
    return ranked


def evaluate_ranker(ds, model, feature_names):
    rows = []
    for rid, dt, base, order, vr in sorted(ds, key=lambda row: (row[1], row[0])):
        if not vr.get("v21_participate"):
            continue
        ranked = rank_one(model, feature_names, base)
        actual_second = int(order[1])
        rank = next(i + 1 for i, (no, _score) in enumerate(ranked) if no == actual_second)
        rows.append(
            {
                "race_id": rid,
                "race_date": dt,
                "actual_first": int(order[0]),
                "actual_second": actual_second,
                "actual_third": int(order[2]),
                "ranking": [no for no, _score in ranked],
                "scores": [{"no": no, "score": score} for no, score in ranked],
                "actual_second_rank": rank,
            }
        )
    return rows


def ranking_metrics(rows):
    n = len(rows)
    if not n:
        return {}
    ranks = [int(r["actual_second_rank"]) for r in rows]
    return {
        "races": n,
        "top1_second_capture": sum(rank <= 1 for rank in ranks) / n,
        "top2_second_capture": sum(rank <= 2 for rank in ranks) / n,
        "top3_second_capture": sum(rank <= 3 for rank in ranks) / n,
        "actual_second_mean_rank": mean(ranks),
        "mrr": mean(1.0 / rank for rank in ranks),
    }


def load_v37_fixed_board():
    rows = json.loads(V37_LOG.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        rid = str(row["race_id"])
        out[rid] = {
            "race_id": rid,
            "race_date": row["race_date"],
            "actual_first": int(row["actual_first"]),
            "actual_second": int(row["actual_second"]),
            "actual_third": int(row["actual_third"]),
            "first_candidates": [int(x) for x in row["first_candidates"]],
            "third_candidates": [int(x) for x in row["third_candidates"]],
            "current_ranking": [int(x["no"]) for x in row["top2_membership"]],
            "current_variable_second_candidates": [int(x) for x in row["second_candidates"]],
        }
    return out


def enrich_with_fixed_board(ranker_rows, fixed):
    out = []
    for row in ranker_rows:
        rid = str(row["race_id"])
        if rid not in fixed:
            continue
        f = fixed[rid]
        # Results must agree exactly or the comparison is invalid.
        if (
            row["actual_first"] != f["actual_first"]
            or row["actual_second"] != f["actual_second"]
            or row["actual_third"] != f["actual_third"]
        ):
            raise RuntimeError(f"result mismatch for race {rid}")
        out.append(
            {
                **row,
                "first_candidates": f["first_candidates"],
                "third_candidates": f["third_candidates"],
                "current_ranking": f["current_ranking"],
                "current_variable_second_candidates": f["current_variable_second_candidates"],
            }
        )
    return out


def compare_metrics(rows):
    if not rows:
        return {}, {}, {}

    def one(which):
        eval_rows = []
        board = {1: 0, 2: 0, 3: 0}
        for row in rows:
            ranking = row[which]
            actual_second = row["actual_second"]
            rank = ranking.index(actual_second) + 1
            eval_rows.append({"actual_second_rank": rank})
            fixed_ok = (
                row["actual_first"] in row["first_candidates"]
                and row["actual_third"] in row["third_candidates"]
            )
            for k in (1, 2, 3):
                if fixed_ok and actual_second in ranking[:k]:
                    board[k] += 1
        m = ranking_metrics(eval_rows)
        n = len(rows)
        m.update(
            {
                "complete_board_top1": board[1] / n,
                "complete_board_top2": board[2] / n,
                "complete_board_top3": board[3] / n,
            }
        )
        return m

    current = one("current_ranking")
    ranker = one("ranking")
    delta = {}
    for key in current:
        if key == "races":
            continue
        delta[key] = ranker[key] - current[key]
    return current, ranker, delta


def variable_current_reference(rows):
    n = len(rows)
    if not n:
        return {}
    hit = 0
    counts = []
    for row in rows:
        candidates = row["current_variable_second_candidates"]
        counts.append(len(candidates))
        hit += int(row["actual_second"] in candidates)
    return {
        "races": n,
        "capture": hit / n,
        "avg_second_candidates": mean(counts),
    }


def markdown_report(summary):
    c = summary["test"]["current_v31_topk"]
    r = summary["test"]["lgbm_ranker_topk"]
    d = summary["test"]["delta_ranker_minus_current"]

    lines = [
        "# Research01: v31 second ranking vs LGBMRanker",
        "",
        f"- test races: {c['races']}",
        f"- selected Ranker variant: {summary['selected_ranker']['name']}",
        f"- features: {len(summary['feature_names'])}",
        "- odds/popularity/payout features: not used",
        "",
        "| metric | current v31 | LGBMRanker | delta |",
        "|---|---:|---:|---:|",
    ]
    ordered = [
        "top1_second_capture",
        "top2_second_capture",
        "top3_second_capture",
        "actual_second_mean_rank",
        "mrr",
        "complete_board_top1",
        "complete_board_top2",
        "complete_board_top3",
    ]
    for key in ordered:
        lines.append(f"| {key} | {c[key]:.6f} | {r[key]:.6f} | {d[key]:+.6f} |")

    lines.extend(
        [
            "",
            "## Frozen current variable-candidate reference",
            "",
            f"- capture: {summary['test']['current_variable_candidate_reference']['capture']:.6f}",
            f"- average candidates: {summary['test']['current_variable_candidate_reference']['avg_second_candidates']:.6f}",
            "",
            "## Leakage / scope guard",
            "",
            "- only seven-rider races are used",
            "- train/calibration/test are chronological and fixed",
            "- no random train_test_split or KFold",
            "- ranker features come only from v25 pre-race rider/line feature map",
            "- no v21 first-probability feature is supplied to Ranker",
            "- row 1 and row 3 candidates are frozen from the existing v37 test log",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    vrows = v25.load_v19_details()
    v25.fit_v21_selector(vrows)
    v25.mark_v21(vrows)
    _, entries_by, results_by = v25.raw_maps()

    train = v25.make_dataset(vrows, entries_by, results_by, TRAIN_START, TRAIN_END, False)
    cal = v25.make_dataset(vrows, entries_by, results_by, CAL_START, CAL_END, False)
    test = v25.make_dataset(vrows, entries_by, results_by, TEST_START, TEST_END, False)

    feature_names = feature_names_from_dataset(train)

    calibration = []
    trained = {}
    for variant in VARIANTS:
        model = fit_ranker(train, feature_names, variant["params"])
        trained[variant["name"]] = model
        rows = evaluate_ranker(cal, model, feature_names)
        metrics = ranking_metrics(rows)
        calibration.append(
            {
                "name": variant["name"],
                "params": variant["params"],
                "metrics": metrics,
            }
        )

    # Calibration selection only. Test is untouched.
    calibration.sort(
        key=lambda item: (
            item["metrics"]["mrr"],
            item["metrics"]["top2_second_capture"],
            item["metrics"]["top3_second_capture"],
            item["metrics"]["top1_second_capture"],
            -item["metrics"]["actual_second_mean_rank"],
        ),
        reverse=True,
    )
    selected = calibration[0]
    model = trained[selected["name"]]

    ranker_test = evaluate_ranker(test, model, feature_names)
    fixed = load_v37_fixed_board()
    comparison_rows = enrich_with_fixed_board(ranker_test, fixed)

    expected_test_participants = len(participant_rows(test))
    if len(comparison_rows) != expected_test_participants:
        missing = sorted(
            set(str(row[0]) for row in participant_rows(test))
            - set(str(row["race_id"]) for row in comparison_rows)
        )
        raise RuntimeError(
            f"fixed-board alignment mismatch: ranker participants={expected_test_participants}, "
            f"aligned={len(comparison_rows)}, missing_sample={missing[:10]}"
        )

    current, ranker, delta = compare_metrics(comparison_rows)

    summary = {
        "research": "research01_lgbm_ranker_second",
        "purpose": "Replace only the second-place ranking engine and compare equal-K capture against frozen current v31.",
        "periods": {
            "train": [TRAIN_START, TRAIN_END],
            "calibration": [CAL_START, CAL_END],
            "test": [TEST_START, TEST_END],
        },
        "dataset_counts": {
            "train_all_7car": len(train),
            "cal_all_7car": len(cal),
            "cal_v21_participants": len(participant_rows(cal)),
            "test_all_7car": len(test),
            "test_v21_participants": expected_test_participants,
        },
        "feature_names": feature_names,
        "selected_ranker": selected,
        "calibration_variants": calibration,
        "test": {
            "current_v31_topk": current,
            "lgbm_ranker_topk": ranker,
            "delta_ranker_minus_current": delta,
            "current_variable_candidate_reference": variable_current_reference(comparison_rows),
        },
        "guards": {
            "seven_rider_only": True,
            "v21_participation_frozen": True,
            "first_row_frozen": True,
            "third_row_frozen": True,
            "odds_used": False,
            "popularity_used": False,
            "payout_used_pre_race": False,
            "v21_first_probability_used_by_ranker": False,
            "random_split_used": False,
            "kfold_used": False,
            "test_used_for_model_selection": False,
        },
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "race_log.json").write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "report.md").write_text(markdown_report(summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Diagnose ninecar v3 probability/result divergence on the 2026 H1 forward set."""

import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "scripts/keirin_shogi_ninecar_v3.py"
MODEL_PATH = ROOT / "results/keirin_shogi/ninecar_v3/model.joblib"
OUT = ROOT / "results/keirin_shogi/ninecar_v3/divergence_2026_h1.json"


def load_module():
    spec = importlib.util.spec_from_file_location("ninecar_v3_diag_base", V3_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v3 = load_module()
v2 = v3.v2


def rank_of(mapping: dict[int, float], car: int) -> int:
    ordered = sorted(mapping.items(), key=lambda x: (-x[1], x[0]))
    return next(i + 1 for i, (no, _) in enumerate(ordered) if no == car)


def value_of(mapping: dict[int, float], car: int) -> float:
    return float(mapping.get(car, 0.0))


def rank_dist(vals):
    c = Counter(vals)
    return {str(k): int(c[k]) for k in sorted(c)}


def quantiles(vals):
    if not vals:
        return {}
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.10)),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
    }


def topk_rate(ranks, ks):
    return {f"top{k}": float(np.mean([r <= k for r in ranks])) for k in ks}


def decile_table(rows, key):
    ordered = sorted(rows, key=lambda r: r[key])
    chunks = np.array_split(np.arange(len(ordered)), 10)
    out = []
    for i, idxs in enumerate(chunks, 1):
        part = [ordered[int(j)] for j in idxs]
        out.append({
            "decile_low_to_high": i,
            "n": len(part),
            "score_min": float(min(r[key] for r in part)),
            "score_max": float(max(r[key] for r in part)),
            "score_mean": float(np.mean([r[key] for r in part])),
            "full_hit_rate": float(np.mean([r["full_hit"] for r in part])),
            "first_hit_rate": float(np.mean([r["hits"][0] for r in part])),
            "second_hit_rate": float(np.mean([r["hits"][1] for r in part])),
            "third_hit_rate": float(np.mean([r["hits"][2] for r in part])),
        })
    return out


def entry_map(race):
    return {v2.ino(e.get("car_no")): e for e in race.entries}


def score_rank(race, car):
    em = entry_map(race)
    vals = [(no, v2.fnum(e.get("score"))) for no, e in em.items()]
    vals.sort(key=lambda x: (-x[1], x[0]))
    return next(i + 1 for i, (no, _) in enumerate(vals) if no == car)


def main():
    races = v2.load_races()
    train_pre2026, eval2026 = v2.split_before(races, 2026)
    models = v3.fit_models(train_pre2026)
    bundle = joblib.load(MODEL_PATH)
    alpha = float(bundle["alpha"])
    beta = float(bundle["beta"])
    policy = bundle["board_policy"]
    part_model = bundle["participation_model"]
    part_threshold = float(bundle["participation_threshold"])

    rows = []
    for race in eval2026:
        comp = v3.probability_components(race, models)
        joint = v3.joint_from_components(comp, alpha, beta)
        board, mass = v3.apply_board(joint, policy)
        hits = v2.captured(race, board)
        p1, p2, p3 = v2.marginals(joint)
        first, second, third = race.order

        pair_mass = defaultdict(float)
        for a, b, c, p in joint:
            pair_mass[(a, b)] += p
        pair_sorted = sorted(pair_mass.items(), key=lambda x: (-x[1], x[0]))
        true_pair_rank = next(i + 1 for i, (ab, _) in enumerate(pair_sorted) if ab == (first, second))
        true_pair_prob = float(pair_mass[(first, second)])

        true_triple_rank = next(i + 1 for i, x in enumerate(joint) if x[:3] == (first, second, third))
        true_triple_prob = float(next(x[3] for x in joint if x[:3] == (first, second, third)))

        # Direct second prior before ordered-pair fusion.
        direct_second = {n: float(comp["p2"][n - 1]) for n in range(1, 10)}

        # Third conditional if the actual first-second pair were known.
        actual_pair_index = comp["pairs"].index((first, second))
        cs = comp["pair_third_cars"][actual_pair_index]
        cond = comp["conditionals"][actual_pair_index]
        conditional_map = {int(c): float(p) for c, p in zip(cs, cond)}
        true_third_cond_rank = rank_of(conditional_map, third)
        true_third_cond_prob = value_of(conditional_map, third)

        feat = v2.prediction_features(race, joint, board, mass)
        participation_score = float(part_model.predict_proba(feat.reshape(1, -1))[0, 1])

        em = entry_map(race)
        row = {
            "race_id": race.race_id,
            "race_date": race.race_date,
            "grade": race.grade,
            "race_type": race.race_type,
            "actual": [first, second, third],
            "board": [list(x) for x in board],
            "row_sizes": [len(x) for x in board],
            "hits": [bool(x) for x in hits],
            "full_hit": bool(all(hits)),
            "board_mass": float(mass),
            "participation_score": participation_score,
            "participate": participation_score >= part_threshold,
            "actual_ranks": {
                "first_joint_marginal": rank_of(p1, first),
                "second_joint_marginal": rank_of(p2, second),
                "third_joint_marginal": rank_of(p3, third),
                "second_direct_prior": rank_of(direct_second, second),
                "true_ordered_pair": true_pair_rank,
                "true_ordered_triple": true_triple_rank,
                "third_given_true_pair": true_third_cond_rank,
                "first_score_rank": score_rank(race, first),
                "second_score_rank": score_rank(race, second),
                "third_score_rank": score_rank(race, third),
            },
            "actual_probabilities": {
                "first_joint_marginal": value_of(p1, first),
                "second_joint_marginal": value_of(p2, second),
                "third_joint_marginal": value_of(p3, third),
                "second_direct_prior": value_of(direct_second, second),
                "true_ordered_pair": true_pair_prob,
                "true_ordered_triple": true_triple_prob,
                "third_given_true_pair": true_third_cond_prob,
            },
            "line": {
                "first_id": v2.ino(em[first].get("line_id")),
                "second_id": v2.ino(em[second].get("line_id")),
                "third_id": v2.ino(em[third].get("line_id")),
                "first_pos": v2.ino(em[first].get("line_position")),
                "second_pos": v2.ino(em[second].get("line_position")),
                "third_pos": v2.ino(em[third].get("line_position")),
                "first_second_same": v2.ino(em[first].get("line_id")) == v2.ino(em[second].get("line_id")),
            },
        }
        rows.append(row)

    hit_patterns = Counter("".join("1" if x else "0" for x in r["hits"]) for r in rows)
    size_patterns = Counter("-".join(map(str, r["row_sizes"])) for r in rows)

    def vals(path):
        group, key = path
        return [r[group][key] for r in rows]

    actual_rank_summary = {}
    for key in (
        "first_joint_marginal","second_joint_marginal","third_joint_marginal",
        "second_direct_prior","true_ordered_pair","true_ordered_triple",
        "third_given_true_pair","first_score_rank","second_score_rank","third_score_rank",
    ):
        ranks = vals(("actual_ranks", key))
        ks = [1,2,3,4,5,6,7,8,9] if max(ranks) <= 9 else [1,3,5,10,20,50,100,200]
        actual_rank_summary[key] = {
            "distribution": rank_dist(ranks),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "topk": topk_rate(ranks, ks),
        }

    actual_prob_summary = {
        key: quantiles(vals(("actual_probabilities", key)))
        for key in (
            "first_joint_marginal","second_joint_marginal","third_joint_marginal",
            "second_direct_prior","true_ordered_pair","true_ordered_triple","third_given_true_pair"
        )
    }

    # Greedy-board loss: actual rider is within marginal top row-size but board omitted it.
    greedy_losses = {}
    marginal_keys = ["first_joint_marginal","second_joint_marginal","third_joint_marginal"]
    for i, key in enumerate(marginal_keys):
        eligible = [r for r in rows if r["actual_ranks"][key] <= r["row_sizes"][i]]
        omitted = [r for r in eligible if not r["hits"][i]]
        greedy_losses[["first","second","third"][i]] = {
            "actual_within_marginal_top_row_size": len(eligible),
            "omitted_by_greedy_board": len(omitted),
            "omission_rate_given_marginally_coverable": len(omitted)/len(eligible) if eligible else None,
        }

    by_line_relation = {}
    for same in (True, False):
        group = [r for r in rows if r["line"]["first_second_same"] == same]
        by_line_relation["same_line" if same else "different_line"] = {
            "n": len(group),
            "first_capture": float(np.mean([r["hits"][0] for r in group])),
            "second_capture": float(np.mean([r["hits"][1] for r in group])),
            "third_capture": float(np.mean([r["hits"][2] for r in group])),
            "full_capture": float(np.mean([r["full_hit"] for r in group])),
            "true_pair_mean_rank": float(np.mean([r["actual_ranks"]["true_ordered_pair"] for r in group])),
        }

    by_second_line_position = {}
    for pos in sorted({r["line"]["second_pos"] for r in rows}):
        group = [r for r in rows if r["line"]["second_pos"] == pos]
        by_second_line_position[str(pos)] = {
            "n": len(group),
            "second_capture": float(np.mean([r["hits"][1] for r in group])),
            "full_capture": float(np.mean([r["full_hit"] for r in group])),
            "second_joint_mean_rank": float(np.mean([r["actual_ranks"]["second_joint_marginal"] for r in group])),
            "second_direct_mean_rank": float(np.mean([r["actual_ranks"]["second_direct_prior"] for r in group])),
        }

    # Largest misses where model was most confident by board mass.
    confident_misses = sorted(
        [r for r in rows if not r["full_hit"]],
        key=lambda r: (-r["board_mass"], -r["participation_score"]),
    )[:40]

    report = {
        "model": "ninecar_v3_direct_second_joint504_fixed7",
        "period": "2026 H1 forward",
        "races": len(rows),
        "alpha": alpha,
        "beta": beta,
        "board_policy": policy,
        "participation_threshold": part_threshold,
        "capture": {
            "first": float(np.mean([r["hits"][0] for r in rows])),
            "second": float(np.mean([r["hits"][1] for r in rows])),
            "third": float(np.mean([r["hits"][2] for r in rows])),
            "full": float(np.mean([r["full_hit"] for r in rows])),
        },
        "hit_patterns": dict(sorted(hit_patterns.items())),
        "row_size_patterns": dict(sorted(size_patterns.items())),
        "actual_rank_summary": actual_rank_summary,
        "actual_probability_summary": actual_prob_summary,
        "greedy_board_losses": greedy_losses,
        "participation_score_deciles": decile_table(rows, "participation_score"),
        "board_mass_deciles": decile_table(rows, "board_mass"),
        "by_first_second_line_relation": by_line_relation,
        "by_second_line_position": by_second_line_position,
        "confident_misses_top40": confident_misses,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

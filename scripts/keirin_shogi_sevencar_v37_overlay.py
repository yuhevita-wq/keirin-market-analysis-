#!/usr/bin/env python3
from __future__ import annotations

"""Runtime-only seven-car v37 state overlay.

Validated policy:
- v21 participation is unchanged.
- v31 second row is unchanged.
- for existing v21 participants only, if the baseline strongest rider is in
  a SOFT_FAIL state (likely 2nd/3rd rather than collapse), append that rider
  to row3 without removing any existing v37 candidate.

The model/rules are frozen from pre-future training/tuning and were validated
on the 2026-07-06..2026-08-30 true-future block.
"""

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results/keirin_shogi/sevencar_state_transfer/validated_model.joblib"


def _rank_desc(base, key):
    order = sorted(base, key=lambda n: (-float(base[n]["x"].get(key, 0.0)), n))
    return {n: i + 1 for i, n in enumerate(order)}


def build_features(
    *,
    base,
    first_candidates,
    second_candidates,
    third_candidates,
    membership,
    top_pairs,
    race_type,
):
    strong = int(first_candidates[0])
    mem = {int(no): float(value) for no, value in membership}
    mem_order = sorted(mem, key=lambda n: (-mem[n], n))
    mr = {n: i + 1 for i, n in enumerate(mem_order)}
    vals = [mem[n] for n in mem_order]

    zkeys = (
        "z_score",
        "z_win_rate",
        "z_top2_rate",
        "z_top3_rate",
        "z_b_count",
        "z_first_count",
        "z_second_count",
        "z_third_count",
        "z_outside_count",
    )
    ranks = {key: _rank_desc(base, key) for key in zkeys}
    xstrong = base[strong]["x"]

    strong_line = str(base[strong]["line_id"])
    line_members = defaultdict(list)
    for no in base:
        line_members[str(base[no]["line_id"])].append(no)

    def line_mean(line_id, key):
        riders = line_members[line_id]
        return float(np.mean([float(base[no]["x"].get(key, 0.0)) for no in riders]))

    strong_line_score = line_mean(strong_line, "z_score")
    rival_scores = [
        line_mean(line_id, "z_score")
        for line_id in line_members
        if line_id != strong_line
    ]
    rival_best = max(rival_scores) if rival_scores else 0.0

    top2_same = 0.0
    if len(first_candidates) >= 2:
        top2_same = float(
            str(base[int(first_candidates[0])]["line_id"])
            == str(base[int(first_candidates[1])]["line_id"])
        )

    top_pair = float(top_pairs[0][2]) if top_pairs else 0.0
    top3_pair = float(sum(float(row[2]) for row in top_pairs[:3]))

    feats = {
        "strong_membership": float(mem.get(strong, 0.0)),
        "strong_membership_rank": float(mr.get(strong, 7)),
        "membership_top1": float(vals[0] if vals else 0.0),
        "membership_gap12": float(vals[0] - vals[1] if len(vals) > 1 else 0.0),
        "membership_top2_mass": float(sum(vals[:2])),
        "top_pair_prob": top_pair,
        "top3_pair_mass": top3_pair,
        "first_count": float(len(first_candidates)),
        "second_count": float(len(second_candidates)),
        "third_count": float(len(third_candidates)),
        "strong_in_second": float(strong in second_candidates),
        "strong_in_third": float(strong in third_candidates),
        "strong_line_pos": float(base[strong]["line_position"]),
        "strong_line_size": float(base[strong]["x"].get("line_size", 1.0)),
        "strong_line_score": strong_line_score,
        "best_rival_line_score": rival_best,
        "strong_vs_rival_line_score": strong_line_score - rival_best,
        "top2_first_same_line": top2_same,
        "num_lines": float(len(line_members)),
    }
    for key in zkeys:
        feats[f"strong_{key}"] = float(xstrong.get(key, 0.0))
        feats[f"strong_{key}_rank"] = float(ranks[key][strong])

    text = str(race_type)
    for label in ("予選", "一般", "準決勝", "特選", "選抜", "決勝"):
        feats[f"race_{label}"] = float(label in text)
    return feats, strong


def predict_overlay(
    *,
    base,
    first_candidates,
    second_candidates,
    third_candidates,
    membership,
    top_pairs,
    race_type,
    participate,
    model_path: Path = MODEL_PATH,
):
    """Return third-row overlay decision. Never changes participation or row1/row2."""
    if not participate:
        return {
            "third_candidates": list(third_candidates),
            "action": "BASE",
            "strong_rider": int(first_candidates[0]),
            "state_probabilities": None,
            "added": False,
        }
    if not model_path.exists():
        raise FileNotFoundError(f"seven-car validated overlay model not found: {model_path}")

    bundle = joblib.load(model_path)
    feats, strong = build_features(
        base=base,
        first_candidates=list(map(int, first_candidates)),
        second_candidates=list(map(int, second_candidates)),
        third_candidates=list(map(int, third_candidates)),
        membership=membership,
        top_pairs=top_pairs,
        race_type=race_type,
    )
    names = bundle["feature_names"]
    x = np.asarray([[float(feats[name]) for name in names]], dtype=float)
    probs = bundle["model"].predict_proba(x)[0]
    state_probs = {
        "WIN": float(probs[0]),
        "SOFT_FAIL": float(probs[1]),
        "COLLAPSE": float(probs[2]),
    }
    rules = bundle["rules"]
    trigger = (
        state_probs["SOFT_FAIL"] >= float(rules["soft_threshold"])
        and state_probs["COLLAPSE"] < float(rules["soft_collapse_max"])
    )

    third = list(map(int, third_candidates))
    added = False
    action = "BASE"
    if trigger and strong not in third:
        third.append(strong)
        third = sorted(set(third))
        added = True
        action = "SOFT_FAIL_APPEND_THIRD"

    return {
        "third_candidates": third,
        "action": action,
        "strong_rider": strong,
        "state_probabilities": state_probs,
        "added": added,
        "rules": {
            "soft_threshold": float(rules["soft_threshold"]),
            "soft_collapse_max": float(rules["soft_collapse_max"]),
        },
        "model_name": str(bundle["model_name"]),
        "validated_through": str(bundle["true_future_validated_through"]),
    }

#!/usr/bin/env python3
from __future__ import annotations

"""Nine-car v3.2 production candidate.

Overlay three mechanical rules on the current v3 joint504 board:
1) COLLAPSE filter: high risk that the baseline strongest rider finishes outside top3 -> skip.
2) SOFT_FAIL overlay: strongest rider likely remains 2nd/3rd -> keep 1st placement, also force into 2nd row.
3) DOMINANT protection: an overwhelmingly strong rider is never demoted from 1st row.

No odds/popularity/payout are used at inference.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "scripts/keirin_shogi_ninecar_v3.py"
OUT = ROOT / "results/keirin_shogi/ninecar_v32"
BASE_MODEL = ROOT / "results/keirin_shogi/ninecar_v3/model.joblib"
STATE_NAMES = ("WIN", "SOFT_FAIL", "COLLAPSE")
METRICS = ("score", "win_rate", "top2_rate", "top3_rate", "s_count", "b_count",
           "nige_count", "makuri_count", "sashi_count", "mark_count",
           "first_count", "second_count", "third_count", "outside_count")


def load_v3():
    spec = importlib.util.spec_from_file_location("ninecar_v32_base", V3_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


v3 = load_v3()
v2 = v3.v2
Race = v2.Race


def rank_map(vals, reverse=True):
    order = sorted(vals, key=lambda n: ((-vals[n]) if reverse else vals[n], n))
    return {n: i + 1 for i, n in enumerate(order)}


def entry_map(race):
    return {v2.ino(e.get("car_no")): e for e in race.entries}


def base_and_joint(race, models, alpha=1.0, beta=0.5):
    base, _ = v2.base_map(race)
    x = np.asarray([base[n] for n in range(1, 10)])
    p1a = v2.normalize(models["first"].predict_proba(x)[:, 1])
    p2a = v2.normalize(models["second"].predict_proba(x)[:, 1])
    p1 = {n: float(p1a[n - 1]) for n in range(1, 10)}
    p2 = {n: float(p2a[n - 1]) for n in range(1, 10)}
    comp = v3.probability_components(race, models)
    joint = v3.joint_from_components(comp, alpha, beta)
    return p1, p2, joint


def state_features(race, models, alpha=1.0, beta=0.5):
    em = entry_map(race)
    p1, p2, joint = base_and_joint(race, models, alpha, beta)
    m1, m2, m3 = v2.marginals(joint)
    ordered = sorted(p1, key=lambda n: (-p1[n], n))
    strong, runner = ordered[:2]
    strong_entry = em[strong]
    strong_line = v2.ino(strong_entry.get("line_id"))

    metric_vals = {
        m: {n: v2.fnum(em[n].get(m)) for n in range(1, 10)}
        for m in METRICS
    }
    metric_ranks = {
        m: rank_map(vals, reverse=(m != "outside_count"))
        for m, vals in metric_vals.items()
    }

    line_mass = {}
    for n in range(1, 10):
        lid = v2.ino(em[n].get("line_id"))
        line_mass[lid] = line_mass.get(lid, 0.0) + p1[n]
    strong_line_mass = float(line_mass[strong_line])
    strongest_rival = float(max((x for lid, x in line_mass.items() if lid != strong_line), default=0.0))
    strong_rank_vec = np.asarray([metric_ranks[m][strong] for m in METRICS], dtype=float)
    field_disagreement = float(np.mean([
        np.std([metric_ranks[m][n] for m in METRICS]) for n in range(1, 10)
    ]))
    p1r = rank_map(p1)
    feats = {
        "strong_p1": p1[strong],
        "strong_p2": p2[strong],
        "strong_joint_first": float(m1[strong]),
        "strong_joint_second": float(m2[strong]),
        "strong_joint_third": float(m3[strong]),
        "strong_joint_down_mass": float(m2[strong] + m3[strong]),
        "strong_p1_gap": float(p1[strong] - p1[runner]),
        "strong_score_gap": float(metric_vals["score"][strong] - max(metric_vals["score"][n] for n in range(1,10) if n != strong)),
        "strong_top2_rate_gap": float(metric_vals["top2_rate"][strong] - max(metric_vals["top2_rate"][n] for n in range(1,10) if n != strong)),
        "strong_top3_rate_gap": float(metric_vals["top3_rate"][strong] - max(metric_vals["top3_rate"][n] for n in range(1,10) if n != strong)),
        "strong_rank_disagreement": float(np.std(strong_rank_vec)),
        "field_rank_disagreement": field_disagreement,
        "strong_good_signal_count": float(sum(metric_ranks[m][strong] <= 2 for m in METRICS)),
        "strong_bad_signal_count": float(sum(metric_ranks[m][strong] >= 5 for m in METRICS)),
        "strong_line_mass": strong_line_mass,
        "strongest_rival_line_mass": strongest_rival,
        "strong_vs_rival_line_gap": strong_line_mass - strongest_rival,
        "strong_line_position": float(v2.ino(strong_entry.get("line_position"))),
        "strong_line_size": float(v2.ino(strong_entry.get("line_size"))),
        "strong_line_top3_count": float(sum(p1r[n] <= 3 for n in range(1,10) if v2.ino(em[n].get("line_id")) == strong_line)),
        "top2_strength_same_line": float(v2.ino(em[runner].get("line_id")) == strong_line),
        "num_lines": float(len(line_mass)),
    }
    for m in METRICS:
        feats[f"strong_{m}_rank"] = float(metric_ranks[m][strong])
        feats[f"strong_{m}_value"] = float(metric_vals[m][strong])
    for g in ("G1", "G2", "G3"):
        feats[f"grade_{g}"] = float(race.grade == g)
    for s in v2.STAGE_PATTERNS:
        feats[f"stage_{s}"] = float(s in race.race_type)
    return feats, strong, p1, p2, joint


def state_label(race, strong):
    if race.order[0] == strong:
        return 0
    if strong in race.order[1:3]:
        return 1
    return 2


def build_oos_rows(year, races):
    models = v3.fit_models([r for r in races if r.year < year])
    rows = []
    for race in [r for r in races if r.year == year]:
        feats, strong, p1, p2, joint = state_features(race, models)
        rows.append({
            "race": race, "features": feats, "strong": strong,
            "state": state_label(race, strong), "p1": p1, "p2": p2, "joint": joint,
        })
    return rows


def fit_state(rows):
    names = list(rows[0]["features"].keys())
    X = np.asarray([[r["features"][f] for f in names] for r in rows], dtype=float)
    y = np.asarray([r["state"] for r in rows], dtype=int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=20260918),
    )
    model.fit(X, y)
    return model, names


def score_state(rows, model, names):
    X = np.asarray([[r["features"][f] for f in names] for r in rows], dtype=float)
    pp = model.predict_proba(X)
    out = []
    for r, p in zip(rows, pp):
        z = dict(r)
        z["state_probs"] = {STATE_NAMES[i]: float(p[i]) for i in range(3)}
        out.append(z)
    return out


def choose_rules(scored):
    collapse_scores = np.asarray([r["state_probs"]["COLLAPSE"] for r in scored])
    collapse_threshold = float(np.quantile(collapse_scores, 0.70))  # cut top 30%

    soft_scores = np.asarray([r["state_probs"]["SOFT_FAIL"] for r in scored])
    soft_candidates = [float(x) for x in np.quantile(soft_scores, [0.70, 0.80, 0.85, 0.90])]
    best_soft = None
    for st in soft_candidates:
        sel = [r for r in scored if r["state_probs"]["SOFT_FAIL"] >= st and r["state_probs"]["COLLAPSE"] < collapse_threshold]
        if len(sel) < max(50, int(0.05 * len(scored))):
            continue
        soft_rate = float(np.mean([r["state"] == 1 for r in sel]))
        collapse_rate = float(np.mean([r["state"] == 2 for r in sel]))
        key = (soft_rate - collapse_rate, soft_rate, -collapse_rate)
        if best_soft is None or key > best_soft[0]:
            best_soft = (key, st)
    soft_threshold = float(best_soft[1] if best_soft else np.quantile(soft_scores, 0.90))

    # Dominant = large rider-only first gap AND strong line concentration AND low collapse.
    gap_vals = np.asarray([r["features"]["strong_p1_gap"] for r in scored])
    line_vals = np.asarray([r["features"]["strong_line_mass"] for r in scored])
    best_dom = None
    for qg in (0.70, 0.80, 0.90):
        for ql in (0.60, 0.70, 0.80):
            gt = float(np.quantile(gap_vals, qg))
            lt = float(np.quantile(line_vals, ql))
            sel = [r for r in scored if r["features"]["strong_p1_gap"] >= gt and r["features"]["strong_line_mass"] >= lt and r["state_probs"]["COLLAPSE"] < collapse_threshold]
            if len(sel) < max(50, int(0.05 * len(scored))):
                continue
            win = float(np.mean([r["state"] == 0 for r in sel]))
            key = (win, len(sel))
            if best_dom is None or key > best_dom[0]:
                best_dom = (key, gt, lt, win, len(sel))
    if best_dom is None:
        best_dom = ((0,0), float(np.quantile(gap_vals,0.9)), float(np.quantile(line_vals,0.8)), 0.0, 0)

    return {
        "collapse_threshold": collapse_threshold,
        "soft_threshold": soft_threshold,
        "dominant_p1_gap_threshold": float(best_dom[1]),
        "dominant_line_mass_threshold": float(best_dom[2]),
        "dominant_training_win_rate": float(best_dom[3]),
        "dominant_training_n": int(best_dom[4]),
    }


def replace_lowest(row, rider, probs):
    row = list(row)
    if rider in row:
        return tuple(sorted(row))
    victim = min(row, key=lambda n: (probs[n], -n))
    row.remove(victim)
    row.append(rider)
    return tuple(sorted(row))


def apply_overlay(r, rules):
    p1m, p2m, p3m = v2.marginals(r["joint"])
    board, mass = v2.greedy_board(r["joint"], 7)
    board = [tuple(x) for x in board]
    collapse = r["state_probs"]["COLLAPSE"]
    soft = r["state_probs"]["SOFT_FAIL"]
    dominant = (
        r["features"]["strong_p1_gap"] >= rules["dominant_p1_gap_threshold"]
        and r["features"]["strong_line_mass"] >= rules["dominant_line_mass_threshold"]
        and collapse < rules["collapse_threshold"]
    )
    action = "BASE"
    if dominant:
        board[0] = replace_lowest(board[0], r["strong"], p1m)
        action = "DOMINANT_FIRST"
    elif soft >= rules["soft_threshold"] and collapse < rules["collapse_threshold"]:
        board[1] = replace_lowest(board[1], r["strong"], p2m)
        action = "SOFT_FAIL_ADD_SECOND"
    participate = collapse < rules["collapse_threshold"]
    return tuple(board), mass, participate, dominant, action


def evaluate(scored, rules):
    vals = []
    for r in scored:
        board, _mass, participate, dominant, action = apply_overlay(r, rules)
        hits = v2.captured(r["race"], board)
        vals.append({
            "hits": hits, "full": bool(all(hits)), "participate": participate,
            "dominant": dominant, "action": action, "state": r["state"],
        })
    part = [x for x in vals if x["participate"]]
    dom = [x for x in vals if x["dominant"]]
    soft = [x for x in vals if x["action"] == "SOFT_FAIL_ADD_SECOND"]
    return {
        "n": len(vals),
        "first_capture": float(np.mean([x["hits"][0] for x in vals])),
        "second_capture": float(np.mean([x["hits"][1] for x in vals])),
        "third_capture": float(np.mean([x["hits"][2] for x in vals])),
        "full_capture": float(np.mean([x["full"] for x in vals])),
        "participation_rate": len(part)/len(vals),
        "participant_full_capture": float(np.mean([x["full"] for x in part])) if part else None,
        "participant_first_capture": float(np.mean([x["hits"][0] for x in part])) if part else None,
        "participant_second_capture": float(np.mean([x["hits"][1] for x in part])) if part else None,
        "dominant_n": len(dom),
        "dominant_rate": len(dom)/len(vals),
        "dominant_actual_win_rate": float(np.mean([x["state"] == 0 for x in dom])) if dom else None,
        "dominant_first_capture": float(np.mean([x["hits"][0] for x in dom])) if dom else None,
        "soft_action_n": len(soft),
        "soft_action_rate": len(soft)/len(vals),
        "soft_action_second_capture": float(np.mean([x["hits"][1] for x in soft])) if soft else None,
        "soft_action_full_capture": float(np.mean([x["full"] for x in soft])) if soft else None,
    }


def rolling_eval():
    races = v2.load_races()
    rows24 = build_oos_rows(2024, races)
    rows25 = build_oos_rows(2025, races)
    rows26 = build_oos_rows(2026, races)

    m24, n24 = fit_state(rows24)
    s24 = score_state(rows24, m24, n24)
    rules24 = choose_rules(s24)
    s25 = score_state(rows25, m24, n24)

    m25, n25 = fit_state(rows25)
    s25_train = score_state(rows25, m25, n25)
    rules25 = choose_rules(s25_train)
    s26 = score_state(rows26, m25, n25)

    # Production for races after 2026 H1: calibrate from 2026 H1 OOS state rows.
    m26, n26 = fit_state(rows26)
    s26_train = score_state(rows26, m26, n26)
    rules26 = choose_rules(s26_train)

    base_bundle = joblib.load(BASE_MODEL)
    prod_bundle = {
        "base_bundle": base_bundle,
        "state_model": m26,
        "state_feature_names": n26,
        "rules": rules26,
        "model_name": "ninecar_v32_strong_state_overlay",
        "trained_through": "2026-06-30",
        "policy_note": "collapse skip + soft-fail second duplication + dominant first protection",
    }

    report = {
        "schema_version": 1,
        "model": prod_bundle["model_name"],
        "rules": {
            "2024_to_2025": rules24,
            "2025_to_2026": rules25,
            "production_after_2026_h1": rules26,
        },
        "evaluation": {
            "2025_forward": evaluate(s25, rules24),
            "2026_h1_forward": evaluate(s26, rules25),
        },
        "reference_v3_2026": {
            "first_capture": 0.5175345377258236,
            "second_capture": 0.4314558979808714,
            "third_capture": 0.4218916046758767,
            "full_capture": 0.17321997874601489,
            "participation_rate": 0.822529224229543,
            "participant_full_capture": 0.1847545219638243,
        },
        "guards": {
            "odds_used": False, "popularity_used": False, "payout_used": False,
            "results_used_at_inference": False,
            "dominant_rider_never_demoted_from_first": True,
            "soft_fail_keeps_first_and_adds_second": True,
            "collapse_only_drives_skip": True,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod_bundle, OUT / "model.joblib", compress=3)
    (OUT / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def live_race_from_payload(payload):
    return v2.live_race_from_payload(payload)


def predict_live(payload: dict, model_path: Path | None = None):
    model_path = model_path or (OUT / "model.joblib")
    bundle = joblib.load(model_path)
    base = bundle["base_bundle"]
    race = live_race_from_payload(payload)
    feats, strong, p1, p2, joint = state_features(
        race, base["models"], float(base["alpha"]), float(base["beta"])
    )
    X = np.asarray([[feats[f] for f in bundle["state_feature_names"]]], dtype=float)
    pp = bundle["state_model"].predict_proba(X)[0]
    scored = {
        "race": race, "features": feats, "strong": strong, "p1": p1, "p2": p2,
        "joint": joint, "state_probs": {STATE_NAMES[i]: float(pp[i]) for i in range(3)}
    }
    board, mass, participate, dominant, action = apply_overlay(scored, bundle["rules"])
    p1m, p2m, p3m = v2.marginals(joint)
    return {
        "race_id": race.race_id,
        "board_generated": True,
        "board_policy": bundle["model_name"],
        "participate": bool(participate),
        "first_candidates": list(board[0]),
        "second_candidates": list(board[1]),
        "third_candidates": list(board[2]),
        "joint_board_mass": mass,
        "strong_rider": strong,
        "strong_state_probabilities": scored["state_probs"],
        "dominant_strong": bool(dominant),
        "overlay_action": action,
        "collapse_threshold": float(bundle["rules"]["collapse_threshold"]),
        "soft_threshold": float(bundle["rules"]["soft_threshold"]),
        "first_ranking": [{"no": n, "probability": p} for n,p in sorted(p1m.items(), key=lambda x:x[1], reverse=True)],
        "second_ranking": [{"no": n, "probability": p} for n,p in sorted(p2m.items(), key=lambda x:x[1], reverse=True)],
        "third_ranking": [{"no": n, "probability": p} for n,p in sorted(p3m.items(), key=lambda x:x[1], reverse=True)],
        "versions": {"ninecar": bundle["model_name"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-evaluate", action="store_true")
    ap.add_argument("--predict-json")
    args = ap.parse_args()
    if args.train_evaluate:
        rolling_eval(); return 0
    if args.predict_json:
        payload = json.loads(Path(args.predict_json).read_text(encoding="utf-8"))
        print(json.dumps(predict_live(payload), ensure_ascii=False, indent=2)); return 0
    ap.error("choose --train-evaluate or --predict-json")


if __name__ == "__main__":
    raise SystemExit(main())

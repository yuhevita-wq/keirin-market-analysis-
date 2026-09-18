#!/usr/bin/env python3
from __future__ import annotations

"""Nine-car v3.1 research: pair reinforcement + variable 7/8/9 board + rolling participation.

Only three changes from v3 are studied:
1. conservative auxiliary ordered-pair model weighted toward cross-line true pairs
   and true pairs whose second-place rider is line position 3;
2. variable 7/8/9-piece greedy board selected only from pre-race uncertainty;
3. participation model recalibrated from prior out-of-sample predictions for
   each forward generation.

No odds, popularity, current-race result, or payout is used at inference.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "scripts/keirin_shogi_ninecar_v3.py"
OUT = ROOT / "results/keirin_shogi/ninecar_v31"


def load_v3():
    spec = importlib.util.spec_from_file_location("keirin_shogi_ninecar_v3_base31", V3_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v3 = load_v3()
v2 = v3.v2
Race = v2.Race

PAIR_CONFIGS = tuple(
    (alpha, beta, rho)
    for alpha in (0.7, 1.0)
    for beta in (0.5, 1.0, 1.5)
    for rho in (0.0, 0.25, 0.5)
)
MAX_AVG_PIECES = 7.57


def fit_models(races: list[Race]):
    core = v3.fit_models(races)
    gw = v2.grade_weights(races)
    xp, yp, wp = [], [], []
    for race in races:
        base, _ = v2.base_map(race)
        first, second, _third = race.order
        first_line = v2.ino(race.entries[first - 1].get("line_id"))
        second_line = v2.ino(race.entries[second - 1].get("line_id"))
        second_pos = v2.ino(race.entries[second - 1].get("line_position"))
        special_weight = 1.0
        if first_line != second_line:
            special_weight *= 2.25
        if second_pos == 3:
            special_weight *= 3.0
        w = gw[race.grade]
        for a in range(1, 10):
            for b in range(1, 10):
                if a == b:
                    continue
                y = int(a == first and b == second)
                xp.append(v2.pair_vec(race, a, b, base))
                yp.append(y)
                wp.append(w * (special_weight if y else 1.0))
    params = dict(
        max_depth=3,
        max_iter=90,
        learning_rate=0.08,
        l2_regularization=0.8,
        random_state=20260918,
    )
    challenger = HistGradientBoostingClassifier(class_weight="balanced", **params)
    challenger.fit(np.asarray(xp), np.asarray(yp), sample_weight=np.asarray(wp))
    return {"core": core, "pair_challenger": challenger}


def probability_components(race: Race, models):
    comp = v3.probability_components(race, models["core"])
    base, _ = v2.base_map(race)
    pair_x = [v2.pair_vec(race, a, b, base) for a, b in comp["pairs"]]
    comp["challenger_raw"] = np.clip(
        models["pair_challenger"].predict_proba(np.asarray(pair_x))[:, 1],
        1e-12,
        None,
    )
    return comp


def joint_from_components(comp, alpha: float, beta: float, rho: float):
    scores = []
    for idx, ((a, b), raw) in enumerate(zip(comp["pairs"], comp["pair_raw"])):
        challenger = comp["challenger_raw"][idx]
        score = (
            raw
            * (comp["p1"][a - 1] ** alpha)
            * (comp["p2"][b - 1] ** beta)
            * (challenger ** rho)
        )
        scores.append(score)
    ppair = v2.normalize(np.asarray(scores, dtype=float))
    joint = []
    for (a, b), pab, cs, cond in zip(
        comp["pairs"], ppair, comp["pair_third_cars"], comp["conditionals"]
    ):
        for c, pc in zip(cs, cond):
            joint.append((a, b, c, float(pab * pc)))
    total = sum(x[3] for x in joint)
    joint = [(a, b, c, p / total) for a, b, c, p in joint]
    joint.sort(key=lambda x: x[3], reverse=True)
    return joint


def actual_relation(race: Race):
    first, second, _ = race.order
    ef = race.entries[first - 1]
    es = race.entries[second - 1]
    return {
        "same_line": v2.ino(ef.get("line_id")) == v2.ino(es.get("line_id")),
        "second_pos3": v2.ino(es.get("line_position")) == 3,
    }


def metric_rows(preds):
    base = v2.evaluate_predictions(preds)
    cross = [p for p in preds if not p["relation"]["same_line"]]
    same = [p for p in preds if p["relation"]["same_line"]]
    pos3 = [p for p in preds if p["relation"]["second_pos3"]]
    base.update({
        "same_line_full": float(np.mean([all(p["hits"]) for p in same])) if same else None,
        "cross_line_full": float(np.mean([all(p["hits"]) for p in cross])) if cross else None,
        "cross_line_second_capture": float(np.mean([p["hits"][1] for p in cross])) if cross else None,
        "second_pos3_cases": len(pos3),
        "second_pos3_capture": float(np.mean([p["hits"][1] for p in pos3])) if pos3 else None,
        "second_pos3_full": float(np.mean([all(p["hits"]) for p in pos3])) if pos3 else None,
    })
    return base


def choose_pair_config(models, tuning_races):
    comps = [probability_components(r, models) for r in tuning_races]
    rows = []
    best = None
    for alpha, beta, rho in PAIR_CONFIGS:
        scored = []
        for comp in comps:
            race = comp["race"]
            joint = joint_from_components(comp, alpha, beta, rho)
            board, mass = v2.greedy_board(joint, 7)
            scored.append({
                "race": race,
                "rows": board,
                "mass": mass,
                "hits": v2.captured(race, board),
                "special": v2.special_second_case(race, board),
                "relation": actual_relation(race),
            })
        m = metric_rows(scored)
        # Full-board remains primary. Special cases only break close ties.
        key = (
            m["full_board_capture"],
            m["cross_line_full"] or 0.0,
            m["second_pos3_capture"] or 0.0,
            m["second_capture"],
        )
        row = {"alpha": alpha, "beta": beta, "rho": rho, "metrics": m}
        rows.append(row)
        if best is None or key > best[0]:
            best = (key, row)
    rows.sort(
        key=lambda x: (
            x["metrics"]["full_board_capture"],
            x["metrics"]["cross_line_full"] or 0.0,
            x["metrics"]["second_pos3_capture"] or 0.0,
        ),
        reverse=True,
    )
    return best[1], rows


def raw_prediction(race, models, config):
    comp = probability_components(race, models)
    joint = joint_from_components(
        comp, float(config["alpha"]), float(config["beta"]), float(config["rho"])
    )
    b7, m7 = v2.greedy_board(joint, 7)
    b8, m8 = v2.greedy_board(joint, 8)
    b9, m9 = v2.greedy_board(joint, 9)
    return {
        "race": race,
        "joint": joint,
        "b7": b7, "m7": m7,
        "b8": b8, "m8": m8,
        "b9": b9, "m9": m9,
        "relation": actual_relation(race),
    }


def choose_budget_policy(raws):
    masses = np.asarray([r["m7"] for r in raws], dtype=float)
    # Low 7-piece mass = uncertainty. Lowest q9 get 9 pieces, next q8 get 8.
    candidates = []
    for q9 in np.linspace(0.0, 0.20, 11):
        for q8 in np.linspace(0.0, 0.57, 20):
            if 7.0 + 2.0 * q9 + q8 > MAX_AVG_PIECES + 1e-9:
                continue
            t9 = float(np.quantile(masses, q9)) if q9 > 0 else float(masses.min() - 1)
            t8q = min(1.0, q9 + q8)
            t8 = float(np.quantile(masses, t8q)) if q8 > 0 else t9
            preds = []
            pieces = []
            for r in raws:
                if r["m7"] <= t9:
                    board = r["b9"]; mass = r["m9"]; piece = 9
                elif r["m7"] <= t8:
                    board = r["b8"]; mass = r["m8"]; piece = 8
                else:
                    board = r["b7"]; mass = r["m7"]; piece = 7
                race = r["race"]
                preds.append({
                    "race": race,
                    "rows": board,
                    "mass": mass,
                    "hits": v2.captured(race, board),
                    "special": v2.special_second_case(race, board),
                    "relation": r["relation"],
                })
                pieces.append(piece)
            m = metric_rows(preds)
            avg = float(np.mean(pieces))
            m["avg_pieces"] = avg
            row = {
                "q9": float(q9),
                "q8": float(q8),
                "threshold9_m7": t9,
                "threshold8_m7": t8,
                "metrics": m,
            }
            key = (
                m["full_board_capture"],
                m["cross_line_full"] or 0.0,
                m["second_pos3_capture"] or 0.0,
                -avg,
            )
            candidates.append(row)
    candidates.sort(
        key=lambda x: (
            x["metrics"]["full_board_capture"],
            x["metrics"]["cross_line_full"] or 0.0,
            x["metrics"]["second_pos3_capture"] or 0.0,
            -x["metrics"]["avg_pieces"],
        ),
        reverse=True,
    )
    return candidates[0], candidates[:30]


def apply_budget(raw, policy):
    m7 = raw["m7"]
    if m7 <= float(policy["threshold9_m7"]):
        return raw["b9"], raw["m9"], 9
    if m7 <= float(policy["threshold8_m7"]):
        return raw["b8"], raw["m8"], 8
    return raw["b7"], raw["m7"], 7


def make_predictions(races, models, pair_config, budget_policy):
    out = []
    for race in races:
        raw = raw_prediction(race, models, pair_config)
        rows, mass, pieces = apply_budget(raw, budget_policy)
        out.append({
            "race": race,
            "joint": raw["joint"],
            "rows": rows,
            "mass": mass,
            "pieces": pieces,
            "hits": v2.captured(race, rows),
            "special": v2.special_second_case(race, rows),
            "relation": raw["relation"],
        })
    return out


def participation_features(p):
    base = list(v2.prediction_features(p["race"], p["joint"], p["rows"], p["mass"]))
    pair_mass = {}
    pos3_mass = 0.0
    for a, b, _c, prob in p["joint"]:
        pair_mass[(a, b)] = pair_mass.get((a, b), 0.0) + prob
    for (a, b), prob in pair_mass.items():
        if v2.ino(p["race"].entries[b - 1].get("line_position")) == 3:
            pos3_mass += prob
    ordered = sorted(pair_mass.values(), reverse=True)
    top_pair = ordered[0] if ordered else 0.0
    pair_gap = top_pair - (ordered[1] if len(ordered) > 1 else 0.0)
    return np.asarray(base + [float(pos3_mass), float(top_pair), float(pair_gap), float(p["pieces"])], dtype=float)


def choose_threshold(scores_hits):
    rows = sorted(scores_hits, key=lambda x: x[0])
    best = None
    for threshold in sorted(set(s for s, _ in rows)):
        selected = [h for s, h in rows if s >= threshold]
        n = len(selected)
        rate_sel = n / len(rows)
        if n < max(40, int(0.15 * len(rows))) or rate_sel > 0.60:
            continue
        hit = float(np.mean(selected))
        z = 1.96
        denom = 1 + z*z/n
        centre = hit + z*z/(2*n)
        lower = (centre - z*math.sqrt(hit*(1-hit)/n + z*z/(4*n*n))) / denom
        key = (lower, hit, n)
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is None:
        return 1.0
    return float(best[1])


def fit_participation(oos_preds):
    X = np.asarray([participation_features(p) for p in oos_preds])
    y = np.asarray([int(all(p["hits"])) for p in oos_preds])
    model = LogisticRegression(max_iter=2000, random_state=20260918)
    model.fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    threshold = choose_threshold(list(zip(scores, y)))
    return model, threshold


def attach_participation(preds, model, threshold):
    out = []
    for p in preds:
        score = float(model.predict_proba(participation_features(p).reshape(1, -1))[0, 1])
        out.append({
            "race_id": p["race"].race_id,
            "race_date": p["race"].race_date,
            "grade": p["race"].grade,
            "race_type": p["race"].race_type,
            "rows": p["rows"],
            "hits": p["hits"],
            "special": p["special"],
            "board_mass": p["mass"],
            "pieces": p["pieces"],
            "relation": p["relation"],
            "participation_score": score,
            "participate": score >= threshold,
        })
    return out


def evaluation(scored):
    m = metric_rows(scored)
    m["avg_pieces"] = float(np.mean([x["pieces"] for x in scored]))
    m["piece_distribution"] = {
        str(k): int(sum(x["pieces"] == k for x in scored)) for k in (7, 8, 9)
    }
    part = [x for x in scored if x["participate"]]
    m["participants"] = len(part)
    m["participation_rate"] = len(part) / len(scored)
    m["participant_full_board_capture"] = (
        float(np.mean([all(x["hits"]) for x in part])) if part else None
    )
    m["participant_cross_line_full"] = (
        float(np.mean([all(x["hits"]) for x in part if not x["relation"]["same_line"]]))
        if any(not x["relation"]["same_line"] for x in part) else None
    )
    return m


def by_grade(scored):
    out = {"ALL": evaluation(scored)}
    for grade in ("G1", "G2", "G3"):
        group = [x for x in scored if x["grade"] == grade]
        if group:
            out[grade] = evaluation(group)
    return out


def train_and_evaluate():
    races = v2.load_races()
    counts = {str(y): sum(r.year == y for r in races) for y in sorted({r.year for r in races})}

    train_pre2024, eval2024 = v2.split_before(races, 2024)
    models24 = fit_models(train_pre2024)
    tune24 = [r for r in eval2024 if r.race_date <= "2024-06-30"]
    pair_config, pair_search = choose_pair_config(models24, tune24)
    raw_tune = [raw_prediction(r, models24, pair_config) for r in tune24]
    budget_policy, budget_search = choose_budget_policy(raw_tune)
    pred24 = make_predictions(eval2024, models24, pair_config, budget_policy)

    # 2025 participation uses only 2024 out-of-sample predictions.
    part25, thr25 = fit_participation(pred24)

    train_pre2025, eval2025 = v2.split_before(races, 2025)
    models25 = fit_models(train_pre2025)
    pred25 = make_predictions(eval2025, models25, pair_config, budget_policy)
    scored25 = attach_participation(pred25, part25, thr25)

    # 2026 participation is recalibrated from 2025 OOS predictions.
    part26, thr26 = fit_participation(pred25)

    train_pre2026, eval2026 = v2.split_before(races, 2026)
    models26 = fit_models(train_pre2026)
    pred26 = make_predictions(eval2026, models26, pair_config, budget_policy)
    scored26 = attach_participation(pred26, part26, thr26)

    # Configuration-period reporting uses a model calibrated on the first half
    # and is not treated as forward evidence.
    h1pred = [p for p in pred24 if p["race"].race_date <= "2024-06-30"]
    h2pred = [p for p in pred24 if p["race"].race_date > "2024-06-30"]
    part24, thr24 = fit_participation(h1pred)
    scored24 = attach_participation(pred24, part24, thr24)

    # Final production-like bundle through 2026-06-30. Participation is fit on
    # the latest complete OOS block (2026 H1) for use on later future races.
    final_train = [r for r in races if r.race_date <= "2026-06-30"]
    final_models = fit_models(final_train)
    part_final, thr_final = fit_participation(pred26)

    bundle = {
        "models": final_models,
        "pair_config": {
            "alpha": float(pair_config["alpha"]),
            "beta": float(pair_config["beta"]),
            "rho": float(pair_config["rho"]),
        },
        "budget_policy": {
            "threshold9_m7": float(budget_policy["threshold9_m7"]),
            "threshold8_m7": float(budget_policy["threshold8_m7"]),
        },
        "participation_model": part_final,
        "participation_threshold": float(thr_final),
        "trained_through": max(r.race_date for r in final_train),
        "model_name": "ninecar_v31_pair_variable_participation",
    }

    ev24 = by_grade(scored24)
    ev25 = by_grade(scored25)
    ev26 = by_grade(scored26)
    g26 = ev26.get("G3", ev26["ALL"])
    report = {
        "schema_version": 1,
        "model": bundle["model_name"],
        "eligible_races_by_year": counts,
        "changes_from_v3": [
            "conservative weighted challenger for ordered 1-2 pair",
            "uncertainty-driven variable 7/8/9-piece greedy board",
            "rolling participation recalibration from prior OOS predictions",
        ],
        "guards": {
            "odds_used": False,
            "popularity_used": False,
            "target_result_used_at_inference": False,
            "payout_used_at_inference": False,
        },
        "selected": {
            "pair_config": bundle["pair_config"],
            "pair_search_top10": pair_search[:10],
            "budget_policy": bundle["budget_policy"],
            "budget_tuning": budget_policy["metrics"],
            "budget_search_top10": budget_search[:10],
            "participation_thresholds": {
                "2024_configuration": thr24,
                "2025_forward": thr25,
                "2026_forward": thr26,
                "production_after_2026_h1": thr_final,
            },
        },
        "evaluation": {
            "2024_configuration_period": ev24,
            "2025_forward": ev25,
            "2026_h1_forward": ev26,
        },
        "references": {
            "v3_2026_full": 0.17321997874601489,
            "v3_2026_avg_pieces": 7.0,
            "v3_2026_participation_rate": 0.822529224229543,
            "v3_2026_participant_full": 0.1847545219638243,
            "old_common923_full": 0.228,
            "old_common923_avg_pieces": 7.57,
        },
        "research_gate": {
            "full_over_v3": bool(g26["full_board_capture"] > 0.17321997874601489),
            "avg_pieces_within_old_reference": bool(g26["avg_pieces"] <= 7.57),
            "participation_below_v3": bool(g26["participation_rate"] < 0.822529224229543),
            "participant_full_over_v3": bool(
                (g26["participant_full_board_capture"] or 0) > 0.1847545219638243
            ),
            "cross_line_full": g26.get("cross_line_full"),
            "second_pos3_capture": g26.get("second_pos3_capture"),
        },
        "production_bundle_trained_through": bundle["trained_through"],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT / "model.joblib", compress=3)
    (OUT / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# ninecar_v31_pair_variable_participation\n\n"
        "v3から変更するのは1-2ペア補助、7/8/9可変盤面、OOS参加再校正の3点だけ。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def live_race_from_payload(payload: dict) -> Race:
    return v2.live_race_from_payload(payload)


def predict_live(payload: dict, model_path: Path | None = None) -> dict:
    model_path = model_path or (OUT / "model.joblib")
    bundle = joblib.load(model_path)
    race = live_race_from_payload(payload)
    config = bundle["pair_config"]
    raw = raw_prediction(race, bundle["models"], config)
    rows, mass, pieces = apply_budget(raw, bundle["budget_policy"])
    feat = participation_features({
        "race": race, "joint": raw["joint"], "rows": rows, "mass": mass, "pieces": pieces
    })
    score = float(bundle["participation_model"].predict_proba(feat.reshape(1, -1))[0, 1])
    p1, p2, p3 = v2.marginals(raw["joint"])
    return {
        "race_id": race.race_id,
        "board_generated": True,
        "board_policy": bundle["model_name"],
        "participate": score >= float(bundle["participation_threshold"]),
        "first_candidates": list(rows[0]),
        "second_candidates": list(rows[1]),
        "third_candidates": list(rows[2]),
        "piece_budget": pieces,
        "joint_board_mass": mass,
        "participation_score": score,
        "participation_threshold": float(bundle["participation_threshold"]),
        "first_ranking": [{"no": n, "probability": p} for n, p in sorted(p1.items(), key=lambda x:x[1], reverse=True)],
        "second_ranking": [{"no": n, "probability": p} for n, p in sorted(p2.items(), key=lambda x:x[1], reverse=True)],
        "third_ranking": [{"no": n, "probability": p} for n, p in sorted(p3.items(), key=lambda x:x[1], reverse=True)],
        "top_joint_triples": [
            {"first": a, "second": b, "third": c, "probability": p}
            for a,b,c,p in raw["joint"][:10]
        ],
        "versions": {"ninecar": bundle["model_name"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-evaluate", action="store_true")
    parser.add_argument("--predict-json")
    args = parser.parse_args()
    if args.train_evaluate:
        train_and_evaluate()
        return 0
    if args.predict_json:
        payload = json.loads(Path(args.predict_json).read_text(encoding="utf-8"))
        print(json.dumps(predict_live(payload), ensure_ascii=False, indent=2))
        return 0
    parser.error("choose --train-evaluate or --predict-json")


if __name__ == "__main__":
    raise SystemExit(main())

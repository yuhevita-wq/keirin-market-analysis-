#!/usr/bin/env python3
from __future__ import annotations

"""Nine-car v3: preserve ordered-pair cross-line strength and recover board capture.

This variant keeps the 504 ordered-triple architecture from v2, but adds a
standalone second-place model and calibrates the ordered pair score with both
first and second rider priors. Board construction is constrained to exactly
seven pieces so forward comparisons cannot improve merely by buying more board
coverage than the 7.57-piece historical reference.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts/keirin_shogi_ninecar_v2.py"
OUT = ROOT / "results/keirin_shogi/ninecar_v3"


def load_base():
    spec = importlib.util.spec_from_file_location("keirin_shogi_ninecar_v2_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v2 = load_base()
Race = v2.Race


def fit_models(races: list[Race]):
    gw = v2.grade_weights(races)
    xf, yf, wf = [], [], []
    xs, ys, ws = [], [], []
    xp, yp, wp = [], [], []
    xt, yt, wt = [], [], []

    for race in races:
        base, _ = v2.base_map(race)
        first, second, third = race.order
        w = gw[race.grade]
        for no in range(1, 10):
            xf.append(base[no]); yf.append(int(no == first)); wf.append(w)
            xs.append(base[no]); ys.append(int(no == second)); ws.append(w)
        for a in range(1, 10):
            for b in range(1, 10):
                if b == a:
                    continue
                xp.append(v2.pair_vec(race, a, b, base))
                yp.append(int(a == first and b == second))
                wp.append(w)
        for c in range(1, 10):
            if c in (first, second):
                continue
            xt.append(v2.third_vec(race, first, second, c, base))
            yt.append(int(c == third))
            wt.append(w)

    params = dict(
        max_depth=3,
        max_iter=90,
        learning_rate=0.08,
        l2_regularization=0.8,
        random_state=20260918,
    )
    first_model = HistGradientBoostingClassifier(class_weight="balanced", **params)
    second_model = HistGradientBoostingClassifier(class_weight="balanced", **params)
    pair_model = HistGradientBoostingClassifier(class_weight="balanced", **params)
    third_model = HistGradientBoostingClassifier(class_weight="balanced", **params)
    first_model.fit(np.asarray(xf), np.asarray(yf), sample_weight=np.asarray(wf))
    second_model.fit(np.asarray(xs), np.asarray(ys), sample_weight=np.asarray(ws))
    pair_model.fit(np.asarray(xp), np.asarray(yp), sample_weight=np.asarray(wp))
    third_model.fit(np.asarray(xt), np.asarray(yt), sample_weight=np.asarray(wt))
    return {
        "first": first_model,
        "second": second_model,
        "pair": pair_model,
        "third": third_model,
    }


def probability_components(race: Race, models):
    base, _ = v2.base_map(race)
    rider_x = np.asarray([base[n] for n in range(1, 10)])
    p1 = v2.normalize(models["first"].predict_proba(rider_x)[:, 1])
    p2 = v2.normalize(models["second"].predict_proba(rider_x)[:, 1])

    pairs, pair_x = [], []
    for a in range(1, 10):
        for b in range(1, 10):
            if a == b:
                continue
            pairs.append((a, b))
            pair_x.append(v2.pair_vec(race, a, b, base))
    pair_raw = np.clip(
        models["pair"].predict_proba(np.asarray(pair_x))[:, 1],
        1e-12,
        None,
    )

    third_x = []
    pair_third_cars = []
    for a, b in pairs:
        cs = [c for c in range(1, 10) if c not in (a, b)]
        pair_third_cars.append(cs)
        third_x.extend(v2.third_vec(race, a, b, c, base) for c in cs)
    third_raw = np.clip(
        models["third"].predict_proba(np.asarray(third_x))[:, 1],
        1e-12,
        None,
    )

    conditionals = []
    offset = 0
    for cs in pair_third_cars:
        raw = third_raw[offset:offset + len(cs)]
        offset += len(cs)
        conditionals.append(v2.normalize(raw))

    return {
        "race": race,
        "pairs": pairs,
        "p1": p1,
        "p2": p2,
        "pair_raw": pair_raw,
        "pair_third_cars": pair_third_cars,
        "conditionals": conditionals,
    }


def joint_from_components(comp, alpha: float, beta: float):
    pair_scores = np.asarray(
        [
            raw * (comp["p1"][a - 1] ** alpha) * (comp["p2"][b - 1] ** beta)
            for (a, b), raw in zip(comp["pairs"], comp["pair_raw"])
        ],
        dtype=float,
    )
    ppair = v2.normalize(pair_scores)
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


def marginal_board(joint, sizes: tuple[int, int, int]):
    p1, p2, p3 = v2.marginals(joint)
    maps = (p1, p2, p3)
    rows = []
    for probs, size in zip(maps, sizes):
        selected = sorted(probs, key=lambda n: (-probs[n], n))[:size]
        rows.append(tuple(sorted(selected)))
    rows = tuple(rows)
    return rows, v2.board_mass(joint, rows)


BOARD_POLICIES = (
    {"type": "greedy", "budget": 7},
    *(
        {"type": "marginal", "sizes": (a, b, c)}
        for a in range(1, 5)
        for b in range(1, 5)
        for c in range(1, 5)
        if a + b + c == 7
    ),
)

BLENDS = (
    (1.0, 0.0),
    (1.0, 0.5),
    (1.0, 1.0),
    (1.0, 1.5),
    (0.7, 0.5),
    (0.7, 1.0),
    (0.35, 1.0),
)


def apply_board(joint, policy):
    if policy["type"] == "greedy":
        return v2.greedy_board(joint, int(policy["budget"]))
    return marginal_board(joint, tuple(policy["sizes"]))


def evaluate_rows(rows):
    return v2.evaluate_predictions(rows)


def choose_configuration(models, tuning_races):
    components = [probability_components(race, models) for race in tuning_races]
    search = []
    best = None
    for alpha, beta in BLENDS:
        joints = [joint_from_components(comp, alpha, beta) for comp in components]
        for policy in BOARD_POLICIES:
            scored = []
            for comp, joint in zip(components, joints):
                race = comp["race"]
                rows, _ = apply_board(joint, policy)
                scored.append(
                    {
                        "rows": rows,
                        "hits": v2.captured(race, rows),
                        "special": v2.special_second_case(race, rows),
                    }
                )
            m = evaluate_rows(scored)
            key = (
                m["full_board_capture"],
                m["special_cross_line_second_capture"] or 0.0,
                m["second_capture"],
                m["first_capture"],
                m["third_capture"],
            )
            row = {
                "alpha": alpha,
                "beta": beta,
                "board_policy": policy,
                "metrics": m,
            }
            search.append(row)
            if best is None or key > best[0]:
                best = (key, row)
    search.sort(
        key=lambda r: (
            r["metrics"]["full_board_capture"],
            r["metrics"]["special_cross_line_second_capture"] or 0.0,
            r["metrics"]["second_capture"],
        ),
        reverse=True,
    )
    return best[1], search[:20]


def make_predictions(races, models, config):
    out = []
    alpha = float(config["alpha"])
    beta = float(config["beta"])
    policy = config["board_policy"]
    for race in races:
        comp = probability_components(race, models)
        joint = joint_from_components(comp, alpha, beta)
        rows, mass = apply_board(joint, policy)
        out.append(
            {
                "race": race,
                "joint": joint,
                "rows": rows,
                "mass": mass,
                "hits": v2.captured(race, rows),
                "special": v2.special_second_case(race, rows),
            }
        )
    return out


def fit_participation(preds):
    ordered = sorted(preds, key=lambda x: x["race"].race_date)
    mid = len(ordered) // 2
    train, cal = ordered[:mid], ordered[mid:]
    X = [
        v2.prediction_features(p["race"], p["joint"], p["rows"], p["mass"])
        for p in train
    ]
    y = [int(all(p["hits"])) for p in train]
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=20260918,
    )
    model.fit(np.asarray(X), np.asarray(y))

    cal_rows = []
    for p in cal:
        feat = v2.prediction_features(p["race"], p["joint"], p["rows"], p["mass"])
        score = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
        cal_rows.append((score, int(all(p["hits"]))))

    best = None
    for threshold in sorted(set(score for score, _ in cal_rows)):
        selected = [hit for score, hit in cal_rows if score >= threshold]
        if len(selected) < max(20, int(0.1 * len(cal_rows))):
            continue
        rate = float(np.mean(selected))
        n = len(selected)
        z = 1.96
        denom = 1 + z * z / n
        centre = rate + z * z / (2 * n)
        lower = (
            centre
            - z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
        ) / denom
        key = (lower, rate, n)
        if best is None or key > best[0]:
            best = (key, threshold)
    return model, float(best[1] if best else 1.0)


def attach_participation(preds, model, threshold):
    out = []
    for p in preds:
        feat = v2.prediction_features(p["race"], p["joint"], p["rows"], p["mass"])
        score = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
        out.append(
            {
                "race_id": p["race"].race_id,
                "race_date": p["race"].race_date,
                "grade": p["race"].grade,
                "rows": p["rows"],
                "hits": p["hits"],
                "special": p["special"],
                "board_mass": p["mass"],
                "participation_score": score,
                "participate": score >= threshold,
            }
        )
    return out


def by_grade(scored):
    out = {"ALL": v2.evaluate_predictions(scored)}
    for grade in ("G1", "G2", "G3"):
        subset = [row for row in scored if row["grade"] == grade]
        if subset:
            out[grade] = v2.evaluate_predictions(subset)
    return out


def train_and_evaluate():
    races = v2.load_races()
    counts = {
        str(year): sum(r.year == year for r in races)
        for year in sorted({r.year for r in races})
    }
    if not races:
        raise RuntimeError("no eligible nine-rider races")

    train_pre2024, eval2024 = v2.split_before(races, 2024)
    models_2024 = fit_models(train_pre2024)
    tune_h1 = [r for r in eval2024 if r.race_date <= "2024-06-30"]
    config, config_search = choose_configuration(models_2024, tune_h1)
    pred2024 = make_predictions(eval2024, models_2024, config)
    part_model, part_threshold = fit_participation(pred2024)
    scored2024 = attach_participation(pred2024, part_model, part_threshold)

    train_pre2025, eval2025 = v2.split_before(races, 2025)
    models_2025 = fit_models(train_pre2025)
    scored2025 = attach_participation(
        make_predictions(eval2025, models_2025, config),
        part_model,
        part_threshold,
    )

    train_pre2026, eval2026 = v2.split_before(races, 2026)
    models_2026 = fit_models(train_pre2026)
    scored2026 = attach_participation(
        make_predictions(eval2026, models_2026, config),
        part_model,
        part_threshold,
    )

    final_train = [r for r in races if r.race_date <= "2026-06-30"]
    final_models = fit_models(final_train)
    model_name = "ninecar_v3_direct_second_joint504_fixed7"
    bundle = {
        "models": final_models,
        "participation_model": part_model,
        "alpha": float(config["alpha"]),
        "beta": float(config["beta"]),
        "board_policy": config["board_policy"],
        "participation_threshold": part_threshold,
        "trained_through": max(r.race_date for r in final_train),
        "model_name": model_name,
    }

    forward_2026 = by_grade(scored2026)
    g3_2026 = forward_2026.get("G3", {})
    production_gate = {
        "full_board_reference": 0.228,
        "max_avg_pieces": 7.57,
        "min_special_cross_line_second": 0.099,
        "full_board_pass": bool(g3_2026.get("full_board_capture", 0.0) > 0.228),
        "pieces_pass": bool(g3_2026.get("avg_pieces", 99.0) <= 7.57),
        "cross_line_pass": bool(
            (g3_2026.get("special_cross_line_second_capture") or 0.0) > 0.099
        ),
    }
    production_gate["eligible"] = all(
        production_gate[key]
        for key in ("full_board_pass", "pieces_pass", "cross_line_pass")
    )

    report = {
        "schema_version": 1,
        "model": model_name,
        "eligible_races_by_year": counts,
        "training_rules": {
            "target": "9-rider S-class G1/G2/G3",
            "odds_used": False,
            "popularity_used": False,
            "target_result_used_at_inference": False,
            "ordered_pair_hypotheses": 72,
            "ordered_triple_hypotheses": 504,
            "board_piece_budget": 7,
            "second_direct_prior": True,
        },
        "selected": {
            "alpha": config["alpha"],
            "beta": config["beta"],
            "board_policy": config["board_policy"],
            "configuration_search_top20": config_search,
            "participation_threshold": part_threshold,
        },
        "evaluation": {
            "2024_configuration_period": by_grade(scored2024),
            "2025_forward": by_grade(scored2025),
            "2026_h1_forward": forward_2026,
        },
        "production_gate": production_gate,
        "reference_metrics": {
            "old_v21_v31_v37_2026_g3_common923_full_board": 0.228,
            "old_v21_v31_v37_2026_g3_common923_avg_pieces": 7.57,
            "ninecar_v1_2026_g3_special_cross_line_second_capture": 0.099,
            "ninecar_v2_2026_g3_full_board": 0.20191285866099895,
            "ninecar_v2_2026_g3_special_cross_line_second_capture": 0.3532934131736527,
        },
        "production_bundle_trained_through": bundle["trained_through"],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT / "model.joblib", compress=3)
    (OUT / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "# ninecar_v3_direct_second_joint504_fixed7\n\n"
        "v2の順序付き504仮説を維持し、直接2着モデルを順序付きペアへ合成。"
        "盤面は常に合計7駒で、2024前半にalpha/betaと行配分方式を固定し、"
        "2025・2026前半を前向き評価する。\n",
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
    comp = probability_components(race, bundle["models"])
    joint = joint_from_components(
        comp,
        float(bundle["alpha"]),
        float(bundle["beta"]),
    )
    rows, mass = apply_board(joint, bundle["board_policy"])
    feat = v2.prediction_features(race, joint, rows, mass)
    score = float(
        bundle["participation_model"].predict_proba(feat.reshape(1, -1))[0, 1]
    )
    p1, p2, p3 = v2.marginals(joint)
    return {
        "race_id": race.race_id,
        "board_generated": True,
        "board_policy": "ninecar_v3_direct_second_joint504_fixed7",
        "participate": score >= float(bundle["participation_threshold"]),
        "first_candidates": list(rows[0]),
        "second_candidates": list(rows[1]),
        "third_candidates": list(rows[2]),
        "joint_board_mass": mass,
        "participation_score": score,
        "participation_threshold": float(bundle["participation_threshold"]),
        "first_ranking": [
            {"no": n, "probability": p}
            for n, p in sorted(p1.items(), key=lambda x: x[1], reverse=True)
        ],
        "second_ranking": [
            {"no": n, "probability": p}
            for n, p in sorted(p2.items(), key=lambda x: x[1], reverse=True)
        ],
        "third_ranking": [
            {"no": n, "probability": p}
            for n, p in sorted(p3.items(), key=lambda x: x[1], reverse=True)
        ],
        "top_joint_triples": [
            {"first": a, "second": b, "third": c, "probability": p}
            for a, b, c, p in joint[:10]
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

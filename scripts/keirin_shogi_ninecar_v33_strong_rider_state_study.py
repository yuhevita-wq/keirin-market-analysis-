#!/usr/bin/env python3
from __future__ import annotations

"""Study the strongest rider's 3-way outcome in nine-car graded races.

States:
- WIN: baseline strongest rider wins.
- SOFT_FAIL: strongest rider does not win but remains 2nd/3rd.
- COLLAPSE: strongest rider finishes outside top3.

The study learns these as separate states and tests whether SOFT_FAIL signals
can improve board placement without treating COLLAPSE races as predictable
reversal opportunities.

Inference features are pre-race only. Results are labels/evaluation only.
"""

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "scripts/keirin_shogi_ninecar_v3.py"
OUT = ROOT / "results/keirin_shogi/ninecar_v33_strong_rider_state_study.json"
OUT_MD = ROOT / "results/keirin_shogi/ninecar_v33_strong_rider_state_study.md"


def load_v3():
    spec = importlib.util.spec_from_file_location("ninecar_v3_state_base", V3_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


v3 = load_v3()
v2 = v3.v2
METRICS = ("score", "win_rate", "top2_rate", "top3_rate", "s_count", "b_count",
           "nige_count", "makuri_count", "sashi_count", "mark_count",
           "first_count", "second_count", "third_count", "outside_count")
STAGES = tuple(v2.STAGE_PATTERNS)
STATE_NAMES = ("WIN", "SOFT_FAIL", "COLLAPSE")


def rank_map(vals: dict[int, float], reverse=True):
    order = sorted(vals, key=lambda n: ((-vals[n]) if reverse else vals[n], n))
    return {n: i + 1 for i, n in enumerate(order)}


def entry_map(race):
    return {v2.ino(e.get("car_no")): e for e in race.entries}


def normalize_dict(a):
    z = sum(a.values())
    return {k: (v / z if z else 0.0) for k, v in a.items()}


def base_probs(race, models):
    base, _ = v2.base_map(race)
    x = np.asarray([base[n] for n in range(1, 10)])
    p1a = v2.normalize(models["first"].predict_proba(x)[:, 1])
    p2a = v2.normalize(models["second"].predict_proba(x)[:, 1])
    return (
        {n: float(p1a[n - 1]) for n in range(1, 10)},
        {n: float(p2a[n - 1]) for n in range(1, 10)},
    )


def joint_probs(race, models):
    comp = v3.probability_components(race, models)
    return v3.joint_from_components(comp, 1.0, 0.5)


def strong_state(race, strong):
    if race.order[0] == strong:
        return 0
    if strong in race.order[1:3]:
        return 1
    return 2


def race_features(race, models):
    em = entry_map(race)
    p1, p2 = base_probs(race, models)
    joint = joint_probs(race, models)
    m1, m2, m3 = v2.marginals(joint)

    ordered = sorted(p1, key=lambda n: (-p1[n], n))
    strong = ordered[0]
    runner = ordered[1]
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

    line_mass = defaultdict(float)
    line_members = defaultdict(list)
    for n in range(1, 10):
        lid = v2.ino(em[n].get("line_id"))
        line_mass[lid] += p1[n]
        line_members[lid].append(n)

    rival_masses = [mass for lid, mass in line_mass.items() if lid != strong_line]
    strongest_rival = max(rival_masses) if rival_masses else 0.0
    strong_line_mass = line_mass[strong_line]

    # Relative/disagreement information. This is the important departure from
    # treating raw ability values as direct answers.
    strong_rank_vec = np.asarray([metric_ranks[m][strong] for m in METRICS], dtype=float)
    field_rank_spread = []
    for n in range(1, 10):
        field_rank_spread.append(float(np.std([metric_ranks[m][n] for m in METRICS])))

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
        "field_rank_disagreement": float(np.mean(field_rank_spread)),
        "strong_good_signal_count": float(sum(metric_ranks[m][strong] <= 2 for m in METRICS)),
        "strong_bad_signal_count": float(sum(metric_ranks[m][strong] >= 5 for m in METRICS)),
        "strong_line_mass": float(strong_line_mass),
        "strongest_rival_line_mass": float(strongest_rival),
        "strong_vs_rival_line_gap": float(strong_line_mass - strongest_rival),
        "strong_line_position": float(v2.ino(strong_entry.get("line_position"))),
        "strong_line_size": float(v2.ino(strong_entry.get("line_size"))),
        "strong_line_top3_count": float(sum(rank_map(p1)[n] <= 3 for n in line_members[strong_line])),
        "top2_strength_same_line": float(v2.ino(em[runner].get("line_id")) == strong_line),
        "num_lines": float(len(line_mass)),
    }

    for m in METRICS:
        feats[f"strong_{m}_rank"] = float(metric_ranks[m][strong])
        feats[f"strong_{m}_value"] = float(metric_vals[m][strong])
    for g in ("G1", "G2", "G3"):
        feats[f"grade_{g}"] = float(race.grade == g)
    for s in STAGES:
        feats[f"stage_{s}"] = float(s in race.race_type)

    return {
        "race": race,
        "strong": strong,
        "state": strong_state(race, strong),
        "features": feats,
        "p1": p1,
        "p2": p2,
        "joint": joint,
    }


def build_year(year, races):
    models = v3.fit_models([r for r in races if r.year < year])
    return [race_features(r, models) for r in races if r.year == year]


def fit_multiclass(rows):
    names = list(rows[0]["features"].keys())
    X = np.asarray([[r["features"][f] for f in names] for r in rows], dtype=float)
    y = np.asarray([r["state"] for r in rows], dtype=int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=20260918,
        ),
    )
    model.fit(X, y)
    lr = model.named_steps["logisticregression"]
    importances = {}
    for state_idx, state_name in enumerate(STATE_NAMES):
        vals = sorted(
            [{"feature": f, "coef": float(c)} for f, c in zip(names, lr.coef_[state_idx])],
            key=lambda x: abs(x["coef"]),
            reverse=True,
        )
        importances[state_name] = vals[:20]
    return model, names, importances


def score_rows(model, names, rows):
    X = np.asarray([[r["features"][f] for f in names] for r in rows], dtype=float)
    probs = model.predict_proba(X)
    out = []
    for r, p in zip(rows, probs):
        z = dict(r)
        z["state_probs"] = {STATE_NAMES[i]: float(p[i]) for i in range(3)}
        out.append(z)
    return out


def one_vs_rest_auc(scored):
    out = {}
    for i, name in enumerate(STATE_NAMES):
        y = np.asarray([int(r["state"] == i) for r in scored], dtype=int)
        s = np.asarray([r["state_probs"][name] for r in scored], dtype=float)
        out[name] = float(roc_auc_score(y, s))
    return out


def deciles(scored, state_name):
    idx = STATE_NAMES.index(state_name)
    order = sorted(scored, key=lambda r: r["state_probs"][state_name])
    chunks = np.array_split(np.arange(len(order)), 10)
    out = []
    for d, inds in enumerate(chunks, 1):
        part = [order[int(i)] for i in inds]
        out.append({
            "decile_low_to_high": d,
            "n": len(part),
            "score_mean": float(np.mean([r["state_probs"][state_name] for r in part])),
            "actual_rate": float(np.mean([r["state"] == idx for r in part])),
        })
    return out


def class_rates(rows):
    c = Counter(r["state"] for r in rows)
    n = len(rows)
    return {STATE_NAMES[i]: {"n": int(c[i]), "rate": c[i] / n} for i in range(3)}


def replace_lowest(row, rider, probs):
    row = list(row)
    if rider in row:
        return tuple(sorted(row))
    if not row:
        return (rider,)
    victim = min(row, key=lambda n: (probs[n], -n))
    row.remove(victim)
    row.append(rider)
    return tuple(sorted(row))


def remove_and_fill(row, rider, probs):
    row = list(row)
    if rider not in row:
        return tuple(sorted(row))
    row.remove(rider)
    candidates = [n for n in range(1,10) if n not in row and n != rider]
    if candidates:
        fill = max(candidates, key=lambda n: (probs[n], -n))
        row.append(fill)
    return tuple(sorted(row))


def board_interventions(scored, soft_threshold, collapse_threshold):
    strategies = ("BASE", "ADD_SECOND", "ADD_SECOND_THIRD", "DEMOTE_TO_SECOND", "DEMOTE_TO_SECOND_THIRD")
    metrics = {s: [] for s in strategies}
    selected_counts = Counter()

    for r in scored:
        race = r["race"]
        strong = r["strong"]
        joint = r["joint"]
        p1, p2, p3 = v2.marginals(joint)
        base, _mass = v2.greedy_board(joint, 7)
        base = tuple(tuple(x) for x in base)

        soft = r["state_probs"]["SOFT_FAIL"]
        collapse = r["state_probs"]["COLLAPSE"]
        selected = soft >= soft_threshold and collapse < collapse_threshold
        if selected:
            selected_counts["soft_action"] += 1

        variants = {"BASE": base}
        if selected:
            b = list(base)
            b[1] = replace_lowest(b[1], strong, p2)
            variants["ADD_SECOND"] = tuple(b)

            b = list(base)
            b[1] = replace_lowest(b[1], strong, p2)
            b[2] = replace_lowest(b[2], strong, p3)
            variants["ADD_SECOND_THIRD"] = tuple(b)

            b = list(base)
            b[0] = remove_and_fill(b[0], strong, p1)
            b[1] = replace_lowest(b[1], strong, p2)
            variants["DEMOTE_TO_SECOND"] = tuple(b)

            b = list(base)
            b[0] = remove_and_fill(b[0], strong, p1)
            b[1] = replace_lowest(b[1], strong, p2)
            b[2] = replace_lowest(b[2], strong, p3)
            variants["DEMOTE_TO_SECOND_THIRD"] = tuple(b)
        else:
            for s in strategies[1:]:
                variants[s] = base

        for s in strategies:
            hits = v2.captured(race, variants[s])
            metrics[s].append({
                "hits": hits,
                "full": bool(all(hits)),
                "state": r["state"],
                "selected": selected,
            })

    out = {}
    for s, vals in metrics.items():
        out[s] = {
            "first_capture": float(np.mean([x["hits"][0] for x in vals])),
            "second_capture": float(np.mean([x["hits"][1] for x in vals])),
            "third_capture": float(np.mean([x["hits"][2] for x in vals])),
            "full_capture": float(np.mean([x["full"] for x in vals])),
            "selected_full_capture": float(np.mean([x["full"] for x in vals if x["selected"]])) if any(x["selected"] for x in vals) else None,
            "selected_soft_fail_second_capture": float(np.mean([x["hits"][1] for x in vals if x["selected"] and x["state"] == 1])) if any(x["selected"] and x["state"] == 1 for x in vals) else None,
            "selected_soft_fail_third_capture": float(np.mean([x["hits"][2] for x in vals if x["selected"] and x["state"] == 1])) if any(x["selected"] and x["state"] == 1 for x in vals) else None,
        }
    out["selected_races"] = int(selected_counts["soft_action"])
    out["selected_rate"] = selected_counts["soft_action"] / len(scored)
    return out


def choose_thresholds(scored):
    # Tune only on 2024. Seek a usable SOFT_FAIL subset while excluding collapse.
    best = None
    grid = []
    soft_scores = np.asarray([r["state_probs"]["SOFT_FAIL"] for r in scored])
    collapse_scores = np.asarray([r["state_probs"]["COLLAPSE"] for r in scored])
    soft_qs = sorted(set(float(x) for x in np.quantile(soft_scores, [0.5,0.6,0.7,0.8,0.9])))
    col_qs = sorted(set(float(x) for x in np.quantile(collapse_scores, [0.3,0.4,0.5,0.6,0.7])))
    for st in soft_qs:
        for ct in col_qs:
            sel = [r for r in scored if r["state_probs"]["SOFT_FAIL"] >= st and r["state_probs"]["COLLAPSE"] < ct]
            if len(sel) < max(80, int(0.08 * len(scored))) or len(sel) > int(0.40 * len(scored)):
                continue
            soft_rate = float(np.mean([r["state"] == 1 for r in sel]))
            collapse_rate = float(np.mean([r["state"] == 2 for r in sel]))
            win_rate = float(np.mean([r["state"] == 0 for r in sel]))
            # We want soft-fail enrichment, but not a subset dominated by collapse.
            key = (soft_rate - collapse_rate, soft_rate, -collapse_rate, len(sel))
            row = {
                "soft_threshold": st,
                "collapse_threshold": ct,
                "n": len(sel),
                "rate": len(sel)/len(scored),
                "win_rate": win_rate,
                "soft_fail_rate": soft_rate,
                "collapse_rate": collapse_rate,
            }
            grid.append(row)
            if best is None or key > best[0]:
                best = (key, row)
    grid.sort(key=lambda x: (x["soft_fail_rate"]-x["collapse_rate"], x["soft_fail_rate"]), reverse=True)
    return best[1], grid[:20]


def base_board_metrics(rows):
    vals = []
    for r in rows:
        board, _ = v2.greedy_board(r["joint"], 7)
        hits = v2.captured(r["race"], board)
        vals.append(hits)
    return {
        "first_capture": float(np.mean([x[0] for x in vals])) if vals else None,
        "second_capture": float(np.mean([x[1] for x in vals])) if vals else None,
        "third_capture": float(np.mean([x[2] for x in vals])) if vals else None,
        "full_capture": float(np.mean([all(x) for x in vals])) if vals else None,
    }


def collapse_cut_table(scored):
    order = sorted(scored, key=lambda r: r["state_probs"]["COLLAPSE"], reverse=True)
    out = []
    for frac in (0.10, 0.20, 0.30, 0.40):
        n = max(1, int(round(len(order)*frac)))
        cut = order[:n]
        keep = order[n:]
        out.append({
            "cut_fraction": frac,
            "cut_n": n,
            "cut_actual_collapse_rate": float(np.mean([r["state"] == 2 for r in cut])),
            "cut_board": base_board_metrics(cut),
            "keep_n": len(keep),
            "keep_actual_collapse_rate": float(np.mean([r["state"] == 2 for r in keep])),
            "keep_win_rate": float(np.mean([r["state"] == 0 for r in keep])),
            "keep_soft_fail_rate": float(np.mean([r["state"] == 1 for r in keep])),
            "keep_board": base_board_metrics(keep),
        })
    return out


def eval_set(scored, thresholds):
    return {
        "n": len(scored),
        "class_rates": class_rates(scored),
        "auc": one_vs_rest_auc(scored),
        "soft_fail_deciles": deciles(scored, "SOFT_FAIL"),
        "collapse_deciles": deciles(scored, "COLLAPSE"),
        "collapse_cut_table": collapse_cut_table(scored),
        "board_interventions": board_interventions(
            scored,
            float(thresholds["soft_threshold"]),
            float(thresholds["collapse_threshold"]),
        ),
    }


def main():
    races = v2.load_races()
    rows24 = build_year(2024, races)
    rows25 = build_year(2025, races)
    rows26 = build_year(2026, races)

    model24, names24, imp24 = fit_multiclass(rows24)
    s24 = score_rows(model24, names24, rows24)
    s25 = score_rows(model24, names24, rows25)
    s26_frozen = score_rows(model24, names24, rows26)
    thresholds, threshold_grid = choose_thresholds(s24)

    # Rolling check: use 2025 OOS labels to refit the state classifier for 2026.
    model25, names25, imp25 = fit_multiclass(rows25)
    s26_roll = score_rows(model25, names25, rows26)

    report = {
        "study": "ninecar_v33_strong_rider_state",
        "state_definition": {
            "strong_rider": "rank1 from rider-only v3 first model trained only on prior years",
            "WIN": "strong rider finishes 1st",
            "SOFT_FAIL": "strong rider finishes 2nd or 3rd",
            "COLLAPSE": "strong rider finishes outside top3",
        },
        "leakage_guards": {
            "odds_used": False,
            "popularity_used": False,
            "payout_used": False,
            "results_used_as_features": False,
            "results_used_only_as_labels": True,
        },
        "2024_training_class_rates": class_rates(rows24),
        "selected_soft_fail_action_thresholds": thresholds,
        "threshold_search_top20": threshold_grid,
        "feature_importance_2024_model": imp24,
        "feature_importance_2025_rolling_model": imp25,
        "evaluation": {
            "2025_forward_frozen_2024_classifier": eval_set(s25, thresholds),
            "2026_h1_forward_frozen_2024_classifier": eval_set(s26_frozen, thresholds),
            "2026_h1_forward_rolling_2025_classifier": eval_set(s26_roll, thresholds),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    md = [
        "# 9車 強者の勝ち切り / 勝ち切れない / 崩壊 研究",
        "",
        "通常評価1位を WIN / SOFT_FAIL(2-3着) / COLLAPSE(4着以下) に分けて前向き検証。",
        "COLLAPSEは見送り候補、SOFT_FAILは下段配置候補として扱う。",
        "",
        f"2024固定SOFT_FAIL閾値: soft>={thresholds['soft_threshold']:.4f}, collapse<{thresholds['collapse_threshold']:.4f}",
    ]
    OUT_MD.write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

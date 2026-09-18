#!/usr/bin/env python3
from __future__ import annotations

"""Research when the nine-car "strength hierarchy" breaks.

The baseline hierarchy is deliberately the rider-only v3 model before ordered-pair
and conditional-third logic. The study asks:
- when does baseline 1st-ranked rider fail to win?
- when does a rider ranked 4th-or-worse by baseline win?
- when does a rider ranked 4th-or-worse by direct 2nd model finish 2nd?

Only pre-race information is used for predictors. Results are labels/evaluation only.
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
OUT = ROOT / "results/keirin_shogi/ninecar_v32_reversal_study.json"
OUT_MD = ROOT / "results/keirin_shogi/ninecar_v32_reversal_study.md"


def load_v3():
    spec = importlib.util.spec_from_file_location("ninecar_v3_reversal_base", V3_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


v3 = load_v3()
v2 = v3.v2

METRICS = ("score", "win_rate", "top2_rate", "top3_rate", "b_count", "first_count", "outside_count")
STAGES = tuple(v2.STAGE_PATTERNS)


def rank_map(values: dict[int, float], reverse=True):
    ordered = sorted(values, key=lambda n: ((-values[n]) if reverse else values[n], n))
    return {n: i + 1 for i, n in enumerate(ordered)}


def entry_map(race):
    return {v2.ino(e.get("car_no")): e for e in race.entries}


def line_pattern(race):
    counts = Counter(v2.ino(e.get("line_id")) for e in race.entries)
    return "-".join(map(str, sorted(counts.values(), reverse=True)))


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else None


def baseline_probabilities(race, models):
    base, _ = v2.base_map(race)
    x = np.asarray([base[n] for n in range(1, 10)])
    p1 = v2.normalize(models["first"].predict_proba(x)[:, 1])
    p2 = v2.normalize(models["second"].predict_proba(x)[:, 1])
    return (
        {n: float(p1[n - 1]) for n in range(1, 10)},
        {n: float(p2[n - 1]) for n in range(1, 10)},
    )


def race_features(race, models):
    em = entry_map(race)
    p1, p2 = baseline_probabilities(race, models)
    r1 = rank_map(p1)
    r2 = rank_map(p2)
    top = min(p1, key=lambda n: (-p1[n], n))
    p1_order = sorted(p1, key=lambda n: (-p1[n], n))
    second_strength = p1_order[1]

    metric_values = {
        m: {n: v2.fnum(em[n].get(m)) for n in range(1, 10)}
        for m in METRICS
    }
    metric_ranks = {m: rank_map(vals, reverse=(m != "outside_count")) for m, vals in metric_values.items()}

    top_rank_vec = np.asarray([metric_ranks[m][top] for m in METRICS], dtype=float)
    rank_disagreement = float(np.std(top_rank_vec))
    top_bad_signals = int(sum(metric_ranks[m][top] >= 5 for m in METRICS))
    top_good_signals = int(sum(metric_ranks[m][top] <= 2 for m in METRICS))

    top_entry = em[top]
    top_line = v2.ino(top_entry.get("line_id"))
    top_pos = v2.ino(top_entry.get("line_position"))
    top_line_size = v2.ino(top_entry.get("line_size"))
    same_line_top2 = int(v2.ino(em[second_strength].get("line_id")) == top_line)

    # Strength mass by line from the rider-only first model.
    line_mass = defaultdict(float)
    line_members = defaultdict(list)
    for n in range(1, 10):
        lid = v2.ino(em[n].get("line_id"))
        line_mass[lid] += p1[n]
        line_members[lid].append(n)
    top_line_mass = float(line_mass[top_line])
    rival_masses = [mass for lid, mass in line_mass.items() if lid != top_line]
    strongest_rival_mass = float(max(rival_masses) if rival_masses else 0.0)
    rival_mass_gap = top_line_mass - strongest_rival_mass

    top3 = p1_order[:3]
    top3_same_line_count = int(sum(v2.ino(em[n].get("line_id")) == top_line for n in top3))
    num_lines = len(line_mass)

    # Hidden challenger: baseline rank 4+ but at least two conventional metrics rank top3.
    hidden = []
    for n in range(1, 10):
        if r1[n] < 4:
            continue
        strong_sub = sum(metric_ranks[m][n] <= 3 for m in ("win_rate", "top2_rate", "top3_rate", "b_count", "first_count"))
        if strong_sub >= 2:
            hidden.append(n)
    hidden_count = len(hidden)
    hidden_best_p1 = max((p1[n] for n in hidden), default=0.0)

    # Pairwise gaps among the conventional hierarchy signals.
    score_order = sorted(metric_values["score"], key=lambda n: (-metric_values["score"][n], n))
    score1, score2 = score_order[:2]
    score_gap = metric_values["score"][score1] - metric_values["score"][score2]
    top_score_gap = metric_values["score"][top] - max(
        metric_values["score"][n] for n in range(1, 10) if n != top
    )
    p1_gap = p1[p1_order[0]] - p1[p1_order[1]]
    p1_top3_mass = sum(p1[n] for n in p1_order[:3])

    # How much the conventional rankings disagree across all riders.
    rider_rank_spreads = []
    for n in range(1, 10):
        rv = np.asarray([metric_ranks[m][n] for m in METRICS], dtype=float)
        rider_rank_spreads.append(float(np.std(rv)))
    field_disagreement = float(np.mean(rider_rank_spreads))

    feats = {
        "p1_top": p1[p1_order[0]],
        "p1_gap_1_2": p1_gap,
        "p1_top3_mass": p1_top3_mass,
        "score_gap_1_2": score_gap,
        "top_score_gap_vs_best_other": top_score_gap,
        "top_rank_disagreement": rank_disagreement,
        "field_rank_disagreement": field_disagreement,
        "top_bad_signal_count": float(top_bad_signals),
        "top_good_signal_count": float(top_good_signals),
        "top_line_mass": top_line_mass,
        "strongest_rival_line_mass": strongest_rival_mass,
        "top_vs_rival_line_mass_gap": rival_mass_gap,
        "top3_same_line_count": float(top3_same_line_count),
        "same_line_top2_strength": float(same_line_top2),
        "num_lines": float(num_lines),
        "hidden_challenger_count": float(hidden_count),
        "hidden_challenger_best_p1": float(hidden_best_p1),
        "top_line_position": float(top_pos),
        "top_line_size": float(top_line_size),
        "grade_G1": float(race.grade == "G1"),
        "grade_G2": float(race.grade == "G2"),
        "grade_G3": float(race.grade == "G3"),
    }
    for stage in STAGES:
        feats[f"stage_{stage}"] = float(stage in race.race_type)

    first, second, third = race.order
    labels = {
        "top_strength_wins": int(first == top),
        "top_strength_loses": int(first != top),
        "winner_rank_ge4": int(r1[first] >= 4),
        "second_rank_ge4": int(r2[second] >= 4),
        "winner_p1_rank": int(r1[first]),
        "second_p2_rank": int(r2[second]),
        "winner_score_rank": int(metric_ranks["score"][first]),
    }

    winner = em[first]
    reversal_detail = {
        "winner_no": first,
        "winner_p1_rank": int(r1[first]),
        "winner_score_rank": int(metric_ranks["score"][first]),
        "winner_line_position": v2.ino(winner.get("line_position")),
        "winner_line_size": v2.ino(winner.get("line_size")),
        "winner_same_line_as_top": bool(v2.ino(winner.get("line_id")) == top_line),
        "winner_metric_ranks": {m: int(metric_ranks[m][first]) for m in METRICS},
        "top_no": top,
        "top_line_position": top_pos,
        "top_metric_ranks": {m: int(metric_ranks[m][top]) for m in METRICS},
        "line_pattern": line_pattern(race),
    }
    return feats, labels, reversal_detail


def build_year(year, races):
    train = [r for r in races if r.year < year]
    test = [r for r in races if r.year == year]
    models = v3.fit_models(train)
    rows = []
    for race in test:
        feats, labels, detail = race_features(race, models)
        rows.append({
            "race_id": race.race_id,
            "race_date": race.race_date,
            "grade": race.grade,
            "race_type": race.race_type,
            "features": feats,
            "labels": labels,
            "detail": detail,
        })
    return rows


def binned_rate(rows, feature, target, bins=5):
    vals = np.asarray([r["features"][feature] for r in rows], dtype=float)
    if len(np.unique(vals)) <= 8:
        out = []
        for v in sorted(np.unique(vals)):
            g = [r for r in rows if r["features"][feature] == v]
            out.append({"value": float(v), "n": len(g), "rate": float(np.mean([r["labels"][target] for r in g]))})
        return out
    qs = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
    out = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        if hi == qs[-1]:
            g = [r for r in rows if lo <= r["features"][feature] <= hi]
        else:
            g = [r for r in rows if lo <= r["features"][feature] < hi]
        if g:
            out.append({
                "low": float(lo), "high": float(hi), "n": len(g),
                "rate": float(np.mean([r["labels"][target] for r in g])),
            })
    return out


def descriptive(rows):
    n = len(rows)
    reversals = [r for r in rows if r["labels"]["winner_rank_ge4"]]
    top_losses = [r for r in rows if r["labels"]["top_strength_loses"]]
    weak_seconds = [r for r in rows if r["labels"]["second_rank_ge4"]]
    win_pos = Counter(str(r["detail"]["winner_line_position"]) for r in reversals)
    win_same = Counter(str(r["detail"]["winner_same_line_as_top"]) for r in reversals)
    patterns = Counter(r["detail"]["line_pattern"] for r in reversals)
    metric_rank_means = {
        m: float(np.mean([r["detail"]["winner_metric_ranks"][m] for r in reversals])) if reversals else None
        for m in METRICS
    }
    conditions = {}
    for f in (
        "p1_gap_1_2", "p1_top", "score_gap_1_2", "top_rank_disagreement",
        "field_rank_disagreement", "top_line_mass", "strongest_rival_line_mass",
        "top_vs_rival_line_mass_gap", "top3_same_line_count", "same_line_top2_strength",
        "num_lines", "hidden_challenger_count", "top_line_position", "top_line_size",
    ):
        conditions[f] = {
            "top_strength_loses": binned_rate(rows, f, "top_strength_loses"),
            "winner_rank_ge4": binned_rate(rows, f, "winner_rank_ge4"),
            "second_rank_ge4": binned_rate(rows, f, "second_rank_ge4"),
        }
    return {
        "races": n,
        "top_strength_win_rate": float(np.mean([r["labels"]["top_strength_wins"] for r in rows])),
        "top_strength_lose_rate": len(top_losses) / n,
        "winner_rank_ge4_rate": len(reversals) / n,
        "second_rank_ge4_rate": len(weak_seconds) / n,
        "winner_p1_rank_distribution": dict(Counter(str(r["labels"]["winner_p1_rank"]) for r in rows)),
        "second_p2_rank_distribution": dict(Counter(str(r["labels"]["second_p2_rank"]) for r in rows)),
        "reversal_winner_line_position": dict(win_pos),
        "reversal_winner_same_line_as_top": dict(win_same),
        "reversal_line_patterns_top10": patterns.most_common(10),
        "reversal_winner_metric_rank_means": metric_rank_means,
        "condition_tables": conditions,
    }


def fit_reversal_models(train_rows, eval_sets):
    feature_names = list(train_rows[0]["features"].keys())
    X = np.asarray([[r["features"][f] for f in feature_names] for r in train_rows], dtype=float)
    out = {"features": feature_names, "targets": {}}
    for target in ("top_strength_loses", "winner_rank_ge4", "second_rank_ge4"):
        y = np.asarray([r["labels"][target] for r in train_rows], dtype=int)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=20260918),
        )
        model.fit(X, y)
        coefs = model.named_steps["logisticregression"].coef_[0]
        importance = sorted(
            [{"feature": f, "coef": float(c)} for f, c in zip(feature_names, coefs)],
            key=lambda x: abs(x["coef"]), reverse=True,
        )
        ev = {}
        for name, rows in eval_sets.items():
            Xe = np.asarray([[r["features"][f] for f in feature_names] for r in rows], dtype=float)
            ye = np.asarray([r["labels"][target] for r in rows], dtype=int)
            score = model.predict_proba(Xe)[:, 1]
            # Deciles to see whether "collapse risk" is rankable forward.
            order = np.argsort(score)
            chunks = np.array_split(order, 10)
            dec = []
            for i, idx in enumerate(chunks, 1):
                if len(idx) == 0:
                    continue
                dec.append({
                    "decile_low_to_high": i,
                    "n": int(len(idx)),
                    "score_mean": float(np.mean(score[idx])),
                    "actual_rate": float(np.mean(ye[idx])),
                })
            ev[name] = {
                "n": len(rows),
                "base_rate": float(np.mean(ye)),
                "auc": safe_auc(ye, score),
                "deciles": dec,
            }
        out["targets"][target] = {
            "coefficient_importance_top15": importance[:15],
            "evaluation": ev,
        }
    return out


def stable_condition_signals(desc25, desc26):
    # Surface conditions whose highest-risk bucket is worse than lowest-risk
    # bucket in the same direction in both 2025 and 2026.
    rows = []
    for f in desc25["condition_tables"]:
        for target in ("top_strength_loses", "winner_rank_ge4", "second_rank_ge4"):
            a = desc25["condition_tables"][f][target]
            b = desc26["condition_tables"][f][target]
            if len(a) < 2 or len(b) < 2:
                continue
            def spread(tbl):
                rates = [x["rate"] for x in tbl]
                return max(rates) - min(rates), int(np.argmax(rates)), int(np.argmin(rates))
            sa, hia, loa = spread(a)
            sb, hib, lob = spread(b)
            rows.append({
                "feature": f,
                "target": target,
                "2025_spread": float(sa),
                "2026_spread": float(sb),
                "2025_high_risk_bucket": a[hia],
                "2025_low_risk_bucket": a[loa],
                "2026_high_risk_bucket": b[hib],
                "2026_low_risk_bucket": b[lob],
                "min_spread": float(min(sa, sb)),
            })
    rows.sort(key=lambda x: x["min_spread"], reverse=True)
    return rows[:30]


def main():
    races = v2.load_races()
    rows24 = build_year(2024, races)
    rows25 = build_year(2025, races)
    rows26 = build_year(2026, races)

    desc24 = descriptive(rows24)
    desc25 = descriptive(rows25)
    desc26 = descriptive(rows26)
    models = fit_reversal_models(rows24, {"2025_forward": rows25, "2026_h1_forward": rows26})
    stable = stable_condition_signals(desc25, desc26)

    report = {
        "study": "ninecar_v32_strength_hierarchy_reversal",
        "definition": {
            "baseline": "v3 rider-only first/second models using current conventional pre-race features before pair/triple logic",
            "top_strength_loses": "baseline p1 rank1 rider does not finish 1st",
            "winner_rank_ge4": "actual winner was baseline p1 rank 4-9",
            "second_rank_ge4": "actual 2nd was direct-second p2 rank 4-9",
        },
        "leakage_guards": {
            "odds_used": False,
            "popularity_used": False,
            "payout_used": False,
            "results_used_as_features": False,
            "results_used_only_as_labels": True,
            "year_models_train_only_on_prior_years": True,
        },
        "descriptive": {
            "2024": desc24,
            "2025_forward": desc25,
            "2026_h1_forward": desc26,
        },
        "reversal_classifier": models,
        "stable_condition_signals_2025_2026": stable,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# 9車 v3.2 序列崩壊・逆転条件研究",
        "",
        "現行の強さベース順位を基準とし、その順位が崩れる条件を別問題として研究する。",
        "結果・オッズ・人気・払戻は推論特徴に使わない。",
        "",
        "## 主要率",
    ]
    for label, d in (("2024", desc24), ("2025 forward", desc25), ("2026 H1 forward", desc26)):
        md.append(
            f"- {label}: 強さ1位勝率 {d['top_strength_win_rate']:.3f}, "
            f"4位以下評価の勝利 {d['winner_rank_ge4_rate']:.3f}, "
            f"2着モデル4位以下の実2着 {d['second_rank_ge4_rate']:.3f}"
        )
    md += ["", "## 2025/2026で安定して差が大きい条件"]
    for x in stable[:12]:
        md.append(
            f"- {x['target']} / {x['feature']}: "
            f"spread 2025={x['2025_spread']:.3f}, 2026={x['2026_spread']:.3f}"
        )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

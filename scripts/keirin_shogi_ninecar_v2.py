#!/usr/bin/env python3
from __future__ import annotations

"""Train/evaluate the dedicated nine-rider Keirin Shogi joint-board model.

Design goals:
- S-class, 9-rider G1/G2/G3 only.
- No odds/popularity/result fields at inference time.
- Model ordered (1st, 2nd) pairs, then 3rd conditional on the ordered pair.
- Enumerate all 9*8*7 = 504 legal ordered triples at prediction time.
- Build the three board rows by maximizing covered joint probability mass.
- Explicitly evaluate 2nd-place capture when the real 2nd is outside the first
  row and comes from a different line.
"""

import argparse
import csv
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/grade_races"
OUT = ROOT / "results/keirin_shogi/ninecar_v2"

NUMERIC = (
    "score", "win_rate", "top2_rate", "top3_rate", "s_count", "b_count",
    "nige_count", "makuri_count", "sashi_count", "mark_count",
    "first_count", "second_count", "third_count", "outside_count",
    "age", "term", "line_position", "line_size",
)
Z_FIELDS = ("score", "win_rate", "top2_rate", "top3_rate", "b_count", "first_count", "outside_count")
STYLE_VALUES = ("逃", "両", "追")
CLASS_VALUES = ("SS", "S1", "S2")
STAGE_PATTERNS = (
    "一次", "二次", "予選", "選抜", "特選", "準決", "決勝", "優秀", "ドリーム", "一般",
)


@dataclass
class Race:
    race_id: str
    race_date: str
    grade: str
    race_type: str
    entries: list[dict]
    order: tuple[int, int, int]

    @property
    def year(self) -> int:
        return int(self.race_date[:4])


def fnum(value, default=0.0) -> float:
    try:
        x = float(str(value).replace("%", "").strip())
        return x if math.isfinite(x) else default
    except Exception:
        return default


def ino(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def archive_paths() -> list[Path]:
    return sorted(DATA_ROOT.rglob("*.zip"))


def read_csv_from_zip(z: zipfile.ZipFile, name: str) -> list[dict]:
    try:
        raw = z.read(name)
    except KeyError:
        return []
    text = raw.decode("utf-8-sig", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def is_s_class(entry: dict) -> bool:
    return str(entry.get("class", "")).strip().upper() in CLASS_VALUES


def valid_line(entry: dict) -> bool:
    return ino(entry.get("line_id")) > 0 and ino(entry.get("line_position")) > 0


def load_races() -> list[Race]:
    entries_by: dict[str, dict[int, dict]] = {}
    meta_by: dict[str, dict] = {}
    result_by: dict[str, dict[int, int]] = {}

    for path in archive_paths():
        try:
            with zipfile.ZipFile(path) as z:
                entries = read_csv_from_zip(z, "entries.csv")
                races = read_csv_from_zip(z, "races.csv")
                results = read_csv_from_zip(z, "results.csv")
        except zipfile.BadZipFile:
            continue

        for row in races:
            rid = str(row.get("race_id", ""))
            if not rid:
                continue
            grade = str(row.get("meeting_grade") or row.get("event_grade") or "").upper()
            if grade not in {"G1", "G2", "G3"}:
                continue
            meta_by[rid] = row

        for row in entries:
            rid = str(row.get("race_id", ""))
            if rid not in meta_by:
                continue
            entries_by.setdefault(rid, {})[ino(row.get("car_no"))] = row

        for row in results:
            rid = str(row.get("race_id", ""))
            if rid not in meta_by:
                continue
            no = ino(row.get("car_no"))
            pos = ino(row.get("finish_position"))
            if no > 0 and pos > 0:
                result_by.setdefault(rid, {})[no] = pos

    out: list[Race] = []
    for rid, meta in meta_by.items():
        entries = list(entries_by.get(rid, {}).values())
        entries.sort(key=lambda r: ino(r.get("car_no")))
        if len(entries) != 9 or {ino(r.get("car_no")) for r in entries} != set(range(1, 10)):
            continue
        if not all(is_s_class(r) and valid_line(r) for r in entries):
            continue
        positions = result_by.get(rid, {})
        top = []
        for p in (1, 2, 3):
            cars = [no for no, pos in positions.items() if pos == p]
            if len(cars) != 1:
                top = []
                break
            top.append(cars[0])
        if len(top) != 3 or len(set(top)) != 3:
            continue
        race_date = str(meta.get("race_date", ""))
        if not re.fullmatch(r"20\d\d-\d\d-\d\d", race_date):
            continue
        out.append(
            Race(
                race_id=rid,
                race_date=race_date,
                grade=str(meta.get("meeting_grade") or meta.get("event_grade") or "").upper(),
                race_type=str(meta.get("race_type", "")),
                entries=entries,
                order=(top[0], top[1], top[2]),
            )
        )
    out.sort(key=lambda r: (r.race_date, r.race_id))
    return out


def race_context(race: Race):
    entries = race.entries
    means = {k: np.mean([fnum(e.get(k)) for e in entries]) for k in Z_FIELDS}
    stds = {k: np.std([fnum(e.get(k)) for e in entries]) for k in Z_FIELDS}
    line_sizes: dict[int, int] = {}
    for e in entries:
        lid = ino(e.get("line_id"))
        line_sizes[lid] = max(line_sizes.get(lid, 0), ino(e.get("line_size")))
    grade_flags = [float(race.grade == g) for g in ("G1", "G2", "G3")]
    stage_flags = [float(p in race.race_type) for p in STAGE_PATTERNS]
    return means, stds, line_sizes, grade_flags, stage_flags


def rider_vec(entry: dict, ctx) -> np.ndarray:
    means, stds, _line_sizes, grade_flags, stage_flags = ctx
    vals = [fnum(entry.get(k)) for k in NUMERIC]
    z = []
    for k in Z_FIELDS:
        sd = stds[k]
        z.append((fnum(entry.get(k)) - means[k]) / sd if sd > 1e-9 else 0.0)
    style = [float(str(entry.get("style", "")).strip() == s) for s in STYLE_VALUES]
    cls = [float(str(entry.get("class", "")).strip().upper() == c) for c in CLASS_VALUES]
    return np.asarray(vals + z + style + cls + grade_flags + stage_flags, dtype=np.float32)


def base_map(race: Race):
    ctx = race_context(race)
    return {ino(e["car_no"]): rider_vec(e, ctx) for e in race.entries}, ctx


def pair_vec(race: Race, a: int, b: int, base=None) -> np.ndarray:
    if base is None:
        base, _ = base_map(race)
    ea = race.entries[a - 1]
    eb = race.entries[b - 1]
    va, vb = base[a], base[b]
    rel = [
        float(ino(ea.get("line_id")) == ino(eb.get("line_id"))),
        float(ino(ea.get("line_id")) != ino(eb.get("line_id"))),
        float(ino(eb.get("line_position")) == ino(ea.get("line_position")) + 1),
        fnum(ea.get("score")) - fnum(eb.get("score")),
        fnum(ea.get("top2_rate")) - fnum(eb.get("top2_rate")),
        fnum(ea.get("b_count")) - fnum(eb.get("b_count")),
        ino(ea.get("line_position")),
        ino(eb.get("line_position")),
        ino(ea.get("line_size")),
        ino(eb.get("line_size")),
    ]
    return np.concatenate([va, vb, np.asarray(rel, dtype=np.float32)])


def third_vec(race: Race, a: int, b: int, c: int, base=None) -> np.ndarray:
    if base is None:
        base, _ = base_map(race)
    ea, eb, ec = race.entries[a - 1], race.entries[b - 1], race.entries[c - 1]
    pair = pair_vec(race, a, b, base)
    vc = base[c]
    rel = [
        float(ino(ec.get("line_id")) == ino(ea.get("line_id"))),
        float(ino(ec.get("line_id")) == ino(eb.get("line_id"))),
        float(ino(ec.get("line_id")) != ino(ea.get("line_id")) and ino(ec.get("line_id")) != ino(eb.get("line_id"))),
        fnum(ec.get("score")) - fnum(ea.get("score")),
        fnum(ec.get("score")) - fnum(eb.get("score")),
        ino(ec.get("line_position")),
        ino(ec.get("line_size")),
    ]
    return np.concatenate([pair, vc, np.asarray(rel, dtype=np.float32)])


def grade_weights(races: list[Race]) -> dict[str, float]:
    counts = {g: sum(r.grade == g for r in races) for g in ("G1", "G2", "G3")}
    present = [v for v in counts.values() if v > 0]
    target = math.sqrt(max(present)) if present else 1.0
    return {g: (target / math.sqrt(n) if n else 1.0) for g, n in counts.items()}


def fit_models(races: list[Race]):
    gw = grade_weights(races)
    xf, yf, wf = [], [], []
    xp, yp, wp = [], [], []
    xt, yt, wt = [], [], []

    for race in races:
        base, _ = base_map(race)
        first, second, third = race.order
        w = gw[race.grade]
        for no in range(1, 10):
            xf.append(base[no]); yf.append(int(no == first)); wf.append(w)
        for a in range(1, 10):
            for b in range(1, 10):
                if b == a:
                    continue
                xp.append(pair_vec(race, a, b, base))
                yp.append(int(a == first and b == second))
                wp.append(w)
        for c in range(1, 10):
            if c in (first, second):
                continue
            xt.append(third_vec(race, first, second, c, base))
            yt.append(int(c == third))
            wt.append(w)

    params = dict(max_depth=3, max_iter=90, learning_rate=0.08, l2_regularization=0.8, random_state=20260918)
    first_model = HistGradientBoostingClassifier(class_weight="balanced", **params)
    pair_model = HistGradientBoostingClassifier(class_weight="balanced", **params)
    third_model = HistGradientBoostingClassifier(class_weight="balanced", **params)
    first_model.fit(np.asarray(xf), np.asarray(yf), sample_weight=np.asarray(wf))
    pair_model.fit(np.asarray(xp), np.asarray(yp), sample_weight=np.asarray(wp))
    third_model.fit(np.asarray(xt), np.asarray(yt), sample_weight=np.asarray(wt))
    return {"first": first_model, "pair": pair_model, "third": third_model}


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 1e-12, None)
    s = float(values.sum())
    return values / s if s > 0 else np.full_like(values, 1.0 / len(values))


def predict_joint(race: Race, models, alpha: float):
    base, _ = base_map(race)
    first_x = np.asarray([base[n] for n in range(1, 10)])
    first_raw = models["first"].predict_proba(first_x)[:, 1]
    p1 = normalize(first_raw)

    pairs, pair_x = [], []
    for a in range(1, 10):
        for b in range(1, 10):
            if a == b:
                continue
            pairs.append((a, b))
            pair_x.append(pair_vec(race, a, b, base))
    pair_raw = models["pair"].predict_proba(np.asarray(pair_x))[:, 1]
    if alpha:
        pair_raw = pair_raw * np.asarray([p1[a - 1] ** alpha for a, _ in pairs])
    ppair = normalize(pair_raw)

    # Score all 504 conditional third hypotheses in one batch. This is
    # mathematically identical to 72 small predict_proba calls but much faster.
    third_meta = []
    third_x = []
    for pair_index, (a, b) in enumerate(pairs):
        for c in range(1, 10):
            if c in (a, b):
                continue
            third_meta.append((pair_index, a, b, c))
            third_x.append(third_vec(race, a, b, c, base))
    third_raw = models["third"].predict_proba(np.asarray(third_x))[:, 1]

    joint = []
    offset = 0
    for pair_index, ((a, b), pab) in enumerate(zip(pairs, ppair)):
        cs = [c for c in range(1, 10) if c not in (a, b)]
        raw = third_raw[offset:offset + len(cs)]
        offset += len(cs)
        pt = normalize(raw)
        for c, pc in zip(cs, pt):
            joint.append((a, b, c, float(pab * pc)))
    total = sum(x[3] for x in joint)
    if total <= 0:
        raise RuntimeError("joint probability mass is zero")
    joint = [(a, b, c, p / total) for a, b, c, p in joint]
    joint.sort(key=lambda x: x[3], reverse=True)
    return joint


def marginals(joint):
    p1 = {n: 0.0 for n in range(1, 10)}
    p2 = {n: 0.0 for n in range(1, 10)}
    p3 = {n: 0.0 for n in range(1, 10)}
    for a, b, c, p in joint:
        p1[a] += p; p2[b] += p; p3[c] += p
    return p1, p2, p3


def board_mass(joint, rows) -> float:
    a, b, c = rows
    return float(sum(p for x, y, z, p in joint if x in a and y in b and z in c))


def greedy_board(joint, budget: int):
    best = joint[0]
    rows = [set([best[0]]), set([best[1]]), set([best[2]])]
    while sum(len(r) for r in rows) < budget:
        current = board_mass(joint, rows)
        best_key, best_move = None, None
        for ri in range(3):
            for no in range(1, 10):
                if no in rows[ri]:
                    continue
                trial = [set(x) for x in rows]
                trial[ri].add(no)
                gain = board_mass(joint, trial) - current
                key = (gain, -ri, -no)
                if best_key is None or key > best_key:
                    best_key = key
                    best_move = (ri, no)
        if best_move is None:
            break
        rows[best_move[0]].add(best_move[1])
    return tuple(tuple(sorted(r)) for r in rows), board_mass(joint, rows)


def entropy(probs: Iterable[float]) -> float:
    arr = np.asarray(list(probs), dtype=float)
    arr = arr[arr > 0]
    return float(-(arr * np.log(arr)).sum())


def prediction_features(race: Race, joint, rows, mass):
    p1, p2, p3 = marginals(joint)
    pair_mass = {}
    cross_pair_mass = 0.0
    for a, b, _c, p in joint:
        pair_mass[(a, b)] = pair_mass.get((a, b), 0.0) + p
    for (a, b), p in pair_mass.items():
        if ino(race.entries[a-1].get("line_id")) != ino(race.entries[b-1].get("line_id")):
            cross_pair_mass += p
    top = joint[0][3]
    return np.asarray([
        mass,
        top,
        entropy(p1.values()) / math.log(9),
        entropy(p2.values()) / math.log(9),
        entropy(p3.values()) / math.log(9),
        entropy(pair_mass.values()) / math.log(72),
        cross_pair_mass,
        len(rows[0]), len(rows[1]), len(rows[2]),
        len(set(rows[0]) & set(rows[1])),
        len(set(rows[1]) & set(rows[2])),
        float(race.grade == "G1"), float(race.grade == "G2"), float(race.grade == "G3"),
    ], dtype=float)


def captured(race: Race, rows):
    a, b, c = race.order
    return a in rows[0], b in rows[1], c in rows[2]


def special_second_case(race: Race, rows):
    first, second, _ = race.order
    first_ok = first in rows[0]
    outside_first = second not in rows[0]
    different_line = ino(race.entries[first-1].get("line_id")) != ino(race.entries[second-1].get("line_id"))
    return first_ok and outside_first and different_line


def evaluate_predictions(preds):
    n = len(preds)
    if not n:
        return {}
    c1 = sum(x["hits"][0] for x in preds)
    c2 = sum(x["hits"][1] for x in preds)
    c3 = sum(x["hits"][2] for x in preds)
    full = sum(all(x["hits"]) for x in preds)
    special = [x for x in preds if x["special"]]
    special_hit = sum(x["hits"][1] for x in special)
    part = [x for x in preds if x.get("participate")]
    return {
        "races": n,
        "first_capture": c1 / n,
        "second_capture": c2 / n,
        "third_capture": c3 / n,
        "full_board_capture": full / n,
        "avg_pieces": float(np.mean([sum(len(r) for r in x["rows"]) for x in preds])),
        "special_cross_line_second_cases": len(special),
        "special_cross_line_second_capture": special_hit / len(special) if special else None,
        "participants": len(part),
        "participation_rate": len(part) / n,
        "participant_full_board_capture": (
            sum(all(x["hits"]) for x in part) / len(part) if part else None
        ),
    }


def build_raw_predictions(races, models, alpha):
    out = []
    for race in races:
        joint = predict_joint(race, models, alpha)
        b7, m7 = greedy_board(joint, 7)
        b8, m8 = greedy_board(joint, 8)
        out.append({"race": race, "joint": joint, "b7": b7, "m7": m7, "b8": b8, "m8": m8})
    return out


def choose_alpha(train_models, tuning_races):
    candidates = (0.0, 0.35, 0.7, 1.0)
    scored = []
    for alpha in candidates:
        preds = build_raw_predictions(tuning_races, train_models, alpha)
        rows = []
        for p in preds:
            hits = captured(p["race"], p["b7"])
            rows.append({"rows": p["b7"], "hits": hits, "special": special_second_case(p["race"], p["b7"])})
        metric = evaluate_predictions(rows)
        scored.append((metric["full_board_capture"], metric["second_capture"], alpha))
    scored.sort(reverse=True)
    return scored[0][2], [{"alpha": a, "full": f, "second": s} for f, s, a in scored]


def choose_budget_policy(raw_preds):
    # Search a 7/8-piece uncertainty policy. 8 pieces are allowed for at most
    # 57% of races, matching the old reference average of about 7.57 pieces.
    vals = np.asarray([p["m7"] for p in raw_preds])
    candidates = []
    for direction in ("low", "high"):
        for q in np.linspace(0.0, 0.57, 20):
            if q <= 0:
                threshold = float(vals.min() - 1.0) if direction == "low" else float(vals.max() + 1.0)
            else:
                threshold = float(np.quantile(vals, q if direction == "low" else 1.0 - q))
            rows = []
            pieces = []
            for p in raw_preds:
                use8 = p["m7"] <= threshold if direction == "low" else p["m7"] >= threshold
                board = p["b8"] if use8 else p["b7"]
                pieces.append(8 if use8 else 7)
                rows.append({"rows": board, "hits": captured(p["race"], board), "special": special_second_case(p["race"], board)})
            m = evaluate_predictions(rows)
            if float(np.mean(pieces)) <= 7.571:
                candidates.append((m["full_board_capture"], m["second_capture"], -float(np.mean(pieces)), direction, threshold, m))
    candidates.sort(reverse=True)
    best = candidates[0]
    return {"direction": best[3], "threshold": best[4]}, best[5]


def apply_budget(raw, policy):
    use8 = raw["m7"] <= policy["threshold"] if policy["direction"] == "low" else raw["m7"] >= policy["threshold"]
    return (raw["b8"], raw["m8"]) if use8 else (raw["b7"], raw["m7"])


def fit_participation(raw_preds, policy):
    ordered = sorted(raw_preds, key=lambda x: x["race"].race_date)
    mid = len(ordered) // 2
    train, cal = ordered[:mid], ordered[mid:]
    X, y = [], []
    for p in train:
        rows, mass = apply_budget(p, policy)
        X.append(prediction_features(p["race"], p["joint"], rows, mass))
        y.append(int(all(captured(p["race"], rows))))
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260918)
    model.fit(np.asarray(X), np.asarray(y))

    cal_rows = []
    for p in cal:
        rows, mass = apply_budget(p, policy)
        score = float(model.predict_proba(prediction_features(p["race"], p["joint"], rows, mass).reshape(1, -1))[0,1])
        cal_rows.append((score, int(all(captured(p["race"], rows)))))
    thresholds = sorted(set(s for s, _ in cal_rows))
    best = None
    for t in thresholds:
        selected = [y for s, y in cal_rows if s >= t]
        if len(selected) < max(20, int(0.1 * len(cal_rows))):
            continue
        rate = float(np.mean(selected))
        n = len(selected)
        z = 1.96
        denom = 1 + z*z/n
        centre = rate + z*z/(2*n)
        lower = (centre - z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))) / denom
        key = (lower, rate, n)
        if best is None or key > best[0]:
            best = (key, t)
    threshold = best[1] if best else 1.0
    return model, float(threshold)


def score_dataset(races, models, alpha, policy, part_model=None, part_threshold=1.0):
    raw = build_raw_predictions(races, models, alpha)
    out = []
    for p in raw:
        rows, mass = apply_budget(p, policy)
        feats = prediction_features(p["race"], p["joint"], rows, mass)
        part_score = float(part_model.predict_proba(feats.reshape(1,-1))[0,1]) if part_model is not None else 0.0
        out.append({
            "race_id": p["race"].race_id,
            "race_date": p["race"].race_date,
            "grade": p["race"].grade,
            "rows": rows,
            "hits": captured(p["race"], rows),
            "special": special_second_case(p["race"], rows),
            "board_mass": mass,
            "participation_score": part_score,
            "participate": bool(part_score >= part_threshold) if part_model is not None else False,
        })
    return out


def split_before(races, year: int):
    return [r for r in races if r.year < year], [r for r in races if r.year == year]


def train_and_evaluate():
    races = load_races()
    counts = {str(y): sum(r.year == y for r in races) for y in sorted({r.year for r in races})}
    if not races:
        raise RuntimeError("no eligible nine-rider races")

    train_pre2024, eval2024 = split_before(races, 2024)
    models_2024 = fit_models(train_pre2024)
    # Use H1 to pick joint pair/first blend; all of 2024 may tune board sizing.
    tune_h1 = [r for r in eval2024 if r.race_date <= "2024-06-30"]
    alpha, alpha_search = choose_alpha(models_2024, tune_h1)
    raw2024 = build_raw_predictions(eval2024, models_2024, alpha)
    policy, policy_metric = choose_budget_policy(raw2024)
    part_model, part_threshold = fit_participation(raw2024, policy)
    scored2024 = score_dataset(eval2024, models_2024, alpha, policy, part_model, part_threshold)

    train_pre2025, eval2025 = split_before(races, 2025)
    models_2025 = fit_models(train_pre2025)
    scored2025 = score_dataset(eval2025, models_2025, alpha, policy, part_model, part_threshold)

    train_pre2026, eval2026 = split_before(races, 2026)
    models_2026 = fit_models(train_pre2026)
    scored2026 = score_dataset(eval2026, models_2026, alpha, policy, part_model, part_threshold)

    # Final production bundle includes every eligible historical race through 2026-06-30.
    final_train = [r for r in races if r.race_date <= "2026-06-30"]
    final_models = fit_models(final_train)
    bundle = {
        "models": final_models,
        "participation_model": part_model,
        "alpha": alpha,
        "budget_policy": policy,
        "participation_threshold": part_threshold,
        "trained_through": max(r.race_date for r in final_train),
        "model_name": "ninecar_v2_joint504",
    }

    def by_grade(scored):
        out = {"ALL": evaluate_predictions(scored)}
        for g in ("G1","G2","G3"):
            subset = [x for x in scored if x["grade"] == g]
            if subset:
                out[g] = evaluate_predictions(subset)
        return out

    report = {
        "schema_version": 1,
        "model": "ninecar_v2_joint504",
        "eligible_races_by_year": counts,
        "training_rules": {
            "target": "9-rider S-class G1/G2/G3",
            "odds_used": False,
            "popularity_used": False,
            "target_result_used_at_inference": False,
            "ordered_pair_hypotheses": 72,
            "ordered_triple_hypotheses": 504,
        },
        "selected": {
            "alpha": alpha,
            "alpha_search": alpha_search,
            "budget_policy": policy,
            "budget_tuning_2024": policy_metric,
            "participation_threshold": part_threshold,
        },
        "evaluation": {
            "2024_configuration_period": by_grade(scored2024),
            "2025_forward": by_grade(scored2025),
            "2026_h1_forward": by_grade(scored2026),
        },
        "reference_metrics": {
            "old_v21_v31_v37_2026_g3_common923_full_board": 0.228,
            "old_v21_v31_v37_2026_g3_common923_avg_pieces": 7.57,
            "ninecar_v1_2026_g3_full_board": 0.176,
            "ninecar_v1_2026_g3_special_cross_line_second_capture": 0.099,
            "note": "Reference values are from the previously audited reports; populations are not asserted identical to this script's eligible set.",
        },
        "production_bundle_trained_through": bundle["trained_through"],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT / "model.joblib", compress=3)
    (OUT / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# ninecar_v2_joint504\n\n"
        "9車S級G1/G2/G3専用。72通りの順序付き1-2着と条件付き3着を組み合わせ、"
        "504通りの同時分布から盤面全体の被覆確率を最大化する。"
        "予測入力にオッズ・人気・当該結果を使用しない。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def live_race_from_payload(payload: dict) -> Race:
    entries = [dict(e) for e in payload.get("entries", [])]
    entries.sort(key=lambda x: ino(x.get("car_no")))
    grade = str(payload.get("meeting_grade", "")).upper()
    if len(entries) != 9 or grade not in {"G1","G2","G3"} or not all(is_s_class(e) and valid_line(e) for e in entries):
        raise ValueError("ninecar_v2 target is 9-rider S-class G1/G2/G3 with valid line data")
    return Race(
        race_id=str(payload.get("race_id", "")),
        race_date=str(payload.get("race_date", date.today().isoformat())),
        grade=grade,
        race_type=str(payload.get("race_type", "")),
        entries=entries,
        order=(0,0,0),
    )


def predict_live(payload: dict, model_path: Path | None = None) -> dict:
    model_path = model_path or (OUT / "model.joblib")
    bundle = joblib.load(model_path)
    race = live_race_from_payload(payload)
    joint = predict_joint(race, bundle["models"], float(bundle["alpha"]))
    raw = {"race": race, "joint": joint}
    raw["b7"], raw["m7"] = greedy_board(joint, 7)
    raw["b8"], raw["m8"] = greedy_board(joint, 8)
    rows, mass = apply_budget(raw, bundle["budget_policy"])
    feats = prediction_features(race, joint, rows, mass)
    score = float(bundle["participation_model"].predict_proba(feats.reshape(1,-1))[0,1])
    p1, p2, p3 = marginals(joint)
    return {
        "race_id": race.race_id,
        "board_generated": True,
        "board_policy": "ninecar_v2_joint504",
        "participate": score >= float(bundle["participation_threshold"]),
        "first_candidates": list(rows[0]),
        "second_candidates": list(rows[1]),
        "third_candidates": list(rows[2]),
        "joint_board_mass": mass,
        "participation_score": score,
        "participation_threshold": float(bundle["participation_threshold"]),
        "first_ranking": [{"no":n,"probability":p} for n,p in sorted(p1.items(), key=lambda x:x[1], reverse=True)],
        "second_ranking": [{"no":n,"probability":p} for n,p in sorted(p2.items(), key=lambda x:x[1], reverse=True)],
        "third_ranking": [{"no":n,"probability":p} for n,p in sorted(p3.items(), key=lambda x:x[1], reverse=True)],
        "top_joint_triples": [{"first":a,"second":b,"third":c,"probability":p} for a,b,c,p in joint[:10]],
        "versions": {"ninecar": "ninecar_v2_joint504"},
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

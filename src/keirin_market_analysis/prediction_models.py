from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import exp, log

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .prediction_features import (
    RelationshipTracker,
    add_race_relative_features,
    build_race_structure,
    pair_features_for_target,
)


PAIR_ID_COLUMNS = {"race_id", "car_a", "car_b", "label_a_beats_b"}


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return log(p / (1.0 - p))


def _softmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    m = max(values.values())
    raw = {k: exp(v - m) for k, v in values.items()}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


def plackett_luce_trifecta(utilities: dict[int, float]) -> pd.DataFrame:
    """Generate all ordered top-three probabilities. Seven riders => exactly 210 rows."""
    cars = sorted(utilities)
    u = {c: max(float(utilities[c]), 1e-12) for c in cars}
    rows: list[dict] = []
    for a, b, c in permutations(cars, 3):
        z1 = sum(u.values())
        p1 = u[a] / z1
        z2 = z1 - u[a]
        p2 = u[b] / z2
        z3 = z2 - u[b]
        p3 = u[c] / z3
        rows.append({"first": a, "second": b, "third": c, "combo": f"{a}{b}{c}", "probability": p1 * p2 * p3})
    out = pd.DataFrame(rows)
    total = float(out["probability"].sum())
    if total > 0:
        out["probability"] /= total
    return out.sort_values("probability", ascending=False).reset_index(drop=True)


def marginals_from_trifecta(trifecta: pd.DataFrame) -> pd.DataFrame:
    cars = sorted(set(trifecta["first"]) | set(trifecta["second"]) | set(trifecta["third"]))
    rows: list[dict] = []
    for car in cars:
        p1 = float(trifecta.loc[trifecta["first"] == car, "probability"].sum())
        p2 = float(trifecta.loc[(trifecta["first"] == car) | (trifecta["second"] == car), "probability"].sum())
        p3 = float(trifecta.loc[
            (trifecta["first"] == car) | (trifecta["second"] == car) | (trifecta["third"] == car), "probability"
        ].sum())
        rows.append({"car_no": car, "p_first": p1, "p_top2": p2, "p_top3": p3})
    return pd.DataFrame(rows).sort_values("p_first", ascending=False).reset_index(drop=True)


@dataclass
class PairwiseKeirinModel:
    c: float = 0.35
    max_iter: int = 2500
    model: Pipeline | None = None
    feature_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()

    def fit(self, pair_rows: pd.DataFrame) -> "PairwiseKeirinModel":
        if pair_rows.empty:
            raise ValueError("pairwise training data is empty")
        candidates = [c for c in pair_rows.columns if c not in PAIR_ID_COLUMNS]
        numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(pair_rows[c])]
        categorical = [c for c in candidates if c not in numeric]
        if not numeric:
            raise ValueError("no numeric pairwise features")

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ])
        preprocess = ColumnTransformer([
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ])
        classifier = LogisticRegression(C=self.c, max_iter=self.max_iter, solver="liblinear")
        self.model = Pipeline([("features", preprocess), ("logit", classifier)])
        x = pair_rows[candidates].replace([np.inf, -np.inf], np.nan)
        y = pair_rows["label_a_beats_b"].astype(int)
        if y.nunique() < 2:
            raise ValueError("pairwise labels contain only one class")
        self.model.fit(x, y)
        self.feature_columns = tuple(candidates)
        self.numeric_columns = tuple(numeric)
        self.categorical_columns = tuple(categorical)
        return self

    def _predict_pair_frame(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        x = pairs.reindex(columns=self.feature_columns).copy()
        for c in self.numeric_columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        for c in self.categorical_columns:
            x[c] = x[c].fillna("").astype(str)
        x = x.replace([np.inf, -np.inf], np.nan)
        p = self.model.predict_proba(x)[:, 1]
        out = pairs[["race_id", "car_a", "car_b"]].copy()
        out["p_a_beats_b"] = p
        return out

    def predict_race(self, entries_for_race: pd.DataFrame, tracker: RelationshipTracker) -> dict[str, pd.DataFrame]:
        pairs = pair_features_for_target(entries_for_race, tracker)
        pp = self._predict_pair_frame(pairs)
        cars = sorted(set(pp["car_a"]) | set(pp["car_b"]))
        score = {int(c): 0.0 for c in cars}
        count = {int(c): 0 for c in cars}
        for r in pp.itertuples(index=False):
            a, b = int(r.car_a), int(r.car_b)
            l = _logit(float(r.p_a_beats_b))
            score[a] += l
            score[b] -= l
            count[a] += 1
            count[b] += 1
        mean_score = {c: score[c] / max(count[c], 1) for c in cars}
        strength_share = _softmax(mean_score)
        utilities = {c: max(p, 1e-12) for c, p in strength_share.items()}
        tri = plackett_luce_trifecta(utilities)
        return {
            "pairwise": pp,
            "strength": pd.DataFrame([
                {"car_no": c, "latent_score": mean_score[c], "strength_share": strength_share[c]} for c in cars
            ]),
            "trifecta": tri,
            "marginals": marginals_from_trifecta(tri),
        }


def assign_structural_roles(entries_for_race: pd.DataFrame) -> pd.DataFrame:
    """Anonymous roles: lines ranked by head score, then position within each line."""
    g = add_race_relative_features(entries_for_race).copy()
    lines: list[tuple[float, object]] = []
    for line_id, lg in g[g["line_id"].notna()].groupby("line_id", sort=False):
        ordered = lg.sort_values(["line_position", "car_no"], na_position="last")
        head_score = float(ordered.iloc[0]["score"])
        lines.append((head_score, line_id))
    lines.sort(reverse=True, key=lambda x: x[0])
    rank = {line_id: i + 1 for i, (_, line_id) in enumerate(lines)}

    roles: list[str] = []
    for r in g.itertuples(index=False):
        line_id = getattr(r, "line_id")
        pos = getattr(r, "line_position")
        car = int(getattr(r, "car_no"))
        if pd.isna(line_id):
            roles.append(f"U{car}")
        else:
            p = int(pos) if pd.notna(pos) else 1
            roles.append(f"L{rank.get(line_id, 9)}P{p}")
    g["structural_role"] = roles
    return g


@dataclass
class SimilarityRoleModel:
    k: int = 250
    distance_floor: float = 0.08
    min_exact_pool: int = 80
    structures: pd.DataFrame | None = None
    outcomes: pd.DataFrame | None = None
    feature_columns: tuple[str, ...] = ()
    medians: pd.Series | None = None
    scales: pd.Series | None = None

    def fit(self, races: pd.DataFrame, entries: pd.DataFrame, results: pd.DataFrame) -> "SimilarityRoleModel":
        structures = build_race_structure(races, entries)
        if "race_no" in structures.columns:
            structures["race_no"] = pd.to_numeric(structures["race_no"], errors="coerce")
        outcome_rows: list[dict] = []
        result_groups = {str(rid): g for rid, g in results.groupby("race_id", sort=False)}
        for raw_race_id, eg in entries.groupby("race_id", sort=False):
            race_id = str(raw_race_id)
            if race_id not in result_groups:
                continue
            role = assign_structural_roles(eg)
            role_by_car = {
                int(r.car_no): r.structural_role
                for r in role[["car_no", "structural_role"]].itertuples(index=False)
            }
            rg = result_groups[race_id].copy()
            rg["order_numeric"] = pd.to_numeric(rg["order_numeric"], errors="coerce")
            rg["car_no"] = pd.to_numeric(rg["car_no"], errors="coerce")
            top = rg.dropna(subset=["order_numeric", "car_no"]).sort_values("order_numeric").head(3)
            if len(top) != 3:
                continue
            cars = [int(x) for x in top["car_no"]]
            if not all(c in role_by_car for c in cars):
                continue
            outcome_rows.append({
                "race_id": race_id,
                "role_first": role_by_car[cars[0]],
                "role_second": role_by_car[cars[1]],
                "role_third": role_by_car[cars[2]],
            })
        outcomes = pd.DataFrame(outcome_rows)
        numeric = [
            c for c in structures.columns
            if pd.api.types.is_numeric_dtype(structures[c]) and c not in {"rider_count"}
        ]
        if not numeric:
            raise ValueError("no race-structure numeric features")
        x = structures[numeric].replace([np.inf, -np.inf], np.nan)
        med = x.median()
        scales = x.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.structures = structures
        self.outcomes = outcomes
        self.feature_columns = tuple(numeric)
        self.medians = med
        self.scales = scales
        return self

    def _candidate_pool(self, target: pd.Series) -> pd.DataFrame:
        assert self.structures is not None
        pool = self.structures
        race_type = str(target.get("race_type", ""))
        if race_type and "race_type" in pool.columns:
            exact_type = pool[pool["race_type"].astype(str) == race_type]
            if len(exact_type) >= self.min_exact_pool:
                pool = exact_type
        if "line_count" in pool.columns and pd.notna(target.get("line_count")):
            exact_line = pool[pd.to_numeric(pool["line_count"], errors="coerce") == float(target["line_count"])]
            if len(exact_line) >= self.min_exact_pool:
                pool = exact_line
        return pool

    def predict_race(self, race_row: pd.DataFrame, entries_for_race: pd.DataFrame) -> pd.DataFrame:
        if self.structures is None or self.outcomes is None or self.medians is None or self.scales is None:
            raise RuntimeError("similarity model is not fitted")
        target_struct = build_race_structure(race_row, entries_for_race)
        if target_struct.empty:
            raise ValueError("target race structure could not be built")
        if "race_no" in target_struct.columns:
            target_struct["race_no"] = pd.to_numeric(target_struct["race_no"], errors="coerce")
        target = target_struct.iloc[0]
        pool = self._candidate_pool(target)
        if pool.empty:
            return pd.DataFrame(columns=["first", "second", "third", "combo", "probability"])

        x = pool[list(self.feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        t = target[list(self.feature_columns)].replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        d = np.sqrt((((x - t) / self.scales) ** 2).mean(axis=1))
        neigh = pool[["race_id"]].copy()
        neigh["distance"] = d
        neigh = neigh.nsmallest(min(self.k, len(neigh)), "distance")
        neigh = neigh.merge(self.outcomes, on="race_id", how="inner")
        if neigh.empty:
            return pd.DataFrame(columns=["first", "second", "third", "combo", "probability"])
        neigh["weight"] = 1.0 / np.maximum(neigh["distance"].astype(float), self.distance_floor)

        target_roles = assign_structural_roles(entries_for_race)
        car_by_role = {
            r.structural_role: int(r.car_no)
            for r in target_roles[["car_no", "structural_role"]].itertuples(index=False)
        }
        probs: dict[tuple[int, int, int], float] = {}
        for r in neigh.itertuples(index=False):
            roles = (r.role_first, r.role_second, r.role_third)
            if not all(x in car_by_role for x in roles):
                continue
            combo = tuple(car_by_role[x] for x in roles)
            if len(set(combo)) != 3:
                continue
            probs[combo] = probs.get(combo, 0.0) + float(r.weight)
        z = sum(probs.values())
        if z <= 0:
            return pd.DataFrame(columns=["first", "second", "third", "combo", "probability"])
        rows = [
            {"first": a, "second": b, "third": c, "combo": f"{a}{b}{c}", "probability": w / z}
            for (a, b, c), w in probs.items()
        ]
        return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def blend_trifecta(stat: pd.DataFrame, similarity: pd.DataFrame, stat_weight: float) -> pd.DataFrame:
    w = float(stat_weight)
    if not 0 <= w <= 1:
        raise ValueError("stat_weight must be in [0,1]")
    base = stat[["first", "second", "third", "combo", "probability"]].rename(columns={"probability": "p_stat"})
    if similarity.empty:
        out = base.copy()
        out["p_similarity"] = 0.0
        out["probability"] = out["p_stat"]
        return out.sort_values("probability", ascending=False).reset_index(drop=True)
    sim = similarity[["combo", "probability"]].rename(columns={"probability": "p_similarity"})
    out = base.merge(sim, on="combo", how="left")
    out["p_similarity"] = out["p_similarity"].fillna(0.0)
    sim_mass = float(out["p_similarity"].sum())
    if sim_mass > 0:
        out["p_similarity"] /= sim_mass
        out["probability"] = w * out["p_stat"] + (1.0 - w) * out["p_similarity"]
    else:
        out["probability"] = out["p_stat"]
    out["probability"] /= out["probability"].sum()
    return out.sort_values("probability", ascending=False).reset_index(drop=True)


def distribution_metrics(trifecta: pd.DataFrame) -> dict[str, float]:
    p = trifecta["probability"].astype(float).to_numpy()
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum()) if len(p) else 0.0
    max_entropy = float(np.log(len(trifecta))) if len(trifecta) > 1 else 1.0
    sorted_p = np.sort(p)[::-1]
    return {
        "entropy": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
        "top1_mass": float(sorted_p[:1].sum()),
        "top3_mass": float(sorted_p[:3].sum()),
        "top5_mass": float(sorted_p[:5].sum()),
        "top10_mass": float(sorted_p[:10].sum()),
    }

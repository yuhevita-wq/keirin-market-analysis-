from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from .prediction_schema import validate_pre_race_columns


IDENTITY_CANDIDATES = (
    "registration_no",
    "registration_number",
    "rider_id",
    "player_id",
    "player_name",
    "rider_name",
    "name",
)

NUMERIC_ALIASES = {
    "score": ("score", "competition_score", "race_score"),
    "age": ("age",),
    "gear": ("gear", "gear_ratio"),
    "s_count": ("s_count", "s"),
    "b_count": ("b_count", "b"),
    "nige": ("nige_count", "nige", "escape", "逃"),
    "makuri": ("makuri_count", "makuri", "捲"),
    "sashi": ("sashi_count", "sashi", "差"),
    "mark": ("mark_count", "mark", "ma", "マ"),
    "first_count": ("first_count",),
    "second_count": ("second_count",),
    "third_count": ("third_count",),
    "outside_count": ("outside_count",),
    "win_rate": ("win_rate", "first_rate"),
    "quinella_rate": ("top2_rate", "quinella_rate", "2ren_rate"),
    "trio_rate": ("top3_rate", "trio_rate", "3ren_rate"),
    "line_id": ("line_id",),
    "line_position": ("line_position",),
    "line_size": ("line_size",),
    "car_no": ("car_no",),
    "race_no": ("race_no",),
}


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    pool = set(columns)
    for c in candidates:
        if c in pool:
            return c
    return None


def rider_identity_column(entries: pd.DataFrame) -> str:
    col = first_existing(entries.columns, IDENTITY_CANDIDATES)
    if col is None:
        raise KeyError(f"No stable rider identity column found. Tried: {IDENTITY_CANDIDATES}")
    return col


def _numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("％", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _value(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def normalize_entries(entries: pd.DataFrame) -> pd.DataFrame:
    validate_pre_race_columns(entries.columns)
    if "race_id" not in entries.columns:
        raise KeyError("entries.csv requires race_id")
    out = entries.copy()
    out["race_id"] = out["race_id"].astype(str)
    rider_col = rider_identity_column(out)
    out["rider_key"] = out[rider_col].astype(str).str.replace(r"\s+", "", regex=True).str.strip()
    if (out["rider_key"] == "").any():
        raise ValueError("blank rider identity found")

    for canonical, aliases in NUMERIC_ALIASES.items():
        src = first_existing(out.columns, aliases)
        if src is not None:
            out[canonical] = _numeric_series(out[src])

    required = {"car_no", "score"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"entries.csv missing required normalized fields: {sorted(missing)}")

    for optional in ("line_id", "line_position", "line_size", "b_count", "s_count", "race_no"):
        if optional not in out.columns:
            out[optional] = np.nan
    for optional in ("track", "race_type", "style", "class", "prefecture", "line_role"):
        if optional not in out.columns:
            out[optional] = ""
        out[optional] = out[optional].fillna("").astype(str).str.strip()
    return out


def add_race_relative_features(entries: pd.DataFrame) -> pd.DataFrame:
    out = normalize_entries(entries)
    numeric = [c for c in (
        "score", "age", "gear", "s_count", "b_count", "nige", "makuri", "sashi", "mark",
        "first_count", "second_count", "third_count", "outside_count",
        "win_rate", "quinella_rate", "trio_rate", "line_position", "line_size",
    ) if c in out.columns]

    for col in numeric:
        g = out.groupby("race_id")[col]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        out[f"{col}_delta"] = out[col] - mean
        out[f"{col}_z"] = (out[col] - mean) / std
        out[f"{col}_rank_pct"] = g.rank(method="average", ascending=False, pct=True)

    valid_line = out["line_id"].notna()
    if valid_line.any():
        line = out.loc[valid_line].groupby(["race_id", "line_id"], dropna=False)
        out.loc[valid_line, "line_score_sum"] = line["score"].transform("sum")
        out.loc[valid_line, "line_score_mean"] = line["score"].transform("mean")
        out.loc[valid_line, "line_b_sum"] = line["b_count"].transform("sum")
        out.loc[valid_line, "line_member_count"] = line["race_id"].transform("size")

    race_score_max = out.groupby("race_id")["score"].transform("max")
    race_score_min = out.groupby("race_id")["score"].transform("min")
    out["race_score_spread"] = race_score_max - race_score_min
    out["race_b_total"] = out.groupby("race_id")["b_count"].transform("sum")
    out["race_line_count"] = out.groupby("race_id")["line_id"].transform(lambda s: s.dropna().nunique())
    out["race_max_line_size"] = out.groupby("race_id")["line_size"].transform("max")

    line_shapes: dict[str, str] = {}
    b_concentrations: dict[str, float] = {}
    for race_id, g in out.groupby("race_id", sort=False):
        sizes = sorted(
            [int(len(lg)) for _, lg in g[g["line_id"].notna()].groupby("line_id", sort=False)],
            reverse=True,
        )
        line_shapes[str(race_id)] = "-".join(str(x) for x in sizes) if sizes else "unpublished"
        b_by_line = [float(lg["b_count"].fillna(0).sum()) for _, lg in g[g["line_id"].notna()].groupby("line_id", sort=False)]
        total_b = float(g["b_count"].fillna(0).sum())
        b_concentrations[str(race_id)] = max(b_by_line) / total_b if b_by_line and total_b > 0 else 0.0
    out["race_line_shape"] = out["race_id"].map(line_shapes)
    out["race_b_line_concentration"] = out["race_id"].map(b_concentrations).astype(float)
    return out


def build_race_structure(races: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    e = add_race_relative_features(entries)
    validate_pre_race_columns(races.columns)
    r = races.copy()
    r["race_id"] = r["race_id"].astype(str)
    meta_cols = [c for c in (
        "race_id", "race_date", "track", "race_no", "race_type", "predicted_line_formation", "segment", "dataset_role"
    ) if c in r.columns]
    base = r[meta_cols].drop_duplicates("race_id").copy()

    rows: list[dict] = []
    for race_id, g in e.groupby("race_id", sort=False):
        row: dict[str, object] = {"race_id": str(race_id)}
        scores = g["score"].dropna().astype(float)
        row["rider_count"] = int(len(g))
        row["score_max"] = float(scores.max()) if len(scores) else np.nan
        row["score_min"] = float(scores.min()) if len(scores) else np.nan
        row["score_mean"] = float(scores.mean()) if len(scores) else np.nan
        row["score_std"] = float(scores.std(ddof=0)) if len(scores) else np.nan
        row["score_spread"] = row["score_max"] - row["score_min"] if len(scores) else np.nan

        line_ids = [x for x in g["line_id"].dropna().unique()]
        row["line_count"] = len(line_ids)
        sizes: list[int] = []
        head_scores: list[float] = []
        line_b: list[float] = []
        for line_id in line_ids:
            lg = g[g["line_id"] == line_id].copy()
            sizes.append(len(lg))
            if lg["line_position"].notna().any():
                lg = lg.sort_values(["line_position", "car_no"])
            else:
                lg = lg.sort_values("car_no")
            head_scores.append(float(lg.iloc[0]["score"]))
            line_b.append(float(lg["b_count"].fillna(0).sum()))
        sizes.sort(reverse=True)
        head_scores.sort(reverse=True)
        line_b.sort(reverse=True)
        for i in range(4):
            row[f"line_size_{i + 1}"] = sizes[i] if i < len(sizes) else 0
            row[f"head_score_{i + 1}"] = head_scores[i] if i < len(head_scores) else np.nan
        row["head_score_gap_12"] = head_scores[0] - head_scores[1] if len(head_scores) >= 2 else np.nan
        total_b = float(g["b_count"].fillna(0).sum())
        row["b_total"] = total_b
        row["b_line_concentration"] = (line_b[0] / total_b) if total_b > 0 and line_b else 0.0
        row["singletons"] = int(sum(1 for x in sizes if x == 1))
        rows.append(row)

    return base.merge(pd.DataFrame(rows), on="race_id", how="inner")


class RelationshipTracker:
    """Strictly chronological head-to-head and same-line history with Bayesian shrinkage."""

    def __init__(self, prior_strength: float = 8.0) -> None:
        self.prior_strength = float(prior_strength)
        self._meetings: dict[tuple[str, str], int] = defaultdict(int)
        self._wins: dict[tuple[str, str], int] = defaultdict(int)
        self._same_line: dict[tuple[str, str], int] = defaultdict(int)
        self._same_line_success: dict[tuple[str, str], int] = defaultdict(int)

    @staticmethod
    def _ordered(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def features(self, a: str, b: str) -> dict[str, float]:
        key = self._ordered(a, b)
        n = self._meetings[key]
        a_wins = self._wins[(a, b)]
        p = (a_wins + 0.5 * self.prior_strength) / (n + self.prior_strength)
        sl_n = self._same_line[key]
        sl_ok = self._same_line_success[key]
        sl_p = (sl_ok + 0.5 * self.prior_strength) / (sl_n + self.prior_strength)
        return {
            "h2h_meetings": float(n),
            "h2h_a_rate_shrunk": float(p),
            "h2h_confidence": float(n / (n + self.prior_strength)),
            "same_line_meetings": float(sl_n),
            "same_line_top3_rate_shrunk": float(sl_p),
            "same_line_confidence": float(sl_n / (sl_n + self.prior_strength)),
        }

    def update_race(self, entry_rows: pd.DataFrame, result_rows: pd.DataFrame) -> None:
        order = {
            int(r.car_no): float(r.order_numeric)
            for r in result_rows[["car_no", "order_numeric"]].itertuples(index=False)
            if pd.notna(r.car_no) and pd.notna(r.order_numeric)
        }
        riders = entry_rows[["rider_key", "car_no", "line_id"]].copy()
        riders["car_no"] = pd.to_numeric(riders["car_no"], errors="coerce")
        records = list(riders.itertuples(index=False))
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                ra, rb = records[i], records[j]
                if pd.isna(ra.car_no) or pd.isna(rb.car_no):
                    continue
                ca, cb = int(ra.car_no), int(rb.car_no)
                if ca not in order or cb not in order or order[ca] == order[cb]:
                    continue
                a, b = str(ra.rider_key), str(rb.rider_key)
                key = self._ordered(a, b)
                self._meetings[key] += 1
                if order[ca] < order[cb]:
                    self._wins[(a, b)] += 1
                else:
                    self._wins[(b, a)] += 1
                same_line = pd.notna(ra.line_id) and pd.notna(rb.line_id) and ra.line_id == rb.line_id
                if same_line:
                    self._same_line[key] += 1
                    if order[ca] <= 3 and order[cb] <= 3:
                        self._same_line_success[key] += 1


def _numeric_pair_columns(e: pd.DataFrame) -> list[str]:
    return [c for c in (
        "score", "age", "gear", "s_count", "b_count", "nige", "makuri", "sashi", "mark",
        "first_count", "second_count", "third_count", "outside_count",
        "win_rate", "quinella_rate", "trio_rate", "line_position", "line_size",
        "line_score_sum", "line_score_mean", "line_b_sum", "line_member_count",
    ) if c in e.columns]


def _pair_row(a: dict, b: dict, tracker: RelationshipTracker, numeric_base: list[str]) -> dict:
    rel = tracker.features(str(a["rider_key"]), str(b["rider_key"]))
    row: dict[str, float | str | int] = {
        "race_id": str(a["race_id"]),
        "car_a": int(a["car_no"]),
        "car_b": int(b["car_no"]),
        "same_line": int(pd.notna(a.get("line_id")) and pd.notna(b.get("line_id")) and a.get("line_id") == b.get("line_id")),
        "same_line_position_gap": _value(a.get("line_position")) - _value(b.get("line_position")),
        "delta_car_no": _value(a.get("car_no")) - _value(b.get("car_no")),
        "race_no": _value(a.get("race_no"), np.nan),
        "race_score_spread": _value(a.get("race_score_spread"), np.nan),
        "race_b_total": _value(a.get("race_b_total"), np.nan),
        "race_line_count": _value(a.get("race_line_count"), np.nan),
        "race_max_line_size": _value(a.get("race_max_line_size"), np.nan),
        "race_b_line_concentration": _value(a.get("race_b_line_concentration"), np.nan),
        "track": str(a.get("track", "")),
        "race_type": str(a.get("race_type", "")),
        "race_line_shape": str(a.get("race_line_shape", "")),
        "a_style": str(a.get("style", "")),
        "b_style": str(b.get("style", "")),
        "a_class": str(a.get("class", "")),
        "b_class": str(b.get("class", "")),
        "a_prefecture": str(a.get("prefecture", "")),
        "b_prefecture": str(b.get("prefecture", "")),
        "same_prefecture": int(str(a.get("prefecture", "")) != "" and str(a.get("prefecture", "")) == str(b.get("prefecture", ""))),
        **rel,
    }
    for col in numeric_base:
        av = _value(a.get(col), np.nan)
        bv = _value(b.get(col), np.nan)
        row[f"delta_{col}"] = av - bv if np.isfinite(av) and np.isfinite(bv) else np.nan
    return row


def build_pairwise_rows(
    entries: pd.DataFrame,
    results: pd.DataFrame,
    *,
    relationship_tracker: RelationshipTracker | None = None,
    update_relationships: bool = True,
) -> tuple[pd.DataFrame, RelationshipTracker]:
    e = add_race_relative_features(entries)
    results = results.copy()
    results["race_id"] = results["race_id"].astype(str)
    tracker = relationship_tracker or RelationshipTracker()
    result_groups = {str(rid): g for rid, g in results.groupby("race_id", sort=False)}
    rows: list[dict[str, float | str | int]] = []

    date_col = "race_date" if "race_date" in e.columns else None
    race_order = e[["race_id"] + ([date_col] if date_col else [])].drop_duplicates()
    if date_col:
        race_order[date_col] = pd.to_datetime(race_order[date_col], errors="coerce")
        race_order = race_order.sort_values([date_col, "race_id"])
    numeric_base = _numeric_pair_columns(e)

    for raw_race_id in race_order["race_id"]:
        race_id = str(raw_race_id)
        if race_id not in result_groups:
            continue
        rg = result_groups[race_id]
        order = {
            int(r.car_no): float(r.order_numeric)
            for r in rg[["car_no", "order_numeric"]].itertuples(index=False)
            if pd.notna(r.car_no) and pd.notna(r.order_numeric)
        }
        g = e[e["race_id"] == race_id].copy().sort_values("car_no")
        recs = list(g.to_dict("records"))
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                ca, cb = int(a["car_no"]), int(b["car_no"])
                if ca not in order or cb not in order or order[ca] == order[cb]:
                    continue
                row = _pair_row(a, b, tracker, numeric_base)
                row["label_a_beats_b"] = int(order[ca] < order[cb])
                rows.append(row)
        if update_relationships:
            tracker.update_race(g, rg)

    return pd.DataFrame(rows), tracker


def pair_features_for_target(entries_for_race: pd.DataFrame, tracker: RelationshipTracker) -> pd.DataFrame:
    g = add_race_relative_features(entries_for_race).sort_values("car_no")
    recs = list(g.to_dict("records"))
    numeric_base = _numeric_pair_columns(g)
    rows: list[dict] = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            rows.append(_pair_row(recs[i], recs[j], tracker, numeric_base))
    return pd.DataFrame(rows)

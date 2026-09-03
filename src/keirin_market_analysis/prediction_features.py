from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log
from typing import Iterable

import numpy as np
import pandas as pd

from .prediction_schema import LeakageError, validate_pre_race_columns


IDENTITY_CANDIDATES = (
    "registration_no",
    "registration_number",
    "rider_id",
    "player_id",
    "rider_name",
    "player_name",
    "name",
)

NUMERIC_ALIASES = {
    "score": ("score", "competition_score", "race_score"),
    "age": ("age",),
    "gear": ("gear", "gear_ratio"),
    "s_count": ("s_count", "s"),
    "b_count": ("b_count", "b"),
    "nige": ("nige", "escape", "逃"),
    "makuri": ("makuri", "捲"),
    "sashi": ("sashi", "差"),
    "mark": ("mark", "ma", "マ"),
    "win_rate": ("win_rate", "first_rate"),
    "quinella_rate": ("quinella_rate", "top2_rate", "2ren_rate"),
    "trio_rate": ("trio_rate", "top3_rate", "3ren_rate"),
    "line_id": ("line_id",),
    "line_position": ("line_position",),
    "line_size": ("line_size",),
    "car_no": ("car_no",),
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


def normalize_entries(entries: pd.DataFrame) -> pd.DataFrame:
    validate_pre_race_columns(entries.columns)
    if "race_id" not in entries.columns:
        raise KeyError("entries.csv requires race_id")
    out = entries.copy()
    rider_col = rider_identity_column(out)
    out["rider_key"] = out[rider_col].astype(str).str.strip()
    if (out["rider_key"] == "").any():
        raise ValueError("blank rider identity found")

    for canonical, aliases in NUMERIC_ALIASES.items():
        src = first_existing(out.columns, aliases)
        if src is not None:
            out[canonical] = pd.to_numeric(out[src], errors="coerce")

    required = {"car_no", "score"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"entries.csv missing required normalized fields: {sorted(missing)}")

    for optional in ("line_id", "line_position", "line_size", "b_count", "s_count"):
        if optional not in out.columns:
            out[optional] = np.nan
    return out


def _safe_std(s: pd.Series) -> float:
    v = float(s.std(ddof=0)) if s.notna().any() else 0.0
    return v if np.isfinite(v) and v > 1e-9 else 1.0


def add_race_relative_features(entries: pd.DataFrame) -> pd.DataFrame:
    out = normalize_entries(entries)
    numeric = [c for c in (
        "score", "age", "gear", "s_count", "b_count", "nige", "makuri", "sashi", "mark",
        "win_rate", "quinella_rate", "trio_rate", "line_position", "line_size",
    ) if c in out.columns]

    for col in numeric:
        g = out.groupby("race_id")[col]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        out[f"{col}_delta"] = out[col] - mean
        out[f"{col}_z"] = (out[col] - mean) / std
        out[f"{col}_rank_pct"] = g.rank(method="average", ascending=False, pct=True)

    if "line_id" in out.columns:
        valid_line = out["line_id"].notna()
        line = out.loc[valid_line].groupby(["race_id", "line_id"], dropna=False)
        if "score" in out.columns:
            out.loc[valid_line, "line_score_sum"] = line["score"].transform("sum")
            out.loc[valid_line, "line_score_mean"] = line["score"].transform("mean")
        if "b_count" in out.columns:
            out.loc[valid_line, "line_b_sum"] = line["b_count"].transform("sum")
        out.loc[valid_line, "line_member_count"] = line["race_id"].transform("size")
    return out


def build_race_structure(races: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    e = add_race_relative_features(entries)
    validate_pre_race_columns(races.columns)
    meta_cols = [c for c in (
        "race_id", "race_date", "track", "race_no", "race_type", "predicted_line_formation", "segment", "dataset_role"
    ) if c in races.columns]
    base = races[meta_cols].drop_duplicates("race_id").copy()

    rows: list[dict] = []
    for race_id, g in e.groupby("race_id", sort=False):
        row: dict[str, object] = {"race_id": race_id}
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
            row[f"line_size_{i+1}"] = sizes[i] if i < len(sizes) else 0
            row[f"head_score_{i+1}"] = head_scores[i] if i < len(head_scores) else np.nan
        row["head_score_gap_12"] = head_scores[0] - head_scores[1] if len(head_scores) >= 2 else np.nan
        total_b = float(g["b_count"].fillna(0).sum())
        row["b_total"] = total_b
        row["b_line_concentration"] = (line_b[0] / total_b) if total_b > 0 and line_b else 0.0
        row["singletons"] = int(sum(1 for x in sizes if x == 1))
        rows.append(row)

    return base.merge(pd.DataFrame(rows), on="race_id", how="inner")


@dataclass
class PairState:
    meetings: int = 0
    a_wins: int = 0
    same_line: int = 0
    same_line_both_top3: int = 0


class RelationshipTracker:
    """Strictly chronological head-to-head and same-line history with shrinkage."""

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
        confidence = n / (n + self.prior_strength)
        sl_n = self._same_line[key]
        sl_ok = self._same_line_success[key]
        sl_p = (sl_ok + 0.5 * self.prior_strength) / (sl_n + self.prior_strength)
        return {
            "h2h_meetings": float(n),
            "h2h_a_rate_shrunk": float(p),
            "h2h_confidence": float(confidence),
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
                if ca not in order or cb not in order:
                    continue
                a, b = str(ra.rider_key), str(rb.rider_key)
                key = self._ordered(a, b)
                self._meetings[key] += 1
                if order[ca] < order[cb]:
                    self._wins[(a, b)] += 1
                elif order[cb] < order[ca]:
                    self._wins[(b, a)] += 1
                same_line = pd.notna(ra.line_id) and pd.notna(rb.line_id) and ra.line_id == rb.line_id
                if same_line:
                    self._same_line[key] += 1
                    if order[ca] <= 3 and order[cb] <= 3:
                        self._same_line_success[key] += 1


def build_pairwise_rows(
    entries: pd.DataFrame,
    results: pd.DataFrame,
    *,
    relationship_tracker: RelationshipTracker | None = None,
    update_relationships: bool = True,
) -> tuple[pd.DataFrame, RelationshipTracker]:
    e = add_race_relative_features(entries)
    tracker = relationship_tracker or RelationshipTracker()
    result_groups = {rid: g for rid, g in results.groupby("race_id", sort=False)}
    rows: list[dict[str, float | str | int]] = []

    date_col = "race_date" if "race_date" in e.columns else None
    race_order = e[["race_id"] + ([date_col] if date_col else [])].drop_duplicates()
    if date_col:
        race_order[date_col] = pd.to_datetime(race_order[date_col], errors="coerce")
        race_order = race_order.sort_values([date_col, "race_id"])

    numeric_base = [c for c in (
        "score", "age", "gear", "s_count", "b_count", "nige", "makuri", "sashi", "mark",
        "win_rate", "quinella_rate", "trio_rate", "line_position", "line_size",
        "line_score_sum", "line_score_mean", "line_b_sum", "line_member_count",
    ) if c in e.columns]

    for race_id in race_order["race_id"]:
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
                if ca not in order or cb not in order:
                    continue
                rel = tracker.features(str(a["rider_key"]), str(b["rider_key"]))
                row: dict[str, float | str | int] = {
                    "race_id": race_id,
                    "car_a": ca,
                    "car_b": cb,
                    "label_a_beats_b": int(order[ca] < order[cb]),
                    "same_line": int(pd.notna(a.get("line_id")) and pd.notna(b.get("line_id")) and a.get("line_id") == b.get("line_id")),
                    "same_line_position_gap": float((a.get("line_position") or 0) - (b.get("line_position") or 0)),
                    **rel,
                }
                for col in numeric_base:
                    av = pd.to_numeric(pd.Series([a.get(col)]), errors="coerce").iloc[0]
                    bv = pd.to_numeric(pd.Series([b.get(col)]), errors="coerce").iloc[0]
                    row[f"delta_{col}"] = float(av - bv) if pd.notna(av) and pd.notna(bv) else np.nan
                rows.append(row)
        if update_relationships:
            tracker.update_race(g, rg)

    return pd.DataFrame(rows), tracker


def pair_features_for_target(entries_for_race: pd.DataFrame, tracker: RelationshipTracker) -> pd.DataFrame:
    g = add_race_relative_features(entries_for_race).sort_values("car_no")
    recs = list(g.to_dict("records"))
    rows: list[dict] = []
    numeric_base = [c for c in (
        "score", "age", "gear", "s_count", "b_count", "nige", "makuri", "sashi", "mark",
        "win_rate", "quinella_rate", "trio_rate", "line_position", "line_size",
        "line_score_sum", "line_score_mean", "line_b_sum", "line_member_count",
    ) if c in g.columns]
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            rel = tracker.features(str(a["rider_key"]), str(b["rider_key"]))
            row = {
                "race_id": a["race_id"],
                "car_a": int(a["car_no"]),
                "car_b": int(b["car_no"]),
                "same_line": int(pd.notna(a.get("line_id")) and pd.notna(b.get("line_id")) and a.get("line_id") == b.get("line_id")),
                "same_line_position_gap": float((a.get("line_position") or 0) - (b.get("line_position") or 0)),
                **rel,
            }
            for col in numeric_base:
                av = pd.to_numeric(pd.Series([a.get(col)]), errors="coerce").iloc[0]
                bv = pd.to_numeric(pd.Series([b.get(col)]), errors="coerce").iloc[0]
                row[f"delta_{col}"] = float(av - bv) if pd.notna(av) and pd.notna(bv) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)

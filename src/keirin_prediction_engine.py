from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


OUTCOME_COLUMNS = {"finish", "rank", "result", "payout", "payoff"}


@dataclass(frozen=True)
class DatasetSplit:
    development: tuple[int, ...] = (2024,)
    destruction_test: tuple[int, ...] = (2025,)
    sealed_validation: tuple[int, ...] = (2026,)


@dataclass(frozen=True)
class EngineConfig:
    seven_rider_only: bool = True
    min_history_races: int = 150
    random_seed: int = 42
    similarity_k: int = 200


class LeakageError(ValueError):
    pass


def _reject_outcome_columns(columns: Iterable[str]) -> None:
    lowered = {c.lower() for c in columns}
    leaking = sorted(c for c in lowered if any(token in c for token in OUTCOME_COLUMNS))
    if leaking:
        raise LeakageError(f"Outcome-like columns are forbidden as pre-race features: {leaking}")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def add_relative_features(entries: pd.DataFrame) -> pd.DataFrame:
    """Convert raw rider rows into race-relative, pre-race features.

    This function only uses columns already present on the entry sheet. It never
    reads results or payouts.
    """
    df = entries.copy()
    _reject_outcome_columns(df.columns)

    race_key = "race_id" if "race_id" in df.columns else "race_url"
    if race_key not in df.columns:
        raise KeyError("entries requires race_id or race_url")

    numeric_candidates = [
        "score", "race_score", "competition_score", "points", "age", "gear",
        "s", "b", "escape", "makuri", "sashi", "mark", "win_rate",
        "quinella_rate", "trio_rate", "line_position", "line_size",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        grp = df.groupby(race_key)[c]
        df[f"{c}_race_mean"] = grp.transform("mean")
        df[f"{c}_race_std"] = grp.transform("std").replace(0, np.nan)
        df[f"{c}_delta_mean"] = df[c] - df[f"{c}_race_mean"]
        df[f"{c}_z"] = df[f"{c}_delta_mean"] / df[f"{c}_race_std"]
        df[f"{c}_race_rank"] = grp.rank(method="average", ascending=False, pct=True)

    if "line_id" in df.columns:
        line_group = df.groupby([race_key, "line_id"], dropna=False)
        if "score" in df.columns:
            df["line_score_sum"] = line_group["score"].transform("sum")
            df["line_score_mean"] = line_group["score"].transform("mean")
        if "b" in df.columns:
            df["line_b_sum"] = line_group["b"].transform("sum")
        df["line_member_count"] = line_group[race_key].transform("size")

    return df


def build_race_table(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Build one row per race with structural features only."""
    e = add_relative_features(entries)
    _reject_outcome_columns(races.columns)

    race_key = "race_id" if "race_id" in e.columns and "race_id" in races.columns else "race_url"
    if race_key not in e.columns or race_key not in races.columns:
        raise KeyError("races/entries need a shared race_id or race_url")

    agg: dict[str, list[str] | str] = {}
    for col in e.columns:
        if col.endswith("_delta_mean") or col.endswith("_z"):
            agg[col] = ["max", "min", "std"]
    if "line_id" in e.columns:
        agg["line_id"] = pd.Series.nunique
    if "line_size" in e.columns:
        agg["line_size"] = ["max", "mean", "std"]
    if "line_position" in e.columns:
        agg["line_position"] = ["mean", "std"]
    if "b" in e.columns:
        agg["b"] = ["sum", "max", "std"]
    if "score" in e.columns:
        agg["score"] = ["max", "min", "mean", "std"]

    if not agg:
        raise ValueError("No recognized structural columns found in entries.csv")

    race_struct = e.groupby(race_key).agg(agg)
    race_struct.columns = ["__".join([str(x) for x in tup if str(x)]) for tup in race_struct.columns]
    race_struct = race_struct.reset_index()

    base_cols = [c for c in [race_key, "date", "race_date", "venue", "race_no", "race_type", "entry_count", "predicted_line_formation"] if c in races.columns]
    out = races[base_cols].drop_duplicates(race_key).merge(race_struct, on=race_key, how="inner")

    if "entry_count" in out.columns:
        out["entry_count"] = pd.to_numeric(out["entry_count"], errors="coerce")
    return out


def chronological_split(race_table: pd.DataFrame, split: DatasetSplit = DatasetSplit()) -> dict[str, pd.DataFrame]:
    date_col = "race_date" if "race_date" in race_table.columns else "date"
    if date_col not in race_table.columns:
        raise KeyError("race table requires date or race_date")
    out = race_table.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = out[date_col].dt.year
    return {
        "development": out[out["year"].isin(split.development)].copy(),
        "destruction_test": out[out["year"].isin(split.destruction_test)].copy(),
        "sealed_validation": out[out["year"].isin(split.sealed_validation)].copy(),
    }


def similarity_neighbors(history: pd.DataFrame, target: pd.Series, k: int = 200) -> pd.DataFrame:
    """Return nearest historical race structures using standardized Euclidean distance.

    Categorical filters should be applied by the caller before this function.
    """
    exclude = {"year", "date", "race_date", "race_id", "race_url", "predicted_line_formation"}
    numeric = [c for c in history.columns if c not in exclude and pd.api.types.is_numeric_dtype(history[c])]
    if not numeric:
        raise ValueError("No numeric similarity features")
    x = history[numeric].astype(float)
    t = pd.to_numeric(target[numeric], errors="coerce").astype(float)
    med = x.median()
    x = x.fillna(med)
    t = t.fillna(med)
    scale = x.std(ddof=0).replace(0, 1.0)
    dist = np.sqrt((((x - t) / scale) ** 2).mean(axis=1))
    result = history.copy()
    result["similarity_distance"] = dist
    return result.sort_values("similarity_distance").head(k)


def validate_pre_race_frame(df: pd.DataFrame) -> None:
    _reject_outcome_columns(df.columns)
    forbidden = [c for c in df.columns if "final_odds" in c.lower() or "confirmed_odds" in c.lower()]
    if forbidden:
        raise LeakageError(f"Final odds are forbidden in the prediction feature frame: {forbidden}")


def walk_forward_indices(race_table: pd.DataFrame, min_history_races: int = 150):
    """Yield (history_index, target_index) so every target only sees earlier races."""
    date_col = "race_date" if "race_date" in race_table.columns else "date"
    table = race_table.copy()
    table[date_col] = pd.to_datetime(table[date_col], errors="coerce")
    table = table.sort_values([date_col, "race_no"] if "race_no" in table.columns else [date_col]).reset_index(drop=True)
    for i in range(min_history_races, len(table)):
        target_date = table.loc[i, date_col]
        history_idx = table.index[(table.index < i) & (table[date_col] < target_date)]
        if len(history_idx) >= min_history_races:
            yield history_idx, i


def engine_manifest() -> dict:
    return {
        "name": "keirin-prediction-engine-v1",
        "purpose": "predict future F1 S-class races and support profitable race selection without outcome leakage",
        "development": "2024",
        "destruction_test": "2025",
        "sealed_validation": "2026_h1",
        "core_models": ["similarity", "statistical", "relationship"],
        "market_policy": "historical final odds are evaluation/reference only, never pre-race features",
        "status": "foundation",
    }

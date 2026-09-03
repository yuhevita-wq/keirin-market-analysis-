from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


class LeakageError(ValueError):
    """Raised when post-race information is about to enter a pre-race feature frame."""


class SealedValidationError(RuntimeError):
    """Raised when sealed 2026H1 labels are requested without an explicit unlock."""


POST_RACE_FILES = {"results.csv", "payouts.csv"}
MARKET_FINAL_FILES = {"trifecta_final_odds.csv", "trio_final_odds.csv"}
FORBIDDEN_FEATURE_TOKENS = (
    "payout_yen",
    "refund",
    "order_numeric",
    "finish_order",
    "result_rank",
    "final_odds",
    "closing_odds",
    "confirmed_odds",
)


@dataclass(frozen=True)
class Segment:
    name: str
    path: Path
    role: str
    year: int
    sealed: bool = False


@dataclass(frozen=True)
class DatasetCatalog:
    root: Path

    @property
    def segments(self) -> tuple[Segment, ...]:
        data = self.root / "data"
        return (
            Segment("2024_q1", data / "2024" / "s_class_f1_all_parts" / "2024_q1", "development", 2024),
            Segment("2024_q2", data / "2024" / "s_class_f1_all_parts" / "2024_q2", "development", 2024),
            Segment("2024_q3", data / "2024" / "s_class_f1_all_parts" / "2024_q3", "development", 2024),
            Segment("2024_q4", data / "2024" / "s_class_f1_all_parts" / "2024_q4", "development", 2024),
            Segment("2025_q1", data / "2025" / "s_class_f1_all_parts" / "2025_q1", "destruction_test", 2025),
            Segment("2025_q2", data / "2025" / "s_class_f1_all_parts" / "2025_q2", "destruction_test", 2025),
            Segment("2025_q3", data / "2025" / "s_class_f1_all_parts" / "2025_q3", "destruction_test", 2025),
            Segment("2025_q4", data / "2025" / "s_class_f1_all_parts" / "2025_q4", "destruction_test", 2025),
            Segment("2026_h1", data / "2026_h1" / "s_class_f1_all", "sealed_validation", 2026, sealed=True),
        )

    def by_role(self, role: str) -> tuple[Segment, ...]:
        return tuple(s for s in self.segments if s.role == role)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def validate_pre_race_columns(columns: Iterable[str]) -> None:
    bad: list[str] = []
    for raw in columns:
        col = raw.lower()
        if any(token in col for token in FORBIDDEN_FEATURE_TOKENS):
            bad.append(raw)
    if bad:
        raise LeakageError(f"post-race/final-market columns are forbidden as features: {sorted(bad)}")


def load_pre_race_segment(segment: Segment, seven_rider_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    races = _read_csv(segment.path / "races.csv")
    entries = _read_csv(segment.path / "entries.csv")
    validate_pre_race_columns(races.columns)
    validate_pre_race_columns(entries.columns)

    if "race_id" not in races.columns or "race_id" not in entries.columns:
        raise KeyError("races.csv and entries.csv must contain race_id")

    races = races.copy()
    entries = entries.copy()
    races["segment"] = segment.name
    races["dataset_role"] = segment.role
    entries["segment"] = segment.name
    entries["dataset_role"] = segment.role

    if seven_rider_only:
        counts = entries.groupby("race_id").size()
        seven_ids = set(counts[counts == 7].index)
        races = races[races["race_id"].isin(seven_ids)].copy()
        entries = entries[entries["race_id"].isin(seven_ids)].copy()

    return races, entries


def load_labels(segment: Segment, *, unlock_sealed: bool = False) -> pd.DataFrame:
    if segment.sealed and not unlock_sealed:
        raise SealedValidationError(
            "2026_h1 is sealed. Its results may only be opened after the strategy/model contract is frozen."
        )
    results = _read_csv(segment.path / "results.csv")
    needed = {"race_id", "car_no", "order_numeric"}
    missing = needed - set(results.columns)
    if missing:
        raise KeyError(f"results.csv missing columns: {sorted(missing)}")
    out = results.copy()
    out["order_numeric"] = pd.to_numeric(out["order_numeric"], errors="coerce")
    out["car_no"] = pd.to_numeric(out["car_no"], errors="coerce")
    out = out[out["order_numeric"].notna() & out["car_no"].notna()].copy()
    return out


def load_payouts(segment: Segment, *, unlock_sealed: bool = False) -> pd.DataFrame:
    if segment.sealed and not unlock_sealed:
        raise SealedValidationError("2026_h1 payouts are sealed with the validation labels.")
    return _read_csv(segment.path / "payouts.csv")


def concat_pre_race(segments: Iterable[Segment], seven_rider_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    race_frames: list[pd.DataFrame] = []
    entry_frames: list[pd.DataFrame] = []
    for segment in segments:
        races, entries = load_pre_race_segment(segment, seven_rider_only=seven_rider_only)
        race_frames.append(races)
        entry_frames.append(entries)
    if not race_frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(race_frames, ignore_index=True), pd.concat(entry_frames, ignore_index=True)


def concat_labels(segments: Iterable[Segment], *, unlock_sealed: bool = False) -> pd.DataFrame:
    frames = [load_labels(s, unlock_sealed=unlock_sealed) for s in segments]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

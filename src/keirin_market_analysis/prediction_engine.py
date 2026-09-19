from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .prediction_features import RelationshipTracker, build_pairwise_rows, normalize_entries
from .prediction_models import PairwiseKeirinModel, SimilarityRoleModel, blend_trifecta, marginals_from_trifecta
from .prediction_schema import DatasetCatalog, Segment, concat_labels, concat_pre_race
from .prediction_strategy import TicketPolicy, select_tickets


@dataclass(frozen=True)
class EngineContract:
    seven_rider_only: bool = True
    statistical_weight: float = 0.80
    relationship_prior_strength: float = 8.0
    similarity_k: int = 250
    model_version: str = "prediction-engine-v1"
    development_year: int = 2024
    destruction_test_year: int = 2025
    sealed_validation: str = "2026_h1"


class KeirinPredictionEngine:
    """Leakage-safe, price-blind core prediction engine.

    Model coefficients are trained only on development data. Historical memory may
    be refreshed at a declared half-year boundary, which matches the repository's
    archival cadence, without refitting the statistical model.
    """

    def __init__(self, contract: EngineContract = EngineContract()) -> None:
        self.contract = contract
        self.tracker = RelationshipTracker(prior_strength=contract.relationship_prior_strength)
        self.statistical = PairwiseKeirinModel()
        self.similarity = SimilarityRoleModel(k=contract.similarity_k)
        self._fitted = False
        self._history_races = pd.DataFrame()
        self._history_entries = pd.DataFrame()
        self._history_results = pd.DataFrame()

    @staticmethod
    def _attach_dates(races: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
        if "race_date" in entries.columns:
            return entries
        if "race_date" not in races.columns:
            raise KeyError("races.csv requires race_date for chronological relationship construction")
        dates = races[["race_id", "race_date"]].drop_duplicates("race_id")
        return entries.merge(dates, on="race_id", how="left")

    @staticmethod
    def _sort_races(races: pd.DataFrame) -> pd.DataFrame:
        out = races.copy()
        out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
        keys = [c for c in ("race_date", "track", "race_no", "race_id") if c in out.columns]
        return out.sort_values(keys).reset_index(drop=True)

    def fit(self, races: pd.DataFrame, entries: pd.DataFrame, results: pd.DataFrame) -> "KeirinPredictionEngine":
        if races.empty or entries.empty or results.empty:
            raise ValueError("development races/entries/results must be non-empty")
        dated_entries = self._attach_dates(races, entries)
        pair_rows, tracker = build_pairwise_rows(
            dated_entries,
            results,
            relationship_tracker=RelationshipTracker(self.contract.relationship_prior_strength),
            update_relationships=True,
        )
        self.statistical.fit(pair_rows)
        self.tracker = tracker
        self.similarity.fit(races, dated_entries, results)
        self._history_races = races.copy()
        self._history_entries = dated_entries.copy()
        self._history_results = results.copy()
        self._fitted = True
        return self

    def predict_race(
        self,
        race_row: pd.DataFrame,
        entries_for_race: pd.DataFrame,
        policy: TicketPolicy | None = None,
    ) -> dict:
        if not self._fitted:
            raise RuntimeError("engine is not fitted")
        if len(entries_for_race) != 7:
            raise ValueError("v1 prediction engine is restricted to seven-rider races")
        rid = str(race_row.iloc[0]["race_id"])
        if set(entries_for_race["race_id"].astype(str)) != {rid}:
            raise ValueError("race_row and entries_for_race do not describe the same race")

        stat_pack = self.statistical.predict_race(entries_for_race, self.tracker)
        sim = self.similarity.predict_race(race_row, entries_for_race)
        blended = blend_trifecta(stat_pack["trifecta"], sim, self.contract.statistical_weight)
        marginals = marginals_from_trifecta(blended)
        decision = select_tickets(blended, stat_pack["trifecta"], sim, policy or TicketPolicy())

        meta = race_row.iloc[0].to_dict()
        return {
            "race_id": rid,
            "meta": meta,
            "pairwise": stat_pack["pairwise"],
            "strength": stat_pack["strength"],
            "statistical_trifecta": stat_pack["trifecta"],
            "similarity_trifecta": sim,
            "trifecta": blended,
            "marginals": marginals,
            "decision": decision,
        }

    def predict_segment(self, races: pd.DataFrame, entries: pd.DataFrame, policy: TicketPolicy | None = None) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("engine is not fitted")
        e = self._attach_dates(races, entries)
        by_race = {rid: g for rid, g in e.groupby("race_id", sort=False)}
        decisions: list[dict] = []
        for r in self._sort_races(races).itertuples(index=False):
            rid = str(getattr(r, "race_id"))
            if rid not in by_race or len(by_race[rid]) != 7:
                continue
            race_row = races[races["race_id"].astype(str) == rid].head(1)
            pack = self.predict_race(race_row, by_race[rid], policy=policy)
            d = pack["decision"]
            meta = pack["meta"]
            decisions.append({
                "race_id": rid,
                "race_date": meta.get("race_date"),
                "track": meta.get("track"),
                "race_no": meta.get("race_no"),
                "race_type": meta.get("race_type"),
                "buy": d["buy"],
                "tickets": d["tickets"],
                "stake_yen": d["stake_yen"],
                "skip_reasons": d["skip_reasons"],
                "metrics": d["metrics"],
                "top_trifecta": pack["trifecta"].head(20).to_dict("records"),
                "marginals": pack["marginals"].to_dict("records"),
            })
        return decisions

    def refresh_history(self, races: pd.DataFrame, entries: pd.DataFrame, results: pd.DataFrame) -> None:
        """Add a completed half-year to history without refitting statistical coefficients."""
        if not self._fitted:
            raise RuntimeError("engine is not fitted")
        dated_entries = self._attach_dates(races, entries)
        normalized = normalize_entries(dated_entries)
        result_groups = {rid: g for rid, g in results.groupby("race_id", sort=False)}
        by_race = {rid: g for rid, g in normalized.groupby("race_id", sort=False)}
        for r in self._sort_races(races).itertuples(index=False):
            rid = getattr(r, "race_id")
            if rid in by_race and rid in result_groups:
                self.tracker.update_race(by_race[rid], result_groups[rid])

        self._history_races = pd.concat([self._history_races, races], ignore_index=True)
        self._history_entries = pd.concat([self._history_entries, dated_entries], ignore_index=True)
        self._history_results = pd.concat([self._history_results, results], ignore_index=True)
        self.similarity.fit(self._history_races, self._history_entries, self._history_results)

    def manifest(self, policy: TicketPolicy | None = None) -> dict:
        return {
            "contract": asdict(self.contract),
            "policy": asdict(policy or TicketPolicy()),
            "history_races": int(self._history_races["race_id"].nunique()) if not self._history_races.empty else 0,
            "statistical_features": list(self.statistical.feature_columns),
            "sealed_rule": "2026_h1 results/payouts cannot be loaded without explicit unlock",
            "market_rule": "historical final odds never enter prediction or ticket selection",
        }


def fit_default_2024(root: str | Path, contract: EngineContract = EngineContract()) -> KeirinPredictionEngine:
    catalog = DatasetCatalog(Path(root))
    segments = catalog.by_role("development")
    races, entries = concat_pre_race(segments, seven_rider_only=contract.seven_rider_only)
    results = concat_labels(segments, unlock_sealed=False)
    valid_ids = set(races["race_id"])
    results = results[results["race_id"].isin(valid_ids)].copy()
    return KeirinPredictionEngine(contract).fit(races, entries, results)

from pathlib import Path

import pandas as pd
import pytest

from keirin_market_analysis.prediction_features import RelationshipTracker, add_race_relative_features, normalize_entries
from keirin_market_analysis.prediction_models import marginals_from_trifecta, plackett_luce_trifecta
from keirin_market_analysis.prediction_schema import (
    DatasetCatalog,
    LeakageError,
    SealedValidationError,
    load_labels,
    validate_pre_race_columns,
)
from keirin_market_analysis.prediction_strategy import TicketPolicy, select_tickets


def sample_entries() -> pd.DataFrame:
    return pd.DataFrame({
        "race_id": ["r1"] * 7,
        "player_name": [f"選手{i}" for i in range(1, 8)],
        "car_no": list(range(1, 8)),
        "age": ["30", "31", "32", "33", "34", "35", "36"],
        "gear": ["3.92"] * 7,
        "score": ["106.0", "104.0", "102.0", "101.0", "100.0", "99.0", "98.0"],
        "s_count": ["2", "1", "0", "0", "1", "0", "0"],
        "b_count": ["10", "1", "0", "7", "0", "3", "0"],
        "nige_count": ["5", "0", "0", "3", "0", "1", "0"],
        "makuri_count": ["2", "0", "0", "2", "0", "1", "0"],
        "sashi_count": ["0", "5", "2", "0", "4", "0", "2"],
        "mark_count": ["0", "3", "4", "0", "3", "0", "2"],
        "win_rate": ["25.0%", "20.0%", "12.0%", "18.0%", "10.0%", "8.0%", "6.0%"],
        "top2_rate": ["45.0%", "42.0%", "33.0%", "35.0%", "28.0%", "22.0%", "20.0%"],
        "top3_rate": ["60.0%", "58.0%", "50.0%", "48.0%", "40.0%", "35.0%", "30.0%"],
        "line_id": [1, 1, 1, 2, 2, 3, 3],
        "line_position": [1, 2, 3, 1, 2, 1, 2],
        "line_size": [3, 3, 3, 2, 2, 2, 2],
    })


def test_final_odds_columns_are_rejected():
    with pytest.raises(LeakageError):
        validate_pre_race_columns(["race_id", "score", "trifecta_final_odds"])


def test_actual_collector_aliases_and_percentages_are_normalized():
    out = normalize_entries(sample_entries())
    assert out.loc[0, "rider_key"] == "選手1"
    assert out.loc[0, "nige"] == 5
    assert out.loc[0, "quinella_rate"] == 45.0
    assert out.loc[0, "trio_rate"] == 60.0


def test_race_relative_score_feature_is_local():
    df = pd.concat([
        sample_entries(),
        sample_entries().assign(race_id="r2", score=["86", "84", "82", "81", "80", "79", "78"]),
    ], ignore_index=True)
    out = add_race_relative_features(df)
    r1 = out[out["race_id"] == "r1"]
    r2 = out[out["race_id"] == "r2"]
    assert round(float(r1.iloc[0]["score_delta"]), 6) == round(float(r2.iloc[0]["score_delta"]), 6)


def test_plackett_luce_is_exactly_210_for_seven_riders_and_normalized():
    tri = plackett_luce_trifecta({i: 8 - i for i in range(1, 8)})
    assert len(tri) == 210
    assert tri["combo"].nunique() == 210
    assert abs(float(tri["probability"].sum()) - 1.0) < 1e-10
    marg = marginals_from_trifecta(tri)
    assert abs(float(marg["p_first"].sum()) - 1.0) < 1e-10
    assert abs(float(marg["p_top2"].sum()) - 2.0) < 1e-10
    assert abs(float(marg["p_top3"].sum()) - 3.0) < 1e-10


def test_relationship_tracker_starts_shrunk_to_neutral():
    t = RelationshipTracker(prior_strength=8.0)
    f = t.features("A", "B")
    assert f["h2h_meetings"] == 0
    assert f["h2h_a_rate_shrunk"] == 0.5
    assert f["h2h_confidence"] == 0.0


def test_2026_labels_are_locked_before_any_file_is_opened(tmp_path: Path):
    catalog = DatasetCatalog(tmp_path)
    sealed = next(s for s in catalog.segments if s.name == "2026_h1")
    with pytest.raises(SealedValidationError):
        load_labels(sealed)


def test_price_blind_policy_can_skip_model_disagreement():
    stat = plackett_luce_trifecta({i: 8 - i for i in range(1, 8)})
    sim = stat.copy()
    sim["probability"] = sim["probability"].iloc[::-1].to_numpy()
    blended = stat.copy()
    policy = TicketPolicy(max_model_tv_distance=0.01)
    decision = select_tickets(blended, stat, sim, policy)
    assert decision["buy"] is False
    assert "model_disagreement" in decision["skip_reasons"]

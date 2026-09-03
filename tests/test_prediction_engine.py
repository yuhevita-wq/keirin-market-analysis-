import pandas as pd
import pytest

from keirin_prediction_engine import LeakageError, add_relative_features, similarity_neighbors, validate_pre_race_frame


def test_rejects_result_columns():
    df = pd.DataFrame({"race_id": ["r1"], "score": [100.0], "finish": [1]})
    with pytest.raises(LeakageError):
        add_relative_features(df)


def test_rejects_final_odds_columns():
    df = pd.DataFrame({"race_id": ["r1"], "score": [100.0], "trifecta_final_odds": [5.2]})
    with pytest.raises(LeakageError):
        validate_pre_race_frame(df)


def test_relative_score_features_are_race_local():
    df = pd.DataFrame({
        "race_id": ["r1", "r1", "r2", "r2"],
        "score": [100.0, 90.0, 80.0, 70.0],
        "line_id": [1, 2, 1, 2],
        "b": [10, 2, 8, 1],
    })
    out = add_relative_features(df)
    assert out.loc[0, "score_delta_mean"] == 5.0
    assert out.loc[2, "score_delta_mean"] == 5.0


def test_similarity_returns_nearest_first():
    history = pd.DataFrame({
        "race_id": ["a", "b", "c"],
        "score__max": [100.0, 110.0, 130.0],
        "score__std": [2.0, 5.0, 10.0],
    })
    target = pd.Series({"race_id": "t", "score__max": 109.0, "score__std": 5.5})
    out = similarity_neighbors(history, target, k=2)
    assert out.iloc[0]["race_id"] == "b"

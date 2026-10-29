"""Unknown competition scores are *not* low competition scores."""
import math
import pytest
from keirin_market_analysis.score_quality import (
    ScoreState, UnknownScoreError, classify_score,
    score_readiness, require_known_score_card,
)

@pytest.mark.parametrize("raw", [0, 0.0, "0", "0.0", "-0"])
def test_zero_score_is_unverified_unknown_not_weak(raw):
    d=classify_score(raw)
    assert d.state is ScoreState.UNVERIFIED_ZERO
    assert d.value is None

@pytest.mark.parametrize("raw", [None, "", "  "])
def test_missing_score_distinguished_from_zero(raw):
    assert classify_score(raw).state is ScoreState.MISSING

@pytest.mark.parametrize("raw", ["abc", -1, "-1", float("inf"),
                                  float("-inf"), float("nan"), True])
def test_invalid_score_does_not_get_score_rank(raw):
    assert classify_score(raw).state is ScoreState.INVALID

@pytest.mark.parametrize("raw", [10.0, "55.1", "70.50", 65])
def test_low_but_positive_is_known(raw):
    d=classify_score(raw)
    assert d.state is ScoreState.KNOWN
    assert d.value > 0

def test_partial_card_ranks_only_known_and_blocks_forecast():
    card={"1":93.1,"2":0,"3":69.2,"4":""}
    q=score_readiness(card)
    assert q.status == "HOLD_FOR_SCORE"
    assert q.known_score_ranks=={"1":1,"3":2}
    assert q.unknown_riders=={"2":"unverified_zero","4":"missing"}
    assert not q.can_predict
    with pytest.raises(UnknownScoreError):
        require_known_score_card(card)

def test_missing_expected_rider_detected_before_prediction():
    q=score_readiness({"1":75}, expected_riders=("1","2","3"))
    assert q.unknown_riders=={"2":"missing","3":"missing"}

def test_full_verified_card_unchanged():
    card={"1":69.4,"2":90.0,"3":80.1}
    q=score_readiness(card)
    assert q.can_predict
    assert q.known_score_ranks=={"2":1,"3":2,"1":3}
    assert require_known_score_card(card)==card

def test_negative_not_interpreted_as_known_low():
    assert score_readiness({"1":1,"2":-1}).status=="HOLD_FOR_SCORE"

def test_duplicate_numbers_cannot_hide_unknown():
    with pytest.raises(ValueError):
        score_readiness({1:0,"1":85.5})

def test_extra_car_not_allowed_when_explicit_lineup_supplied():
    with pytest.raises(ValueError):
        score_readiness({"1":80.0,"2":60.0},("1",))

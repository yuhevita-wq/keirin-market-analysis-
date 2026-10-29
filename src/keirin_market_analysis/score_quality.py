"""Score quality: unknown score != low strength. Never rank 0 as weakest.

A recorded score of 0.0 is an UNVERIFIED_ZERO (its raw source semantics
must be checked) rather than a verified measure of low ability. Missing
and invalid scores are separate states. Score-based prediction is placed on
HOLD until a separately validated uncertainty model is available.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping

class ScoreState(str, Enum):
    KNOWN = "known"
    UNVERIFIED_ZERO = "unverified_zero"
    MISSING = "missing"
    INVALID = "invalid"

@dataclass(frozen=True)
class RiderScore:
    state: ScoreState
    value: float | None
    @property
    def is_known(self) -> bool:
        return self.state is ScoreState.KNOWN

def classify_score(raw: object) -> RiderScore:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return RiderScore(ScoreState.MISSING, None)
    if isinstance(raw, bool):
        return RiderScore(ScoreState.INVALID, None)
    try:
        value = float(raw)
    except (ValueError, TypeError, OverflowError):
        return RiderScore(ScoreState.INVALID, None)
    if not isfinite(value) or value < 0:
        return RiderScore(ScoreState.INVALID, None)
    if value == 0:
        return RiderScore(ScoreState.UNVERIFIED_ZERO, None)
    return RiderScore(ScoreState.KNOWN, value)

@dataclass(frozen=True)
class ScoreReadiness:
    status: str
    known_scores: dict[str, float]
    known_score_ranks: dict[str, int]
    unknown_riders: dict[str, str]
    expected_riders: tuple[str, ...]
    @property
    def can_predict(self) -> bool:
        return self.status == "READY"

class UnknownScoreError(ValueError):
    """Never silently score unknown athletes as weak athletes."""

def score_readiness(
    scores: Mapping[str | int, object],
    expected_riders: tuple[str | int, ...] | None = None,
) -> ScoreReadiness:
    canonical: dict[str, object] = {}
    for car, value in scores.items():
        label = str(car)
        if label in canonical:
            raise ValueError(f"Duplicate car number: {label}")
        canonical[label] = value
    expected = (tuple(str(c) for c in expected_riders)
                if expected_riders is not None else tuple(canonical))
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Missing or duplicated expected rider list")
    if set(canonical) - set(expected):
        raise ValueError("Found scores for riders outside expected lineup")
    known: dict[str, float] = {}
    unknown: dict[str, str] = {}
    for car in expected:
        datum = classify_score(canonical.get(car))
        if datum.is_known:
            assert datum.value is not None
            known[car] = datum.value
        else:
            unknown[car] = datum.state.value
    # These are partial known-only ranks in HOLD cases. NEVER use them for
    # a full-field forecast before checking .can_predict.
    ordered = sorted(known, key=lambda c: (-known[c], c))
    ranks = {c: i for i, c in enumerate(ordered, 1)}
    return ScoreReadiness(
        status="READY" if not unknown else "HOLD_FOR_SCORE",
        known_scores=known, known_score_ranks=ranks,
        unknown_riders=unknown, expected_riders=expected)

def require_known_score_card(
    scores: Mapping[str | int, object],
    expected_riders: tuple[str | int, ...] | None = None,
) -> dict[str, float]:
    result = score_readiness(scores, expected_riders)
    if not result.can_predict:
        raise UnknownScoreError(
            f"Cannot give numeric score ranks to unknown rider scores: {result.unknown_riders}")
    return result.known_scores

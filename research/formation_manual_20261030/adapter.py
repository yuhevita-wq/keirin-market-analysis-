"""Formation proposal adapter for frozen V6/V6-C1 week-by-week studies.

Replaces odds-seeded single-ticket expansion with structural scenarios.
No result labels, payout files or odds influence shape construction.
No automatic purchase: admission needs independently validated evidence.
"""
from __future__ import annotations
from dataclasses import dataclass

try:
    from .formation import evaluate_set
    from .scenarios import construct_candidates
except ImportError:
    from formation import evaluate_set
    from scenarios import construct_candidates


@dataclass(frozen=True)
class ManualCandidate:
    label: str
    explanation: str
    forms: tuple[str, ...]
    points: int
    overlap_points: int
    ticket_set: frozenset
    estimated_hit_probability: float
    estimated_roi_pct: float
    synthetic_odds: float
    stake_yen: int
    low_EV_components: int


def generate_preview(event, model_probabilities, full_ordered_market, *, max_shapes=160,
                     stake_per_point=100, lowest_composite_odds=3.5):
    """Return competing independently described shapes and NO purchase.

    Event must contain ONLY pre-race rider scores and line roles. Odds are
    applied only after candidates are generated, to price the whole set.
    """
    if any(k in event for k in ("order", "outcome", "payout", "finish_order", "winner")):
        raise ValueError("Result label leaked into pre-race formation event")
    scores=event.get("scores")
    positions=event.get("positions")
    if not isinstance(scores,dict) or not positions:
        raise ValueError("Need genuine pre-race scores and published line positions")
    cars=tuple(sorted(scores,key=int))
    proposals=construct_candidates(cars,model_probabilities,positions,
                                   max_scenario_groups=max_shapes)
    result=[]
    for p in proposals:
        z=evaluate_set(p.pack.tickets,model_probabilities,full_ordered_market,
                       stake_yen=stake_per_point)
        if z["composite_odds"] < lowest_composite_odds:
            continue
        result.append(ManualCandidate(
            label=p.scenario,
            explanation=p.explanation,
            forms=tuple(sh.notation for sh in p.pack.shapes),
            points=z["points"],
            overlap_points=len(p.pack.overlap),
            ticket_set=p.pack.tickets,
            estimated_hit_probability=z["model_hit_probability"],
            estimated_roi_pct=z["model_expected_roi_pct"],
            synthetic_odds=z["composite_odds"],
            stake_yen=z["stake_yen"],
            low_EV_components=z["ticket_ev_under_100_count"],
        ))
    # Preliminary coverage ordering, NOT highest-EV seed driving form creation.
    result.sort(key=lambda a:(-a.estimated_hit_probability,
                              -a.estimated_roi_pct,a.points,a.label))
    bytype={}
    for p in result:
        if p.label not in bytype:
            bytype[p.label]=p
    return {
        "status":"候補のみ（未購入）",
        "decision":"見送り",
        "reason":"市場誤差を実測で確認する独立した購入判定が未承認",
        "total_scenario_candidates_generated":len(proposals),
        "price_floor_compatible_candidates":len(result),
        "alternatives":tuple(bytype.values()),
        "no_single_form_for_all_races":True,
        "all_purchase_logic_deleted":True,
        "source_model_unchanged":True,
    }


def authorize_purchase(preview, *, external_approval=None):
    """Fail closed. A bare positive expected ROI cannot trigger purchase."""
    if external_approval is None:
        return {"decision":"見送り","reason":"no independently validated race-entry decision"}
    if not isinstance(external_approval,dict) or external_approval.get("approved") is not True:
        return {"decision":"見送り","reason":"external validation absent/negative"}
    key=external_approval.get("candidate_label")
    proof=external_approval.get("pre_declared_validation_id")
    if not isinstance(key,str) or not isinstance(proof,str) or len(proof.strip())<8:
        return {"decision":"見送り","reason":"missing frozen validation/provenance"}
    alternatives=preview["alternatives"]
    matches=[x for x in alternatives if x.label==key]
    if len(matches)!=1:
        return {"decision":"見送り","reason":"candidate unknown or ambiguous"}
    pick=matches[0]
    return {
        "decision":"採用",
        "validation_reference":proof,
        "formation_package":pick.forms,
        "ticket_set":pick.ticket_set,
        "ticket_points":pick.points,
        "stake_yen":pick.stake_yen,
        "contains_sub100_EV_components":pick.low_EV_components>0,
        "reason":"independent scenario and externally verified race-entry reason",
    }

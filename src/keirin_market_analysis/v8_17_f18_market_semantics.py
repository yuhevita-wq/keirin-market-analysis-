from __future__ import annotations

"""v8.17-F18: race-type market semantics.

Core idea:
- Do not force one interpretation of market hierarchy across every race type.
- FINAL trusts strong first-place concentration after PS_AB.
- SELECTION does NOT treat PS_AB/H concentration as a winning-strength signal.
  It is a market-disagreement race type and is analysed through the gap between
  the 35-way trio market and the 210-way trifecta market aggregated back to
  the same 35 unordered three-rider sets.

No result or payout is used by this engine.
No Q1-fitted Selection threshold is introduced here.
"""

from math import log
from typing import Mapping

from v7_0_f01_market_hierarchy import implied_probabilities, trio_implied_probabilities
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_16_f17_final_revival import build_v8_16_f17

SCHEME_VERSION = "v8.17-F18"
STATUS = "DEVELOPMENT_RACE_TYPE_MARKET_SEMANTICS"


def _unordered(combo):
    return tuple(sorted(combo))


def selection_divergence_diagnostics(
    trio_odds: Mapping[tuple[int, int, int], float],
    trifecta_odds: Mapping[tuple[int, int, int], float],
) -> dict[str, object]:
    """Compare the same 35 rider-sets across trio and trifecta markets.

    P_trio(c): normalized trio implied probability for unordered set c.
    Q_tf_set(c): normalized trifecta implied probability aggregated across the
                 six permutations belonging to unordered set c.
    D_log(c): log(Q_tf_set(c) / P_trio(c)).

    Positive D means the trifecta market supports that three-rider set more
    strongly than the trio market; negative D means the opposite.
    """
    p = trio_implied_probabilities(trio_odds)
    q = implied_probabilities(trifecta_odds)

    q_set = {c: 0.0 for c in p}
    for order, prob in q.items():
        c = _unordered(order)
        if c in q_set:
            q_set[c] += prob

    d_log = {}
    abs_gap = {}
    for c, pv in p.items():
        qv = q_set.get(c, 0.0)
        if pv > 0.0 and qv > 0.0:
            d_log[c] = log(qv / pv)
        else:
            d_log[c] = None
        abs_gap[c] = abs(qv - pv)

    p_rank = tuple(sorted(p, key=lambda c: (-p[c], c)))
    q_rank = tuple(sorted(q_set, key=lambda c: (-q_set[c], c)))
    valid_d = [c for c in d_log if d_log[c] is not None]
    pos_d_rank = tuple(sorted(valid_d, key=lambda c: (-d_log[c], c)))
    neg_d_rank = tuple(sorted(valid_d, key=lambda c: (d_log[c], c)))

    tv_distance = 0.5 * sum(abs_gap.values())
    top3_overlap = len(set(p_rank[:3]) & set(q_rank[:3]))
    top5_overlap = len(set(p_rank[:5]) & set(q_rank[:5]))

    rows = []
    for c in p_rank:
        rows.append({
            "set": c,
            "P_trio": p[c],
            "Q_tf_set": q_set[c],
            "D_log": d_log[c],
            "abs_gap": abs_gap[c],
            "trio_rank": p_rank.index(c) + 1,
            "tf_set_rank": q_rank.index(c) + 1,
            "rank_shift": (p_rank.index(c) + 1) - (q_rank.index(c) + 1),
        })

    return {
        "selection_market_model": "TRIO_VS_TRIFECTA_SET_DIVERGENCE",
        "tv_distance": tv_distance,
        "top_set_same": p_rank[0] == q_rank[0],
        "top3_overlap": top3_overlap,
        "top5_overlap": top5_overlap,
        "trio_top_sets": p_rank[:5],
        "tf_set_top_sets": q_rank[:5],
        "largest_tf_over_trio": pos_d_rank[:5],
        "largest_trio_over_tf": neg_d_rank[:5],
        "set_rows": rows,
        "threshold_policy": "NO_Q1_FITTED_THRESHOLD",
    }


def build_v8_17_f18(
    trio_odds,
    trifecta_odds,
    predicted_line_formation: str,
    race_type: str,
):
    group = classify_race_type(race_type)

    # All live branches, including the adopted FINAL branch, are inherited
    # unchanged from v8.16-F17.
    if not (group == "SPECIAL" and race_type == "Ｓ級選抜"):
        out = build_v8_16_f17(
            trio_odds,
            trifecta_odds,
            predicted_line_formation,
            race_type,
        )
        return {**out, "scheme_version": SCHEME_VERSION}

    # Selection is formally redefined as a disagreement-model branch.
    # It remains diagnostic-only until a semantic, pre-race rule is established.
    diag = selection_divergence_diagnostics(trio_odds, trifecta_odds)
    return {
        "scheme_version": SCHEME_VERSION,
        "buy": False,
        "reason": "SELECTION_DIVERGENCE_MODEL_DIAGNOSTIC_ONLY",
        "race_type": race_type,
        "race_type_group": "SPECIAL",
        "special_subtype": "SELECTION",
        "branch_policy": "READ_MARKET_DISAGREEMENT_NOT_MARKET_STRENGTH",
        "selection_diagnostics": diag,
        "development_status": STATUS,
    }

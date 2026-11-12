from __future__ import annotations

from typing import Dict, Mapping, Tuple

from v8_20_f21_initial_special_unified import build_v8_20_f21, INITIAL_SPECIAL_LABELS

SCHEME_VERSION = 'v8.21-F22'
FINAL_LABEL = 'Ｓ級決勝'


def build_v8_21_f22(
    trio_odds: Mapping[Tuple[int, int, int], float],
    trifecta_odds: Mapping[Tuple[int, int, int], float],
    predicted_line_formation: str,
    race_type: str,
) -> Dict[str, object]:
    """v8.21-F22.

    Everything from v8.20-F21 is unchanged except FINAL.

    FINAL is quarantined after Q1+Q2 development diagnostics found no
    reproducible market-only heat structure. Total vote heat, heat location,
    price viability, line structure, trio/trifecta agreement, structural
    compression, and head-certainty x tail-diffusion all failed to reproduce
    across Q1 and Q2.

    This is deliberately a no-bet branch rather than an outcome-fitted rescue.
    """
    if race_type == FINAL_LABEL:
        base = build_v8_20_f21(
            trio_odds,
            trifecta_odds,
            predicted_line_formation,
            race_type,
        )
        return {
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'reason': 'FINAL_HEAT_NO_REPRODUCIBLE_EDGE_Q1Q2',
            'branch_policy': 'FINAL_HEAT_QUARANTINE_DIAGNOSTIC_ONLY',
            'prior_candidate_buy': bool(base.get('buy')),
            'prior_candidate_reason': base.get('reason'),
            'prior_candidate_formation': base.get('formation'),
            'heat_research_status': 'TOTAL_AND_LOCATION_HEAT_OBSERVED_BUT_NO_REPRODUCIBLE_VALUE_EDGE',
            'individual_ticket_pruning': False,
            'outcome_fitted_final_cutoff': False,
        }

    out = build_v8_20_f21(
        trio_odds,
        trifecta_odds,
        predicted_line_formation,
        race_type,
    )
    out = dict(out)
    out['scheme_version'] = SCHEME_VERSION
    return out

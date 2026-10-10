from __future__ import annotations

"""v8.22-F23 candidate: QUALIFYING single-pole hard-narrow rebuild.

Only QUALIFYING changes from v8.21-F22.

Development semantics from Q1+Q2 diagnostics:
- qualifying is better suited to a hard/narrow branch than a broad hole branch;
- generic H concentration is not enough;
- the promising structural state is one-sided head concentration rather than neat A/B head agreement.

QUALIFYING rule:
1. F09 PS_AB must pass.
2. Require H_CONCENTRATED: H1 >= 2*H2.
3. Require H_AB == False. This means the two strongest H riders are NOT split one each across the two strongest lines A/B; the state is interpreted as a single dominant head block rather than balanced two-line consensus.
4. Translate that state into the formation: first place is H1 only.
5. Preserve F09 second/third pools initially.
6. Structural price contraction may remove whole riders only from second/third pools; first-place H1 is protected.
7. No individual ticket pruning and no fitted odds cutoff.

Q1 and Q2 are development data for this branch. Q3 remains the next untouched validation set only after this rule is frozen.
"""

from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _select_structural_knee
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_18_f19_semifinal_h_adaptive import _protected_price_path
from v8_21_f22_final_heat_quarantine import build_v8_21_f22

SCHEME_VERSION = 'v8.22-F23'
BASE_SCHEME = 'v8.21-F22'
STATUS = 'DEVELOPMENT_Q1Q2_QUALIFYING_SINGLE_POLE'


def _build_qualifying(trio_odds, trifecta_odds, predicted_line_formation, race_type: str):
    base = build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get('buy'):
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'race_type': race_type,
            'race_type_group': 'QUALIFYING',
            'branch_policy': 'PS_AB + H_CONCENTRATED + H_AB0 + H1_PROTECTED_FIRST',
        }

    eg = dict(base.get('entry_gate') or {})
    h_concentrated = not bool(eg.get('H_RATIO'))
    h_ab = bool(eg.get('H_AB'))
    if not h_concentrated:
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'race_type': race_type,
            'race_type_group': 'QUALIFYING',
            'reason': 'QUALIFYING_H_NOT_CONCENTRATED',
            'branch_policy': 'PS_AB + H_CONCENTRATED + H_AB0 + H1_PROTECTED_FIRST',
        }
    if h_ab:
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'race_type': race_type,
            'race_type_group': 'QUALIFYING',
            'reason': 'QUALIFYING_TWO_LINE_HEAD_AGREEMENT_REJECTED',
            'branch_policy': 'PS_AB + H_CONCENTRATED + H_AB0 + H1_PROTECTED_FIRST',
        }

    top2 = tuple(int(x) for x in eg.get('top2_H') or ())
    if len(top2) < 2:
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'race_type': race_type,
            'race_type_group': 'QUALIFYING',
            'reason': 'QUALIFYING_MISSING_H_TOP2',
        }

    first = (top2[0],)
    q = implied_probabilities(trifecta_odds)
    s0 = _state(first, base['second'], base['third'], q)
    if s0 is None:
        return {
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'race_type': race_type,
            'race_type_group': 'QUALIFYING',
            'reason': 'QUALIFYING_INVALID_SINGLE_POLE_RECTANGLE',
            'branch_policy': 'PS_AB + H_CONCENTRATED + H_AB0 + H1_PROTECTED_FIRST',
        }

    p0 = _price_state(s0, s0.q_mass, q, trifecta_odds)
    states, moves = _protected_price_path(p0, q, trifecta_odds)
    idx = _select_structural_knee(states)
    chosen = states[idx]

    return {
        **base,
        'scheme_version': SCHEME_VERSION,
        'base_scheme': BASE_SCHEME,
        'buy': True,
        'race_type': race_type,
        'race_type_group': 'QUALIFYING',
        'formation_before_price': s0.display,
        'ticket_count_before_price': s0.ticket_count,
        'formation': chosen.display,
        'first': chosen.state.first,
        'second': chosen.state.second,
        'third': chosen.state.third,
        'tickets': chosen.state.tickets,
        'ticket_count': chosen.state.ticket_count,
        'q_mass': chosen.state.q_mass,
        'H_CONCENTRATED': True,
        'H_AB': False,
        'H1_anchor': first[0],
        'price_compression_steps': idx,
        'price_path_steps_total': len(moves),
        'price_q_retention': chosen.q_retention,
        'price_weighted_gm_odds': chosen.weighted_gm_odds,
        'price_gm_return_multiple': chosen.gm_return_multiple,
        'price_profitable_q_share': chosen.profitable_q_share,
        'first_pool_protected_during_price_phase': True,
        'individual_ticket_pruning': False,
        'fixed_first_count_rule': False,
        'branch_policy': 'QUALIFYING single-pole: PS_AB + H_CONCENTRATED + H_AB=0; first=H1; keep F09 2nd/3rd then whole-rider price contraction only in 2nd/3rd',
        'development_status': STATUS,
    }


def build_v8_22_f23(trio_odds, trifecta_odds, predicted_line_formation: str, race_type: str):
    if classify_race_type(race_type) == 'QUALIFYING':
        return _build_qualifying(trio_odds, trifecta_odds, predicted_line_formation, race_type)

    out = build_v8_21_f22(trio_odds, trifecta_odds, predicted_line_formation, race_type)
    out = dict(out)
    out['scheme_version'] = SCHEME_VERSION
    return out

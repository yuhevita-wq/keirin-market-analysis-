from __future__ import annotations

"""v8.19-F20: exact-race-type SPECIAL rebuild.

Only the SPECIAL group changes from v8.18-F19.

Development interpretation from Q1+Q2:
- S級初特選 is the only SPECIAL exact type given an active rebuilt branch.
- S級特選, S級初日特選, and S級選抜 remain no-bet development branches.

Active S級初特選 rule:
1. F09 PS_AB must pass.
2. Require pre-existing H-state consistency: H_AB == H_CONCENTRATED.
3. Translate the H state into the first-place pool:
   - concentrated: H1 only
   - balanced: top two H riders
4. Keep F09 second/third pools.
5. Price contraction may remove whole riders only from second/third pools;
   the H-derived first pool is protected.

No fitted numeric cutoff, rider ability, result, payout, or individual-ticket
pruning is used to generate a bet.
"""

from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _select_structural_knee
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_18_f19_semifinal_h_adaptive import build_v8_18_f19, _protected_price_path

SCHEME_VERSION = 'v8.19-F20'
BASE_SCHEME = 'v8.18-F19'
STATUS = 'DEVELOPMENT_Q1Q2_SPECIAL_EXACT_TYPE_REBUILD'


def _build_initial_special(trio_odds, trifecta_odds, predicted_line_formation):
    base = build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get('buy'):
        return {**base, 'scheme_version': SCHEME_VERSION, 'branch_policy': 'INITIAL_SPECIAL_PS_AB_H_STATE_ADAPTIVE'}

    eg = dict(base.get('entry_gate') or {})
    h_ab = bool(eg.get('H_AB'))
    h_conc = not bool(eg.get('H_RATIO'))
    if h_ab != h_conc:
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'reason': 'INITIAL_SPECIAL_H_STATE_MISMATCH',
            'branch_policy': 'PS_AB + H_STATE_CONSISTENCY + H_ADAPTIVE_FIRST',
        }

    top2 = tuple(int(x) for x in eg.get('top2_H') or ())
    if len(top2) < 2:
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'reason': 'INITIAL_SPECIAL_MISSING_H_TOP2',
        }

    first = (top2[0],) if h_conc else tuple(sorted(top2[:2]))
    q = implied_probabilities(trifecta_odds)
    s0 = _state(first, base['second'], base['third'], q)
    if s0 is None:
        return {
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'reason': 'INITIAL_SPECIAL_INVALID_H_ADAPTIVE_RECTANGLE',
            'branch_policy': 'PS_AB + H_STATE_CONSISTENCY + H_ADAPTIVE_FIRST',
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
        'formation_before_price': s0.display,
        'ticket_count_before_price': s0.ticket_count,
        'formation': chosen.display,
        'first': chosen.state.first,
        'second': chosen.state.second,
        'third': chosen.state.third,
        'tickets': chosen.state.tickets,
        'ticket_count': chosen.state.ticket_count,
        'q_mass': chosen.state.q_mass,
        'H_CONCENTRATED': h_conc,
        'H_AB': h_ab,
        'H_first_pool': first,
        'price_compression_steps': idx,
        'price_path_steps_total': len(moves),
        'price_q_retention': chosen.q_retention,
        'price_weighted_gm_odds': chosen.weighted_gm_odds,
        'price_gm_return_multiple': chosen.gm_return_multiple,
        'price_profitable_q_share': chosen.profitable_q_share,
        'first_pool_protected_during_price_phase': True,
        'individual_ticket_pruning': False,
        'branch_policy': 'S級初特選: PS_AB + (H_AB == H_CONCENTRATED); concentrated=>1st H1, balanced=>1st top2 H; protect first pool',
        'development_status': STATUS,
    }


def build_v8_19_f20(trio_odds, trifecta_odds, predicted_line_formation: str, race_type: str):
    if classify_race_type(race_type) != 'SPECIAL':
        out = build_v8_18_f19(trio_odds, trifecta_odds, predicted_line_formation, race_type)
        return {**out, 'scheme_version': SCHEME_VERSION}

    rt = (race_type or '').strip()
    if rt == 'Ｓ級初特選':
        return _build_initial_special(trio_odds, trifecta_odds, predicted_line_formation)

    if rt == 'Ｓ級選抜':
        # Preserve the existing divergence diagnostic-only branch.
        out = build_v8_18_f19(trio_odds, trifecta_odds, predicted_line_formation, race_type)
        return {**out, 'scheme_version': SCHEME_VERSION}

    if rt == 'Ｓ級特選':
        return {
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'reason': 'ORDINARY_SPECIAL_QUARANTINED_Q1Q2_UNSTABLE',
            'race_type': rt,
            'race_type_group': 'SPECIAL',
            'branch_policy': 'NO_BET_PENDING_DISTINCT_ORDINARY_SPECIAL_MODEL',
            'development_status': STATUS,
        }

    if rt == 'Ｓ級初日特選':
        return {
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'reason': 'INITIAL_DAY_SPECIAL_QUARANTINED_Q1Q2_UNSTABLE',
            'race_type': rt,
            'race_type_group': 'SPECIAL',
            'branch_policy': 'NO_BET_PENDING_DISTINCT_INITIAL_DAY_MODEL',
            'development_status': STATUS,
        }

    return {
        'scheme_version': SCHEME_VERSION,
        'buy': False,
        'reason': 'UNMAPPED_SPECIAL_EXACT_TYPE',
        'race_type': rt,
        'race_type_group': 'SPECIAL',
        'development_status': STATUS,
    }

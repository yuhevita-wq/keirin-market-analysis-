from __future__ import annotations

"""v8.20-F21: unify S級初特選 and S級初日特選 as one INITIAL_SPECIAL branch.

Correction to v8.19-F20: these two labels are alternate names for the same
first-day S-class special-selection race concept and must not be modeled as
separate race regimes.

Only the SPECIAL routing changes from v8.19-F20.

Unified INITIAL_SPECIAL rule for both labels:
1. F09 PS_AB must pass.
2. Require existing H-state consistency: H_AB == H_CONCENTRATED.
3. Translate H state into first-place pool:
   - concentrated: H1 only
   - balanced: top two H riders
4. Keep F09 second/third pools.
5. Price contraction may remove whole riders only from second/third pools;
   the H-derived first pool is protected.

S級特選 remains quarantined. S級選抜 remains divergence diagnostic only.
No fitted numeric cutoff, rider ability, result, payout, or individual-ticket
pruning is used to generate a bet.
"""

from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _select_structural_knee
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_18_f19_semifinal_h_adaptive import _protected_price_path
from v8_19_f20_special_exact_type_rebuild import build_v8_19_f20

SCHEME_VERSION = 'v8.20-F21'
BASE_SCHEME = 'v8.19-F20'
STATUS = 'DEVELOPMENT_Q1Q2_INITIAL_SPECIAL_LABEL_UNIFIED'
INITIAL_SPECIAL_LABELS = {'Ｓ級初特選', 'Ｓ級初日特選'}


def _build_initial_special(trio_odds, trifecta_odds, predicted_line_formation, race_type: str):
    base = build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get('buy'):
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'race_type': race_type,
            'branch_policy': 'INITIAL_SPECIAL_UNIFIED_PS_AB_H_STATE_ADAPTIVE',
        }

    eg = dict(base.get('entry_gate') or {})
    h_ab = bool(eg.get('H_AB'))
    h_conc = not bool(eg.get('H_RATIO'))
    if h_ab != h_conc:
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'race_type': race_type,
            'reason': 'INITIAL_SPECIAL_H_STATE_MISMATCH',
            'branch_policy': 'PS_AB + H_STATE_CONSISTENCY + H_ADAPTIVE_FIRST',
        }

    top2 = tuple(int(x) for x in eg.get('top2_H') or ())
    if len(top2) < 2:
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'race_type': race_type,
            'reason': 'INITIAL_SPECIAL_MISSING_H_TOP2',
        }

    first = (top2[0],) if h_conc else tuple(sorted(top2[:2]))
    q = implied_probabilities(trifecta_odds)
    s0 = _state(first, base['second'], base['third'], q)
    if s0 is None:
        return {
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'race_type': race_type,
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
        'race_type': race_type,
        'race_type_group': 'SPECIAL',
        'special_subtype': 'INITIAL_SPECIAL',
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
        'branch_policy': 'INITIAL_SPECIAL unified (初特選/初日特選): PS_AB + (H_AB == H_CONCENTRATED); concentrated=>1st H1, balanced=>1st top2 H; protect first pool',
        'development_status': STATUS,
    }


def build_v8_20_f21(trio_odds, trifecta_odds, predicted_line_formation: str, race_type: str):
    if classify_race_type(race_type) != 'SPECIAL':
        out = build_v8_19_f20(trio_odds, trifecta_odds, predicted_line_formation, race_type)
        return {**out, 'scheme_version': SCHEME_VERSION}

    rt = (race_type or '').strip()
    if rt in INITIAL_SPECIAL_LABELS:
        return _build_initial_special(trio_odds, trifecta_odds, predicted_line_formation, rt)

    out = build_v8_19_f20(trio_odds, trifecta_odds, predicted_line_formation, race_type)
    return {**out, 'scheme_version': SCHEME_VERSION}

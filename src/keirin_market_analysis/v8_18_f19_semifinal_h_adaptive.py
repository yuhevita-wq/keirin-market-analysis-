from __future__ import annotations

"""v8.18-F19: semifinal H-adaptive first-place market translation.

Only the SEMIFINAL branch changes from v8.17-F18.

Semifinal rule:
1. Require F09 PS_AB entry structure.
2. Require H_AB: the two strongest first-place H riders must map one each to
   the two strongest lines A/B.
3. Translate the H concentration state into the first-place pool itself:
   - H1 >= 2*H2: first = H1 only.
   - H1 < 2*H2: first = top two H riders.
4. Preserve F09 second/third pools.
5. After the complete clean formation exists, allow structural whole-rider
   price contraction only in second/third pools. The H-derived first pool is
   protected and cannot be removed by the price phase.

No rider ability, result, payout, individual-ticket pruning, fixed point count,
or Q1/Q2-optimized numeric threshold is used by this engine.
"""

from dataclasses import dataclass
from math import log

from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _select_structural_knee
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_17_f18_market_semantics import build_v8_17_f18

SCHEME_VERSION = 'v8.18-F19'
BASE_SCHEME = 'v8.17-F18'
STATUS = 'DEVELOPMENT_Q1Q2_SEMIFINAL_REBUILD'


@dataclass(frozen=True)
class _Move:
    place: int
    rider: int
    q_loss: float
    gain: float
    efficiency: float
    next_state: object


def _protected_price_path(p0, q, trifecta_odds):
    states = [p0]
    moves = []
    cur = p0
    base_q_mass = p0.state.q_mass
    while True:
        candidates = []
        s = cur.state
        for place, pool in ((2, s.second), (3, s.third)):
            if len(pool) <= 1:
                continue
            for rider in pool:
                f2, f3 = s.second, s.third
                if place == 2:
                    f2 = tuple(x for x in f2 if x != rider)
                else:
                    f3 = tuple(x for x in f3 if x != rider)
                nxt = _state(s.first, f2, f3, q)
                if nxt is None:
                    continue
                pn = _price_state(nxt, base_q_mass, q, trifecta_odds)
                q_loss = s.q_mass - nxt.q_mass
                if q_loss <= 0:
                    continue
                gain = log(pn.gm_return_multiple) - log(cur.gm_return_multiple)
                if gain <= 0:
                    continue
                candidates.append(_Move(place, rider, q_loss, gain, gain / q_loss, pn))
        if not candidates:
            break
        move = max(candidates, key=lambda m: (m.efficiency, -m.q_loss, -m.place, -m.rider))
        moves.append(move)
        cur = move.next_state
        states.append(cur)
    return states, moves


def _build_semifinal(trio_odds, trifecta_odds, predicted_line_formation):
    base = build_v8_8_f09(trio_odds, trifecta_odds, predicted_line_formation)
    if not base.get('buy'):
        return {**base, 'scheme_version': SCHEME_VERSION, 'branch_policy': 'SEMIFINAL_PS_AB_H_ADAPTIVE'}

    eg = dict(base.get('entry_gate') or {})
    if not bool(eg.get('H_AB')):
        return {
            **base,
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'reason': 'SEMIFINAL_H_TOP2_NOT_AB',
            'branch_policy': 'SEMIFINAL_PS_AB_PLUS_H_AB_H_ADAPTIVE_FIRST',
        }

    h_top2 = tuple(int(x) for x in eg['top2_H'])
    h_concentrated = not bool(eg.get('H_RATIO'))
    first = (h_top2[0],) if h_concentrated else tuple(sorted(h_top2[:2]))

    q = implied_probabilities(trifecta_odds)
    base_state = _state(first, base['second'], base['third'], q)
    if base_state is None:
        return {
            'scheme_version': SCHEME_VERSION,
            'buy': False,
            'reason': 'SEMIFINAL_INVALID_H_ADAPTIVE_RECTANGLE',
            'branch_policy': 'SEMIFINAL_PS_AB_PLUS_H_AB_H_ADAPTIVE_FIRST',
        }

    p0 = _price_state(base_state, base_state.q_mass, q, trifecta_odds)
    states, moves = _protected_price_path(p0, q, trifecta_odds)
    selected_i = _select_structural_knee(states)
    chosen = states[selected_i]

    return {
        **base,
        'scheme_version': SCHEME_VERSION,
        'base_scheme': BASE_SCHEME,
        'buy': True,
        'formation_before_price': base_state.display,
        'ticket_count_before_price': base_state.ticket_count,
        'formation': chosen.display,
        'first': chosen.state.first,
        'second': chosen.state.second,
        'third': chosen.state.third,
        'tickets': chosen.state.tickets,
        'ticket_count': chosen.state.ticket_count,
        'q_mass': chosen.state.q_mass,
        'H_CONCENTRATED': h_concentrated,
        'H_AB': True,
        'H_first_pool': first,
        'price_compression_steps': selected_i,
        'price_path_steps_total': len(moves),
        'price_q_retention': chosen.q_retention,
        'price_weighted_gm_odds': chosen.weighted_gm_odds,
        'price_gm_return_multiple': chosen.gm_return_multiple,
        'price_profitable_q_share': chosen.profitable_q_share,
        'individual_ticket_pruning': False,
        'first_pool_protected_during_price_phase': True,
        'branch_policy': 'PS_AB + H_AB; H concentrated=>1st H1, balanced=>1st top2 H; keep F09 2nd/3rd; price may contract only 2nd/3rd',
        'development_status': STATUS,
    }


def build_v8_18_f19(trio_odds, trifecta_odds, predicted_line_formation: str, race_type: str):
    if classify_race_type(race_type) != 'SEMIFINAL':
        out = build_v8_17_f18(trio_odds, trifecta_odds, predicted_line_formation, race_type)
        return {**out, 'scheme_version': SCHEME_VERSION}
    return _build_semifinal(trio_odds, trifecta_odds, predicted_line_formation)

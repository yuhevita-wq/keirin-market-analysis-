from __future__ import annotations

"""Semifinal rebuild diagnostic using 2024Q1+Q2 as development data.

Q2 has already been observed, so it is no longer treated as untouched OOS here.
No optimized numeric threshold is searched. Candidate variants use only existing
market semantics already defined in the project:
- PS_AB
- H_CONCENTRATED (H1_top >= 2*H1_second)
- H_AB
- H-state consistency
- preservation of the H1 concentration cliff as a first-place anchor
- existing whole-rider structural price compression
"""

from collections import defaultdict
import json

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _compression_path, _select_structural_knee
from v7_0_f01_market_hierarchy import implied_probabilities

STAKE = 100


def compress_state(state, q, tf_odds):
    p0 = _price_state(state, state.q_mass, q, tf_odds)
    states, moves = _compression_path(p0, q, tf_odds)
    k = _select_structural_knee(states)
    chosen = states[k]
    return {
        'buy': True,
        'formation': chosen.display,
        'tickets': chosen.state.tickets,
        'ticket_count': chosen.state.ticket_count,
        'q_mass': chosen.state.q_mass,
        'price_steps': k,
        'price_gm_return_multiple': chosen.gm_return_multiple,
        'price_profitable_q_share': chosen.profitable_q_share,
        'price_q_retention': chosen.q_retention,
    }


def candidate(tf_odds, trio_odds, line, mode):
    base = build_v8_8_f09(trio_odds, tf_odds, line)
    if not base.get('buy'):
        return {'buy': False, 'reason': base.get('reason')}

    eg = dict(base.get('entry_gate') or {})
    h_conc = not bool(eg.get('H_RATIO'))
    h_ab = bool(eg.get('H_AB'))
    h_consistent = (h_ab == h_conc)

    if mode in {'H_CONC_F09','H_CONC_HAB_F09','H_CONC_ANCHOR','H_CONC_HAB_ANCHOR'} and not h_conc:
        return {'buy': False, 'reason': 'H_NOT_CONCENTRATED'}
    if mode in {'H_CONC_HAB_F09','H_CONC_HAB_ANCHOR'} and not h_ab:
        return {'buy': False, 'reason': 'H_NOT_AB'}
    if mode == 'H_STATE_CONSISTENT_F09' and not h_consistent:
        return {'buy': False, 'reason': 'H_STATE_MISMATCH'}

    q = implied_probabilities(tf_odds)
    if mode in {'H_CONC_ANCHOR','H_CONC_HAB_ANCHOR'}:
        anchor = int(eg['top2_H'][0])
        state = _state((anchor,), base['second'], base['third'], q)
        if state is None:
            return {'buy': False, 'reason': 'INVALID_ANCHORED_RECTANGLE'}
        out = compress_state(state, q, tf_odds)
        out.update({'anchor': anchor, 'H_CONCENTRATED': h_conc, 'H_AB': h_ab, 'H_STATE_CONSISTENT': h_consistent})
        return out

    state = _state(base['first'], base['second'], base['third'], q)
    if state is None:
        return {'buy': False, 'reason': 'INVALID_BASE_FORMATION'}
    out = compress_state(state, q, tf_odds)
    out.update({'H_CONCENTRATED': h_conc, 'H_AB': h_ab, 'H_STATE_CONSISTENT': h_consistent})
    return out


def summary(rows):
    n = len(rows)
    h = sum(r['hit'] for r in rows)
    t = sum(r['tickets'] for r in rows)
    p = sum(r['payout'] for r in rows)
    s = t * STAKE
    profitable_hits = sum(1 for r in rows if r['hit'] and r['payout'] > r['tickets'] * STAKE)
    losing_hits = sum(1 for r in rows if r['hit'] and r['payout'] < r['tickets'] * STAKE)
    return {
        'bets': n,
        'hits': h,
        'hit_rate_pct': 100*h/n if n else None,
        'tickets': t,
        'avg_tickets': t/n if n else None,
        'stake_yen': s,
        'payout_yen': p,
        'profit_yen': p-s,
        'roi_pct': 100*p/s if s else None,
        'profitable_hits': profitable_hits,
        'losing_hits': losing_hits,
    }


def valid_semifinals(data):
    races, trio, tf, pay = data
    for rid, r in sorted(races.items(), key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade') != 'F1' or r.get('race_type') != 'Ｓ級準決勝':
            continue
        if pi(r.get('entry_count')) != 7 or len(trio.get(rid,{})) != 35 or len(tf.get(rid,{})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get('predicted_line_formation'))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars) or rid not in pay or not pay[rid]:
            continue
        yield rid, r, trio[rid], tf[rid], pay[rid]


def evaluate_dataset(label, data, modes):
    rows = {m: [] for m in modes}
    population = 0
    ps_ab_pass = 0
    h_state = defaultdict(int)

    for rid, r, trio, tf, pay in valid_semifinals(data):
        population += 1
        base = build_v8_8_f09(trio, tf, r.get('predicted_line_formation') or '')
        if base.get('buy'):
            ps_ab_pass += 1
            eg = base.get('entry_gate') or {}
            key = f"HAB{int(bool(eg.get('H_AB')))}_HCONC{int(not bool(eg.get('H_RATIO')))}"
            h_state[key] += 1

        for mode in modes:
            d = candidate(tf, trio, r.get('predicted_line_formation') or '', mode)
            if not d.get('buy'):
                continue
            wins = [t for t in d['tickets'] if t in pay]
            payout = sum(pay[t] for t in wins)
            rows[mode].append({
                'dataset': label,
                'race_id': rid,
                'race_date': r.get('race_date'),
                'hit': int(bool(wins)),
                'payout': payout,
                'tickets': int(d['ticket_count']),
                'formation': d.get('formation'),
                'gm': d.get('price_gm_return_multiple'),
                'pqs': d.get('price_profitable_q_share'),
                'qret': d.get('price_q_retention'),
            })

    return {
        'dataset': label,
        'population': population,
        'ps_ab_pass': ps_ab_pass,
        'h_state_population_among_ps_ab': dict(sorted(h_state.items())),
        'candidates': {m: summary(rows[m]) for m in modes},
        '_rows': rows,
    }


def main():
    modes = [
        'PS_AB_F09',
        'H_CONC_F09',
        'H_CONC_HAB_F09',
        'H_STATE_CONSISTENT_F09',
        'H_CONC_ANCHOR',
        'H_CONC_HAB_ANCHOR',
    ]
    q1 = evaluate_dataset('2024Q1', load(), modes)
    q2 = evaluate_dataset('2024Q2', load_q2(), modes)

    combined = {}
    for m in modes:
        combined[m] = summary(q1['_rows'][m] + q2['_rows'][m])

    # Natural, non-optimized price diagnostics on the anchored candidate only.
    # Boundaries are semantic: break-even GM >= 1 and majority profitable q >= 0.5.
    anchor_price = {}
    for label, block in [('2024Q1', q1), ('2024Q2', q2)]:
        base_rows = block['_rows']['H_CONC_ANCHOR']
        anchor_price[label] = {
            'ALL': summary(base_rows),
            'GM_GE_1': summary([r for r in base_rows if r['gm'] is not None and r['gm'] >= 1.0]),
            'PQS_GE_0_5': summary([r for r in base_rows if r['pqs'] is not None and r['pqs'] >= 0.5]),
            'GM_GE_1_AND_PQS_GE_0_5': summary([r for r in base_rows if r['gm'] is not None and r['gm'] >= 1.0 and r['pqs'] is not None and r['pqs'] >= 0.5]),
        }
    all_anchor = q1['_rows']['H_CONC_ANCHOR'] + q2['_rows']['H_CONC_ANCHOR']
    anchor_price['Q1Q2_COMBINED'] = {
        'ALL': summary(all_anchor),
        'GM_GE_1': summary([r for r in all_anchor if r['gm'] is not None and r['gm'] >= 1.0]),
        'PQS_GE_0_5': summary([r for r in all_anchor if r['pqs'] is not None and r['pqs'] >= 0.5]),
        'GM_GE_1_AND_PQS_GE_0_5': summary([r for r in all_anchor if r['gm'] is not None and r['gm'] >= 1.0 and r['pqs'] is not None and r['pqs'] >= 0.5]),
    }

    for block in (q1,q2):
        block.pop('_rows',None)

    result = {
        'analysis': 'SEMIFINAL_REBUILD_Q1Q2_DEVELOPMENT',
        'q2_status_note': 'Q2 has already been observed and is now development data; next untouched OOS must be Q3.',
        'candidate_policy': 'existing semantic market states only; no optimized numeric threshold search',
        'Q1': q1,
        'Q2': q2,
        'Q1Q2_COMBINED': combined,
        'ANCHOR_PRICE_NATURAL_BOUNDARIES': anchor_price,
    }
    print('SEMIFINAL_REBUILD_Q1Q2_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('SEMIFINAL_REBUILD_Q1Q2_END')


if __name__ == '__main__':
    main()

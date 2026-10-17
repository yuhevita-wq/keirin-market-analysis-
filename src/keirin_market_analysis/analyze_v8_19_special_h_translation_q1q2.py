from __future__ import annotations

import json
from collections import defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _select_structural_knee
from v8_15_f16_special_split_rebuild import special_subtype
from v8_18_f19_semifinal_h_adaptive import _protected_price_path

STAKE = 100

CANDIDATES = (
    'CURRENT',
    'H1_ANCHOR',
    'H1_ANCHOR_HAB',
    'H_ADAPTIVE_HAB',
    'INIT_ADAPT_CONSISTENT',
    'INIT_H1_HAB_CONC',
    'INIT_BALANCED_HAB0',
)


def summarize(rows):
    n = len(rows)
    hits = sum(r['hit'] for r in rows)
    tickets = sum(r['tickets'] for r in rows)
    payout = sum(r['payout'] for r in rows)
    stake = tickets * STAKE
    return {
        'bets': n,
        'hits': hits,
        'hit_rate_pct': 100 * hits / n if n else None,
        'tickets': tickets,
        'avg_tickets': tickets / n if n else None,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else None,
        'profitable_hits': sum(r['hit'] and r['payout'] > r['tickets'] * STAKE for r in rows),
        'losing_hits': sum(r['hit'] and r['payout'] < r['tickets'] * STAKE for r in rows),
    }


def translated(base, trio_odds, trifecta_odds, mode):
    eg = dict(base.get('entry_gate') or {})
    h_ab = bool(eg.get('H_AB'))
    h_conc = not bool(eg.get('H_RATIO'))
    top2 = tuple(int(x) for x in eg.get('top2_H') or ())
    if len(top2) < 2:
        return None

    if mode == 'H1':
        first = (top2[0],)
    elif mode == 'ADAPT':
        first = (top2[0],) if h_conc else tuple(sorted(top2[:2]))
    elif mode == 'TOP2':
        first = tuple(sorted(top2[:2]))
    else:
        raise ValueError(mode)

    q = implied_probabilities(trifecta_odds)
    s0 = _state(first, base['second'], base['third'], q)
    if s0 is None:
        return None
    p0 = _price_state(s0, s0.q_mass, q, trifecta_odds)
    states, moves = _protected_price_path(p0, q, trifecta_odds)
    idx = _select_structural_knee(states)
    chosen = states[idx]
    return {
        'tickets': chosen.state.tickets,
        'ticket_count': chosen.state.ticket_count,
        'formation': chosen.display,
        'h_ab': h_ab,
        'h_conc': h_conc,
        'first_pool': first,
        'price_steps': idx,
    }


def candidate_decisions(base, trio_odds, trifecta_odds, subtype):
    if not base.get('buy'):
        return {}
    eg = dict(base.get('entry_gate') or {})
    h_ab = bool(eg.get('H_AB'))
    h_conc = not bool(eg.get('H_RATIO'))
    out = {}

    if subtype == 'SPECIAL':
        if h_conc:
            out['CURRENT'] = {'tickets': base['tickets'], 'ticket_count': base['ticket_count'], 'formation': base['formation']}
            x = translated(base, trio_odds, trifecta_odds, 'H1')
            if x:
                out['H1_ANCHOR'] = x
            if h_ab and x:
                out['H1_ANCHOR_HAB'] = x
        if h_ab:
            x = translated(base, trio_odds, trifecta_odds, 'ADAPT')
            if x:
                out['H_ADAPTIVE_HAB'] = x

    elif subtype == 'INITIAL_SPECIAL':
        if h_ab == h_conc:
            out['CURRENT'] = {'tickets': base['tickets'], 'ticket_count': base['ticket_count'], 'formation': base['formation']}
            x = translated(base, trio_odds, trifecta_odds, 'ADAPT')
            if x:
                out['INIT_ADAPT_CONSISTENT'] = x
        if h_ab and h_conc:
            x = translated(base, trio_odds, trifecta_odds, 'H1')
            if x:
                out['INIT_H1_HAB_CONC'] = x
        if (not h_ab) and (not h_conc):
            x = translated(base, trio_odds, trifecta_odds, 'TOP2')
            if x:
                out['INIT_BALANCED_HAB0'] = x
    return out


def evaluate(label, data):
    races, trio, tf, pay = data
    rows = defaultdict(list)
    by_type = defaultdict(lambda: defaultdict(list))
    populations = defaultdict(int)
    ps_pass = defaultdict(int)
    h_states = defaultdict(lambda: defaultdict(int))

    for rid, r in sorted(races.items(), key=lambda x: (x[1].get('race_date', ''), x[0])):
        rt = r.get('race_type') or ''
        st = special_subtype(rt)
        if st not in ('SPECIAL', 'INITIAL_SPECIAL'):
            continue
        if r.get('meeting_grade') != 'F1' or not rt.startswith('Ｓ級'):
            continue
        if pi(r.get('entry_count')) != 7 or len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get('predicted_line_formation'))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars) or rid not in pay or not pay[rid]:
            continue

        populations[rt] += 1
        base = build_v8_8_f09(trio[rid], tf[rid], r.get('predicted_line_formation') or '')
        if not base.get('buy'):
            continue
        ps_pass[rt] += 1
        eg = dict(base.get('entry_gate') or {})
        h_ab = int(bool(eg.get('H_AB')))
        h_conc = int(not bool(eg.get('H_RATIO')))
        h_states[rt][f'HAB{h_ab}_HCONC{h_conc}'] += 1

        for name, d in candidate_decisions(base, trio[rid], tf[rid], st).items():
            wins = [t for t in d['tickets'] if t in pay[rid]]
            payout = sum(pay[rid][t] for t in wins)
            row = {
                'race_id': rid,
                'race_type': rt,
                'subtype': st,
                'candidate': name,
                'hit': int(bool(wins)),
                'payout': payout,
                'tickets': int(d['ticket_count']),
                'formation': d.get('formation'),
                'h_ab': h_ab,
                'h_conc': h_conc,
            }
            rows[(st, name)].append(row)
            by_type[rt][name].append(row)

    return {
        'dataset': label,
        'population_by_type': dict(populations),
        'ps_ab_pass_by_type': dict(ps_pass),
        'h_state_by_type': {rt: dict(v) for rt, v in h_states.items()},
        'by_subtype_candidate': {
            f'{st}:{name}': summarize(rs)
            for (st, name), rs in sorted(rows.items())
        },
        'by_exact_type_candidate': {
            rt: {name: summarize(rs) for name, rs in sorted(cands.items())}
            for rt, cands in sorted(by_type.items())
        },
    }


def combine(q1, q2):
    out = {}
    keys = set(q1['by_subtype_candidate']) | set(q2['by_subtype_candidate'])
    for key in sorted(keys):
        # Combined metrics from summary totals; enough for stability screening.
        a = q1['by_subtype_candidate'].get(key, {})
        b = q2['by_subtype_candidate'].get(key, {})
        bets = (a.get('bets') or 0) + (b.get('bets') or 0)
        hits = (a.get('hits') or 0) + (b.get('hits') or 0)
        tickets = (a.get('tickets') or 0) + (b.get('tickets') or 0)
        payout = (a.get('payout_yen') or 0) + (b.get('payout_yen') or 0)
        stake = tickets * STAKE
        out[key] = {
            'bets': bets,
            'hits': hits,
            'hit_rate_pct': 100 * hits / bets if bets else None,
            'tickets': tickets,
            'avg_tickets': tickets / bets if bets else None,
            'stake_yen': stake,
            'payout_yen': payout,
            'profit_yen': payout - stake,
            'roi_pct': 100 * payout / stake if stake else None,
            'Q1_profit_yen': a.get('profit_yen'),
            'Q2_profit_yen': b.get('profit_yen'),
            'Q1_roi_pct': a.get('roi_pct'),
            'Q2_roi_pct': b.get('roi_pct'),
        }
    return out


def main():
    q1 = evaluate('2024Q1_DEVELOPMENT', load())
    q2 = evaluate('2024Q2_DEVELOPMENT_AFTER_OBSERVED', load_q2())
    result = {
        'analysis': 'SPECIAL_H_TRANSLATION_Q1Q2_DEVELOPMENT',
        'policy': 'No fitted numeric cutoff. Compare existing H semantics translated into first-place pool. Selection remains quarantined.',
        'Q1': q1,
        'Q2': q2,
        'Q1Q2_COMBINED': combine(q1, q2),
    }
    print('SPECIAL_H_TRANSLATION_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('SPECIAL_H_TRANSLATION_END')


if __name__ == '__main__':
    main()

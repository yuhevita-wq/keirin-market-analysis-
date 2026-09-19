from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2, DATA as Q2_DATA
from v7_0_f01_market_hierarchy import implied_probabilities, positional_support
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _select_structural_knee
from v8_18_f19_semifinal_h_adaptive import _protected_price_path
from v8_20_f21_initial_special_unified import build_v8_20_f21

ROOT = Path(__file__).resolve().parents[2]
Q1_DATA = ROOT / 'data/2024/s_class_f1_all_parts/2024_q1'
STAKE = 100


def read_votes(data: Path, name: str):
    out = {}
    with (data / name).open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                v = int(float(r.get('total_votes') or 0))
            except (TypeError, ValueError):
                continue
            if v > 0:
                out.setdefault(r['race_id'], v)
    return out


def valid_rows(data):
    races, trio, tf, pay = data
    out = []
    for rid, r in races.items():
        if r.get('meeting_grade') != 'F1' or not (r.get('race_type') or '').startswith('Ｓ級'):
            continue
        if pi(r.get('entry_count')) != 7 or len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get('predicted_line_formation'))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars) or rid not in pay or not pay[rid]:
            continue
        out.append((rid, r, cars))
    return out


def market_metrics(trio_odds, tf_odds, line, cars):
    base = build_v8_8_f09(trio_odds, tf_odds, line)
    if not base.get('buy'):
        return None
    q = implied_probabilities(tf_odds)
    h1, _, _ = positional_support(q, cars)
    ranked = sorted(cars, key=lambda c: (-h1[c], c))
    htop, hsecond = h1[ranked[0]], h1[ranked[1]]
    qmass = float(base.get('q_mass') or 0.0)
    tickets = int(base.get('ticket_count') or 0)
    eg = base.get('entry_gate') or {}
    a = tuple(eg.get('A_line') or ())
    b = tuple(eg.get('B_line') or ())
    ab = set(a) | set(b)
    return {
        'h1': htop,
        'h2': hsecond,
        'h_cliff': max(0.0, htop - hsecond),
        'h_ratio': htop / hsecond if hsecond > 0 else None,
        'q_mass': qmass,
        'q_density': qmass / tickets if tickets else None,
        'tickets': tickets,
        'ab_head_share': sum(h1[c] for c in ab),
        'H1_rider': ranked[0],
        'H2_rider': ranked[1],
    }


def summary(rows):
    n = len(rows)
    h = sum(r['hit'] for r in rows)
    t = sum(r['tickets'] for r in rows)
    p = sum(r['payout'] for r in rows)
    s = t * STAKE
    return {
        'bet_races': n,
        'hits': h,
        'hit_rate_pct': 100*h/n if n else None,
        'tickets': t,
        'avg_tickets': t/n if n else None,
        'stake_yen': s,
        'payout_yen': p,
        'profit_yen': p-s,
        'roi_pct': 100*p/s if s else None,
    }


def anchored_row(d, tf_odds, pay):
    eg = d.get('entry_gate') or {}
    top2 = tuple(int(x) for x in eg.get('top2_H') or ())
    if len(top2) < 2:
        return None
    q = implied_probabilities(tf_odds)
    s0 = _state((top2[0],), d['second'], d['third'], q)
    if s0 is None:
        return None
    wins0 = [t for t in s0.tickets if t in pay]
    plain = {
        'hit': int(bool(wins0)),
        'tickets': s0.ticket_count,
        'payout': sum(pay[t] for t in wins0),
        'formation': s0.display,
    }
    p0 = _price_state(s0, s0.q_mass, q, tf_odds)
    states, _ = _protected_price_path(p0, q, tf_odds)
    idx = _select_structural_knee(states)
    chosen = states[idx]
    wins1 = [t for t in chosen.state.tickets if t in pay]
    compressed = {
        'hit': int(bool(wins1)),
        'tickets': chosen.state.ticket_count,
        'payout': sum(pay[t] for t in wins1),
        'formation': chosen.display,
    }
    return plain, compressed


def analyze(label, data, data_path):
    races, trio, tf, pay = data
    tf_votes = read_votes(data_path, 'trifecta_final_odds.csv')
    trio_votes = read_votes(data_path, 'trio_final_odds.csv')
    vals = valid_rows(data)

    peer_market = {}
    by_day = defaultdict(list)
    for rid, r, cars in vals:
        m = market_metrics(trio[rid], tf[rid], r.get('predicted_line_formation') or '', cars)
        if m is not None and rid in tf_votes and rid in trio_votes:
            peer_market[rid] = m
            by_day[(r.get('race_date'), r.get('track'))].append(rid)

    rows = []
    anchor_plain = []
    anchor_comp = []
    for rid, r, cars in vals:
        if (r.get('race_type') or '') != 'Ｓ級決勝':
            continue
        d = build_v8_20_f21(trio[rid], tf[rid], r.get('predicted_line_formation') or '', 'Ｓ級決勝')
        if not d.get('buy'):
            continue
        m = peer_market.get(rid)
        if m is None:
            continue
        peers = [x for x in by_day[(r.get('race_date'), r.get('track'))] if x != rid and x in peer_market]
        if not peers:
            continue
        def med(k):
            xs = [peer_market[x][k] for x in peers if peer_market[x].get(k) is not None]
            return median(xs) if xs else None
        def mx(k):
            xs = [peer_market[x][k] for x in peers if peer_market[x].get(k) is not None]
            return max(xs) if xs else None
        h1_med, cliff_med, qmass_med, qden_med, ab_med = med('h1'), med('h_cliff'), med('q_mass'), med('q_density'), med('ab_head_share')
        h1_max, cliff_max = mx('h1'), mx('h_cliff')
        wins = [t for t in d['tickets'] if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        n = int(d['ticket_count'])
        tv = tf_votes[rid]
        row = {
            'race_id': rid,
            'race_date': r.get('race_date'),
            'track': r.get('track'),
            'hit': int(bool(wins)),
            'payout': payout,
            'tickets': n,
            'formation': d.get('formation'),
            'first_pool_size': len(tuple(d.get('first') or ())),
            'H1_rider': m['H1_rider'],
            'H1_in_first': int(m['H1_rider'] in set(d.get('first') or ())),
            'tf_votes': tv,
            'trio_votes': trio_votes[rid],
            'h1_share': m['h1'],
            'h_cliff_share': m['h_cliff'],
            'q_mass': m['q_mass'],
            'q_density': m['q_density'],
            'ab_head_share': m['ab_head_share'],
            'head_heat_abs': tv*m['h1'],
            'head_cliff_heat_abs': tv*m['h_cliff'],
            'formation_heat_abs': tv*m['q_mass'],
            'formation_heat_per_ticket_abs': tv*m['q_density'] if m['q_density'] is not None else None,
            'h1_vs_peer_median': m['h1']/h1_med if h1_med else None,
            'h_cliff_vs_peer_median': m['h_cliff']/cliff_med if cliff_med else None,
            'qmass_vs_peer_median': m['q_mass']/qmass_med if qmass_med else None,
            'qdensity_vs_peer_median': m['q_density']/qden_med if qden_med and m['q_density'] is not None else None,
            'ab_head_vs_peer_median': m['ab_head_share']/ab_med if ab_med else None,
            'h1_dominates_peer_max': int(h1_max is not None and m['h1'] >= h1_max),
            'cliff_dominates_peer_max': int(cliff_max is not None and m['h_cliff'] >= cliff_max),
            'peer_count_psab': len(peers),
        }
        row['head_location_boost'] = int((row['h1_vs_peer_median'] or 0) >= 1)
        row['cliff_location_boost'] = int((row['h_cliff_vs_peer_median'] or 0) >= 1)
        row['formation_location_boost'] = int((row['qmass_vs_peer_median'] or 0) >= 1)
        row['density_location_boost'] = int((row['qdensity_vs_peer_median'] or 0) >= 1)
        row['head_and_form_boost'] = int(row['head_location_boost'] and row['formation_location_boost'])
        row['cliff_and_form_boost'] = int(row['cliff_location_boost'] and row['formation_location_boost'])
        row['all_location_boost'] = int(row['head_location_boost'] and row['cliff_location_boost'] and row['formation_location_boost'] and row['density_location_boost'])
        rows.append(row)

        ar = anchored_row(d, tf[rid], pay[rid])
        if ar:
            ap, ac = ar
            anchor_plain.append({**ap, 'race_id':rid})
            anchor_comp.append({**ac, 'race_id':rid})

    states = {'CURRENT_ALL': summary(rows)}
    flags = [
        'head_location_boost','cliff_location_boost','formation_location_boost','density_location_boost',
        'head_and_form_boost','cliff_and_form_boost','all_location_boost','h1_dominates_peer_max','cliff_dominates_peer_max'
    ]
    for f in flags:
        states[f.upper()] = summary([r for r in rows if r[f]])
    for sz in (1,2,3):
        states[f'FIRST_POOL_SIZE_{sz}'] = summary([r for r in rows if r['first_pool_size']==sz])
    states['H1_ANCHOR_PLAIN_ALL_CURRENT_ENTRY'] = summary(anchor_plain)
    states['H1_ANCHOR_PROTECTED_PRICE_ALL_CURRENT_ENTRY'] = summary(anchor_comp)
    return {'dataset':label,'states':states,'rows':rows}


def main():
    q1 = analyze('2024Q1_DEVELOPMENT', load(), Q1_DATA)
    q2_data = load_q2()
    q2 = analyze('2024Q2_DEVELOPMENT_AFTER_OBSERVED', q2_data, Q2_DATA)
    result = {
        'analysis':'FINAL_HEAT_LOCATION_Q1Q2',
        'status':'DEVELOPMENT_DIAGNOSTIC_NO_NEW_FITTED_CUTOFF',
        'concept':'Final-specific extra money must be located inside head/formation structure, not merely total race volume.',
        'semantic_boundaries':'relative-to-same-day PS_AB peer median >= 1.0; no outcome-fitted numeric cutoff',
        'Q1':q1,'Q2':q2,
    }
    print('FINAL_HEAT_LOCATION_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('FINAL_HEAT_LOCATION_END')


if __name__ == '__main__':
    main()

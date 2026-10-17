from __future__ import annotations

import json
from collections import defaultdict
from math import exp, log
from statistics import median

from simulate_v8_1_f02_2024q1 import STAKE, load, pi, pl
from v7_0_f01_market_hierarchy import implied_probabilities
from v8_9_f10_compact_entry import build_v8_9_f10


def effective_ticket_count(tickets, q):
    mass = sum(q[t] for t in tickets)
    if mass <= 0:
        return None
    w = [q[t] / mass for t in tickets if q[t] > 0]
    if len(w) != len(tickets):
        return None
    entropy = -sum(x * log(x) for x in w)
    return exp(entropy)


def metrics(rows):
    rows = list(rows)
    races = len(rows)
    hits = sum(r['hit'] for r in rows)
    tickets = sum(r['ticket_count'] for r in rows)
    stake = tickets * STAKE
    payout = sum(r['payout_yen'] for r in rows)
    hit_mult = [r['return_multiple'] for r in rows if r['hit']]
    return {
        'races': races,
        'hits': hits,
        'hit_rate_pct': 100 * hits / races if races else None,
        'avg_tickets': tickets / races if races else None,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else None,
        'median_effective_tickets': median([r['effective_tickets'] for r in rows]) if rows else None,
        'median_compression_ratio': median([r['compression_ratio'] for r in rows]) if rows else None,
        'median_normalized_entropy': median([r['normalized_entropy'] for r in rows]) if rows else None,
        'median_top_ticket_share': median([r['top_ticket_share'] for r in rows]) if rows else None,
        'median_q_mass': median([r['q_mass'] for r in rows]) if rows else None,
        'median_hit_return_multiple': median(hit_mult) if hit_mult else None,
    }


def quartiles(rows, field):
    ordered = sorted(rows, key=lambda r: (r[field], r['race_date'], r['race_id']))
    n = len(ordered)
    out = {}
    for i in range(4):
        lo = i * n // 4
        hi = (i + 1) * n // 4
        part = ordered[lo:hi]
        m = metrics(part)
        m['field_min'] = min((r[field] for r in part), default=None)
        m['field_max'] = max((r[field] for r in part), default=None)
        m['field_median'] = median([r[field] for r in part]) if part else None
        out[f'Q{i+1}'] = m
    return out


def main():
    races, trio, tf, pay = load()
    rows = []

    for rid, r in sorted(races.items(), key=lambda kv: (kv[1].get('race_date', ''), kv[0])):
        if r.get('meeting_grade') != 'F1' or not (r.get('race_type') or '').startswith('Ｓ級'):
            continue
        if pi(r.get('entry_count')) != 7:
            continue
        if len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get('predicted_line_formation'))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars):
            continue
        if rid not in pay or not pay[rid]:
            continue

        d = build_v8_9_f10(trio[rid], tf[rid], r.get('predicted_line_formation') or '')
        if not d.get('buy'):
            continue

        tickets = tuple(d['tickets'])
        q = implied_probabilities(tf[rid])
        n = len(tickets)
        neff = effective_ticket_count(tickets, q)
        if neff is None:
            continue
        qmass = sum(q[t] for t in tickets)
        weights = [q[t] / qmass for t in tickets]
        top_share = max(weights)
        norm_entropy = (log(neff) / log(n)) if n > 1 else 1.0
        compression = n / neff

        wins = [t for t in tickets if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        hit = int(bool(wins))
        stake = n * STAKE
        multiple = payout / stake if hit and stake else 0.0

        rows.append({
            'race_id': rid,
            'race_date': r.get('race_date'),
            'track': r.get('track'),
            'race_no': pi(r.get('race_no')),
            'race_type': r.get('race_type'),
            'formation': d['formation'],
            'ticket_count': n,
            'q_mass': qmass,
            'effective_tickets': neff,
            'compression_ratio': compression,
            'normalized_entropy': norm_entropy,
            'top_ticket_share': top_share,
            'hit': hit,
            'payout_yen': payout,
            'return_multiple': multiple,
        })

    diagnostic_groups = {
        'MISSES': metrics([r for r in rows if not r['hit']]),
        'HIT_LOSS_LT1X': metrics([r for r in rows if r['hit'] and r['return_multiple'] < 1.0]),
        'HIT_1X_TO_LT2X': metrics([r for r in rows if r['hit'] and 1.0 <= r['return_multiple'] < 2.0]),
        'HIT_2X_PLUS': metrics([r for r in rows if r['hit'] and r['return_multiple'] >= 2.0]),
    }

    result = {
        'scheme': 'v8.9-F10',
        'dataset': '2024Q1',
        'analysis': 'PRICE_COMPRESSION_RISK',
        'definition': {
            'effective_tickets': 'exp(Shannon entropy of normalized q weights inside selected formation)',
            'compression_ratio': 'ticket_count / effective_tickets; higher means paid ticket count is much wider than market-effective support count',
            'normalized_entropy': 'log(effective_tickets)/log(ticket_count); lower means selected support is more concentrated',
            'top_ticket_share': 'largest selected-ticket q divided by selected formation q mass',
        },
        'overall': metrics(rows),
        'diagnostic_groups': diagnostic_groups,
        'compression_ratio_quartiles': quartiles(rows, 'compression_ratio'),
        'normalized_entropy_quartiles': quartiles(rows, 'normalized_entropy'),
        'top_ticket_share_quartiles': quartiles(rows, 'top_ticket_share'),
        'notes': [
            'This is exploratory Q1 development analysis, not a validated entry cutoff.',
            'No individual-ticket odds threshold is used; only normalized market-support shape inside the already selected formation is measured.',
        ],
    }

    print('V8_9_F10_PRICE_COMPRESSION_Q1_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V8_9_F10_PRICE_COMPRESSION_Q1_END')


if __name__ == '__main__':
    main()

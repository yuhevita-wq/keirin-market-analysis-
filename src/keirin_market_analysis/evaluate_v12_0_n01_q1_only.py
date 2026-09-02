from __future__ import annotations

import json
from collections import Counter, defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v9_0_q1q2q3 import load_entries
from v8_11_f12_race_type_adaptive import classify_race_type
from v12_0_n01_market_connection_topology import build_v12_0_n01

STAKE = 100


def summarize(rows):
    n = len(rows)
    hits = sum(r['hit'] for r in rows)
    tickets = sum(r['tickets'] for r in rows)
    payout = sum(r['payout'] for r in rows)
    stake = tickets * STAKE
    return {
        'bet_races': n,
        'hits': hits,
        'hit_rate_pct': (100 * hits / n) if n else None,
        'tickets': tickets,
        'avg_tickets': (tickets / n) if n else None,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': (100 * payout / stake) if stake else None,
    }


def main():
    races, trio, tf, pay = load()
    entries = load_entries('2024_q1')

    rows = []
    by_group = defaultdict(list)
    by_ticket_count = defaultdict(list)
    by_head_count = defaultdict(list)
    reason_counts = Counter()
    population = 0
    valid_complete_market = 0

    for rid, r in sorted(races.items(), key=lambda x: (x[1].get('race_date',''), x[0])):
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

        population += 1
        valid_complete_market += 1
        rt = r.get('race_type') or ''
        group = classify_race_type(rt)

        d = build_v12_0_n01(
            trio[rid],
            tf[rid],
            entries.get(rid, []),
            race_type=rt,
        )
        reason_counts[str(d.get('reason'))] += 1
        if not d.get('buy'):
            continue

        tickets = [tuple(t) for t in d['tickets']]
        wins = [t for t in tickets if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        n = int(d['ticket_count'])
        head_count = len(d.get('head_block') or [])

        row = {
            'race_id': rid,
            'race_date': r.get('race_date'),
            'race_type': rt,
            'group': group,
            'tickets': n,
            'head_count': head_count,
            'hit': int(bool(wins)),
            'payout': payout,
        }
        rows.append(row)
        by_group[group].append(row)
        by_ticket_count[n].append(row)
        by_head_count[head_count].append(row)

    result = {
        'scheme': 'v12.0-N01',
        'dataset': '2024Q1',
        'mode': 'SIMULATION_REQUESTED_BY_USER',
        'scheme_changed_for_run': False,
        'scientific_status': 'DEVELOPMENT_TEST; v12 was designed after earlier Q1-Q3 project outcomes were known, so Q1 is not pristine OOS.',
        'population': population,
        'complete_market_population': valid_complete_market,
        'overall': summarize(rows),
        'buy_rate_pct': (100 * len(rows) / population) if population else None,
        'by_group': {k: summarize(v) for k, v in sorted(by_group.items())},
        'by_ticket_count': {str(k): summarize(v) for k, v in sorted(by_ticket_count.items())},
        'by_head_count': {str(k): summarize(v) for k, v in sorted(by_head_count.items())},
        'decision_reasons': dict(reason_counts),
        'rule_note': 'Fresh-root connection topology. No v8-v11 betting gate, no race-type branch, no line-based ticket generation, no result/payout input.',
    }

    print('V12_0_N01_Q1_ONLY_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V12_0_N01_Q1_ONLY_END')


if __name__ == '__main__':
    main()

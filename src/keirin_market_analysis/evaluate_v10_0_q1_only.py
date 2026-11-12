from __future__ import annotations

import json
from collections import Counter, defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_25_q1q2 import summary
from evaluate_v9_0_q1q2q3 import load_entries
from v8_11_f12_race_type_adaptive import classify_race_type
from v10_0_f29_universal_market_fundamental_value import build_v10_0_f29

GROUPS = ('QUALIFYING','GENERAL','SEMIFINAL','SPECIAL','FINAL','OTHER')


def evaluate_q1():
    races, trio, tf, pay = load()
    entries = load_entries('2024_q1')

    rows = []
    by_group = defaultdict(list)
    by_type = defaultdict(list)
    pop_group = Counter()
    pop_type = Counter()
    decision_reasons = Counter()
    population = 0

    estimated_roi_bought = []
    estimated_roi_no_bet = []
    ticket_count_dist = Counter()

    for rid, r in sorted(races.items(), key=lambda x: (x[1].get('race_date',''), x[0])):
        if r.get('meeting_grade') != 'F1' or not (r.get('race_type') or '').startswith('Ｓ級'):
            continue
        if pi(r.get('entry_count')) != 7 or len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get('predicted_line_formation'))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars):
            continue
        if rid not in pay or not pay[rid]:
            continue

        population += 1
        rt = r.get('race_type') or ''
        group = classify_race_type(rt)
        pop_group[group] += 1
        pop_type[rt] += 1

        d = build_v10_0_f29(
            trio[rid],
            tf[rid],
            r.get('predicted_line_formation') or '',
            rt,
            entries.get(rid, []),
        )
        decision_reasons[d.get('reason') or 'UNKNOWN'] += 1

        eroi = d.get('estimated_roi')
        if d.get('buy'):
            if eroi is not None:
                estimated_roi_bought.append(float(eroi))
            wins = [tuple(t) for t in d.get('tickets', ()) if tuple(t) in pay[rid]]
            payout = sum(pay[rid][t] for t in wins)
            n = int(d.get('ticket_count') or 0)
            ticket_count_dist[n] += 1
            row = {
                'race_id': rid,
                'race_date': r.get('race_date'),
                'race_type': rt,
                'group': group,
                'hit': int(bool(wins)),
                'payout': payout,
                'tickets': n,
                'formation': d.get('formation'),
                'estimated_roi': eroi,
                'estimated_edge': d.get('estimated_edge'),
                'formation_mass': d.get('formation_mass'),
            }
            rows.append(row)
            by_group[group].append(row)
            by_type[rt].append(row)
        else:
            if eroi is not None:
                estimated_roi_no_bet.append(float(eroi))

    def stat(xs):
        if not xs:
            return {'count': 0, 'min': None, 'mean': None, 'max': None}
        return {
            'count': len(xs),
            'min': min(xs),
            'mean': sum(xs)/len(xs),
            'max': max(xs),
        }

    return {
        'scheme': 'v10.0-F29',
        'dataset': '2024Q1',
        'mode': 'SIMULATION_REQUESTED_BY_USER',
        'scheme_changed_for_run': False,
        'population': population,
        'overall': summary(rows),
        'by_group': {
            g: {**summary(by_group[g]), 'population': pop_group[g]}
            for g in GROUPS
        },
        'by_race_type': {
            rt: {**summary(by_type[rt]), 'population': pop_type[rt]}
            for rt in sorted(pop_type)
        },
        'decision_reasons': dict(decision_reasons),
        'estimated_roi_bought': stat(estimated_roi_bought),
        'estimated_roi_no_bet': stat(estimated_roi_no_bet),
        'ticket_count_distribution': {str(k): v for k, v in sorted(ticket_count_dist.items())},
        'scientific_note': (
            'v10.0-F29 was frozen before this Q1 run. This evaluator does not alter '
            'the scheme. Q1 is diagnostic/development because prior project work has '
            'already observed 2024Q1 outcomes.'
        ),
    }


def main():
    result = evaluate_q1()
    print('V10_0_F29_Q1_ONLY_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V10_0_F29_Q1_ONLY_END')


if __name__ == '__main__':
    main()

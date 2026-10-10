from __future__ import annotations

import json
from collections import Counter, defaultdict

from evaluate_v8_25_q3_oos import load_q3
from evaluate_v9_0_q1q2q3 import load_entries
from evaluate_v8_25_q1q2 import summary
from v8_11_f12_race_type_adaptive import classify_race_type
from v11_0_c01_market_psychology_racecard_context import build_v11_0_c01
from simulate_v8_1_f02_2024q1 import pi, pl


def main():
    races, trio, tf, pay = load_q3()
    entries = load_entries('2024_q3')

    rows = []
    by_context = defaultdict(list)
    by_head_context = defaultdict(list)
    by_line_context = defaultdict(list)
    by_h_state = defaultdict(list)
    by_group = defaultdict(list)
    population = 0
    context_population = Counter()
    context_buy_population = Counter()
    fail = Counter()

    for rid, r in sorted(races.items(), key=lambda x: (x[1].get('race_date',''), x[0])):
        if r.get('meeting_grade') != 'F1' or not (r.get('race_type') or '').startswith('Ｓ級'):
            continue
        if pi(r.get('entry_count')) != 7 or len(trio.get(rid, {})) != 35 or len(tf.get(rid, {})) != 210:
            continue
        cars = sorted({v for c in trio[rid] for v in c})
        lines = pl(r.get('predicted_line_formation'))
        if len(cars) != 7 or lines is None or set(v for line in lines for v in line) != set(cars) or rid not in pay or not pay[rid]:
            continue

        population += 1
        rt = r.get('race_type') or ''
        g = classify_race_type(rt)
        d = build_v11_0_c01(
            trio[rid], tf[rid], r.get('predicted_line_formation') or '', rt,
            entries.get(rid, []),
        )
        ctx = d.get('racecard_context') if d.get('racecard_context_ok') else 'CONTEXT_INVALID'
        context_population[ctx] += 1

        if not d.get('buy'):
            fail[f'{rt}:{d.get("reason")}'] += 1
            continue

        context_buy_population[ctx] += 1
        wins = [t for t in d['tickets'] if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        n = int(d['ticket_count'])
        row = {
            'race_id': rid,
            'race_date': r.get('race_date'),
            'race_type': rt,
            'group': g,
            'hit': int(bool(wins)),
            'payout': payout,
            'tickets': n,
            'formation': d.get('formation'),
            'context': ctx,
            'h_state': d.get('racecard_h_state'),
            'head_context': d.get('racecard_head_context'),
            'line_context': d.get('racecard_line_context'),
            'market_h1_fundamental_rank': d.get('racecard_market_h1_fundamental_rank'),
            'market_top_line_fundamental_rank': d.get('racecard_market_top_line_fundamental_rank'),
        }
        rows.append(row)
        by_context[ctx].append(row)
        by_h_state[row['h_state']].append(row)
        by_head_context[row['head_context']].append(row)
        by_line_context[row['line_context']].append(row)
        by_group[g].append(row)

    result = {
        'scheme': 'v11.0-C01',
        'base_scheme': 'v8.25-F26',
        'dataset': '2024Q3',
        'mode': 'SIMULATION_REQUESTED_BY_USER',
        'context_definition_changed_from_q1_q2': False,
        'bet_logic_changed_vs_v8_25': False,
        'scientific_status': 'FIXED_V11_CONTEXT_APPLIED_TO_Q3; Q3 outcomes were already known from prior v8.25 project validation, so this is not a pristine project-level OOS test.',
        'population': population,
        'overall': summary(rows),
        'by_context': {k: {**summary(v), 'population': context_population[k], 'buy_population': context_buy_population[k]} for k, v in sorted(by_context.items())},
        'by_h_state': {k: summary(v) for k, v in sorted(by_h_state.items())},
        'by_head_context': {k: summary(v) for k, v in sorted(by_head_context.items())},
        'by_line_context': {k: summary(v) for k, v in sorted(by_line_context.items())},
        'by_group': {k: summary(v) for k, v in sorted(by_group.items())},
        'context_population': dict(context_population),
        'context_buy_population': dict(context_buy_population),
        'entry_fail_reasons': dict(fail),
    }

    print('V11_0_C01_Q3_ONLY_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V11_0_C01_Q3_ONLY_END')


if __name__ == '__main__':
    main()

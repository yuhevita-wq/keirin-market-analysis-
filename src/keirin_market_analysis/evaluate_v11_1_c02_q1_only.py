from __future__ import annotations

import json
from collections import Counter, defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v9_0_q1q2q3 import load_entries
from evaluate_v8_25_q1q2 import summary
from v8_11_f12_race_type_adaptive import classify_race_type
from v11_1_c02_head_survives_line_fade import build_v11_1_c02, TARGET_CONTEXT


def main():
    races, trio, tf, pay = load()
    entries = load_entries('2024_q1')

    rows = []
    c02_rows = []
    by_group = defaultdict(list)
    c02_by_group = defaultdict(list)
    population = 0
    target_context_buys_before_c02 = 0
    c02_applied = 0
    target_context_skipped_by_c02 = Counter()

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
        d = build_v11_1_c02(
            trio[rid], tf[rid], r.get('predicted_line_formation') or '', rt,
            entries.get(rid, []),
        )

        if d.get('racecard_context') == TARGET_CONTEXT:
            # v11.0 only assigned context to races already processed by base engine;
            # C02 then either applies or explicitly rejects the sharp geometry.
            if d.get('c02_override_applied'):
                c02_applied += 1
            elif str(d.get('reason','')).startswith('C02_'):
                target_context_skipped_by_c02[str(d.get('reason'))] += 1

        if not d.get('buy'):
            continue

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
            'c02_override_applied': bool(d.get('c02_override_applied')),
            'reason': d.get('reason'),
            'context': d.get('racecard_context'),
        }
        rows.append(row)
        by_group[g].append(row)
        if row['c02_override_applied']:
            c02_rows.append(row)
            c02_by_group[g].append(row)

    # Count the target context as seen under v11.0 before C02 geometry can reject it.
    # For applied rows + explicit C02 skip reasons, each represents one original target-context buy.
    target_context_buys_before_c02 = c02_applied + sum(target_context_skipped_by_c02.values())

    result = {
        'scheme': 'v11.1-C02',
        'dataset': '2024Q1',
        'mode': 'SIMULATION_REQUESTED_BY_USER',
        'scheme_changed_for_run': False,
        'development_note': 'C02 was designed after Q1-Q3 v11 context outcomes were observed; Q1 is development data for C02, not OOS.',
        'population': population,
        'overall_after_c02': summary(rows),
        'c02_applied_only': summary(c02_rows),
        'c02_applied_races': c02_applied,
        'target_context_buys_before_c02': target_context_buys_before_c02,
        'target_context_skipped_by_c02': dict(target_context_skipped_by_c02),
        'overall_by_group': {k: summary(v) for k, v in sorted(by_group.items())},
        'c02_by_group': {k: summary(v) for k, v in sorted(c02_by_group.items())},
        'c02_rule': 'H1 fixed first; top two F riders from strongest rival fundamental line in second/third; exactly 2 tail-swap tickets.',
    }

    print('V11_1_C02_Q1_ONLY_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V11_1_C02_Q1_ONLY_END')


if __name__ == '__main__':
    main()

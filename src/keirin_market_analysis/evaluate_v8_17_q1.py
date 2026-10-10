from __future__ import annotations
import json
from collections import Counter, defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_15_f16_special_split_rebuild import special_subtype
from v8_17_f18_market_semantics import build_v8_17_f18

STAKE = 100
GROUPS = ('QUALIFYING','GENERAL','SEMIFINAL','SPECIAL','FINAL','OTHER')


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


def main():
    races, trio, tf, pay = load()
    rows = []
    by_group = defaultdict(list)
    by_type = defaultdict(list)
    population = 0
    pop_group = Counter()
    pop_type = Counter()
    fail = Counter()
    selection_diag_count = 0

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
        pop_group[g] += 1
        pop_type[rt] += 1

        d = build_v8_17_f18(trio[rid], tf[rid], r.get('predicted_line_formation') or '', rt)
        if d.get('selection_diagnostics') is not None:
            selection_diag_count += 1

        if not d.get('buy'):
            fail[f'{rt}:{d.get("reason")}'] += 1
            continue

        wins = [t for t in d['tickets'] if t in pay[rid]]
        payout = sum(pay[rid][t] for t in wins)
        n = int(d['ticket_count'])
        row = {
            'race_id': rid,
            'race_type': rt,
            'group': g,
            'special_subtype': special_subtype(rt) if g == 'SPECIAL' else None,
            'hit': int(bool(wins)),
            'payout': payout,
            'tickets': n,
            'formation': d.get('formation'),
            'branch_policy': d.get('branch_policy'),
        }
        rows.append(row)
        by_group[g].append(row)
        by_type[rt].append(row)

    result = {
        'scheme': 'v8.17-F18',
        'dataset': '2024Q1',
        'population': population,
        'overall': summary(rows),
        'by_group': {g: {**summary(by_group[g]), 'population': pop_group[g]} for g in GROUPS},
        'by_race_type': {rt: {**summary(by_type[rt]), 'population': pop_type[rt]} for rt in sorted(pop_type)},
        'selection_diagnostic_races': selection_diag_count,
        'selection_action': 'NO_BET_DIAGNOSTIC_ONLY',
        'final_rule': 'PS_AB + H1_top >= 2*H1_second; unchanged F09 formation',
        'entry_fail_reasons': dict(fail),
    }
    result['acceptance'] = 'PASS_Q1_PROFIT' if result['overall']['profit_yen'] > 0 else 'REJECT_Q1_NOT_PROFITABLE'
    print('V8_17_F18_Q1_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V8_17_F18_Q1_END')


if __name__ == '__main__':
    main()

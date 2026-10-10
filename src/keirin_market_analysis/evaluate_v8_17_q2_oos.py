from __future__ import annotations
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from simulate_v8_1_f02_2024q1 import pi, pf, pt, p3, pl
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_15_f16_special_split_rebuild import special_subtype
from v8_17_f18_market_semantics import build_v8_17_f18

STAKE = 100
GROUPS = ('QUALIFYING','GENERAL','SEMIFINAL','SPECIAL','FINAL','OTHER')
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/2024/s_class_f1_all_parts/2024_q2'


def ensure_q2_data():
    if (DATA / 'races.csv').exists():
        return
    subprocess.run(
        ['git', 'sparse-checkout', 'add', 'data/2024/s_class_f1_all_parts/2024_q2'],
        cwd=ROOT,
        check=True,
    )


def rr(name):
    with (DATA / name).open('r', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def load_q2():
    ensure_q2_data()
    races = {r['race_id']: r for r in rr('races.csv')}
    trio = defaultdict(dict)
    tf = defaultdict(dict)
    pay = defaultdict(dict)
    for r in rr('trio_final_odds.csv'):
        c, o = pt(r.get('combination')), pf(r.get('odds'))
        if c and o and o > 0 and r.get('odds_status') == 'available':
            trio[r['race_id']][c] = o
    for r in rr('trifecta_final_odds.csv'):
        c, o = p3(r.get('combination')), pf(r.get('odds'))
        if c and o and o > 0 and r.get('odds_status') == 'available':
            tf[r['race_id']][c] = o
    for r in rr('payouts.csv'):
        if r.get('bet_code') != 'trifecta' and r.get('ticket_type') != '3連単':
            continue
        if r.get('status') != 'paid':
            continue
        c, y = p3(r.get('combination')), pi(r.get('payout_yen'))
        if c and y is not None:
            pay[r['race_id']][c] = y
    return races, trio, tf, pay


def losing_streak(rows):
    cur = best = 0
    for r in sorted(rows, key=lambda x: (x.get('race_date') or '', x['race_id'])):
        if r['hit']:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


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
        'max_losing_streak': losing_streak(rows) if rows else None,
    }


def main():
    races, trio, tf, pay = load_q2()
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
            'race_date': r.get('race_date'),
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
        'dataset': '2024Q2',
        'validation_status': 'OUT_OF_SAMPLE_FIXED_AFTER_Q1',
        'no_q2_tuning': True,
        'population': population,
        'overall': summary(rows),
        'by_group': {g: {**summary(by_group[g]), 'population': pop_group[g]} for g in GROUPS},
        'by_race_type': {rt: {**summary(by_type[rt]), 'population': pop_type[rt]} for rt in sorted(pop_type)},
        'selection_diagnostic_races': selection_diag_count,
        'selection_action': 'NO_BET_DIAGNOSTIC_ONLY',
        'final_rule': 'PS_AB + H1_top >= 2*H1_second; unchanged F09 formation',
        'entry_fail_reasons': dict(fail),
    }
    result['acceptance'] = 'PASS_Q2_OOS_PROFIT' if result['overall']['profit_yen'] > 0 else 'FAIL_Q2_OOS_NOT_PROFITABLE'
    print('V8_17_F18_Q2_OOS_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V8_17_F18_Q2_OOS_END')


if __name__ == '__main__':
    main()

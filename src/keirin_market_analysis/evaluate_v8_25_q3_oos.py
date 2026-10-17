from __future__ import annotations
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from simulate_v8_1_f02_2024q1 import pi, pf, pt, p3
from evaluate_v8_25_q1q2 import evaluate

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/2024/s_class_f1_all_parts/2024_q3'


def ensure_q3_data():
    if (DATA / 'races.csv').exists():
        return
    subprocess.run(
        ['git', 'sparse-checkout', 'add', 'data/2024/s_class_f1_all_parts/2024_q3'],
        cwd=ROOT,
        check=True,
    )


def rr(name):
    with (DATA / name).open('r', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def load_q3():
    ensure_q3_data()
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


def main():
    q3 = evaluate('2024Q3_OUT_OF_SAMPLE_FIXED_V8_25', load_q3())
    result = {
        'scheme': 'v8.25-F26',
        'validation_dataset': '2024Q3',
        'validation_status': 'OUT_OF_SAMPLE_FIXED_AFTER_Q1Q2_DEVELOPMENT',
        'frozen_before_q3': True,
        'no_q3_tuning': True,
        'Q3': q3,
        'acceptance': {
            'overall_profit_positive': q3['overall']['profit_yen'] > 0,
            'overall_roi_ge_100': (q3['overall']['roi_pct'] or 0) >= 100,
            'FINAL_profit_positive': q3['FINAL_SET_LOCK_ORDER_SPLIT']['profit_yen'] > 0,
        },
    }
    print('V8_25_F26_Q3_OOS_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V8_25_F26_Q3_OOS_END')


if __name__ == '__main__':
    main()

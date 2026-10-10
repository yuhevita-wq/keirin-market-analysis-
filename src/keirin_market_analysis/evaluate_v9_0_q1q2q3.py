from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2, ensure_q2_data
from evaluate_v8_25_q3_oos import load_q3, ensure_q3_data
from evaluate_v8_25_q1q2 import evaluate as evaluate_v8_25, summary
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_20_f21_initial_special_unified import INITIAL_SPECIAL_LABELS
from v9_0_f27_fundamental_first_branch import build_v9_0_f27

ROOT = Path(__file__).resolve().parents[2]
BASE_DATA = ROOT / 'data/2024/s_class_f1_all_parts'
GROUPS = ('QUALIFYING','GENERAL','SEMIFINAL','SPECIAL','FINAL','OTHER')


def load_entries(quarter: str):
    if quarter == '2024_q2':
        ensure_q2_data()
    elif quarter == '2024_q3':
        ensure_q3_data()
    path = BASE_DATA / quarter / 'entries.csv'
    out = defaultdict(list)
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            out[r['race_id']].append(r)
    return out


def evaluate_v9(label, data, entries):
    races, trio, tf, pay = data
    rows = []
    by_group = defaultdict(list)
    by_type = defaultdict(list)
    initial = []
    selection = []
    final = []
    pop_group = Counter()
    pop_type = Counter()
    fail = Counter()
    actions = Counter()
    base_buy_races = 0
    population = 0

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

        d = build_v9_0_f27(
            trio[rid], tf[rid], r.get('predicted_line_formation') or '', rt,
            entries.get(rid, []),
        )
        action = d.get('fundamental_filter_action') or 'UNKNOWN'
        actions[action] += 1
        if action != 'BASE_NO_BET_PASSTHROUGH':
            base_buy_races += 1

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
            'hit': int(bool(wins)),
            'payout': payout,
            'tickets': n,
            'formation': d.get('formation'),
            'branch_policy': d.get('branch_policy'),
            'fundamental_filter_action': action,
            'base_first_riders': d.get('base_first_riders'),
            'surviving_first_riders': d.get('surviving_first_riders'),
            'fundamental_head_block': d.get('fundamental_head_block'),
        }
        rows.append(row)
        by_group[g].append(row)
        by_type[rt].append(row)
        if rt in INITIAL_SPECIAL_LABELS:
            initial.append(row)
        if rt == 'Ｓ級選抜':
            selection.append(row)
        if rt == 'Ｓ級決勝':
            final.append(row)

    return {
        'dataset': label,
        'population': population,
        'overall': summary(rows),
        'by_group': {g: {**summary(by_group[g]), 'population': pop_group[g]} for g in GROUPS},
        'by_race_type': {rt: {**summary(by_type[rt]), 'population': pop_type[rt]} for rt in sorted(pop_type)},
        'INITIAL_SPECIAL_UNIFIED': {**summary(initial), 'population': sum(pop_type[x] for x in INITIAL_SPECIAL_LABELS)},
        'SELECTION_ULTRA_ROI_AFTER_F': {**summary(selection), 'population': pop_type['Ｓ級選抜']},
        'FINAL_SET_LOCK_AFTER_F': {**summary(final), 'population': pop_type['Ｓ級決勝']},
        'fundamental_filter_actions': dict(actions),
        'base_market_buy_candidates': base_buy_races,
        'entry_fail_reasons': dict(fail),
    }


def delta(new, old):
    a, b = new['overall'], old['overall']
    return {
        'bet_races': a['bet_races'] - b['bet_races'],
        'hits': a['hits'] - b['hits'],
        'tickets': a['tickets'] - b['tickets'],
        'stake_yen': a['stake_yen'] - b['stake_yen'],
        'payout_yen': a['payout_yen'] - b['payout_yen'],
        'profit_yen': a['profit_yen'] - b['profit_yen'],
        'roi_pp': (a['roi_pct'] or 0) - (b['roi_pct'] or 0),
    }


def main():
    q1_data = load()
    q2_data = load_q2()
    q3_data = load_q3()

    q1 = evaluate_v9('2024Q1_DEVELOPMENT_V9', q1_data, load_entries('2024_q1'))
    q2 = evaluate_v9('2024Q2_DEVELOPMENT_V9', q2_data, load_entries('2024_q2'))
    q3 = evaluate_v9('2024Q3_DEVELOPMENT_V9_AFTER_V8_OOS_OBSERVED', q3_data, load_entries('2024_q3'))

    b1 = evaluate_v8_25('2024Q1_V8_25_BASELINE', q1_data)
    b2 = evaluate_v8_25('2024Q2_V8_25_BASELINE', q2_data)
    b3 = evaluate_v8_25('2024Q3_V8_25_BASELINE_ALREADY_OBSERVED', q3_data)

    result = {
        'scheme': 'v9.0-F27',
        'base_scheme': 'v8.25-F26',
        'status': 'DEVELOPMENT_Q1Q2Q3_UNIVERSAL_FUNDAMENTAL_LAYER',
        'scientific_status': (
            'Q1-Q3 are development for v9.0. Q3 was previously consumed as the '
            'v8.25 out-of-sample test and is NOT out-of-sample for v9.0. '
            '2024Q4 remains reserved for a future untouched fixed validation.'
        ),
        'fundamental_definition': {
            'inputs': ['score','top3_rate','b_count','nige_count','makuri_count','sashi_count','mark_count','line_position','line_size'],
            'normalization': 'within-race ordinal rank; best=1 worst=0; ties equal',
            'attack': 'equal mean of B, nige, makuri ranks',
            'follow': 'equal mean of sashi, mark ranks',
            'role_fit': 'line leader=attack; followers=follow; singleton=mean(attack,follow)',
            'F': 'equal mean(score_rank, top3_rank, role_fit)',
            'head_block': 'largest adjacent F-gap cut; last max-gap on ties for conservative larger block',
            'action': 'only remove complete first-position rider branches from v8.25 candidate tickets',
            'fitted_thresholds': False,
            'race_type_specific_fundamental_weights': False,
            'uses_prediction_mark_or_evaluation': False,
            'uses_result_or_payout_to_build_candidate': False,
        },
        'Q1': q1,
        'Q2': q2,
        'Q3': q3,
        'delta_vs_v8_25': {
            'Q1': delta(q1, b1),
            'Q2': delta(q2, b2),
            'Q3': delta(q3, b3),
        },
        'next_untouched_validation': '2024Q4',
    }

    print('V9_0_F27_Q1Q2Q3_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V9_0_F27_Q1Q2Q3_END')


if __name__ == '__main__':
    main()

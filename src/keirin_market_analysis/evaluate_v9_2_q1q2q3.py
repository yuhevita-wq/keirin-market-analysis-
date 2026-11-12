from __future__ import annotations

import json
from collections import Counter, defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from evaluate_v8_25_q3_oos import load_q3
from evaluate_v8_25_q1q2 import evaluate as evaluate_v8_25, summary
from evaluate_v9_0_q1q2q3 import load_entries
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_20_f21_initial_special_unified import INITIAL_SPECIAL_LABELS
from v9_2_f28_fundamental_line_set_mispricing import build_v9_2_f28

GROUPS = ('QUALIFYING','GENERAL','SEMIFINAL','SPECIAL','FINAL','OTHER')


def evaluate_v9_2(label, data, entries):
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
    gate_actions = Counter()
    population = 0
    base_market_buy_candidates = 0

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

        d = build_v9_2_f28(
            trio[rid], tf[rid], r.get('predicted_line_formation') or '', rt,
            entries.get(rid, []),
        )
        action = d.get('fundamental_gate_action') or 'UNKNOWN'
        gate_actions[action] += 1
        if action != 'BASE_NO_BET_PASSTHROUGH':
            base_market_buy_candidates += 1

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
            'fundamental_gate_action': action,
            'market_top2_line_indices': d.get('market_top2_line_indices'),
            'fundamental_top2_line_indices': d.get('fundamental_top2_line_indices'),
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
        'SELECTION_ULTRA_ROI_AFTER_LINE_SET_GATE': {**summary(selection), 'population': pop_type['Ｓ級選抜']},
        'FINAL_SET_LOCK_AFTER_LINE_SET_GATE': {**summary(final), 'population': pop_type['Ｓ級決勝']},
        'fundamental_gate_actions': dict(gate_actions),
        'base_market_buy_candidates': base_market_buy_candidates,
        'entry_fail_reasons': dict(fail),
        '_rows': rows,
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


def strip_rows(d):
    return {k: v for k, v in d.items() if k != '_rows'}


def main():
    q1_data, q2_data, q3_data = load(), load_q2(), load_q3()
    q1 = evaluate_v9_2('2024Q1_DEVELOPMENT_V9_2', q1_data, load_entries('2024_q1'))
    q2 = evaluate_v9_2('2024Q2_DEVELOPMENT_V9_2', q2_data, load_entries('2024_q2'))
    q3 = evaluate_v9_2('2024Q3_DEVELOPMENT_V9_2_AFTER_V8_OOS_OBSERVED', q3_data, load_entries('2024_q3'))

    b1 = evaluate_v8_25('2024Q1_V8_25_BASELINE', q1_data)
    b2 = evaluate_v8_25('2024Q2_V8_25_BASELINE', q2_data)
    b3 = evaluate_v8_25('2024Q3_V8_25_BASELINE_ALREADY_OBSERVED', q3_data)

    combined_rows = q1['_rows'] + q2['_rows'] + q3['_rows']
    combined = summary(combined_rows)

    result = {
        'scheme': 'v9.2-F28',
        'base_scheme': 'v8.25-F26',
        'status': 'DEVELOPMENT_Q1Q2Q3_UNIVERSAL_LINE_SET_MISPRICING_GATE',
        'scientific_status': (
            'Q1-Q3 are development for v9.2. Q3 was already observed as the '
            'v8.25 out-of-sample test and is therefore NOT out-of-sample for '
            'v9.2. 2024Q4 remains untouched and reserved for a future fixed validation.'
        ),
        'fundamental_definition': {
            'rider_inputs': ['score','top3_rate','b_count','nige_count','makuri_count','sashi_count','mark_count','line_position','line_size'],
            'normalization': 'within-race ordinal rank; best=1 worst=0; ties equal',
            'attack': 'equal mean of B, nige, makuri ranks',
            'follow': 'equal mean of sashi, mark ranks',
            'role_fit': 'line leader=attack; followers=follow; singleton=mean(attack,follow)',
            'F': 'equal mean(score_rank, top3_rank, role_fit)',
            'fundamental_line_support': 'sum(F) across riders in each predicted line',
            'market_line_support': 'sum of trio-derived rider support S across riders in each predicted line',
            'gate': 'BUY unchanged v8.25 candidate iff market top-two line set != fundamental top-two line set',
            'fitted_numeric_thresholds': False,
            'race_type_specific_fundamental_weights': False,
            'individual_ticket_pruning': False,
            'uses_prediction_mark_or_evaluation': False,
            'uses_result_or_payout_to_build_candidate': False,
        },
        'Q1': strip_rows(q1),
        'Q2': strip_rows(q2),
        'Q3': strip_rows(q3),
        'COMBINED_Q1_Q3': combined,
        'delta_vs_v8_25': {
            'Q1': delta(q1, b1),
            'Q2': delta(q2, b2),
            'Q3': delta(q3, b3),
        },
        'rejected_predecessor': {
            'scheme': 'v9.0-F27',
            'reason': 'direct fundamental first-rider veto worsened Q3 from ROI87.09% to ROI67.46%; not adopted',
        },
        'next_untouched_validation': '2024Q4',
    }

    print('V9_2_F28_Q1Q2Q3_BEGIN')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('V9_2_F28_Q1Q2Q3_END')


if __name__ == '__main__':
    main()

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .research_high_payout_structures_v1 import parse_combination
from .research_weakest_line_2023_2025 import weakest_line
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival
from .simulate_mainline_v1 import num, segment_for

DATA = Path('data/2026_h1/s_class_yosen')
SPEC = Path('data/strategy_specs/dynamic_weak_middle6_proposed_pre2026_eval.json')
OUT = DATA / 'dynamic_weak_middle6_evaluation.json'

ORDERS = {
    ('W1L','A','B'), ('W1L','B','A'),
    ('W1B','A','B'), ('W1B','B','A'),
    ('A','W1B','B'), ('B','W1B','A'),
}


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f(v):
    x = num(v)
    return 0.0 if x == float('-inf') else float(x)


def load_data():
    races = read_csv(DATA / 'races.csv')
    entries = read_csv(DATA / 'entries.csv')
    payouts = read_csv(DATA / 'payouts.csv')

    eb = defaultdict(list)
    for e in entries:
        eb[e['race_id']].append(e)

    tri = defaultdict(list)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            try:
                tri[p['race_id']].append((p['combination'], int(p['payout_yen'])))
            except (TypeError, ValueError):
                pass

    groups = defaultdict(list)
    for r in races:
        groups[(r['race_date'], r['track'])].append(r)
    seg = {}
    for g in groups.values():
        g.sort(key=lambda r: int(r['race_no']))
        for i, r in enumerate(g, 1):
            seg[r['race_id']] = segment_for(i, len(g))
    return races, eb, tri, seg


def assert_spec(spec):
    assert spec['strategy'] == 's_class_yosen_dynamic_weak_middle6'
    assert spec['status'] == 'FROZEN_BEFORE_2026_H1_EVALUATION'
    assert spec['development_years'] == [2023, 2024, 2025]
    assert spec['evaluation_data_used_for_this_formation_search'] is False
    assert spec['scope']['segment'] == '中盤'
    assert spec['pre_race_condition']['name'] == 'HIDDEN_ATTACK'
    assert spec['formation']['points'] == 6
    assert spec['formation']['stake_per_point_yen'] == 100
    assert set(tuple(x.split('-')) for x in spec['formation']['orders']) == ORDERS
    assert spec['no_post_evaluation_changes_allowed'] is True


def hidden_attack(weak_mem, rival):
    w1l = weak_mem[0]
    r1l = rival[0]
    weak_pair_score = f(weak_mem[0].get('score')) + f(weak_mem[1].get('score'))
    rival_pair_score = f(rival[0].get('score')) + f(rival[1].get('score'))
    weak_attack = f(w1l.get('nige_count')) + f(w1l.get('makuri_count'))
    rival_attack = f(r1l.get('nige_count')) + f(r1l.get('makuri_count'))
    return (
        weak_pair_score < rival_pair_score and weak_attack >= rival_attack,
        weak_pair_score, rival_pair_score, weak_attack, rival_attack,
    )


def summarize(rows):
    ordered = sorted(rows, key=lambda r: (r['race_date'], r['track'], r['race_no']))
    stake = sum(r['stake_yen'] for r in ordered)
    payout = sum(r['payout_yen'] for r in ordered)
    hits = [r for r in ordered if r['hit']]
    hitp = [r['payout_yen'] for r in hits]
    cur = mx = 0
    for r in ordered:
        if r['hit']:
            cur = 0
        else:
            cur += 1
            mx = max(mx, cur)
    bym = defaultdict(list)
    for r in ordered:
        bym[r['race_date'][:7]].append(r)
    monthly = {}
    for m, rs in sorted(bym.items()):
        s = sum(x['stake_yen'] for x in rs)
        p = sum(x['payout_yen'] for x in rs)
        monthly[m] = {
            'races': len(rs), 'hits': sum(x['hit'] for x in rs),
            'stake_yen': s, 'payout_yen': p, 'profit_yen': p-s,
            'roi': p/s if s else 0.0,
        }
    return {
        'races': len(ordered),
        'hits': len(hits),
        'hit_rate': len(hits)/len(ordered) if ordered else 0.0,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout-stake,
        'roi': payout/stake if stake else 0.0,
        'high5000_hits': sum(x >= 5000 for x in hitp),
        'high10000_hits': sum(x >= 10000 for x in hitp),
        'high20000_hits': sum(x >= 20000 for x in hitp),
        'median_hit_payout_yen': sorted(hitp)[len(hitp)//2] if hitp else 0,
        'max_hit_payout_yen': max(hitp) if hitp else 0,
        'top1_payout_share': max(hitp)/payout if hitp and payout else 0.0,
        'max_losing_streak': mx,
        'monthly': monthly,
        'hits_detail': [
            {
                'race_date': r['race_date'], 'track': r['track'], 'race_no': r['race_no'],
                'race_id': r['race_id'], 'payout_yen': r['payout_yen'],
                'winning_orders_hit': r['winning_orders_hit'],
                'weak_pair_score': r['weak_pair_score'], 'rival_pair_score': r['rival_pair_score'],
                'weak_attack': r['weak_attack'], 'rival_attack': r['rival_attack'],
            }
            for r in hits
        ],
    }


def main():
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    assert_spec(spec)
    races, eb, tri, seg = load_data()

    rows = []
    middle_with_result = 0
    structural_eligible = 0
    excluded = defaultdict(int)

    for race in races:
        rid = race['race_id']
        if seg.get(rid) != '中盤' or not tri.get(rid):
            continue
        middle_with_result += 1

        chosen = choose_main_line(eb[rid])
        if not chosen:
            excluded['no_main'] += 1
            continue
        main_id, main = chosen
        if len(main) < 3:
            excluded['main_under_3'] += 1
            continue
        rival = strongest_rival(eb[rid], main_id)
        if not rival or len(rival) < 2:
            excluded['no_rival'] += 1
            continue
        weak, reason = weakest_line(eb[rid], main_id)
        if not weak:
            excluded[reason] += 1
            continue
        weak_id, weak_mem, _ = weak
        rival_id = int(rival[0]['line_id'])
        if weak_id == rival_id:
            excluded['weak_equals_rival'] += 1
            continue
        structural_eligible += 1

        qualifies, weak_score, rival_score, weak_attack, rival_attack = hidden_attack(weak_mem, rival)
        if not qualifies:
            continue

        A, B = main[0], main[1]
        W1L, W1B = weak_mem[0], weak_mem[1]
        role_by_car = {
            int(A['car_no']): 'A', int(B['car_no']): 'B',
            int(W1L['car_no']): 'W1L', int(W1B['car_no']): 'W1B',
        }

        payout_sum = 0
        winning_orders_hit = []
        for combo, pay in tri[rid]:
            cars = parse_combination(combo)
            order = tuple(role_by_car.get(c, 'OTHER') for c in cars)
            if order in ORDERS:
                payout_sum += pay
                winning_orders_hit.append('-'.join(order))

        stake = 600
        rows.append({
            'race_date': race['race_date'], 'track': race['track'], 'race_no': int(race['race_no']),
            'race_id': rid, 'hit': payout_sum > 0, 'stake_yen': stake,
            'payout_yen': payout_sum,
            'winning_orders_hit': winning_orders_hit,
            'weak_pair_score': weak_score, 'rival_pair_score': rival_score,
            'weak_attack': weak_attack, 'rival_attack': rival_attack,
        })

    summary = summarize(rows)
    summary['middle_with_result'] = middle_with_result
    summary['structural_eligible'] = structural_eligible
    summary['purchase_rate_within_structural_middle'] = len(rows)/structural_eligible if structural_eligible else 0.0
    summary['purchase_rate_within_all_middle_with_result'] = len(rows)/middle_with_result if middle_with_result else 0.0

    out = {
        'status': 'DYNAMIC_WEAK_MIDDLE6_2026_H1_EVALUATION_COMPLETE',
        'strategy_spec': str(SPEC),
        'strategy_spec_asserted_unchanged': True,
        'development_years': [2023, 2024, 2025],
        'evaluation_period': '2026-01-01 through 2026-06-30',
        'dataset_races': len(races),
        'segment': '中盤',
        'points_per_purchase': 6,
        'stake_per_purchase_yen': 600,
        'no_condition_formation_or_stake_changes': True,
        'important_caveat': 'This strategy family was developed after other 2026 H1 results had already been observed. This is fixed-spec backtesting, not pristine family-level out-of-sample validation.',
        'excluded_structural_reasons': dict(excluded),
        'summary': summary,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'middle_with_result': middle_with_result,
        'structural_eligible': structural_eligible,
        'purchases': summary['races'],
        'hits': summary['hits'],
        'stake_yen': summary['stake_yen'],
        'payout_yen': summary['payout_yen'],
        'profit_yen': summary['profit_yen'],
        'roi': summary['roi'],
        'median_hit_payout_yen': summary['median_hit_payout_yen'],
        'high10000_hits': summary['high10000_hits'],
        'high20000_hits': summary['high20000_hits'],
        'max_losing_streak': summary['max_losing_streak'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

from .research_weakest_line_2023_2025 import weakest_line
from .simulate_early_candidate_v0 import pair_strength
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival, segment_for, num

DATA = Path('data/2026_h1/s_class_yosen')
SPEC = Path('data/strategy_specs/weakest_line_mid_v1_pre2026.json')
OUT = DATA / 'oos_weakest_line_mid_v1' / 'summary.json'


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def score(e):
    x = num(e.get('score'))
    if x == float('-inf'):
        raise ValueError(f'missing score: race_id={e.get("race_id")} car={e.get("car_no")}')
    return float(x)


def formation_orders():
    x = set(itertools.permutations(('A', 'B', 'W1L'), 3))
    x |= set(itertools.permutations(('A', 'B', 'W1B'), 3))
    assert len(x) == 12
    return x


def parse_combo(s: str):
    return tuple(int(x) for x in s.split('-'))


def main():
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    assert spec['status'] == 'FROZEN_PRE_2026_EVALUATION'
    assert spec['segment'] == '中盤'
    assert spec['conditions'] == [
        {'feature': 'weak_lead_score', 'operator': '<=', 'threshold': 98},
        {'feature': 'rival_minus_weak_score', 'operator': '<=', 'threshold': 6},
    ]
    assert spec['formation']['id'] == 'WEAK1_MAINPAIR_12'
    assert spec['formation']['points'] == 12
    assert spec['formation']['stake_yen_per_point'] == 100

    races = read_csv(DATA / 'races.csv')
    entries = read_csv(DATA / 'entries.csv')
    payouts = read_csv(DATA / 'payouts.csv')

    eb = defaultdict(list)
    for e in entries:
        eb[e['race_id']].append(e)

    paid3 = defaultdict(list)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            try:
                paid3[p['race_id']].append((parse_combo(p['combination']), int(p['payout_yen'])))
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

    orders = formation_orders()
    selected = []
    structural_middle = 0
    exclusion = defaultdict(int)

    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        if seg.get(rid) != '中盤':
            continue

        chosen = choose_main_line(eb[rid])
        if not chosen:
            exclusion['no_main'] += 1
            continue
        main_id, main = chosen
        if len(main) < 3:
            exclusion['main_under_3'] += 1
            continue

        rival = strongest_rival(eb[rid], main_id)
        if not rival or len(rival) < 2:
            exclusion['no_strong_rival'] += 1
            continue
        rival_id = int(rival[0]['line_id'])

        weak, reason = weakest_line(eb[rid], main_id)
        if not weak:
            exclusion[reason or 'no_weakest'] += 1
            continue
        weak_id, weak_mem, weak_pair_score = weak
        if weak_id == rival_id:
            exclusion['weak_equals_rival'] += 1
            continue

        structural_middle += 1
        weak_lead_score = score(weak_mem[0])
        rival_minus_weak_score = pair_strength(rival) - weak_pair_score
        if not (weak_lead_score <= 98 and rival_minus_weak_score <= 6):
            continue

        role_by_car = {
            int(main[0]['car_no']): 'A',
            int(main[1]['car_no']): 'B',
            int(weak_mem[0]['car_no']): 'W1L',
            int(weak_mem[1]['car_no']): 'W1B',
        }
        bought = set()
        role_to_car = {v: k for k, v in role_by_car.items()}
        for ro in orders:
            bought.add(tuple(role_to_car[x] for x in ro))
        assert len(bought) == 12

        matched = []
        race_payout = 0
        for combo, pay in paid3.get(rid, []):
            if combo in bought:
                matched.append({'combination': '-'.join(map(str, combo)), 'payout_yen': pay})
                race_payout += pay

        selected.append({
            'race_id': rid,
            'race_date': race['race_date'],
            'track': race['track'],
            'race_no': int(race['race_no']),
            'weak_lead_score': weak_lead_score,
            'rival_minus_weak_score': rival_minus_weak_score,
            'stake_yen': 1200,
            'hit': bool(matched),
            'payout_yen': race_payout,
            'matched_winners': matched,
        })

    stake = sum(r['stake_yen'] for r in selected)
    payout = sum(r['payout_yen'] for r in selected)
    hits = sum(r['hit'] for r in selected)
    hit_payouts = [r['payout_yen'] for r in selected if r['hit']]

    cur = max_ls = 0
    for r in selected:
        if r['hit']:
            cur = 0
        else:
            cur += 1
            max_ls = max(max_ls, cur)

    monthly = {}
    for m in range(1, 7):
        key = f'2026-{m:02d}'
        rows = [r for r in selected if r['race_date'].startswith(key)]
        st = sum(r['stake_yen'] for r in rows)
        py = sum(r['payout_yen'] for r in rows)
        monthly[key] = {
            'races': len(rows),
            'hits': sum(r['hit'] for r in rows),
            'stake_yen': st,
            'payout_yen': py,
            'profit_yen': py - st,
            'roi_pct': (py / st * 100) if st else None,
        }

    sorted_hp = sorted(hit_payouts)
    out = {
        'status': 'WEAKEST_LINE_MID_V1_2026_H1_OOS_EVALUATION',
        'spec_path': str(SPEC),
        'spec_frozen_before_evaluation': True,
        'evaluation_window': '2026-01-01/2026-06-30',
        'strategy': {
            'segment': '中盤',
            'conditions': spec['conditions'],
            'formation': spec['formation'],
        },
        'structural_middle_races': structural_middle,
        'exclusions': dict(exclusion),
        'result': {
            'races': len(selected),
            'hits': hits,
            'hit_rate_pct': (hits / len(selected) * 100) if selected else 0.0,
            'stake_yen': stake,
            'payout_yen': payout,
            'profit_yen': payout - stake,
            'roi_pct': (payout / stake * 100) if stake else 0.0,
            'median_hit_payout_yen': sorted_hp[len(sorted_hp)//2] if sorted_hp else 0,
            'mean_hit_payout_yen': (sum(sorted_hp) / len(sorted_hp)) if sorted_hp else 0.0,
            'max_hit_payout_yen': max(sorted_hp) if sorted_hp else 0,
            'high5000_hits': sum(x >= 5000 for x in sorted_hp),
            'high10000_hits': sum(x >= 10000 for x in sorted_hp),
            'high20000_hits': sum(x >= 20000 for x in sorted_hp),
            'top1_payout_share': (max(sorted_hp) / payout) if sorted_hp and payout else 0.0,
            'max_losing_streak': max_ls,
        },
        'monthly': monthly,
        'selected_races': selected,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': out['status'],
        'structural_middle_races': structural_middle,
        'result': out['result'],
        'monthly': monthly,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

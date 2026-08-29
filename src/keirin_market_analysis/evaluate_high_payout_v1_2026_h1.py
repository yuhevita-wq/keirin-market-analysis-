from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .research_high_payout_structures_v1 import exclusive_roles, parse_combination
from .search_high_payout_strategy_v1 import FORMS
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival, feature_row, segment_for

DATA = Path('data/2026_h1/s_class_yosen')
SPEC = Path('data/strategy_specs/high_payout_v1_proposed_pre_evaluation.json')
OUT = DATA / 'high_payout_v1_evaluation.json'


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_2026_h1():
    races = read_csv(DATA / 'races.csv')
    entries = read_csv(DATA / 'entries.csv')
    payouts = read_csv(DATA / 'payouts.csv')

    eb = defaultdict(list)
    for e in entries:
        eb[e['race_id']].append(e)

    win3 = {}
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            try:
                win3[p['race_id']] = (p['combination'], int(p['payout_yen']))
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
    return races, eb, win3, seg


def assert_spec(spec):
    assert spec['strategy'] == 's_class_yosen_high_payout_v1'
    assert spec['development_years'] == [2023, 2024, 2025]
    assert spec['evaluation_data_used_for_design'] is False
    e = spec['segments']['early']
    m = spec['segments']['middle']
    l = spec['segments']['late']
    assert e['conditions'] == ['B score >= 104', 'A+B top2 rate sum >= 70']
    assert e['formation'] == 'O1O2_KEY_18' and e['points'] == 18
    assert m['conditions'] == ['B score >= 104']
    assert m['formation'] == 'O1_MAINPAIR_6' and m['points'] == 6
    assert l['conditions'] == ['R1L+R1B score sum <= 200', 'A+B top3 rate sum >= 110']
    assert l['formation'] == 'O1_MAINPAIR_6' and l['points'] == 6


def qualifies(seg_name, row):
    if seg_name == 'early':
        return row['b_score'] >= 104 and row['main_top2'] >= 70
    if seg_name == 'middle':
        return row['b_score'] >= 104
    if seg_name == 'late':
        return row['rival_score'] <= 200 and row['main_top3'] >= 110
    raise KeyError(seg_name)


def summarize(rows):
    ordered = sorted(rows, key=lambda r: (r['race_date'], r['track'], r['race_no']))
    stake = sum(r['stake_yen'] for r in ordered)
    payout = sum(r['payout_yen'] for r in ordered)
    hits = [r for r in ordered if r['hit']]
    hitp = [r['payout_yen'] for r in hits]
    rec = [r['recovery_multiple'] for r in hits]
    cur = mx = 0
    for r in ordered:
        if r['hit']:
            cur = 0
        else:
            cur += 1
            mx = max(mx, cur)
    monthly = {}
    bym = defaultdict(list)
    for r in ordered:
        bym[r['race_date'][:7]].append(r)
    for month, rs in sorted(bym.items()):
        s = sum(x['stake_yen'] for x in rs)
        p = sum(x['payout_yen'] for x in rs)
        monthly[month] = {
            'races': len(rs),
            'hits': sum(x['hit'] for x in rs),
            'stake_yen': s,
            'payout_yen': p,
            'profit_yen': p - s,
            'roi': p / s if s else 0.0,
        }
    return {
        'races': len(ordered),
        'hits': len(hits),
        'hit_rate': len(hits) / len(ordered) if ordered else 0.0,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi': payout / stake if stake else 0.0,
        'high5000_hits': sum(p >= 5000 for p in hitp),
        'high10000_hits': sum(p >= 10000 for p in hitp),
        'high20000_hits': sum(p >= 20000 for p in hitp),
        'mean_hit_payout_yen': sum(hitp) / len(hitp) if hitp else 0.0,
        'median_hit_payout_yen': sorted(hitp)[len(hitp)//2] if hitp else 0,
        'mean_recovery_multiple': sum(rec) / len(rec) if rec else 0.0,
        'median_recovery_multiple': sorted(rec)[len(rec)//2] if rec else 0.0,
        'max_hit_payout_yen': max(hitp) if hitp else 0,
        'top1_payout_share': max(hitp) / payout if payout and hitp else 0.0,
        'max_losing_streak': mx,
        'monthly': monthly,
        'hits_detail': [
            {
                'race_date': r['race_date'], 'track': r['track'], 'race_no': r['race_no'],
                'race_id': r['race_id'], 'combination': r['combination'],
                'winner_role_order': '-'.join(r['winner_role_order']),
                'payout_yen': r['payout_yen'],
                'stake_yen': r['stake_yen'],
                'recovery_multiple': r['recovery_multiple'],
            }
            for r in hits
        ],
    }


def main():
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    assert_spec(spec)
    races, eb, win3, seg = load_2026_h1()

    mapseg = {'前半': 'early', '中盤': 'middle', '後半': 'late'}
    formation = {'early': 'O1O2_KEY_18', 'middle': 'O1_MAINPAIR_6', 'late': 'O1_MAINPAIR_6'}
    rows = {k: [] for k in ('early', 'middle', 'late')}
    eligible_base = {k: 0 for k in rows}

    for race in races:
        rid = race['race_id']
        jpseg = seg.get(rid)
        if jpseg not in mapseg or rid not in win3:
            continue
        chosen = choose_main_line(eb[rid])
        if not chosen:
            continue
        mid, main = chosen
        if len(main) < 3:
            continue
        rival = strongest_rival(eb[rid], mid)
        if not rival or len(rival) < 2:
            continue
        seg_name = mapseg[jpseg]
        eligible_base[seg_name] += 1
        roles = exclusive_roles(eb[rid], main, rival)
        combo, pay = win3[rid]
        cars = parse_combination(combo)
        if any(c not in roles for c in cars):
            continue
        order = tuple(roles[c] for c in cars)
        row = feature_row(race, eb[rid], mid, main, rival)
        if not qualifies(seg_name, row):
            continue
        form = formation[seg_name]
        available = set(roles.values())
        valid_orders = [o for o in FORMS[form] if set(o) <= available]
        points = len(valid_orders)
        expected_points = spec['segments'][seg_name]['points']
        assert points == expected_points, (rid, seg_name, points, expected_points, available)
        hit = order in FORMS[form]
        stake = points * 100
        rows[seg_name].append({
            'race_id': rid, 'race_date': race['race_date'], 'track': race['track'],
            'race_no': int(race['race_no']), 'combination': combo,
            'winner_role_order': order, 'hit': hit, 'stake_yen': stake,
            'payout_yen': pay if hit else 0,
            'recovery_multiple': pay / stake if hit else 0.0,
        })

    sections = {}
    all_rows = []
    for k in ('early', 'middle', 'late'):
        st = summarize(rows[k])
        st['eligible_base'] = eligible_base[k]
        st['purchase_rate'] = len(rows[k]) / eligible_base[k] if eligible_base[k] else 0.0
        sections[k] = st
        all_rows.extend(rows[k])
    combined = summarize(all_rows)
    combined['eligible_base'] = sum(eligible_base.values())
    combined['purchase_rate'] = len(all_rows) / combined['eligible_base'] if combined['eligible_base'] else 0.0

    out = {
        'status': 'HIGH_PAYOUT_V1_2026_H1_EVALUATION_COMPLETE',
        'strategy_spec': str(SPEC),
        'strategy_spec_asserted_unchanged': True,
        'development_years': [2023, 2024, 2025],
        'evaluation_period': '2026-01-01 through 2026-06-30',
        'dataset_races': len(races),
        'no_threshold_formation_or_stake_changes': True,
        'important_caveat': 'The high-payout family was conceived after earlier 2026 H1 strategy failures had already been observed; this is a fixed-spec backtest, not a pristine family-level OOS.',
        'sections': sections,
        'combined': combined,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'sections': {k: {x: sections[k][x] for x in ('races','hits','stake_yen','payout_yen','profit_yen','roi','high10000_hits','high20000_hits','median_hit_payout_yen','median_recovery_multiple','max_losing_streak')} for k in sections},
        'combined': {x: combined[x] for x in ('races','hits','stake_yen','payout_yen','profit_yen','roi','high10000_hits','high20000_hits','median_hit_payout_yen','median_recovery_multiple','max_losing_streak')}
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

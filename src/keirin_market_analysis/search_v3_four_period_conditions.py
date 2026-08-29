from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .audit_v3_failure_decomposition import PERIODS, load_period
from .simulate_mainline_v1 import choose_main_line
from .simulate_early_candidate_v0 import strongest_rival
from .search_three_year_conditions_v1 import ATOMS, compatible, feature_row, formations, passes as atom_passes

OUT = Path('data/audits/v3_four_period_condition_search.json')
SPEC = Path('data/audits/v3_condition_search_spec.json')
SEGMENTS = {'early': '前半', 'middle': '中盤', 'late': '後半'}
FORMS = {'early': 'MIX2', 'middle': 'MIX2', 'late': 'MAIN4_X'}
MIN_RACES = {'2023': 20, '2024': 20, '2025': 20, '2026_H1': 8}
MIN_HITS = {'2023': 3, '2024': 3, '2025': 3, '2026_H1': 2}


def fin(rows):
    n = len(rows)
    stake = sum(r['stake'] for r in rows)
    payout = sum(r['payout'] for r in rows)
    hit_pays = [r['payout'] for r in rows if r['payout'] > 0]
    return {
        'races': n,
        'hits': len(hit_pays),
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi': payout / stake if stake else 0.0,
        'hit_rate': len(hit_pays) / n if n else 0.0,
        'top1_payout_share': max(hit_pays) / payout if hit_pays and payout else 0.0,
    }


def main():
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    assert spec['status'] == 'FROZEN_BEFORE_V3_FOUR_PERIOD_CONDITION_SEARCH'
    assert spec['formations_fixed_for_search'] == FORMS

    data = {s: {p: [] for p in PERIODS} for s in SEGMENTS}
    for period, path in PERIODS.items():
        races, eb, tri, _rb, seg = load_period(path)
        for race in races:
            strategy = next((s for s, sg in SEGMENTS.items() if seg.get(race['race_id']) == sg), None)
            if strategy is None:
                continue
            es = eb[race['race_id']]
            chosen = choose_main_line(es)
            if not chosen:
                continue
            main_id, main = chosen
            if len(main) < 3:
                continue
            rival = strongest_rival(es, main_id)
            if not rival or len(rival) < 2:
                continue
            row = feature_row(race, es, main_id, main, rival)
            bets = formations(row, es)[FORMS[strategy]]
            row['stake'] = 100 * len(bets)
            row['payout'] = sum(tri[race['race_id']].get(b, 0) for b in bets)
            data[strategy][period].append(row)

    rules = [(a[0], [a]) for a in ATOMS]
    rules += [
        (a[0] + '__AND__' + b[0], [a, b])
        for i, a in enumerate(ATOMS)
        for b in ATOMS[i + 1:]
        if compatible(a, b)
    ]

    out = {
        'status': 'V3_FOUR_PERIOD_CONDITION_SEARCH_COMPLETE',
        'spec': str(SPEC),
        'atom_count': len(ATOMS),
        'rule_count': len(rules),
        'structural_counts': {s: {p: len(data[s][p]) for p in PERIODS} for s in SEGMENTS},
        'strategies': {},
    }

    for strategy in SEGMENTS:
        qualified = []
        evaluated = 0
        for name, atoms in rules:
            selected = {
                p: [r for r in data[strategy][p] if all(atom_passes(r, a) for a in atoms)]
                for p in PERIODS
            }
            if any(len(selected[p]) < MIN_RACES[p] for p in PERIODS):
                continue
            evaluated += 1
            stats = {p: fin(selected[p]) for p in PERIODS}
            if any(stats[p]['hits'] < MIN_HITS[p] for p in PERIODS):
                continue
            if any(stats[p]['roi'] <= 1.0 for p in PERIODS):
                continue
            if any(stats[p]['top1_payout_share'] > 0.70 for p in PERIODS):
                continue
            rois = sorted(stats[p]['roi'] for p in PERIODS)
            stake = sum(stats[p]['stake_yen'] for p in PERIODS)
            payout = sum(stats[p]['payout_yen'] for p in PERIODS)
            qualified.append({
                'rule': name,
                'complexity': len(atoms),
                'formation': FORMS[strategy],
                'periods': stats,
                'worst_period_roi': rois[0],
                'median_period_roi': (rois[1] + rois[2]) / 2,
                'combined_roi': payout / stake if stake else 0.0,
                'combined_profit_yen': payout - stake,
            })
        qualified.sort(
            key=lambda x: (x['worst_period_roi'], x['median_period_roi'], x['combined_roi'], -x['complexity']),
            reverse=True,
        )
        out['strategies'][strategy] = {
            'segment': SEGMENTS[strategy],
            'formation': FORMS[strategy],
            'evaluated_after_min_race_filter': evaluated,
            'qualified_count': len(qualified),
            'top_candidates': qualified[:30],
        }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({s: {'qualified': out['strategies'][s]['qualified_count'], 'top': out['strategies'][s]['top_candidates'][:5]} for s in SEGMENTS}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

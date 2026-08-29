from __future__ import annotations

import json
from pathlib import Path

from .audit_v3_failure_decomposition import PERIODS, load_period
from .simulate_mainline_v1 import choose_main_line
from .simulate_early_candidate_v0 import strongest_rival
from .search_three_year_conditions_v1 import ATOMS, compatible, feature_row, formations, passes as atom_passes
from .search_v3_four_period_conditions import SEGMENTS, FORMS, MIN_RACES, MIN_HITS, fin

OUT = Path('data/audits/v3_leave_one_period_out.json')


def build_data():
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
    return data


def main():
    spec = json.loads(Path('data/audits/v3_condition_search_spec.json').read_text(encoding='utf-8'))
    assert spec['status'] == 'FROZEN_BEFORE_V3_FOUR_PERIOD_CONDITION_SEARCH'
    data = build_data()
    rules = [(a[0], [a]) for a in ATOMS]
    rules += [(a[0]+'__AND__'+b[0], [a,b]) for i,a in enumerate(ATOMS) for b in ATOMS[i+1:] if compatible(a,b)]
    out = {'status':'V3_LEAVE_ONE_PERIOD_OUT_COMPLETE','formations':FORMS,'strategies':{}}

    for strategy in SEGMENTS:
        cases = {}
        for heldout in PERIODS:
            train = [p for p in PERIODS if p != heldout]
            candidates = []
            for name, atoms in rules:
                selected = {p:[r for r in data[strategy][p] if all(atom_passes(r,a) for a in atoms)] for p in PERIODS}
                if any(len(selected[p]) < MIN_RACES[p] for p in train):
                    continue
                stats = {p:fin(selected[p]) for p in PERIODS}
                if any(stats[p]['hits'] < MIN_HITS[p] for p in train):
                    continue
                if any(stats[p]['roi'] <= 1.0 for p in train):
                    continue
                if any(stats[p]['top1_payout_share'] > 0.70 for p in train):
                    continue
                rois = sorted(stats[p]['roi'] for p in train)
                st = sum(stats[p]['stake_yen'] for p in train)
                py = sum(stats[p]['payout_yen'] for p in train)
                candidates.append({
                    'rule':name,'complexity':len(atoms),'stats':stats,
                    'train_worst_roi':rois[0],
                    'train_median_roi':rois[len(rois)//2],
                    'train_combined_roi':py/st if st else 0.0,
                })
            candidates.sort(key=lambda x:(x['train_worst_roi'],x['train_median_roi'],x['train_combined_roi'],-x['complexity']),reverse=True)
            if not candidates:
                cases[heldout] = {'selected':None,'heldout_profitable':False,'candidate_count':0}
                continue
            best = candidates[0]
            h = best['stats'][heldout]
            cases[heldout] = {
                'candidate_count':len(candidates),
                'selected':{
                    'rule':best['rule'],'complexity':best['complexity'],
                    'train_worst_roi':best['train_worst_roi'],
                    'train_median_roi':best['train_median_roi'],
                    'train_combined_roi':best['train_combined_roi'],
                },
                'heldout':h,
                'heldout_profitable':h['roi']>1.0,
            }
        out['strategies'][strategy] = {
            'segment':SEGMENTS[strategy],
            'cases':cases,
            'profitable_heldouts':sum(1 for x in cases.values() if x['heldout_profitable']),
        }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())

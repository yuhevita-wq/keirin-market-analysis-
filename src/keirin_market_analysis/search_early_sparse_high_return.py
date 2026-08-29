from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_early_candidate_v0 import strongest_rival

YEARS = (2024, 2025)
OUT = Path('data/audits/early_sparse_high_return_search.json')


def read_csv(path: Path):
    norm = str(path).replace('\\', '/')
    if '/2023/' in norm:
        raise RuntimeError('2023 OOS is prohibited during early-condition development')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f(v):
    x = num(v)
    return 0.0 if x == float('-inf') else float(x)


def dataset(year: int):
    d = Path(f'data/{year}/s_class_yosen')
    races = read_csv(d/'races.csv')
    entries = read_csv(d/'entries.csv')
    payouts = read_csv(d/'payouts.csv')
    eb = defaultdict(list)
    groups = defaultdict(list)
    tri = defaultdict(dict)
    for e in entries:
        eb[e['race_id']].append(e)
    for r in races:
        groups[(r['race_date'], r['track'])].append(r)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            try:
                tri[p['race_id']][p['combination']] = int(p['payout_yen'])
            except ValueError:
                pass
    seg = {}
    for g in groups.values():
        g.sort(key=lambda r: int(r['race_no']))
        for i, r in enumerate(g, 1):
            seg[r['race_id']] = segment_for(i, len(g))
    return races, eb, tri, seg


def rows_for(year: int):
    races, eb, tri, seg = dataset(year)
    out = []
    for race in races:
        rid = race['race_id']
        if seg.get(rid) != '前半':
            continue
        main = choose_main_line(eb[rid])
        if not main:
            continue
        mid, m = main
        if len(m) < 3:
            continue
        rival = strongest_rival(eb[rid], mid)
        if not rival or len(rival) < 2:
            continue
        A, B = m[0], m[1]
        R1L, R1B = rival[0], rival[1]
        a, b = int(A['car_no']), int(B['car_no'])
        l, rb = int(R1L['car_no']), int(R1B['car_no'])
        bets = [f'{l}-{rb}-{a}', f'{l}-{rb}-{b}', f'{rb}-{l}-{a}', f'{rb}-{l}-{b}']
        payout = sum(tri[rid].get(c, 0) for c in bets)
        vals = {
            'main_win': f(A.get('win_rate')) + f(B.get('win_rate')),
            'main_top2': f(A.get('top2_rate')) + f(B.get('top2_rate')),
            'main_top3': f(A.get('top3_rate')) + f(B.get('top3_rate')),
            'rival_win': f(R1L.get('win_rate')) + f(R1B.get('win_rate')),
            'rival_top2': f(R1L.get('top2_rate')) + f(R1B.get('top2_rate')),
            'rival_top3': f(R1L.get('top3_rate')) + f(R1B.get('top3_rate')),
            'pair_score_gap': (f(A.get('score')) + f(B.get('score'))) - (f(R1L.get('score')) + f(R1B.get('score'))),
            'leader_score_gap': f(A.get('score')) - f(R1L.get('score')),
            'second_score_gap': f(B.get('score')) - f(R1B.get('score')),
            'pair_win_gap': (f(A.get('win_rate')) + f(B.get('win_rate'))) - (f(R1L.get('win_rate')) + f(R1B.get('win_rate'))),
            'pair_top2_gap': (f(A.get('top2_rate')) + f(B.get('top2_rate'))) - (f(R1L.get('top2_rate')) + f(R1B.get('top2_rate'))),
            'pair_top3_gap': (f(A.get('top3_rate')) + f(B.get('top3_rate'))) - (f(R1L.get('top3_rate')) + f(R1B.get('top3_rate'))),
        }
        out.append({
            'race_id': rid,
            'race_date': race['race_date'],
            'payout': payout,
            'hit': int(payout > 0),
            **vals,
        })
    return out


# Coarse predeclared threshold families. No result-driven fine thresholds.
ATOMS = []

def add(name, fn):
    ATOMS.append((name, fn))

for t in (16, 20, 24, 28, 32, 36):
    add(f'main_win_le_{t}', lambda r, t=t: r['main_win'] <= t)
for t in (45, 55, 65, 75, 85):
    add(f'main_top2_le_{t}', lambda r, t=t: r['main_top2'] <= t)
for t in (80, 90, 100, 110, 120):
    add(f'main_top3_le_{t}', lambda r, t=t: r['main_top3'] <= t)
for t in (20, 30, 40, 50):
    add(f'rival_win_ge_{t}', lambda r, t=t: r['rival_win'] >= t)
for t in (45, 55, 65, 75):
    add(f'rival_top2_ge_{t}', lambda r, t=t: r['rival_top2'] >= t)
for t in (80, 90, 100, 110):
    add(f'rival_top3_ge_{t}', lambda r, t=t: r['rival_top3'] >= t)
for t in (-6, -3, 0, 3, 6):
    add(f'pair_score_gap_le_{t}', lambda r, t=t: r['pair_score_gap'] <= t)
for t in (-4, -2, 0, 2, 4):
    add(f'leader_score_gap_le_{t}', lambda r, t=t: r['leader_score_gap'] <= t)
for t in (-4, -2, 0, 2, 4):
    add(f'second_score_gap_le_{t}', lambda r, t=t: r['second_score_gap'] <= t)
for t in (-25, -15, -5, 5):
    add(f'pair_win_gap_le_{t}', lambda r, t=t: r['pair_win_gap'] <= t)
for t in (-40, -20, 0, 20):
    add(f'pair_top2_gap_le_{t}', lambda r, t=t: r['pair_top2_gap'] <= t)
for t in (-40, -20, 0, 20):
    add(f'pair_top3_gap_le_{t}', lambda r, t=t: r['pair_top3_gap'] <= t)


def stats(rows, fn):
    x = [r for r in rows if fn(r)]
    n = len(x)
    stake = n * 400
    payout = sum(r['payout'] for r in x)
    hits = sum(r['hit'] for r in x)
    ordered = sorted(x, key=lambda r: r['race_date'])
    half = max(1, len(ordered)//2)
    halves = []
    for part in (ordered[:half], ordered[half:]):
        s = len(part)*400
        p = sum(r['payout'] for r in part)
        halves.append({'races': len(part), 'hits': sum(r['hit'] for r in part), 'roi': p/s if s else None})
    paid = sorted((r['payout'] for r in x if r['payout'] > 0), reverse=True)
    return {
        'races': n,
        'hits': hits,
        'hit_rate': hits/n if n else 0.0,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout-stake,
        'roi': payout/stake if stake else 0.0,
        'max_hit_payout_yen': paid[0] if paid else 0,
        'top1_payout_share': (paid[0]/payout) if paid and payout else 0.0,
        'halves': halves,
    }


def main():
    ys = {y: rows_for(y) for y in YEARS}
    candidates = []

    rules = []
    for name, fn in ATOMS:
        rules.append((name, fn, 1))
    # Pair only atoms from different metric families to avoid redundant threshold stacking.
    for i, (n1, f1) in enumerate(ATOMS):
        fam1 = n1.rsplit('_', 2)[0]
        for n2, f2 in ATOMS[i+1:]:
            fam2 = n2.rsplit('_', 2)[0]
            if fam1 == fam2:
                continue
            rules.append((f'{n1}__AND__{n2}', lambda r, f1=f1, f2=f2: f1(r) and f2(r), 2))

    for name, fn, complexity in rules:
        ss = {str(y): stats(ys[y], fn) for y in YEARS}
        if min(ss[str(y)]['races'] for y in YEARS) < 10:
            continue
        if min(ss[str(y)]['hits'] for y in YEARS) < 2:
            continue
        if min(ss[str(y)]['roi'] for y in YEARS) <= 1.0:
            continue
        worst_roi = min(ss[str(y)]['roi'] for y in YEARS)
        combined_stake = sum(ss[str(y)]['stake_yen'] for y in YEARS)
        combined_payout = sum(ss[str(y)]['payout_yen'] for y in YEARS)
        candidates.append({
            'rule': name,
            'complexity': complexity,
            '2024': ss['2024'],
            '2025': ss['2025'],
            'worst_year_roi': worst_roi,
            'combined': {
                'races': ss['2024']['races'] + ss['2025']['races'],
                'hits': ss['2024']['hits'] + ss['2025']['hits'],
                'stake_yen': combined_stake,
                'payout_yen': combined_payout,
                'profit_yen': combined_payout-combined_stake,
                'roi': combined_payout/combined_stake if combined_stake else 0.0,
            },
        })

    candidates.sort(key=lambda c: (c['worst_year_roi'], c['combined']['roi'], -c['complexity']), reverse=True)
    out = {
        'scope': '2024+2025 development only. 2023 is prohibited and not read.',
        'goal': 'Find sparse early conditions with high return while keeping the original 4-bet rival-pair formation fixed.',
        'formation': ['R1L-R1B-A','R1L-R1B-B','R1B-R1L-A','R1B-R1L-B'],
        'search_guardrails': {
            'thresholds': 'coarse predeclared grids only',
            'max_conditions': 2,
            'min_races_each_year': 10,
            'min_hits_each_year': 2,
            'required_roi_each_year': '> 100%',
            'ranking': 'worst-year ROI, then combined ROI, then lower complexity',
            'candidate_atom_count': len(ATOMS),
            'evaluated_rule_count': len(rules),
        },
        'qualified_candidate_count': len(candidates),
        'top_candidates': candidates[:30],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

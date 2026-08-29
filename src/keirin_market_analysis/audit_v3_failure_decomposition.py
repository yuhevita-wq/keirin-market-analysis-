from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .simulate_mainline_v1 import choose_main_line, segment_for
from .simulate_early_candidate_v0 import strongest_rival
from .search_three_year_conditions_v1 import feature_row, formations, f

OUT = Path('data/audits/v3_failure_decomposition.json')
PERIODS = {
    '2023': Path('data/2023/s_class_yosen'),
    '2024': Path('data/2024/s_class_yosen'),
    '2025': Path('data/2025/s_class_yosen'),
    '2026_H1': Path('data/2026_h1/s_class_yosen'),
}
STRATEGIES = {
    'early': {'segment': '前半', 'formation': 'MIX2'},
    'middle': {'segment': '中盤', 'formation': 'MIX2'},
    'late': {'segment': '後半', 'formation': 'MAIN4_X'},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def passes(strategy: str, row: dict[str, object]) -> bool:
    if strategy == 'early':
        return float(row['pair_score_gap']) <= 6 and float(row['pair_win_gap']) <= 0
    if strategy == 'middle':
        return float(row['rival_top3']) >= 80 and float(row['pair_top3_gap']) >= 10
    if strategy == 'late':
        return float(row['r1l_score']) >= 100 and float(row['main_top2']) <= 70
    raise KeyError(strategy)


def load_period(path: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, dict[str, int]], dict[str, list[dict[str, str]]], dict[str, str]]:
    races = read_csv(path / 'races.csv')
    entries = read_csv(path / 'entries.csv')
    payouts = read_csv(path / 'payouts.csv')
    results = read_csv(path / 'results.csv')

    eb: dict[str, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        eb[e['race_id']].append(e)

    tri: dict[str, dict[str, int]] = defaultdict(dict)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            tri[p['race_id']][p['combination']] = int(p['payout_yen'])

    rb: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in results:
        rb[r['race_id']].append(r)

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in races:
        groups[(r['race_date'], r['track'])].append(r)
    seg: dict[str, str] = {}
    for g in groups.values():
        g.sort(key=lambda r: int(r['race_no']))
        for i, r in enumerate(g, 1):
            seg[r['race_id']] = segment_for(i, len(g))
    return races, eb, tri, rb, seg


def actual_triplet(results: list[dict[str, str]], role_by_car: dict[int, str]) -> str:
    placed = []
    for r in results:
        fp = str(r.get('finish_position', '')).strip()
        if fp.isdigit() and int(fp) in (1, 2, 3):
            placed.append((int(fp), int(r['car_no'])))
    placed.sort()
    if len(placed) != 3:
        return 'INCOMPLETE'
    return '-'.join(role_by_car.get(car, 'O') for _, car in placed)


def role_map(row: dict[str, object], es: list[dict[str, str]]) -> dict[int, str]:
    m = {
        int(row['A']): 'A', int(row['B']): 'B', int(row['M3']): 'M3',
        int(row['R1L']): 'R1L', int(row['R1B']): 'R1B',
    }
    excluded = {int(row['A']), int(row['B']), int(row['M3'])}
    remain = [e for e in es if int(e['car_no']) not in excluded]
    if remain:
        x = max(remain, key=lambda e: (f(e.get('score')), -int(e['car_no'])))
        xc = int(x['car_no'])
        if xc not in m:
            m[xc] = 'X'
    return m


def summarize(selected: list[dict[str, object]], structural_n: int, formation: str) -> dict[str, object]:
    n = len(selected)
    if not n:
        return {
            'structural_eligible_races': structural_n, 'entrance_races': 0, 'entrance_rate': 0.0,
            'formation': formation, 'formation_points': 0, 'formation_hits': 0, 'formation_hit_rate': 0.0,
            'stake_yen': 0, 'payout_yen': 0, 'profit_yen': 0, 'roi': 0.0,
            'hit_payout_mean_yen': 0.0, 'hit_payout_median_yen': 0.0,
            'break_even_hit_rate_at_observed_mean_payout': None,
            'break_even_mean_payout_at_observed_hit_rate_yen': None,
            'top1_payout_share': 0.0, 'top_role_triplets': [],
        }
    points = int(selected[0]['points'])
    stake = sum(int(r['stake']) for r in selected)
    pay = sum(int(r['payout']) for r in selected)
    hit_pays = [int(r['payout']) for r in selected if int(r['payout']) > 0]
    hits = len(hit_pays)
    hit_rate = hits / n
    mean_pay = sum(hit_pays) / hits if hits else 0.0
    triplets = Counter(str(r['triplet']) for r in selected)
    return {
        'structural_eligible_races': structural_n,
        'entrance_races': n,
        'entrance_rate': n / structural_n if structural_n else 0.0,
        'formation': formation,
        'formation_points': points,
        'formation_hits': hits,
        'formation_hit_rate': hit_rate,
        'stake_yen': stake,
        'payout_yen': pay,
        'profit_yen': pay - stake,
        'roi': pay / stake if stake else 0.0,
        'hit_payout_mean_yen': mean_pay,
        'hit_payout_median_yen': float(median(hit_pays)) if hit_pays else 0.0,
        'break_even_hit_rate_at_observed_mean_payout': (points * 100 / mean_pay) if mean_pay else None,
        'break_even_mean_payout_at_observed_hit_rate_yen': (points * 100 / hit_rate) if hit_rate else None,
        'top1_payout_share': max(hit_pays) / pay if pay and hit_pays else 0.0,
        'top_role_triplets': [{'triplet': t, 'count': c, 'rate': c / n} for t, c in triplets.most_common(10)],
    }


def main() -> int:
    protocol = json.loads(Path('data/audits/v3_research_protocol.json').read_text(encoding='utf-8'))
    assert protocol['status'] == 'V3_RESEARCH_PROTOCOL_FROZEN_BEFORE_DETAILED_2026_H1_OUTCOME_INSPECTION'

    out: dict[str, object] = {
        'status': 'V3_PHASE1_FAILURE_DECOMPOSITION_COMPLETE',
        'development_periods': list(PERIODS),
        'protocol': 'data/audits/v3_research_protocol.json',
        'strategies': {},
    }
    accum: dict[str, dict[str, list[dict[str, object]]]] = {
        s: {p: [] for p in PERIODS} for s in STRATEGIES
    }
    structural: dict[str, dict[str, int]] = {s: {p: 0 for p in PERIODS} for s in STRATEGIES}

    for period, path in PERIODS.items():
        races, eb, tri, rb, seg = load_period(path)
        for race in races:
            rid = race['race_id']
            segment = seg.get(rid)
            matching = [s for s, spec in STRATEGIES.items() if spec['segment'] == segment]
            if not matching:
                continue
            es = eb[rid]
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
            forms = formations(row, es)
            roles = role_map(row, es)
            triplet = actual_triplet(rb[rid], roles)
            for strategy in matching:
                structural[strategy][period] += 1
                if not passes(strategy, row):
                    continue
                form = STRATEGIES[strategy]['formation']
                bets = forms[form]
                payout = sum(tri[rid].get(b, 0) for b in bets)
                accum[strategy][period].append({
                    'race_id': rid, 'points': len(bets), 'stake': 100 * len(bets),
                    'payout': payout, 'triplet': triplet,
                })

    for strategy, spec in STRATEGIES.items():
        period_stats = {
            p: summarize(accum[strategy][p], structural[strategy][p], spec['formation'])
            for p in PERIODS
        }
        dev_rows = [r for p in ('2023', '2024', '2025') for r in accum[strategy][p]]
        dev_structural = sum(structural[strategy][p] for p in ('2023', '2024', '2025'))
        pooled_2325 = summarize(dev_rows, dev_structural, spec['formation'])
        h1 = period_stats['2026_H1']
        out['strategies'][strategy] = {
            'segment': spec['segment'],
            'formation': spec['formation'],
            'periods': period_stats,
            'pooled_2023_2025': pooled_2325,
            '2026_h1_vs_pooled_2023_2025': {
                'entrance_rate_ratio': h1['entrance_rate'] / pooled_2325['entrance_rate'] if pooled_2325['entrance_rate'] else None,
                'formation_hit_rate_ratio': h1['formation_hit_rate'] / pooled_2325['formation_hit_rate'] if pooled_2325['formation_hit_rate'] else None,
                'hit_payout_mean_ratio': h1['hit_payout_mean_yen'] / pooled_2325['hit_payout_mean_yen'] if pooled_2325['hit_payout_mean_yen'] else None,
                'roi_ratio': h1['roi'] / pooled_2325['roi'] if pooled_2325['roi'] else None,
            },
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

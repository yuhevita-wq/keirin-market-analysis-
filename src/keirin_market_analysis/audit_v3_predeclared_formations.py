from __future__ import annotations

import json
from pathlib import Path

from .audit_v3_failure_decomposition import PERIODS, load_period, passes
from .simulate_mainline_v1 import choose_main_line
from .simulate_early_candidate_v0 import strongest_rival
from .search_three_year_conditions_v1 import feature_row, formations, f

OUT = Path('data/audits/v3_predeclared_formation_audit.json')
SPECS = {
    'early': ('前半', ['MIX2', 'RIVAL4', 'MAIN4_RIVAL']),
    'middle': ('中盤', ['MIX2', 'MAIN4_RIVAL', 'MAIN6', 'RIVAL4']),
    'late': ('後半', ['MAIN2_M3', 'MAIN2_X', 'MAIN4_X', 'MAIN6']),
}


def all_forms(row, es):
    out = formations(row, es)
    A, B, M3 = [int(row[k]) for k in ('A', 'B', 'M3')]
    out['MAIN2_M3'] = [f'{A}-{B}-{M3}', f'{B}-{A}-{M3}']
    remain = [e for e in es if int(e['car_no']) not in {A, B, M3}]
    if remain:
        X = max(remain, key=lambda e: (f(e.get('score')), -int(e['car_no'])))
        x = int(X['car_no'])
        out['MAIN2_X'] = [f'{A}-{B}-{x}', f'{B}-{A}-{x}']
    return out


def stats(rows):
    n = len(rows)
    stake = sum(s for s, _ in rows)
    payout = sum(p for _, p in rows)
    hit_pays = [p for _, p in rows if p > 0]
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
    protocol = json.loads(Path('data/audits/v3_research_protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase2_predeclared_formation_families'] == {
        'early': ['MIX2', 'RIVAL4', 'MAIN4_RIVAL'],
        'middle': ['MIX2', 'MAIN4_RIVAL', 'MAIN6', 'RIVAL4'],
        'late': ['MAIN2_M3', 'MAIN2_X', 'MAIN4_X', 'MAIN6'],
    }
    acc = {s: {form: {p: [] for p in PERIODS} for form in forms} for s, (_, forms) in SPECS.items()}

    for period, path in PERIODS.items():
        races, eb, tri, _rb, seg = load_period(path)
        for race in races:
            segment = seg.get(race['race_id'])
            selected = [s for s, (sg, _) in SPECS.items() if sg == segment]
            if not selected:
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
            fs = all_forms(row, es)
            for strategy in selected:
                if not passes(strategy, row):
                    continue
                for form in SPECS[strategy][1]:
                    bets = fs[form]
                    payout = sum(tri[race['race_id']].get(b, 0) for b in bets)
                    acc[strategy][form][period].append((100 * len(bets), payout))

    out = {'status': 'V3_PREDECLARED_FORMATION_AUDIT_COMPLETE', 'strategies': {}}
    for strategy, (segment, forms) in SPECS.items():
        rows = []
        for form in forms:
            periods = {p: stats(acc[strategy][form][p]) for p in PERIODS}
            stake = sum(x['stake_yen'] for x in periods.values())
            payout = sum(x['payout_yen'] for x in periods.values())
            rows.append({
                'formation': form,
                'periods': periods,
                'profitable_periods': sum(x['roi'] > 1.0 for x in periods.values()),
                'worst_period_roi': min(x['roi'] for x in periods.values()),
                'combined_roi': payout / stake if stake else 0.0,
                'combined_profit_yen': payout - stake,
            })
        rows.sort(key=lambda x: (x['profitable_periods'], x['worst_period_roi'], x['combined_roi']), reverse=True)
        out['strategies'][strategy] = {'segment': segment, 'formations': rows}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

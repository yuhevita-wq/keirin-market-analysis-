"""Apply D=sum(y)-sum(q) to archived, corrected sufficient statistics.
No new decryption, result-label audit, probability model, or significance claim.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'research/snowberg_observation_corrected_20261027/summary.json'


def calculate(tickets, hits, expected, returns):
    if tickets <= 0 or not 0 <= hits <= tickets or not 0 <= expected <= tickets:
        raise ValueError('Invalid sufficient statistics')
    return {'tickets': tickets, 'hits': hits, 'market_expected_hits': expected,
            'D': hits - expected, 'actual_over_market': hits / expected if expected else None,
            'roi_pct_final_odds_proxy': 100 * returns / tickets}


def run(source=SOURCE, out=OUT):
    raw = source.read_bytes()
    data = json.loads(raw)
    if data['status'] != 'COMPLETE_WITH_INDEPENDENT_ORIENTATION_CHECKS':
        raise ValueError('Source is incomplete or orientation is unverified')
    cells = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for row in data['groups']:
        source_name, year, kind, family, *tail = row['group']
        if source_name != 'sealed_four_markets_orientation_verified' or family != 'price':
            continue  # Never add overlapping price_cars/price_type/cross partitions.
        band = int(tail[0])
        for key in ((year, kind, band), ('ALL', kind, band)):
            for i, name in enumerate(('tickets', 'hits', 'market_expected_hits', 'return_units')):
                cells[key][i] += row[name]
    if not cells:
        raise ValueError('No disjoint price cells')
    cuts = data['price_upper_bounds_inclusive']
    def label(b):
        return f'({cuts[b-1] if b else 0}, {cuts[b]}]' if b < len(cuts) else f'({cuts[-1]}, inf)'
    rows = [{'year': y, 'market': k, 'band_index': b, 'odds_band': label(b),
             **calculate(*v)} for (y, k, b), v in sorted(cells.items())]
    controls = []
    for year in sorted({r['year'] for r in rows}):
        for kind in sorted({r['market'] for r in rows}):
            rs = [r for r in rows if r['year'] == year and r['market'] == kind]
            if not rs:
                continue
            hits = sum(r['hits'] for r in rs)
            expected = math.fsum(r['market_expected_hits'] for r in rs)
            required = (data['audit']['usable_races_sealed_four_markets_orientation_verified']
                        if year == 'ALL' else data['audit']['year_' + year])
            if hits != required or not math.isclose(expected, required, abs_tol=1e-5):
                raise ValueError(f'Partition does not reconcile: {year} {kind}')
            controls.append({'year': year, 'market': kind, 'hits': hits,
                             'market_expected_hits': expected, 'D': hits - expected})
    report = {'status': 'COMPLETE_AGGREGATE_FORMULA_RECALCULATION',
              'source_path': str(source.relative_to(ROOT)),
              'source_sha256': hashlib.sha256(raw).hexdigest(),
              'source_generation_commit': data['source_commit'],
              'source_audit': data['audit'], 'formula': 'D=sum(y)-sum(q_proxy)',
              'q_proxy': '(1/O)/sum_all_selections(1/O), separately within each race and market',
              'scope': 'All disjoint archived price cells, 4 markets, 2021 through 2026-H1',
              'limits': ['Reuses archived sufficient statistics; raw races NOT reprocessed or independently re-audited in this run.',
                         'q_proxy is not exact observed betting-money share.',
                         'D is a count residual, not money profit; positive D need not imply ROI > 100%.',
                         'Annual signs are descriptive, not untouched out-of-sample validation.',
                         'No race/day residuals in source: no dependence-aware confidence intervals or significance claims.',
                         'Return uses final quoted odds, not independently checked official payouts or pre-close executable prices.',
                         'Source label/orientation checks are inherited, not repeated.'],
              'partition_controls': controls, 'rows': rows}
    out.mkdir(parents=True, exist_ok=True)
    (out/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (out/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'status': report['status'], 'races': data['audit']['usable_races_sealed_four_markets_orientation_verified'],
                      'cells': len(rows), 'sha256': report['source_sha256']}, ensure_ascii=False))
    for r in rows:
        if r['year'] == 'ALL' and r['market'] == '3rentan':
            print(json.dumps(r, ensure_ascii=False))
    return report

if __name__ == '__main__':
    run()

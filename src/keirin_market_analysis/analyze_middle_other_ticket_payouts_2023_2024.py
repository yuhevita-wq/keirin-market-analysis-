from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

YEARS = (2023, 2024)
TICKET_TYPES = ('2枠複','2枠単','2車複','2車単','3連複','3連単','ワイド')
OUT = Path('data/audits/middle_other_ticket_payouts_2023_2024.json')
COMPACT = Path('data/audits/middle_other_ticket_payouts_2023_2024_compact.json')


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def q(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    pos = (len(ys)-1)*p
    lo = int(pos)
    hi = min(lo+1, len(ys)-1)
    frac = pos-lo
    return ys[lo] + (ys[hi]-ys[lo])*frac


def summary(vals):
    xs = [int(x) for x in vals]
    if not xs:
        return {'n':0}
    return {
        'n': len(xs),
        'median_yen': statistics.median(xs),
        'mean_yen': sum(xs)/len(xs),
        'p25_yen': q(xs,.25),
        'p75_yen': q(xs,.75),
        'p90_yen': q(xs,.90),
        'max_yen': max(xs),
        'ge1000_pct': 100*sum(x>=1000 for x in xs)/len(xs),
        'ge5000_pct': 100*sum(x>=5000 for x in xs)/len(xs),
        'ge10000_pct': 100*sum(x>=10000 for x in xs)/len(xs),
        'ge20000_pct': 100*sum(x>=20000 for x in xs)/len(xs),
    }


def load_year(year):
    decisions = read_csv(Path(f'data/audits/market_scenario_portfolio_{year}_v1_decisions.csv'))
    payouts = read_csv(Path(f'data/{year}/s_class_yosen/payouts.csv'))
    middle_ids = {r['race_id'] for r in decisions if int(r.get('middle_count') or 0) > 0}
    no_middle_ids = {r['race_id'] for r in decisions if int(r.get('middle_count') or 0) == 0}
    all_ids = {r['race_id'] for r in decisions}

    by_group = {'middle':middle_ids, 'no_middle':no_middle_ids, 'all':all_ids}
    out = {
        'decision_races': len(all_ids),
        'middle_races': len(middle_ids),
        'no_middle_races': len(no_middle_ids),
        'middle_rate_pct': 100*len(middle_ids)/len(all_ids) if all_ids else 0.0,
        'groups': {},
    }
    paid = []
    for p in payouts:
        if p.get('status') != 'paid':
            continue
        if p.get('ticket_type') not in TICKET_TYPES:
            continue
        try:
            py = int(float(p.get('payout_yen') or 0))
        except ValueError:
            continue
        if py <= 0:
            continue
        paid.append((p['race_id'], p['ticket_type'], py))

    for gname, ids in by_group.items():
        gout = {}
        for tt in TICKET_TYPES:
            rows = [(rid,py) for rid,t,py in paid if t == tt and rid in ids]
            row_vals = [py for _,py in rows]
            per_race = defaultdict(list)
            for rid,py in rows:
                per_race[rid].append(py)
            race_max = [max(v) for v in per_race.values()]
            race_med = [statistics.median(v) for v in per_race.values()]
            gout[tt] = {
                'offered_races': len(per_race),
                'published_paid_rows': len(row_vals),
                'all_published_payout_rows': summary(row_vals),
                'per_race_max_payout': summary(race_max),
                'per_race_median_payout': summary(race_med),
            }
        out['groups'][gname] = gout

    comparisons = {}
    for tt in TICKET_TYPES:
        m = out['groups']['middle'][tt]['per_race_max_payout']
        n = out['groups']['no_middle'][tt]['per_race_max_payout']
        comparisons[tt] = {
            'middle_vs_no_middle_median_ratio': (m.get('median_yen')/n.get('median_yen')) if m.get('median_yen') and n.get('median_yen') else None,
            'middle_minus_no_middle_ge5000_pp': (m.get('ge5000_pct',0)-n.get('ge5000_pct',0)) if m.get('n') and n.get('n') else None,
            'middle_minus_no_middle_ge10000_pp': (m.get('ge10000_pct',0)-n.get('ge10000_pct',0)) if m.get('n') and n.get('n') else None,
            'middle_minus_no_middle_ge20000_pp': (m.get('ge20000_pct',0)-n.get('ge20000_pct',0)) if m.get('n') and n.get('n') else None,
        }
    out['middle_vs_no_middle'] = comparisons
    return out


def compact_year(y):
    z = {
        'decision_races': y['decision_races'],
        'middle_races': y['middle_races'],
        'no_middle_races': y['no_middle_races'],
        'middle_rate_pct': y['middle_rate_pct'],
        'tickets': {},
    }
    for tt in TICKET_TYPES:
        m = y['groups']['middle'][tt]['per_race_max_payout']
        n = y['groups']['no_middle'][tt]['per_race_max_payout']
        z['tickets'][tt] = {
            'middle': {k:m.get(k) for k in ('n','median_yen','mean_yen','max_yen','ge1000_pct','ge5000_pct','ge10000_pct','ge20000_pct')},
            'no_middle': {k:n.get(k) for k in ('n','median_yen','mean_yen','max_yen','ge1000_pct','ge5000_pct','ge10000_pct','ge20000_pct')},
            'comparison': y['middle_vs_no_middle'][tt],
        }
    return z


def main():
    yearly = {str(y): load_year(y) for y in YEARS}
    out = {
        'status': 'MIDDLE_SCENARIO_OTHER_TICKET_PAYOUT_AUDIT_2023_2024',
        'years_read': list(YEARS),
        'evaluation_year_2025_used': False,
        'evaluation_year_2026_used': False,
        'definition': 'middle race = frozen market_scenario_portfolio_v1 decision row with middle_count > 0. Payouts are actual published winning payouts, not full-market odds. For ワイド and dead-heat/multi-row cases, both all paid rows and per-race maximum/median are summarized.',
        'years': yearly,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    compact = {
        'status': out['status'] + '_COMPACT',
        'years_read': list(YEARS),
        'evaluation_year_2025_used': False,
        'evaluation_year_2026_used': False,
        'metric_note': 'Per-race maximum published winning payout. For normal single-row ticket types this equals the winning payout; for ワイド it is the highest of the three paid combinations.',
        'years': {str(y): compact_year(yearly[str(y)]) for y in YEARS},
    }
    COMPACT.write_text(json.dumps(compact, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(compact, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

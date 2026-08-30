from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from .simulate_market_scenario_portfolio_2023_v1 import build_race, middle_scenarios

YEARS = (2023, 2024)
OUT = Path('data/audits/middle_ticket_hit_anatomy_2023_validate_2024.json')


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def parse_tickets(row):
    out = []
    for part in (row.get('tickets') or '').split(' | '):
        m = re.match(r'^(main|middle|hole):([0-9-]+)@([0-9.]+)x([0-9]+)u$', part.strip())
        if not m:
            continue
        kind, cars_s, odds_s, units_s = m.groups()
        cars = tuple(sorted(int(x) for x in cars_s.split('-')))
        out.append({'kind': kind, 'cars': cars, 'odds': float(odds_s), 'units': int(units_s)})
    return out


def entropy(ps):
    s = sum(x for x in ps if x > 0)
    if s <= 0:
        return 0.0
    qs = [x / s for x in ps if x > 0]
    return -sum(q * math.log(q) for q in qs)


def rank_desc(market, field):
    ordered = sorted(
        market,
        key=lambda c: (
            -(market[c][field] if market[c][field] is not None else -1e99),
            c,
        ),
    )
    return {c: i + 1 for i, c in enumerate(ordered)}


def rank_odds(market):
    ordered = sorted(market, key=lambda c: (market[c]['odds'], c))
    return {c: i + 1 for i, c in enumerate(ordered)}, ordered


def load_year(year):
    data = Path(f'data/{year}/s_class_yosen')
    decisions = read_csv(f'data/audits/market_scenario_portfolio_{year}_v1_decisions.csv')
    tri = defaultdict(list)
    trio = defaultdict(list)
    for r in read_csv(data / 'trifecta_final_odds.csv'):
        tri[r['race_id']].append(r)
    for r in read_csv(data / 'trio_final_odds.csv'):
        trio[r['race_id']].append(r)

    winners = {}
    payouts = {}
    for p in read_csv(data / 'payouts.csv'):
        if p.get('ticket_type') != '3連複' or p.get('status') != 'paid':
            continue
        nums = tuple(sorted(int(x) for x in re.findall(r'\d+', p.get('combination', ''))))
        if len(nums) != 3 or len(set(nums)) != 3:
            continue
        try:
            py = int(float(p.get('payout_yen') or 0))
        except Exception:
            continue
        if py <= 0:
            continue
        winners.setdefault(p['race_id'], set()).add(nums)
        payouts[(p['race_id'], nums)] = py

    rows = []
    race_meta = {}
    for d in decisions:
        ts = parse_tickets(d)
        mids = [t for t in ts if t['kind'] == 'middle']
        mains = [t for t in ts if t['kind'] == 'main']
        if not mids or not mains:
            continue
        rid = d['race_id']
        market = build_race(tri.get(rid, []), trio.get(rid, []))
        if not market or rid not in winners:
            continue
        main = mains[0]['cars']
        if main not in market or any(t['cars'] not in market for t in mids):
            continue

        pop_rank, pop_order = rank_odds(market)
        delta_rank = rank_desc(market, 'delta')
        ratio_rank = rank_desc(market, 'ratio')
        cons_rank = rank_desc(market, 'consensus')
        tri_set_rank = rank_desc(market, 'p_tri_set')
        trio_rank = rank_desc(market, 'p_trio')
        favorite = pop_order[0]
        second = pop_order[1]
        fav = market[favorite]
        mainv = market[main]
        scenarios = {s['cars']: s for s in middle_scenarios(market, main)}
        trio_ent = entropy([market[c]['p_trio'] for c in market])
        tri_ent = entropy([market[c]['p_tri_set'] for c in market])
        top3_share = sum(market[c]['p_trio'] for c in pop_order[:3])
        top5_share = sum(market[c]['p_trio'] for c in pop_order[:5])
        fav_hit = int(favorite in winners[rid])

        race_meta[rid] = {
            'favorite': favorite,
            'favorite_hit': fav_hit,
            'fav_odds': fav['odds'],
            'fav_p_trio': fav['p_trio'],
            'fav_consensus': fav['consensus'],
            'second_to_fav_odds_ratio': market[second]['odds'] / fav['odds'],
            'top3_market_share': top3_share,
            'top5_market_share': top5_share,
            'trio_entropy': trio_ent,
            'trifecta_set_entropy': tri_ent,
            'main_is_favorite': int(main == favorite),
        }

        for t in mids:
            c = t['cars']
            x = market[c]
            sc = scenarios.get(c)
            score = float(sc['score']) if sc else x['delta']
            support = int(sc['support_sets']) if sc else 0
            hit = int(c in winners[rid])
            pay = payouts.get((rid, c), 0)
            rows.append({
                'year': year,
                'race_id': rid,
                'cars': c,
                'hit': hit,
                'payout': pay,
                'odds': x['odds'],
                'popularity_rank': pop_rank[c],
                'p_trio': x['p_trio'],
                'p_tri_set': x['p_tri_set'],
                'delta': x['delta'],
                'ratio': x['ratio'] if x['ratio'] is not None else 0.0,
                'consensus': x['consensus'],
                'delta_rank': delta_rank[c],
                'ratio_rank': ratio_rank[c],
                'consensus_rank': cons_rank[c],
                'tri_set_rank': tri_set_rank[c],
                'trio_rank': trio_rank[c],
                'scenario_score': score,
                'support_sets': support,
                'representative_delta_share': x['delta'] / score if score > 0 else 0.0,
                'mid_to_main_consensus': x['consensus'] / mainv['consensus'] if mainv['consensus'] > 0 else 0.0,
                'mid_to_main_trio': x['p_trio'] / mainv['p_trio'] if mainv['p_trio'] > 0 else 0.0,
                'mid_to_main_tri_set': x['p_tri_set'] / mainv['p_tri_set'] if mainv['p_tri_set'] > 0 else 0.0,
                'mid_to_fav_trio': x['p_trio'] / fav['p_trio'] if fav['p_trio'] > 0 else 0.0,
                'overlap_with_favorite': len(set(c) & set(favorite)),
                'fav_odds': fav['odds'],
                'fav_p_trio': fav['p_trio'],
                'fav_consensus': fav['consensus'],
                'second_to_fav_odds_ratio': market[second]['odds'] / fav['odds'],
                'top3_market_share': top3_share,
                'top5_market_share': top5_share,
                'trio_entropy': trio_ent,
                'trifecta_set_entropy': tri_ent,
                'main_is_favorite': int(main == favorite),
                'favorite_hit': fav_hit,
            })
    return rows, race_meta


FEATURES = [
    'odds', 'popularity_rank', 'p_trio', 'p_tri_set', 'delta', 'ratio', 'consensus',
    'delta_rank', 'ratio_rank', 'consensus_rank', 'tri_set_rank', 'trio_rank',
    'scenario_score', 'support_sets', 'representative_delta_share',
    'mid_to_main_consensus', 'mid_to_main_trio', 'mid_to_main_tri_set', 'mid_to_fav_trio',
    'overlap_with_favorite', 'fav_odds', 'fav_p_trio', 'fav_consensus',
    'second_to_fav_odds_ratio', 'top3_market_share', 'top5_market_share',
    'trio_entropy', 'trifecta_set_entropy', 'main_is_favorite',
]


def describe(vals):
    xs = [float(x) for x in vals]
    if not xs:
        return {'n': 0, 'mean': None, 'median': None}
    return {'n': len(xs), 'mean': sum(xs) / len(xs), 'median': median(xs)}


def anatomy(rows):
    hits = [r for r in rows if r['hit']]
    misses = [r for r in rows if not r['hit']]
    out = {}
    for f in FEATURES:
        out[f] = {
            'hit': describe([r[f] for r in hits]),
            'miss': describe([r[f] for r in misses]),
        }
    return out


def flat_stats(rows):
    stake = 100 * len(rows)
    payout = sum(r['payout'] for r in rows)
    hits = sum(r['hit'] for r in rows)
    races = len({r['race_id'] for r in rows})
    hit_races = len({r['race_id'] for r in rows if r['hit']})
    return {
        'races': races,
        'hit_races': hit_races,
        'race_hit_rate_pct': 100 * hit_races / races if races else 0.0,
        'tickets': len(rows),
        'hits': hits,
        'ticket_hit_rate_pct': 100 * hits / len(rows) if rows else 0.0,
        'avg_odds': sum(r['odds'] for r in rows) / len(rows) if rows else None,
        'median_odds': median([r['odds'] for r in rows]) if rows else None,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else 0.0,
    }


def q(vals, p):
    xs = sorted(float(x) for x in vals)
    if not xs:
        return None
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def match(r, c):
    v = float(r[c['feature']])
    t = float(c['threshold'])
    return v <= t if c['op'] == '<=' else v >= t


def candidate_thresholds(rows23, rows24):
    continuous = [
        'odds', 'popularity_rank', 'p_trio', 'p_tri_set', 'delta', 'ratio', 'consensus',
        'delta_rank', 'ratio_rank', 'consensus_rank', 'tri_set_rank', 'trio_rank',
        'scenario_score', 'representative_delta_share', 'mid_to_main_consensus',
        'mid_to_main_trio', 'mid_to_main_tri_set', 'mid_to_fav_trio', 'fav_odds',
        'fav_p_trio', 'fav_consensus', 'second_to_fav_odds_ratio', 'top3_market_share',
        'top5_market_share', 'trio_entropy', 'trifecta_set_entropy',
    ]
    cs = []
    for f in continuous:
        vals = [r[f] for r in rows23]
        for t in sorted(set(q(vals, p) for p in (0.25, 0.50, 0.75))):
            for op in ('<=', '>='):
                c = {'feature': f, 'op': op, 'threshold': t}
                a = [r for r in rows23 if match(r, c)]
                b = [r for r in rows24 if match(r, c)]
                sa = flat_stats(a)
                sb = flat_stats(b)
                if sa['tickets'] < 300 or sa['hits'] < 10:
                    continue
                cs.append({**c, 'train_2023': sa, 'validation_2024': sb})
    for f, ops in [
        ('support_sets', [('>=', 3), ('<=', 2)]),
        ('overlap_with_favorite', [('>=', 2), ('<=', 1)]),
        ('main_is_favorite', [('>=', 1), ('<=', 0)]),
    ]:
        for op, t in ops:
            c = {'feature': f, 'op': op, 'threshold': t}
            a = [r for r in rows23 if match(r, c)]
            b = [r for r in rows24 if match(r, c)]
            sa = flat_stats(a)
            sb = flat_stats(b)
            if sa['tickets'] >= 300 and sa['hits'] >= 10:
                cs.append({**c, 'train_2023': sa, 'validation_2024': sb})
    cs.sort(key=lambda x: (-x['train_2023']['roi_pct'], -x['train_2023']['tickets']))
    return cs


def stable_candidates(candidates, base23, base24):
    out = []
    for c in candidates:
        s23 = c['train_2023']
        s24 = c['validation_2024']
        if s24['tickets'] < 250 or s24['hits'] < 8:
            continue
        if s23['roi_pct'] > base23['roi_pct'] and s24['roi_pct'] > base24['roi_pct']:
            out.append(c)
    out.sort(key=lambda x: (-(x['train_2023']['roi_pct'] - base23['roi_pct'] + x['validation_2024']['roi_pct'] - base24['roi_pct']), -x['train_2023']['tickets']))
    return out


def main():
    rows23, meta23 = load_year(2023)
    rows24, meta24 = load_year(2024)
    base23 = flat_stats(rows23)
    base24 = flat_stats(rows24)
    cands = candidate_thresholds(rows23, rows24)
    stable = stable_candidates(cands, base23, base24)

    miss23 = [r for r in rows23 if not r['favorite_hit']]
    miss24 = [r for r in rows24 if not r['favorite_hit']]

    out = {
        'status': 'MIDDLE_TICKET_HIT_ANATOMY_2023_VALIDATE_2024',
        'years_read': [2023, 2024],
        'evaluation_year_2025_used': False,
        'evaluation_year_2026_used': False,
        'purpose': 'Use results only as labels to study which pre-result market features distinguish winning selected middle tickets from losing selected middle tickets. 2023 discovers simple ticket-level thresholds; exact thresholds are frozen into 2024.',
        'important_limit': 'Descriptive hit-vs-miss anatomy is not itself a betting rule. Candidate thresholds are exploratory 2023 training rules and must survive unchanged in 2024. Actual favorite-miss subsets are anatomy only because favorite miss is known only after the race.',
        'baseline': {'2023': base23, '2024': base24},
        'hit_vs_miss_anatomy': {'2023': anatomy(rows23), '2024': anatomy(rows24)},
        'actual_favorite_miss_anatomy_only': {
            '2023': {'stats': flat_stats(miss23), 'hit_vs_miss': anatomy(miss23)},
            '2024': {'stats': flat_stats(miss24), 'hit_vs_miss': anatomy(miss24)},
        },
        'candidate_count': len(cands),
        'top20_by_2023_ticket_roi_with_frozen_2024': cands[:20],
        'stable_improvement_both_years': stable[:20],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'baseline': out['baseline'],
        'actual_favorite_miss': {y: out['actual_favorite_miss_anatomy_only'][y]['stats'] for y in ('2023', '2024')},
        'stable_improvement_both_years': stable[:10],
        'top10': cands[:10],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

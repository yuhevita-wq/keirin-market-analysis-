from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .research_high_payout_structures_v1 import load_year, parse_combination
from .simulate_early_candidate_v0 import line_map, pair_strength
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival

YEARS = (2023, 2024, 2025)
SEGMENTS = ('前半', '中盤', '後半')
THRESHOLDS = (5000, 10000, 20000)
OUT = Path('data/audits/weakest_line_structure_2023_2025.json')


def weakest_line(entries, main_id):
    by = line_map(entries)
    cands = [(lid, mem, pair_strength(mem)) for lid, mem in by.items() if lid != main_id and len(mem) >= 2]
    if len(cands) < 2:  # need main + at least two non-main lines
        return None, 'fewer_than_3_lines'
    min_score = min(x[2] for x in cands)
    lows = [x for x in cands if x[2] == min_score]
    if len(lows) != 1:
        return None, 'weakest_pair_score_tie'
    lid, mem, score = lows[0]
    return (lid, mem, score), None


def pct(n, d):
    return n / d if d else 0.0


def payout_summary(rows):
    pays = sorted(r['payout_yen'] for r in rows)
    if not pays:
        return {'races': 0}
    return {
        'races': len(rows),
        'median_payout_yen': pays[len(pays)//2],
        'mean_payout_yen': sum(pays)/len(pays),
        'max_payout_yen': max(pays),
        **{f'ge_{t}_count': sum(p >= t for p in pays) for t in THRESHOLDS},
        **{f'ge_{t}_rate': pct(sum(p >= t for p in pays), len(pays)) for t in THRESHOLDS},
    }


def summarize(rows):
    n = len(rows)
    buckets = {}
    for k in (0, 1, 2):
        x = [r for r in rows if r['weak_count_top3'] == k]
        buckets[str(k)] = payout_summary(x)
        buckets[str(k)]['share'] = pct(len(x), n)

    weak_inc = [r for r in rows if r['weak_count_top3'] >= 1]
    weak_both = [r for r in rows if r['weak_count_top3'] == 2]
    weak_head = [r for r in rows if r['weak_head']]
    weak_pair_top2 = [r for r in rows if r['weak_pair_top2']]

    def enrich(x, t):
        base = pct(sum(r['payout_yen'] >= t for r in rows), n)
        xr = pct(sum(r['payout_yen'] >= t for r in x), len(x))
        return xr / base if base else 0.0

    mixes = Counter()
    orders = Counter()
    high_orders = {str(t): Counter() for t in THRESHOLDS}
    for r in weak_inc:
        mixes[r['weak_mix']] += 1
        orders[r['role_order']] += 1
        for t in THRESHOLDS:
            if r['payout_yen'] >= t:
                high_orders[str(t)][r['role_order']] += 1

    return {
        'eligible_races': n,
        'overall': payout_summary(rows),
        'weakest_top3_count': buckets,
        'weakest_included': {
            **payout_summary(weak_inc),
            'share': pct(len(weak_inc), n),
            **{f'ge_{t}_enrichment_vs_all': enrich(weak_inc, t) for t in THRESHOLDS},
        },
        'both_weakest_in_top3': {
            **payout_summary(weak_both),
            'share': pct(len(weak_both), n),
            **{f'ge_{t}_enrichment_vs_all': enrich(weak_both, t) for t in THRESHOLDS},
        },
        'weakest_head': {
            **payout_summary(weak_head),
            'share': pct(len(weak_head), n),
            **{f'ge_{t}_enrichment_vs_all': enrich(weak_head, t) for t in THRESHOLDS},
        },
        'weakest_pair_top2': {
            **payout_summary(weak_pair_top2),
            'share': pct(len(weak_pair_top2), n),
            **{f'ge_{t}_enrichment_vs_all': enrich(weak_pair_top2, t) for t in THRESHOLDS},
        },
        'weak_mix_counts': dict(mixes),
        'top_weak_role_orders': orders.most_common(25),
        'top_high_payout_weak_orders': {k: v.most_common(20) for k, v in high_orders.items()},
    }


def main():
    all_rows = []
    exclusions = defaultdict(Counter)
    structural = {}

    for year in YEARS:
        races, eb, win3, seg = load_year(year)
        year_rows = []
        for race in races:
            rid = race['race_id']
            s = seg.get(rid)
            if s not in SEGMENTS or rid not in win3:
                continue
            chosen = choose_main_line(eb[rid])
            if not chosen:
                exclusions[str(year)]['no_main'] += 1
                continue
            main_id, main = chosen
            if len(main) < 3:
                exclusions[str(year)]['main_under_3'] += 1
                continue
            rival = strongest_rival(eb[rid], main_id)
            if not rival or len(rival) < 2:
                exclusions[str(year)]['no_strong_rival'] += 1
                continue
            rival_id = int(rival[0]['line_id'])

            weak, reason = weakest_line(eb[rid], main_id)
            if not weak:
                exclusions[str(year)][reason] += 1
                continue
            weak_id, weak_mem, weak_pair_score = weak
            if weak_id == rival_id:
                # This can only occur if all non-main candidate lines collapse to the same strength ordering.
                exclusions[str(year)]['weak_equals_strong_rival'] += 1
                continue

            roles = {}
            for name, e in [('A', main[0]), ('B', main[1]), ('M3', main[2]), ('R1L', rival[0]), ('R1B', rival[1]), ('W1L', weak_mem[0]), ('W1B', weak_mem[1])]:
                roles[int(e['car_no'])] = name

            combo, payout = win3[rid]
            cars = parse_combination(combo)
            role_order = tuple(roles.get(c, 'OTHER') for c in cars)
            weak_count = sum(x in {'W1L', 'W1B'} for x in role_order)
            main_count = sum(x in {'A', 'B', 'M3'} for x in role_order)
            rival_count = sum(x in {'R1L', 'R1B'} for x in role_order)
            other_count = 3 - weak_count - main_count - rival_count
            weak_head = role_order[0] in {'W1L', 'W1B'}
            weak_pair_top2 = set(role_order[:2]) == {'W1L', 'W1B'}
            if weak_count:
                weak_mix = f'W{weak_count}_M{main_count}_R{rival_count}_O{other_count}'
            else:
                weak_mix = 'NO_WEAK'

            row = {
                'year': year,
                'segment': s,
                'race_id': rid,
                'race_date': race['race_date'],
                'track': race['track'],
                'race_no': int(race['race_no']),
                'combination': combo,
                'payout_yen': payout,
                'role_order': '-'.join(role_order),
                'weak_count_top3': weak_count,
                'weak_head': weak_head,
                'weak_pair_top2': weak_pair_top2,
                'weak_mix': weak_mix,
                'weak_pair_score': weak_pair_score,
                'strong_rival_pair_score': pair_strength(rival),
                'main_pair_score': pair_strength(main),
            }
            year_rows.append(row)
            all_rows.append(row)

        structural[str(year)] = {
            s: sum(r['segment'] == s for r in year_rows) for s in SEGMENTS
        } | {'total': len(year_rows)}

    out = {
        'status': 'WEAKEST_LINE_STRUCTURE_RESEARCH_2023_2025_ONLY',
        'years_read': list(YEARS),
        'evaluation_year_2026_used': False,
        'definition': {
            'candidate_line': 'predicted line with at least 2 riders',
            'weakest_line': 'unique minimum of lead+second competition-score sum among non-main candidate lines',
            'requires': 'main line size>=3 and at least two non-main 2+ lines, so strongest rival and weakest line are distinct',
            'tie_policy': 'exclude races tied for minimum non-main pair score; never break the tie post hoc',
            'roles': {'W1L': 'weakest line lead', 'W1B': 'weakest line second'},
        },
        'thresholds_yen_per_100': list(THRESHOLDS),
        'structural_counts': structural,
        'exclusions': {y: dict(c) for y, c in exclusions.items()},
        'segments': {},
        'combined': summarize(all_rows),
    }

    for s in SEGMENTS:
        sr = [r for r in all_rows if r['segment'] == s]
        out['segments'][s] = {
            'combined_2023_2025': summarize(sr),
            'by_year': {str(y): summarize([r for r in sr if r['year'] == y]) for y in YEARS},
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': out['status'],
        'structural_counts': structural,
        'exclusions': out['exclusions'],
        'segments': {s: out['segments'][s]['combined_2023_2025'] for s in SEGMENTS},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

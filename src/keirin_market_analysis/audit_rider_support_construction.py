from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'rider_support_construction_audit.json'
DATASETS = {
    '2023': ROOT / 'data' / '2023' / 's_class_yosen' / 'trio_final_odds.csv',
    '2024': ROOT / 'data' / '2024' / 's_class_yosen' / 'trio_final_odds.csv',
    '2025': ROOT / 'data' / '2025' / 's_class_yosen' / 'trio_final_odds.csv',
    '2026_h1': ROOT / 'data' / '2026_h1' / 's_class_yosen' / 'trio_final_odds.csv',
}


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def combo(s: str):
    s = str(s).strip().replace('=', '-').replace(',', '-')
    try:
        c = tuple(sorted(int(x) for x in s.split('-') if x.strip()))
    except Exception:
        return ()
    return c if len(c) == 3 and len(set(c)) == 3 else ()


def close(a: float, b: float, tol: float = 1e-12):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def audit_dataset(label: str, path: Path):
    rows = read_csv(path)
    by_race = defaultdict(list)
    for r in rows:
        if r.get('odds_status') not in ('', 'available', None):
            continue
        try:
            o = float(r.get('odds') or r.get('final_odds') or r.get('trio_odds'))
        except Exception:
            continue
        c = combo(r.get('combination') or r.get('bet_code') or '')
        if len(c) == 3 and o > 0:
            by_race[str(r.get('race_id'))].append((c, o))

    counts = defaultdict(int)
    rank_mismatches = []
    invariant_failures = []
    incomplete_markets = []
    duplicate_combos = []
    sample = None

    for rid in sorted(by_race):
        pairs = by_race[rid]
        odds_by_combo = {}
        dup = False
        for c, o in pairs:
            if c in odds_by_combo:
                dup = True
            odds_by_combo[c] = o
        if dup:
            duplicate_combos.append(rid)

        riders = sorted({x for c in odds_by_combo for x in c})
        n = len(riders)
        expected = math.comb(n, 3) if n >= 3 else 0
        actual = len(odds_by_combo)
        if actual != expected:
            incomplete_markets.append({'race_id': rid, 'riders_observed': riders, 'actual_available_combos': actual, 'expected_from_observed_riders': expected})
            continue

        expected_set = set(itertools.combinations(riders, 3))
        if set(odds_by_combo) != expected_set:
            incomplete_markets.append({'race_id': rid, 'riders_observed': riders, 'actual_available_combos': actual, 'expected_from_observed_riders': expected, 'set_mismatch': True})
            continue

        raw_weight = {c: 1.0/o for c, o in odds_by_combo.items()}
        z = sum(raw_weight.values())
        prob = {c: w/z for c, w in raw_weight.items()}

        support_prob = {r: sum(p for c, p in prob.items() if r in c) for r in riders}
        support_raw = {r: sum(w for c, w in raw_weight.items() if r in c) for r in riders}
        share = {r: support_prob[r]/3.0 for r in riders}

        ok_prob = close(sum(prob.values()), 1.0)
        ok_support = close(sum(support_prob.values()), 3.0)
        ok_share = close(sum(share.values()), 1.0)
        ok_identity = all(close(support_prob[r], support_raw[r]/z) for r in riders)
        if not (ok_prob and ok_support and ok_share and ok_identity):
            invariant_failures.append({
                'race_id': rid,
                'sum_trio_prob': sum(prob.values()),
                'sum_rider_support': sum(support_prob.values()),
                'sum_rider_share': sum(share.values()),
                'raw_normalized_identity': ok_identity,
            })
            continue

        rank_prob = sorted(riders, key=lambda r: (-support_prob[r], r))
        rank_raw = sorted(riders, key=lambda r: (-support_raw[r], r))
        rank_share = sorted(riders, key=lambda r: (-share[r], r))
        if not (rank_prob == rank_raw == rank_share):
            rank_mismatches.append({'race_id': rid, 'prob': rank_prob, 'raw': rank_raw, 'share': rank_share})
            continue

        counts[f'{n}_rider_complete_markets'] += 1
        counts['fully_verified_races'] += 1
        if sample is None:
            sample = {
                'race_id': rid,
                'riders': riders,
                'available_combo_count': actual,
                'normalization_z_sum_inverse_odds': z,
                'sum_trio_prob': sum(prob.values()),
                'sum_rider_support': sum(support_prob.values()),
                'sum_rider_share': sum(share.values()),
                'support_rank': rank_prob,
                'rider_support': {str(r): support_prob[r] for r in rank_prob},
                'rider_share_pct': {str(r): 100.0*share[r] for r in rank_prob},
                'raw_inverse_odds_support': {str(r): support_raw[r] for r in rank_prob},
                'note': 'Ranking is mathematically identical whether using raw sum(1/odds), normalized trio probability mass, or rider share divided by 3.'
            }

    return {
        'dataset': label,
        'source_file': str(path.relative_to(ROOT)),
        'race_count_with_any_available_trio_odds': len(by_race),
        'fully_verified_races': counts['fully_verified_races'],
        'complete_market_counts_by_observed_riders': {k: v for k, v in sorted(counts.items()) if k.endswith('_rider_complete_markets')},
        'incomplete_market_count': len(incomplete_markets),
        'duplicate_combo_race_count': len(duplicate_combos),
        'invariant_failure_count': len(invariant_failures),
        'rank_mismatch_count': len(rank_mismatches),
        'incomplete_market_examples': incomplete_markets[:20],
        'duplicate_combo_examples': duplicate_combos[:20],
        'invariant_failure_examples': invariant_failures[:20],
        'rank_mismatch_examples': rank_mismatches[:20],
        'sample_verified_race': sample,
    }


def main():
    datasets = {}
    for label, path in DATASETS.items():
        if path.exists():
            datasets[label] = audit_dataset(label, path)

    total_verified = sum(v['fully_verified_races'] for v in datasets.values())
    total_incomplete = sum(v['incomplete_market_count'] for v in datasets.values())
    total_invariant_fail = sum(v['invariant_failure_count'] for v in datasets.values())
    total_rank_mismatch = sum(v['rank_mismatch_count'] for v in datasets.values())

    out = {
        'status': 'RIDER_SUPPORT_CONSTRUCTION_AUDIT',
        'definition': {
            'trio_weight': 'w(combo) = 1 / trio_odds(combo)',
            'normalized_trio_mass': 'p(combo) = w(combo) / sum_all_valid_combos w',
            'rider_support': 'S(rider) = sum p(combo) for every trio containing rider',
            'rider_share': 'share(rider) = S(rider) / 3; shares sum to 1 because each trio contains exactly 3 riders',
            'ranking_equivalence': 'Ranking by S(rider), share(rider), or raw sum 1/odds over containing trios is exactly equivalent because all transformations use positive race-level constants.'
        },
        'important_semantics': 'This is a derived market-support share from 3連複 odds, not a literal probability that the rider finishes top-3 and not a published market field.',
        'datasets': datasets,
        'totals': {
            'fully_verified_races': total_verified,
            'incomplete_market_races_excluded_from_invariant_check': total_incomplete,
            'invariant_failures': total_invariant_fail,
            'rank_mismatches': total_rank_mismatch,
        },
        'verdict': 'PASS' if total_verified > 0 and total_invariant_fail == 0 and total_rank_mismatch == 0 else 'CHECK',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

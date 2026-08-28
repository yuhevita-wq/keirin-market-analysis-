from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_early_candidate_v0 import formation as early_formation

ROOT = Path('.')
DATA25 = ROOT / 'data/2025/s_class_yosen'
DATA24 = ROOT / 'data/2024/s_class_yosen'
OLD_EARLY = DATA25 / 'simulations/early_candidate_v0/races.csv'
OLD_MIDDLE = DATA25 / 'simulations/middle_candidate_v0/races.csv'
OLD_LATE_RACES = DATA25 / 'simulations/branching_v3_confidence/race_simulation.csv'
OLD_LATE_BETS = DATA25 / 'simulations/branching_v3_confidence/bets.csv'
FROZEN25 = Path('/tmp/frozen-2025-audit')
FROZEN24 = DATA24 / 'oos_frozen_2025_strategies'
REPORT = ROOT / 'data/audits/frozen_strategy_equivalence.json'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def as_int(v: str) -> int:
    return int(float(v))


def as_float(v: str) -> float:
    return float(v) if str(v).strip() else 0.0


def same_float(a: str, b: str, tol: float = 1e-9) -> bool:
    return abs(as_float(a) - as_float(b)) <= tol


def audit_2025_equivalence() -> dict[str, object]:
    frozen_decisions = read_csv(FROZEN25 / 'race_decisions.csv')
    frozen_bets = read_csv(FROZEN25 / 'bets.csv')

    f_by_strategy = defaultdict(list)
    for r in frozen_decisions:
        f_by_strategy[r['strategy']].append(r)

    # Early V0: compare every purchased race, role assignment and accounting.
    old_early = read_csv(OLD_EARLY)
    new_early = [r for r in f_by_strategy['early_v0'] if as_int(r['purchased']) == 1]
    oe = {r['race_id']: r for r in old_early}
    ne = {r['race_id']: r for r in new_early}
    assert set(oe) == set(ne), {'old_only': sorted(set(oe)-set(ne)), 'new_only': sorted(set(ne)-set(oe))}
    early_bet_mismatches = []
    for rid in sorted(oe):
        o, n = oe[rid], ne[rid]
        for key in ('a_car','b_car','r1l_car','r1b_car'):
            assert as_int(o[key]) == as_int(n[key]), (rid, key, o[key], n[key])
        expected = early_formation(as_int(n['a_car']), as_int(n['b_car']), as_int(n['r1l_car']), as_int(n['r1b_car']))
        actual = sorted(b['combination'] for b in frozen_bets if b['race_id'] == rid and b['strategy'] == 'early_v0')
        if sorted(expected) != actual:
            early_bet_mismatches.append({'race_id': rid, 'expected': sorted(expected), 'actual': actual})
        for key in ('stake_yen','payout_yen','hit'):
            assert as_int(o[key]) == as_int(n[key]), (rid, key, o[key], n[key])
    assert not early_bet_mismatches, early_bet_mismatches

    # Middle V0: compare purchased race ids, exact two combinations and accounting.
    old_middle = read_csv(OLD_MIDDLE)
    new_middle = [r for r in f_by_strategy['middle_v0'] if as_int(r['purchased']) == 1]
    om = {r['race_id']: r for r in old_middle}
    nm = {r['race_id']: r for r in new_middle}
    assert set(om) == set(nm), {'old_only': sorted(set(om)-set(nm)), 'new_only': sorted(set(nm)-set(om))}
    for rid in sorted(om):
        o, n = om[rid], nm[rid]
        actual = sorted(b['combination'] for b in frozen_bets if b['race_id'] == rid and b['strategy'] == 'middle_v0')
        expected = sorted([o['bet_1'], o['bet_2']])
        assert expected == actual, (rid, expected, actual)
        assert same_float(o['b_score'], n['b_score']), (rid, o['b_score'], n['b_score'])
        for key in ('stake_yen','payout_yen','hit'):
            assert as_int(o[key]) == as_int(n[key]), (rid, key, o[key], n[key])

    # Late V1: compare every eligible race decision and every bet row.
    old_late = read_csv(OLD_LATE_RACES)
    new_late = f_by_strategy['late_v1']
    ol = {r['race_id']: r for r in old_late}
    nl = {r['race_id']: r for r in new_late}
    assert set(ol) == set(nl), {'old_only': sorted(set(ol)-set(nl)), 'new_only': sorted(set(nl)-set(ol))}
    for rid in sorted(ol):
        o, n = ol[rid], nl[rid]
        assert o['strategy'] == n['decision'], (rid, o['strategy'], n['decision'])
        assert same_float(o['pair_top3_sum'], n['pair_top3_sum']), (rid, 'pair_top3')
        assert same_float(o['pair_win_sum'], n['pair_win_sum']), (rid, 'pair_win')
        assert same_float(o['main_second_top3'], n['b_top3']), (rid, 'b_top3')
        assert same_float(o['score_gap_A_vs_R1L'], n['score_gap_A_vs_R1L']), (rid, 'score_gap')
        for key in ('bet_count','stake_yen','payout_yen','hit'):
            assert as_int(o[key]) == as_int(n[key]), (rid, key, o[key], n[key])

    old_lb = sorted((r['race_id'], r['combination'], as_int(r['stake_yen']), as_int(r['payout_yen']), as_int(r['hit'])) for r in read_csv(OLD_LATE_BETS))
    new_lb = sorted((r['race_id'], r['combination'], as_int(r['stake_yen']), as_int(r['payout_yen']), as_int(r['hit'])) for r in frozen_bets if r['strategy'] == 'late_v1')
    assert old_lb == new_lb, 'Late bet rows differ from original 2025 implementation'

    return {
        'early_purchased_races_exact_match': len(oe),
        'middle_purchased_races_exact_match': len(om),
        'late_eligible_races_exact_match': len(ol),
        'late_bet_rows_exact_match': len(old_lb),
        'status': 'PASS',
    }


def audit_2024_integrity() -> dict[str, object]:
    races = read_csv(DATA24 / 'races.csv')
    entries = read_csv(DATA24 / 'entries.csv')
    payouts = read_csv(DATA24 / 'payouts.csv')
    decisions = read_csv(FROZEN24 / 'race_decisions.csv')
    bets = read_csv(FROZEN24 / 'bets.csv')
    summary = json.loads((FROZEN24 / 'summary.json').read_text(encoding='utf-8'))

    race_ids = {r['race_id'] for r in races}
    assert len(races) == 1461 and len(race_ids) == 1461
    assert all(r['race_id'] in race_ids for r in decisions)
    assert len({(r['race_id'], r['strategy']) for r in decisions}) == len(decisions)
    assert all(r['race_date'].startswith('2024-') for r in decisions)

    # Entries must cover every race and all published line fields required for decisions.
    entries_by = defaultdict(list)
    for e in entries:
        entries_by[e['race_id']].append(e)
    assert set(entries_by) == race_ids
    assert all(e.get('line_id','').isdigit() and e.get('line_position','').isdigit() for e in entries)

    # Build exact published 3連単 payout lookup and verify every bet/accounting row.
    tri = defaultdict(dict)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            tri[p['race_id']][p['combination']] = as_int(p['payout_yen'])

    assert len({(b['race_id'], b['strategy'], b['combination']) for b in bets}) == len(bets), 'duplicate bet rows'
    for b in bets:
        assert as_int(b['stake_yen']) == 100
        expected = tri[b['race_id']].get(b['combination'], 0)
        assert as_int(b['payout_yen']) == expected, (b['race_id'], b['combination'], b['payout_yen'], expected)
        assert as_int(b['hit']) == (1 if expected else 0)

    bets_by = defaultdict(list)
    for b in bets:
        bets_by[(b['race_id'], b['strategy'])].append(b)

    for d in decisions:
        key = (d['race_id'], d['strategy'])
        rows = bets_by.get(key, [])
        assert len(rows) == as_int(d['bet_count'])
        assert sum(as_int(x['stake_yen']) for x in rows) == as_int(d['stake_yen'])
        assert sum(as_int(x['payout_yen']) for x in rows) == as_int(d['payout_yen'])
        assert as_int(d['hit']) == (1 if as_int(d['payout_yen']) > 0 else 0)

        if d['strategy'] == 'early_v0':
            if d['decision'] == 'buy':
                assert as_float(d['pair_win_sum']) <= 22.35 and as_int(d['bet_count']) == 4
            else:
                assert d['decision'] == 'skip_threshold' and as_float(d['pair_win_sum']) > 22.35 and as_int(d['bet_count']) == 0
        elif d['strategy'] == 'middle_v0':
            if d['decision'] == 'buy':
                assert as_float(d['b_score']) > 105.0 and as_int(d['bet_count']) == 2
            else:
                assert d['decision'] == 'skip_threshold' and as_float(d['b_score']) <= 105.0 and as_int(d['bet_count']) == 0
        elif d['strategy'] == 'late_v1':
            p3, pw, bt = as_float(d['pair_top3_sum']), as_float(d['pair_win_sum']), as_float(d['b_top3'])
            if d['decision'] == 'mainline_4pt':
                assert p3 > 106.1 and pw > 54.7 and as_int(d['bet_count']) == 4
            elif d['decision'] == 'rough_18pt':
                assert p3 <= 106.1 and bt <= 35.7 and as_float(d['score_gap_A_vs_R1L']) < 10.0 and as_int(d['bet_count']) == 18
            elif d['decision'] == 'skip_rough_score_gap_ge_10':
                assert p3 <= 106.1 and bt <= 35.7 and as_float(d['score_gap_A_vs_R1L']) >= 10.0 and as_int(d['bet_count']) == 0
            elif d['decision'] == 'skip_middle':
                assert not (p3 > 106.1 and pw > 54.7)
                assert not (p3 <= 106.1 and bt <= 35.7)
                assert as_int(d['bet_count']) == 0
            else:
                raise AssertionError(('unexpected late decision', d['decision'], d['race_id']))

    bought = [d for d in decisions if as_int(d['purchased']) == 1]
    assert len(bought) == summary['combined']['purchased_races']
    assert sum(as_int(d['stake_yen']) for d in bought) == summary['combined']['stake_yen']
    assert sum(as_int(d['payout_yen']) for d in bought) == summary['combined']['payout_yen']
    assert sum(as_int(d['hit']) for d in bought) == summary['combined']['hits']

    return {
        'dataset_races_verified': len(races),
        'decision_rows_verified': len(decisions),
        'bet_rows_verified': len(bets),
        'published_trifecta_payouts_reconciled': True,
        'threshold_and_branch_invariants_verified': True,
        'summary_accounting_reconciled': True,
        'status': 'PASS',
    }


def main() -> None:
    report = {
        'purpose': 'Audit implementation equivalence and accounting only. No thresholds, filters, branch rules, or bets are changed.',
        'equivalence_2025_old_vs_generic': audit_2025_equivalence(),
        'integrity_2024_oos': audit_2024_integrity(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

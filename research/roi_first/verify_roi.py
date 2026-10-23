#!/usr/bin/env python3
"""Audit a proposed wager policy WITHOUT selecting or tuning its bets.

The caller must freeze a candidate policy first. All wagers represent actual
hypothetical placements made from timestamped data. Never fill official
settlements with final odds * 100. ROI is undefined if total stake is zero.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

REQUIRED = {
    'race_id', 'race_date', 'bet_type', 'combination', 'stake_yen',
    'returned_yen', 'settlement_verified', 'price_phase'
}
D = Decimal


def amount(value: str) -> Decimal:
    try:
        parsed = D(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError('non-numeric amount') from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError('amount must be nonnegative and finite')
    return parsed


def audit(csv_path: Path, policy: dict) -> dict:
    """Outcome-dependent optimization and strategy selection are deliberately absent."""
    if policy.get('goal') != 'maximize_realized_official_roi':
        raise ValueError('ROI_FIRST_GOAL_REQUIRED')
    if not policy.get('strategy_sha256'):
        raise ValueError('FROZEN_STRATEGY_CHECKSUM_REQUIRED')
    test_from = dt.date.fromisoformat(policy['test_start'])
    test_to = dt.date.fromisoformat(policy['test_end'])
    if test_from > test_to:
        raise ValueError('INVALID_TEST_WINDOW')
    if test_from <= dt.date(2025, 4, 19) and policy.get('claimed_new_untouched_test'):
        raise ValueError('2025_04_19_AND_EARLIER_ALREADY_EXAMINED')
    if policy.get('policy_tuned_using_test_results'):
        raise ValueError('HINDSIGHT_OPTIMIZATION_NOT_A_FRESH_TEST')
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED <= set(reader.fieldnames or []):
            raise ValueError('MISSING_LEDGER_COLUMNS: ' + str(sorted(REQUIRED-set(reader.fieldnames or []))))
        rows = list(reader)
    stake = D(0)
    returned = D(0)
    markets = collections.defaultdict(lambda: {'bets': 0, 'stake': D(0), 'returned': D(0), 'races': set()})
    day = collections.defaultdict(lambda: {'stake': D(0), 'returned': D(0)})
    n_races = set()
    phase_counts = collections.Counter()
    for row in rows:
        date = dt.date.fromisoformat(row['race_date'])
        if not test_from <= date <= test_to:
            raise ValueError('WAGER_OUTSIDE_FROZEN_TEST_WINDOW')
        if row['settlement_verified'].strip().lower() not in ('true', '1', 'yes'):
            raise ValueError('UNVERIFIED_OFFICIAL_SETTLEMENT')
        invest, paid = amount(row['stake_yen']), amount(row['returned_yen'])
        if invest == 0:
            raise ValueError('ZERO_STAKE_BET_ROW')
        phase = row['price_phase'].strip()
        if phase not in ('preclose_timestamp_verified', 'retrospective_final'):
            raise ValueError('UNKNOWN_PRICE_PHASE')
        key = row['bet_type'].strip()
        if not key or not row['combination'].strip() or not row['race_id'].strip():
            raise ValueError('INCOMPLETE_BET_KEY')
        stake += invest
        returned += paid
        markets[key]['bets'] += 1
        markets[key]['stake'] += invest
        markets[key]['returned'] += paid
        markets[key]['races'].add(row['race_id'])
        n_races.add(row['race_id'])
        day[str(date)]['stake'] += invest
        day[str(date)]['returned'] += paid
        phase_counts[phase] += 1
    market_result = {}
    for key, obj in sorted(markets.items()):
        market_result[key] = {
            'bets': obj['bets'], 'races': len(obj['races']),
            'stake_yen': str(obj['stake']), 'official_returned_yen': str(obj['returned']),
            'net_yen': str(obj['returned']-obj['stake']),
            'realized_roi': float(obj['returned']/obj['stake'])
        }
    roi = float(returned/stake) if stake else None
    # Reporting no purchase as 100% or 0% ROI is prohibited.
    status = 'NO_BETS_ROI_UNDEFINED' if stake == 0 else 'OBSERVED_POSITIVE_ROI_NOT_PROVEN' if roi > 1 else 'NO_REALIZED_PROFIT'
    if 'retrospective_final' in phase_counts and stake:
        status = 'RETROSPECTIVE_RESEARCH_ONLY_' + status
    return {
        'strategy_sha256': policy['strategy_sha256'],
        'test_period': {'start': str(test_from), 'end': str(test_to)},
        'status': status, 'total_bets': len(rows), 'races_bet': len(n_races),
        'days_with_bets': len(day),
        'stake_yen': str(stake), 'official_returned_yen': str(returned),
        'net_yen': str(returned-stake), 'realized_roi': roi,
        'price_phase_count': dict(phase_counts),
        'market_breakdown': market_result,
        'warning': ('Neither a positive ROI nor any other in-sample success '
                    'proves future profitability. A never-examined test period '
                    'and executable preclose prices must be verified.')
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', required=True, type=Path)
    parser.add_argument('--policy', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.ledger, json.loads(args.policy.read_text(encoding='utf-8')))
    dump = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(dump, encoding='utf-8')
    print(dump)


if __name__ == '__main__':
    main()

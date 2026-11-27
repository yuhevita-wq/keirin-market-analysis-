"""Train a price-aware classifier and tune a purchase policy on earlier races."""
from collections import defaultdict
from datetime import datetime, timedelta
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from train import load, outcomes_for_model, published_inputs, winning_keys

POOLS = {'umaren': 'quinella', 'umatan': 'exacta', 'trio': 'trio'}
BASE_CAPS = {'quinella': 2, 'exacta': 2, 'trio': 4}
THRESHOLDS = (1.00, 1.05, 1.10, 1.20, 1.35, 1.55, 1.80, 2.10)


def archives(root):
    for month in ('07', '08', '09', '10'):
        folder = root / month
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run(['gh', 'release', 'download', f'path-odds-2026-{month}',
                        '--pattern', '2026-*.json.gz', '--dir', str(folder), '--skip-existing'],
                       check=True, capture_output=True)
        for file in sorted(folder.glob('*.json.gz')):
            with gzip.open(file, 'rt') as stream:
                yield from json.load(stream).values()


def canonical(pool, key):
    values = list(map(int, key.replace('=', '-').split('-')))
    if pool != 'umatan':
        values.sort()
    return '-'.join(map(str, values))


def observed_quotes(record, pool, deadline):
    target = deadline - timedelta(minutes=5)
    frames = defaultdict(list)
    for frame in record.get('pools', {}).get(pool, {}).get('frames', []):
        if frame['final'] or not re.fullmatch(r'\d{2}:\d{2}', frame['label']):
            continue
        h, m = map(int, frame['label'].split(':'))
        if h > 23 or m > 59:
            continue
        time = deadline.replace(hour=h, minute=m, second=0, microsecond=0)
        if 0 <= (target - time).total_seconds() <= 120:
            frames[time].append(frame['odds'])
    if not frames or len(frames[max(frames)]) != 1:
        return {}
    return {canonical(pool, key): odds for key, odds in frames[max(frames)][0].items()
            if isinstance(odds, (int, float)) and math.isfinite(odds) and odds >= 1}


def probability(market, order, distribution):
    if market == 'exacta':
        return distribution[order[0], order[1], :].sum()
    if market == 'quinella':
        a, b = order
        return distribution[a, b, :].sum() + distribution[b, a, :].sum()
    a, b, c = order
    return sum(distribution[i, j, a + b + c - i - j]
               for i in order for j in order if i != j)


def features(p, odds):
    p = min(max(float(p), 1e-8), 1 - 1e-8)
    price = max(1.000001, float(odds))
    return [1., math.log(p / (1 - p)), -math.log(price - 1), math.log(price) ** 2]


def matched_rows(archived, races, beta):
    seen = set()
    for record in archived:
        official = record.get('official', {})
        source = official.get('source')
        link = record.get('official_url') or (source.get('url', '') if isinstance(source, dict) else '')
        match = re.search(r'/racedetail/(\d{16})/', link)
        if not match or match[1] in seen or match[1] not in races or not official.get('deadline'):
            continue
        race_id = match[1]
        _, day, cars, individual, groups, outcome, _, _ = races[race_id]
        if day != record['race_date'] or cars != record['cars'] or int(race_id[-2:]) != int(record['race_no']):
            continue
        seen.add(race_id)
        distribution = outcomes_for_model(individual, groups, beta)
        winners = winning_keys(cars, outcome)
        deadline = datetime.fromisoformat(official['deadline'])
        for pool, market in POOLS.items():
            if pool not in official.get('complete_pools', []):
                continue
            paid = {canonical(pool, key): amount for key, amount in official['payouts_per_100_yen'][pool].items()}
            if set(paid) != winners[market] or any(not isinstance(amount, int) or amount < 100 for amount in paid.values()):
                continue
            quotes = observed_quotes(record, pool, deadline)
            rows = []
            for key, odds in sorted(quotes.items()):
                order = [cars.index(int(car)) for car in key.split('-')]
                chance = probability(market, order, distribution)
                rows.append((key, odds, chance, paid.get(key, 0)))
            if rows:
                yield market, day, rows


def fit(races):
    tickets = [ticket for _, rows in races for ticket in rows]
    x = np.array([features(chance, odds) for _, odds, chance, _ in tickets])
    y = np.array([payout > 0 for _, _, _, payout in tickets])
    penalty = np.array([0., 0.001, 0.001, 0.001])

    def loss(beta):
        z = x @ beta
        return (np.mean(np.logaddexp(0., z) - y * z) + np.sum(penalty * beta ** 2) / 2,
                x.T @ (expit(z) - y) / len(y) + penalty * beta)

    fitted = minimize(loss, np.zeros(x.shape[1]), method='L-BFGS-B', jac=True)
    if not fitted.success:
        raise RuntimeError(f'Probability calibration did not converge: {fitted.message}')
    return fitted.x


def scored(races, beta):
    for _, rows in races:
        matrix = np.array([features(chance, odds) for _, odds, chance, _ in rows])
        probs = expit(matrix @ beta)
        yield list(zip(rows, (probs * np.array([row[1] for row in rows])).tolist()))


def summarize(races, threshold, limit):
    stake, payout, base_stake, base_payout, purchases, hits = 0, 0, 0, 0, 0, 0
    for rows in races:
        ranked = sorted(rows, key=lambda row: (-row[1], row[0][0]))
        selected = [row for row, score in ranked if score >= threshold][:limit]
        if not selected:
            continue
        purchases += 1
        hits += int(any(ticket[3] > 0 for ticket in selected))
        peers = sorted((row for row, _ in rows), key=lambda row: (-row[2], row[0]))[:len(selected)]
        stake += 100 * len(selected)
        payout += sum(row[3] for row in selected)
        base_stake += 100 * len(peers)
        base_payout += sum(row[3] for row in peers)
    return {'eligible_races': len(races), 'purchased_races': purchases, 'hit_races': hits,
            'stake_yen': stake, 'payout_yen': payout, 'return_rate': payout / stake if stake else None,
            'probability_only_return_rate_same_races_and_points': base_payout / base_stake if base_stake else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((args.output / 'model.json').read_text())
    beta = np.array(model['coefficients'])
    lines, _ = published_inputs(args.cache)
    races = {row[0]: row for row in load(args.cache, lines) if '2026-07-01' <= row[1] <= '2026-10-31'}
    data = defaultdict(list)
    for market, day, rows in matched_rows(archives(args.cache / 'timed'), races, beta):
        period = 'fit' if day < '2026-08-01' else 'validation' if day < '2026-09-01' else 'test'
        data[market, period].append((day, rows))
    report = {}
    for market, cap in BASE_CAPS.items():
        fitting, validation, test = (data[market, period] for period in ('fit', 'validation', 'test'))
        counts = dict(fit_races=len(fitting), validation_races=len(validation), test_races=len(test))
        if min(counts.values()) < 100:
            report[market] = {**counts, 'status': 'insufficient_coverage'}
            continue
        fitted = fit(fitting)
        validation_scores = list(scored(validation, fitted))
        candidates = [(threshold, summarize(validation_scores, threshold, cap)) for threshold in THRESHOLDS]
        usable = [(threshold, result) for threshold, result in candidates if result['purchased_races'] >= 100]
        if not usable:
            report[market] = {**counts, 'status': 'insufficient_validation_purchases', 'validation_grid': candidates}
            continue
        threshold, validation_result = max(usable, key=lambda row: (row[1]['return_rate'], row[1]['purchased_races']))
        test_result = summarize(list(scored(test, fitted)), threshold, cap)
        report[market] = {**counts, 'status': 'evaluated', 'calibration_coefficients': fitted.tolist(),
                          'threshold': threshold, 'max_tickets': cap, 'validation': validation_result,
                          'test': test_result}
    result = {'decision_time': 'cutoff minus 5 minutes, up to 120 seconds stale',
              'fit': '2026-07', 'threshold_selection': '2026-08', 'test': '2026-09 through 2026-10',
              'markets': report}
    print(json.dumps(result, indent=2, allow_nan=False))
    (args.output / 'purchase-evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

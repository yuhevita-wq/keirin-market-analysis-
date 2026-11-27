"""Fit a conditional logit for each place and evaluate chronological holdouts."""
import argparse
from collections import defaultdict
import csv
from datetime import date, timedelta
import gzip
from itertools import combinations, groupby
import json
from pathlib import Path
import re
import shutil
import sqlite3
from urllib.request import urlopen

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

SOURCES = {
    'riders': ('keirin-market-analysis-', 'c0f043625ca6ee51bad9d9fd261661762ad1d2ac'),
    'F1': ('F1', 'c959979c79cdd56766dd769d5df99c6ff92306f6'),
    'F2': ('F2', '09e1005a38cc9bd2bd7a62d3b4314f76f0a140d3'),
}
FEATURES = [
    'year_first', 'year_second', 'year_third', 'recent_first', 'recent_top3',
    'log_same_class_starts', 'car', 'term', 'top_class', 'second_class',
    'line_head', 'line_second', 'line_later', 'solo', 'line_size',
    'same_line_as_first', 'same_line_as_second',
]
MARKETS = ('wide', 'quinella', 'exacta', 'trio', 'trifecta')
POINTS = dict(zip(MARKETS, (2, 2, 2, 4, 6)))


def prepare(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for year in (2025, 2026):
        for kind, (repo, revision) in SOURCES.items():
            if kind == 'riders':
                local = f'rider{year}.csv.gz'
                remote = f'data/rider_condition_2025_202610_full/rider_races_{year}.csv.gz'
            else:
                local = f'{kind}-{year}.sqlite.gz'
                remote = f'data/ready/{kind.lower()}_{year}.sqlite.gz'
            target = cache / local
            if not target.exists():
                address = f'https://raw.githubusercontent.com/yuhevita-wq/{repo}/{revision}/{remote}'
                temp = target.with_suffix('.partial')
                with urlopen(address, timeout=120) as source, temp.open('wb') as out:
                    shutil.copyfileobj(source, out)
                temp.replace(target)


def published_inputs(cache):
    lines, payouts = {}, defaultdict(dict)
    for kind in ('F1', 'F2'):
        for year in (2025, 2026):
            compressed = cache / f'{kind}-{year}.sqlite.gz'
            dbfile = compressed.with_suffix('')
            with gzip.open(compressed, 'rb') as source, dbfile.open('wb') as out:
                shutil.copyfileobj(source, out)
            with sqlite3.connect(dbfile) as db:
                for race_id, raw in db.execute(
                    "SELECT race_id, predicted_line_formation FROM races WHERE line_status = 'published'"
                ):
                    if re.fullmatch(r'[1-9](?:[-/][1-9])*', raw or ''):
                        lines[race_id] = [[int(car) for car in line.split('-')] for line in raw.split('/')]
                for race_id, market, key, amount, status in db.execute(
                    "SELECT race_id, bet_code, combination, payout_yen, status FROM payouts"
                ):
                    if market in MARKETS and status == 'paid' and re.fullmatch(r'[1-9](?:-[1-9])+', key or ''):
                        if str(amount).isdigit() and int(amount) >= 100:
                            payouts[race_id].setdefault(market, {})[key] = int(amount)
    return lines, payouts


def category(code):
    if code in ('SS', 'S1', 'S2'):
        return 'S'
    if code in ('A1', 'A2'):
        return 'A12'
    return code


def rider_features(row, history, day, line):
    car, term, cls = int(row['car_no']), int(row['term'] or 0), row['class']
    period = [r for r in history if day - 365 <= r[0] and r[2] == category(cls)]
    recent = [r for r in period if day - 90 <= r[0]]
    rates = [sum(r[1] == rank for r in period) / max(1, len(period)) for rank in (1, 2, 3)]
    baseline_rows = [r for r in history if day - 365 <= r[0]]
    baseline_exact = [r for r in baseline_rows if r[3] == cls]
    baseline = baseline_exact if len(baseline_exact) >= 10 else baseline_rows
    _, position, size = line
    return (
        [*rates, sum(r[1] == 1 for r in recent) / max(1, len(recent)),
         sum(r[1] in (1, 2, 3) for r in recent) / max(1, len(recent)),
         np.log1p(len(period)), car / 9, term / 130,
         int(cls in ('SS', 'S1', 'A1')), int(cls in ('S2', 'A2')),
         int(position == 1 and size > 1), int(position == 2),
         int(position >= 3), int(size == 1), size / 4],
        len(baseline), sum(r[1] in (1, 2, 3) for r in baseline),
    )


def load(cache, lines):
    records = []
    for year in (2025, 2026):
        with gzip.open(cache / f'rider{year}.csv.gz', 'rt', encoding='utf-8-sig') as file:
            records.extend(csv.DictReader(file))
    records.sort(key=lambda r: (r['race_date'], r['race_id'], int(r['car_no'])))
    history = defaultdict(list)
    for day_text, day_rows in groupby(records, lambda r: r['race_date']):
        day = date.fromisoformat(day_text).toordinal()
        day_rows = list(day_rows)
        for race_id, rows in groupby(day_rows, lambda r: r['race_id']):
            rows = [r for r in rows if r['finish_label'] != '欠']
            cars = [int(r['car_no']) for r in rows]
            if not 5 <= len(cars) <= 9 or len(set(cars)) != len(cars):
                continue
            if any(r['result_status'] != 'full_result' or r['class'] not in ('SS','S1','S2','A1','A2','A3') for r in rows):
                continue
            ranks = [r['finish_position'] for r in rows]
            if any(ranks.count(str(rank)) != 1 for rank in (1, 2, 3)):
                continue
            published = lines.get(race_id, [])
            if sorted(car for line in published for car in line) != cars:
                published = []
            roles = {car: (i, p, len(line)) for i, line in enumerate(published) for p, car in enumerate(line, 1)}
            race_features = [rider_features(row, history[row['provisional_player_key']], day,
                            roles.get(car, (-1, 0, 0))) for row, car in zip(rows, cars)]
            x, counts, hits = zip(*race_features)
            groups = [roles.get(car, (-1, 0, 0))[0] for car in cars]
            outcome = tuple(ranks.index(str(rank)) for rank in (1, 2, 3))
            yield race_id, day_text, cars, np.array(x), np.array(groups), outcome, counts, hits
        for row in day_rows:
            if row['result_status'] == 'full_result' and row['finish_label'] != '欠' and row['provisional_player_key']:
                history[row['provisional_player_key']].append((
                    day, int(row['finish_position'] or 0), category(row['class']), row['class'],
                ))


def stage_features(x, groups, previous):
    flags = [((groups >= 0) & (groups == groups[index])).astype(float) for index in previous]
    flags += [np.zeros(len(x)) for _ in range(2 - len(flags))]
    return np.column_stack((x, *flags))


def fit(races, stage):
    features = np.zeros((len(races), 9, len(FEATURES)))
    allowed = np.zeros((len(races), 9), dtype=bool)
    target = np.zeros(len(races), dtype=int)
    for i, (_, _, cars, x, groups, outcome, _, _) in enumerate(races):
        previous = outcome[:stage]
        features[i, :len(cars)] = stage_features(x, groups, previous)
        allowed[i, :len(cars)] = True
        allowed[i, list(previous)] = False
        target[i] = outcome[stage]
    indices = np.arange(len(races))
    actual = features[indices, target]
    regularization = 0.0002

    def objective(beta):
        scores = np.where(allowed, features @ beta, -np.inf)
        normalization = logsumexp(scores, axis=1)
        probabilities = np.exp(scores - normalization[:, None])
        loss = np.mean(normalization - scores[indices, target]) + regularization * (beta @ beta) / 2
        gradient = (np.einsum('ij,ijk->k', probabilities, features) - actual.sum(axis=0)) / len(races)
        return loss, gradient + regularization * beta

    result = minimize(objective, np.zeros(len(FEATURES)), jac=True, method='L-BFGS-B',
                      options={'maxiter': 250, 'ftol': 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'Fit failed for place {stage + 1}: {result.message}')
    return result.x.tolist()


def outcomes_for_model(x, groups, beta):
    n = len(x)
    base = x @ beta[:, :-2].T
    same = (groups[:, None] == groups) & (groups[:, None] >= 0)
    first = np.exp(base[:, 0] - logsumexp(base[:, 0]))
    second = base[:, 1] + beta[1, -2] * same
    second[np.arange(n), np.arange(n)] = -np.inf
    second = np.exp(second - logsumexp(second, axis=1)[:, None])
    third = base[:, 2] + beta[2, -2] * same[:, None, :] + beta[2, -1] * same[None, :, :]
    third[np.arange(n), :, np.arange(n)] = -np.inf
    third[:, np.arange(n), np.arange(n)] = -np.inf
    third = np.exp(third - logsumexp(third, axis=2)[:, :, None])
    return first[:, None, None] * second[:, :, None] * third


def outcomes_for_baseline(counts, hits):
    n = len(counts)
    rate = (np.array(hits) + 20 * 3 / n) / (np.array(counts) + 20)
    weights = rate / (1 - rate)
    triple = weights[:, None, None] * weights[None, :, None] * weights
    idx = np.arange(n)
    triple[idx, idx, :] = 0
    triple[idx, :, idx] = 0
    triple[:, idx, idx] = 0
    return triple / triple.sum()


def market_picks(distribution, market, cars):
    if market == 'trifecta':
        tickets = {f'{cars[a]}-{cars[b]}-{cars[c]}': distribution[a, b, c]
                   for a, b, c in zip(*np.nonzero(distribution))}
    elif market == 'exacta':
        tickets = {f'{a}-{b}': probability for a, row in zip(cars, distribution.sum(axis=2))
                   for b, probability in zip(cars, row) if a != b}
    else:
        tickets = {}
        for a, b in combinations(range(len(cars)), 2):
            key = f'{cars[a]}-{cars[b]}'
            tickets[key] = distribution[a, b, :].sum() + distribution[b, a, :].sum()
            if market == 'wide':
                tickets[key] += (distribution[a, :, b].sum() + distribution[b, :, a].sum()
                                 + distribution[:, a, b].sum() + distribution[:, b, a].sum())
        if market == 'trio':
            tickets = {f'{cars[a]}-{cars[b]}-{cars[c]}': sum(
                distribution[i, j, a+b+c-i-j] for i in (a,b,c) for j in (a,b,c) if i != j
            ) for a, b, c in combinations(range(len(cars)), 3)}
    return sorted(tickets, key=lambda k: (-tickets[k], k))[:POINTS[market]]


def winning_keys(cars, outcome):
    a, b, c = [cars[i] for i in outcome]
    pair = lambda x, y: f'{min(x, y)}-{max(x, y)}'
    return {'wide': {pair(a,b), pair(a,c), pair(b,c)}, 'quinella': {pair(a,b)},
            'exacta': {f'{a}-{b}'}, 'trio': {'-'.join(map(str, sorted((a,b,c))))},
            'trifecta': {f'{a}-{b}-{c}'}}


def evaluate(races, beta, payouts):
    beta = np.array(beta)
    metrics = defaultdict(lambda: {'races': 0, 'podium_log_loss': 0., 'top3_brier': 0.,
                                   'markets': {key: defaultdict(int) for key in MARKETS}})
    coverage = {'published_line_races': 0, 'baseline_common_races': 0}
    for race_id, _, cars, x, groups, outcome, counts, hits in races:
        coverage['published_line_races'] += int(np.any(groups >= 0))
        if not all(counts):
            continue
        coverage['baseline_common_races'] += 1
        winners = winning_keys(cars, outcome)
        actual = np.zeros(len(cars))
        actual[list(outcome)] = 1
        for name, distribution in (
            ('ml', outcomes_for_model(x, groups, beta)),
            ('site_probability_baseline', outcomes_for_baseline(counts, hits)),
        ):
            row = metrics[name]
            row['races'] += 1
            row['podium_log_loss'] -= np.log(distribution[outcome])
            marginals = distribution.sum(axis=(0,1)) + distribution.sum(axis=(0,2)) + distribution.sum(axis=(1,2))
            row['top3_brier'] += np.mean((marginals - actual) ** 2)
            for market in MARKETS:
                selected = market_picks(distribution, market, cars)
                scored = row['markets'][market]
                scored['hit_races'] += bool(winners[market].intersection(selected))
                official = payouts.get(race_id, {}).get(market, {})
                if set(official) == winners[market]:
                    scored['settled_races'] += 1
                    scored['stake_yen'] += 100 * len(selected)
                    scored['payout_yen'] += sum(official.get(key, 0) for key in selected)
    for row in metrics.values():
        for key in ('podium_log_loss', 'top3_brier'):
            row[key] /= row['races']
        for market in row['markets'].values():
            market['hit_rate'] = market['hit_races'] / row['races']
            market['return_rate'] = market['payout_yen'] / market['stake_yen'] if market['stake_yen'] else None
    return {'eligible_races': len(races), **coverage, 'results': metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cache)
    lines, payouts = published_inputs(args.cache)
    races = list(load(args.cache, lines))
    training = [r for r in races if '2025-04-01' <= r[1] <= '2026-06-30']
    august = [r for r in races if '2026-07-01' <= r[1] <= '2026-08-31']
    october = [r for r in races if '2026-09-01' <= r[1] <= '2026-10-31']
    print(f'training={len(training)} validation={len(august)} test={len(october)}', flush=True)
    beta = [fit(training, stage) for stage in range(3)]
    args.output.mkdir(parents=True, exist_ok=True)
    model = {'version': 'LABOQ_ML_PODIUM_V1', 'trainedThrough': '2026-06-30',
             'features': FEATURES, 'coefficients': beta, 'sources': SOURCES}
    (args.output / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
    evaluation = {
        'training_races': len(training), 'points': POINTS,
        '2026-07_to_08': evaluate(august, beta, payouts),
        '2026-09_to_10': evaluate(october, beta, payouts),
    }
    output = json.dumps(evaluation, indent=2, allow_nan=False) + '\n'
    (args.output / 'evaluation.json').write_text(output)
    print(output, flush=True)


if __name__ == '__main__':
    main()

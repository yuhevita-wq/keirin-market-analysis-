from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / '2023' / 's_class_yosen'
OUT = ROOT / 'data' / 'audits' / 'discount_band_formations_2023.json'


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def combo(s):
    s = str(s).strip().replace('=', '-').replace(',', '-')
    try:
        return tuple(sorted(int(x) for x in s.split('-') if x.strip()))
    except Exception:
        return ()


def q_from_weights(combos, weights):
    raw = {c: math.prod(weights[x] for x in c) for c in combos}
    z = sum(raw.values())
    return {c: v / z for c, v in raw.items()}


def fit_maxent(riders, combos, target, tol=1e-12, max_iter=20000):
    weights = {r: 1.0 for r in riders}
    q = q_from_weights(combos, weights)
    for _ in range(max_iter):
        for r in riders:
            q = q_from_weights(combos, weights)
            cur = sum(v for c, v in q.items() if r in c)
            t = target[r]
            if not (0 < t < 1) or not (0 < cur < 1):
                raise RuntimeError(f'bad marginal rider={r} target={t} current={cur}')
            factor = (t * (1 - cur)) / (cur * (1 - t))
            weights[r] *= factor
        g = math.exp(sum(math.log(max(weights[r], 1e-300)) for r in riders) / len(riders))
        for r in riders:
            weights[r] /= g
        q = q_from_weights(combos, weights)
        marg = {r: sum(v for c, v in q.items() if r in c) for r in riders}
        err = max(abs(marg[r] - target[r]) for r in riders)
        if err < tol:
            return q, err
    raise RuntimeError(f'maxent did not converge; err={err}')


def build_market():
    by_race = defaultdict(list)
    for r in read_csv(DATA / 'trio_final_odds.csv'):
        if r.get('odds_status') != 'available':
            continue
        try:
            o = float(r['odds'])
        except Exception:
            continue
        c = combo(r.get('combination', ''))
        if len(c) == 3 and o > 0:
            by_race[str(r['race_id'])].append((c, o))
    return by_race


def build_payouts():
    wins = defaultdict(dict)
    for r in read_csv(DATA / 'payouts.csv'):
        if r.get('ticket_type') != '3連複' or r.get('status') != 'paid':
            continue
        c = combo(r.get('combination', ''))
        if len(c) != 3:
            continue
        try:
            pay = int(r['payout_yen'])
        except Exception:
            continue
        wins[str(r['race_id'])][c] = pay
    return wins


def band_candidates(rank, ratios):
    # A band is a support-rank pair core plus the four most-discounted third ranks.
    # Score uses the geometric mean R of those four tickets. Lower means more discounted.
    pair_rows = []
    rank_combos = {tuple(sorted(rank[x] for x in c)): c for c in ratios}
    for a, b in itertools.combinations(range(1, 8), 2):
        members = []
        for pat, physical in rank_combos.items():
            if a in pat and b in pat:
                members.append((ratios[physical], pat, physical))
        assert len(members) == 5
        members.sort(key=lambda x: (x[0], x[1]))
        chosen = members[:4]
        score = math.exp(sum(math.log(x[0]) for x in chosen) / 4)
        pair_rows.append({
            'core': (a, b),
            'score': score,
            'patterns': [x[1] for x in chosen],
            'tickets': [x[2] for x in chosen],
            'ratios': [x[0] for x in chosen],
            'excluded_pattern': members[4][1],
            'excluded_ratio': members[4][0],
        })
    pair_rows.sort(key=lambda x: (x['score'], x['core']))
    return pair_rows


def max_losing_streak(rows):
    best = cur = 0
    for r in rows:
        if r['hit_race']:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def summarize_strategy(rows, name):
    stake = sum(r['ticket_count'] * 100 for r in rows)
    payout = sum(r['payout_yen'] for r in rows)
    hits = sum(r['hit_race'] for r in rows)
    ticket_hits = sum(r['hit_ticket_count'] for r in rows)
    n = len(rows)
    return {
        'name': name,
        'races': n,
        'tickets': sum(r['ticket_count'] for r in rows),
        'avg_tickets_per_race': sum(r['ticket_count'] for r in rows) / n if n else None,
        'min_tickets_per_race': min((r['ticket_count'] for r in rows), default=None),
        'max_tickets_per_race': max((r['ticket_count'] for r in rows), default=None),
        'hit_races': hits,
        'race_hit_rate_pct': 100 * hits / n if n else None,
        'hit_tickets': ticket_hits,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else None,
        'max_losing_streak_races': max_losing_streak(rows),
    }


def main():
    markets = build_market()
    payouts = build_payouts()

    one_band_rows = []
    two_band_rows = []
    solver_failures = []
    excluded_non7_or_incomplete = 0
    no_payout = 0
    core1_counter = Counter()
    core2_counter = Counter()
    selected_pattern_one = Counter()
    selected_pattern_two = Counter()

    for rid in sorted(markets):
        rows = markets[rid]
        riders = sorted({x for c, _ in rows for x in c})
        if len(riders) != 7:
            excluded_non7_or_incomplete += 1
            continue
        combos = list(itertools.combinations(riders, 3))
        odds = {c: o for c, o in rows}
        if set(odds) != set(combos):
            excluded_non7_or_incomplete += 1
            continue
        if rid not in payouts:
            no_payout += 1
            continue

        inv = {c: 1 / odds[c] for c in combos}
        z = sum(inv.values())
        m = {c: inv[c] / z for c in combos}
        support = {r: sum(v for c, v in m.items() if r in c) for r in riders}
        ranked = sorted(riders, key=lambda r: (-support[r], r))
        rank = {r: i + 1 for i, r in enumerate(ranked)}
        try:
            q, err = fit_maxent(riders, combos, support)
        except Exception as e:
            solver_failures.append({'race_id': rid, 'error': repr(e)})
            continue
        ratios = {c: m[c] / q[c] for c in combos}
        bands = band_candidates(rank, ratios)

        b1 = bands[0]
        b2 = bands[1]
        core1_counter[''.join(map(str, b1['core']))] += 1
        core2_counter[''.join(map(str, b2['core']))] += 1

        tickets1 = set(b1['tickets'])
        tickets2 = set(b1['tickets']) | set(b2['tickets'])
        for c in tickets1:
            selected_pattern_one[''.join(map(str, sorted(rank[x] for x in c)))] += 1
        for c in tickets2:
            selected_pattern_two[''.join(map(str, sorted(rank[x] for x in c)))] += 1

        wins = payouts[rid]

        def score(ticket_set):
            hit_tickets = [c for c in ticket_set if c in wins]
            pay = sum(wins[c] for c in hit_tickets)
            return len(hit_tickets) > 0, len(hit_tickets), pay

        hit1, hitn1, pay1 = score(tickets1)
        hit2, hitn2, pay2 = score(tickets2)
        one_band_rows.append({
            'race_id': rid,
            'core1': ''.join(map(str, b1['core'])),
            'band1_score': b1['score'],
            'ticket_count': len(tickets1),
            'hit_race': hit1,
            'hit_ticket_count': hitn1,
            'payout_yen': pay1,
            'solver_error': err,
        })
        two_band_rows.append({
            'race_id': rid,
            'core1': ''.join(map(str, b1['core'])),
            'core2': ''.join(map(str, b2['core'])),
            'band1_score': b1['score'],
            'band2_score': b2['score'],
            'ticket_count': len(tickets2),
            'hit_race': hit2,
            'hit_ticket_count': hitn2,
            'payout_yen': pay2,
            'solver_error': err,
        })

    s1 = summarize_strategy(one_band_rows, 'ONE_DISCOUNT_BAND_4')
    s2 = summarize_strategy(two_band_rows, 'TWO_DISCOUNT_BANDS_7_OR_8')

    out = {
        'status': 'DISCOUNT_BAND_FORMATION_SIMULATION_2023',
        'year': 2023,
        'scope': '7-rider exact S-class qualifying races with complete 35-way trio final odds and paid trio result.',
        'methodology_lock': {
            'ticket_selection_uses_results_or_payouts': False,
            'market_probability_M': 'normalized inverse odds across all 35 trios',
            'support': 'rider marginal of M',
            'maxent_Q': 'maximum-entropy 35-trio distribution reproducing the seven rider marginals',
            'residual_R': 'M/Q; lower means market discount relative to rider supports',
            'band_definition': 'For each of the 21 support-rank pairs, inspect its 5 containing trios; keep the 4 lowest-R trios and score the band by their geometric-mean R.',
            'one_band_rule': 'Buy the 4 tickets from the single lowest-score pair-band.',
            'two_band_rule': 'Buy the union of the 4 tickets from each of the two lowest-score pair-bands; duplicates removed, yielding 7 or 8 tickets.',
            'stake_per_ticket_yen': 100,
            'payout_scoring': 'actual published 3連複 payout_yen; ties/multiple paid trios summed if multiple selected tickets win.',
            'warning': '2023 market structure motivated this family before outcome scoring. These results are development/in-sample for profitability and are not validation.'
        },
        'accounting': {
            'market_races': len(markets),
            'excluded_non7_or_incomplete': excluded_non7_or_incomplete,
            'no_payout': no_payout,
            'solver_failures': len(solver_failures),
            'evaluated_races': len(one_band_rows),
            'max_solver_error': max((r['solver_error'] for r in one_band_rows), default=None),
        },
        'strategies': [s1, s2],
        'formation_diagnostics': {
            'first_band_core_frequency': core1_counter.most_common(),
            'second_band_core_frequency': core2_counter.most_common(),
            'one_band_selected_support_patterns': selected_pattern_one.most_common(),
            'two_band_selected_support_patterns': selected_pattern_two.most_common(),
        },
        'solver_failure_examples': solver_failures[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'accounting': out['accounting'],
        'one_band': s1,
        'two_band': s2,
        'first_band_cores_top10': core1_counter.most_common(10),
        'second_band_cores_top10': core2_counter.most_common(10),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

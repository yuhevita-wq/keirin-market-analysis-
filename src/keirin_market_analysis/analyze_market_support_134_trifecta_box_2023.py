from __future__ import annotations

import itertools
import json
from pathlib import Path

from fake_favorite_formation_sim_2023 import (
    load_audited_gate,
    load_trio_odds,
    load_trifecta_odds,
    load_results,
    entrants_from_trio,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'market_support_134_trifecta_box_2023.json'
STAKE = 100


def rider_supports(trio_race, entrants):
    support = {r: 0.0 for r in entrants}
    for combo, odds in trio_race.items():
        if odds <= 0:
            continue
        mass = 1.0 / odds
        for r in combo:
            support[r] += mass
    return support


def max_losing_streak(flags):
    best = cur = 0
    for hit in flags:
        if hit:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    return ys[n//2] if n % 2 else (ys[n//2-1] + ys[n//2]) / 2


def main():
    gate = load_audited_gate()
    trio = load_trio_odds()
    tf = load_trifecta_odds()
    results = load_results()

    bet_races = tickets_total = hit_races = payout = 0
    hit_flags = []
    winning_odds = []
    examples = []

    for rid in sorted(gate):
        entrants = entrants_from_trio(trio[rid])
        support = rider_supports(trio[rid], entrants)
        ranked = sorted(entrants, key=lambda r: (-support[r], r))
        r1, r3, r4 = ranked[0], ranked[2], ranked[3]
        box_cars = (r1, r3, r4)
        tickets = list(itertools.permutations(box_cars, 3))
        result_order = tuple(results[rid])
        hit = result_order in tickets

        bet_races += 1
        tickets_total += len(tickets)
        hit_races += int(hit)
        hit_flags.append(hit)
        if hit:
            odds = tf[rid][result_order]
            winning_odds.append(odds)
            payout += round(STAKE * odds)

        if len(examples) < 10:
            examples.append({
                'race_id': rid,
                'market_rank_1': r1,
                'market_rank_3': r3,
                'market_rank_4': r4,
                'box_cars': list(box_cars),
                'result_order': list(result_order),
                'hit': hit,
                'winning_trifecta_odds': tf[rid][result_order] if hit else None,
            })

    stake = tickets_total * STAKE
    out = {
        'status': 'MARKET_SUPPORT_134_TRIFECTA_BOX_2023',
        'year': 2023,
        'years_read': [2023],
        'population': 'Audited 208 fake-favorite races only.',
        'market_support_definition': 'For each rider, sum 1/trio_final_odds across every valid trio containing that rider; rank descending, lower car number only breaks exact ties.',
        'strategy': 'Use market support ranks 1,3,4 as a 3-car trifecta BOX; exactly 6 tickets/race.',
        'payout_method': '100 yen times final trifecta odds; actual trifecta payout CSV not yet used.',
        'result': {
            'bet_races': bet_races,
            'tickets': tickets_total,
            'avg_tickets_per_race': tickets_total / bet_races,
            'hit_races': hit_races,
            'race_hit_rate_pct': 100 * hit_races / bet_races,
            'stake_yen': stake,
            'payout_yen': payout,
            'profit_yen': payout - stake,
            'roi_pct': 100 * payout / stake,
            'max_losing_streak': max_losing_streak(hit_flags),
            'mean_winning_odds': sum(winning_odds) / len(winning_odds) if winning_odds else None,
            'median_winning_odds': median(winning_odds),
            'min_winning_odds': min(winning_odds) if winning_odds else None,
            'max_winning_odds': max(winning_odds) if winning_odds else None,
        },
        'examples_first_10': examples,
        'note': '2023 development simulation only. No 2024/2025/2026 data read.'
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

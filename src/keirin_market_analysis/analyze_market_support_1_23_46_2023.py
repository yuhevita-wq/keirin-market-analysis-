from __future__ import annotations

import json
from pathlib import Path

from fake_favorite_formation_sim_2023 import (
    load_audited_gate,
    load_trio_odds,
    load_results,
    entrants_from_trio,
    norm_combo,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'market_support_1_23_46_2023.json'
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


def max_losing_streak(hit_flags):
    best = cur = 0
    for hit in hit_flags:
        if hit:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def main():
    gate = load_audited_gate()
    trio = load_trio_odds()
    results = load_results()

    bet_races = tickets_total = hit_races = payout = 0
    hit_flags = []
    winning_odds = []
    examples = []

    for rid in sorted(gate):
        entrants = entrants_from_trio(trio[rid])
        support = rider_supports(trio[rid], entrants)
        ranked = sorted(entrants, key=lambda r: (-support[r], r))
        if len(ranked) < 7:
            continue

        r1 = ranked[0]
        mids = ranked[1:3]
        tails = [ranked[3], ranked[5]]  # support ranks 4 and 6

        tickets = sorted({norm_combo(r1, m, t) for m in mids for t in tails})
        result = norm_combo(*results[rid])
        hit = result in tickets

        bet_races += 1
        tickets_total += len(tickets)
        hit_races += int(hit)
        hit_flags.append(hit)
        if hit:
            odds = trio[rid][result]
            winning_odds.append(odds)
            payout += round(STAKE * odds)

        if len(examples) < 10:
            examples.append({
                'race_id': rid,
                'market_rank_1': r1,
                'market_rank_2': ranked[1],
                'market_rank_3': ranked[2],
                'market_rank_4': ranked[3],
                'market_rank_6': ranked[5],
                'tickets': tickets,
                'result': list(result),
                'hit': hit,
            })

    stake = tickets_total * STAKE
    win_sorted = sorted(winning_odds)
    median = None
    if win_sorted:
        n = len(win_sorted)
        median = win_sorted[n // 2] if n % 2 else (win_sorted[n // 2 - 1] + win_sorted[n // 2]) / 2

    out = {
        'status': 'MARKET_SUPPORT_1_23_46_2023',
        'year': 2023,
        'years_read': [2023],
        'population': 'Audited 208 fake-favorite races only.',
        'market_support_definition': 'For each rider, sum 1/trio_final_odds across every valid trio containing that rider; rank descending, lower car number only breaks exact ties.',
        'formation': 'trio market support rank 1 - ranks 2,3 - ranks 4,6',
        'strategy_detail': 'Exactly 4 tickets in a 7-rider race: rank1-rank2-rank4, rank1-rank2-rank6, rank1-rank3-rank4, rank1-rank3-rank6.',
        'payout_method': '100 yen times final trio odds for apples-to-apples comparison with prior development simulations; actual payout CSV not yet used.',
        'result': {
            'bet_races': bet_races,
            'tickets': tickets_total,
            'avg_tickets_per_race': tickets_total / bet_races if bet_races else 0,
            'hit_races': hit_races,
            'race_hit_rate_pct': 100 * hit_races / bet_races if bet_races else 0,
            'stake_yen': stake,
            'payout_yen': payout,
            'profit_yen': payout - stake,
            'roi_pct': 100 * payout / stake if stake else 0,
            'max_losing_streak': max_losing_streak(hit_flags),
            'mean_winning_odds': sum(winning_odds) / len(winning_odds) if winning_odds else None,
            'median_winning_odds': median,
        },
        'examples_first_10': examples,
        'note': '2023 development simulation only. No 2024/2025/2026 data read.'
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

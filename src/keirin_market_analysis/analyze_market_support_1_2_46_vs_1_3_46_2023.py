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
OUT = ROOT / 'data' / 'audits' / 'market_support_1_2_46_vs_1_3_46_2023.json'
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


def summarize(name, rows):
    tickets = sum(len(r['tickets']) for r in rows)
    hits = sum(int(r['hit']) for r in rows)
    payout = sum(r['payout_yen'] for r in rows)
    stake = tickets * STAKE
    win_odds = sorted(r['winning_odds'] for r in rows if r['hit'])
    median = None
    if win_odds:
        n = len(win_odds)
        median = win_odds[n//2] if n % 2 else (win_odds[n//2-1] + win_odds[n//2]) / 2
    return {
        'name': name,
        'bet_races': len(rows),
        'tickets': tickets,
        'avg_tickets_per_race': tickets / len(rows) if rows else 0,
        'hit_races': hits,
        'race_hit_rate_pct': 100 * hits / len(rows) if rows else 0,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else 0,
        'max_losing_streak': max_losing_streak([r['hit'] for r in rows]),
        'mean_winning_odds': (sum(win_odds)/len(win_odds)) if win_odds else None,
        'median_winning_odds': median,
    }


def main():
    gate = load_audited_gate()
    trio = load_trio_odds()
    results = load_results()

    rows_12 = []
    rows_13 = []
    examples = []

    for rid in sorted(gate):
        entrants = entrants_from_trio(trio[rid])
        support = rider_supports(trio[rid], entrants)
        ranked = sorted(entrants, key=lambda r: (-support[r], r))
        if len(ranked) < 7:
            continue

        r1, r2, r3 = ranked[0], ranked[1], ranked[2]
        r4, r6 = ranked[3], ranked[5]
        result = norm_combo(*results[rid])

        for bucket, mid in ((rows_12, r2), (rows_13, r3)):
            tickets = sorted({norm_combo(r1, mid, r4), norm_combo(r1, mid, r6)})
            hit = result in tickets
            odds = trio[rid][result] if hit else None
            payout_yen = round(STAKE * odds) if hit else 0
            bucket.append({
                'race_id': rid,
                'tickets': tickets,
                'result': list(result),
                'hit': hit,
                'winning_odds': odds,
                'payout_yen': payout_yen,
            })

        if len(examples) < 10:
            examples.append({
                'race_id': rid,
                'market_rank_1': r1,
                'market_rank_2': r2,
                'market_rank_3': r3,
                'market_rank_4': r4,
                'market_rank_6': r6,
                'tickets_1_2_46': rows_12[-1]['tickets'],
                'tickets_1_3_46': rows_13[-1]['tickets'],
                'result': list(result),
                'hit_1_2_46': rows_12[-1]['hit'],
                'hit_1_3_46': rows_13[-1]['hit'],
            })

    out = {
        'status': 'MARKET_SUPPORT_1_2_46_VS_1_3_46_2023',
        'year': 2023,
        'years_read': [2023],
        'population': 'Audited 208 fake-favorite races only.',
        'market_support_definition': 'For each rider, sum 1/trio_final_odds across every valid trio containing that rider; rank descending, lower car number only breaks exact ties.',
        'strategies': {
            '1-2-46': 'rank1-rank2-rank4 and rank1-rank2-rank6; exactly 2 tickets/race.',
            '1-3-46': 'rank1-rank3-rank4 and rank1-rank3-rank6; exactly 2 tickets/race.',
        },
        'payout_method': '100 yen times final trio odds for apples-to-apples comparison with prior development simulations; actual payout CSV not yet used.',
        'results': {
            '1-2-46': summarize('1-2-46', rows_12),
            '1-3-46': summarize('1-3-46', rows_13),
        },
        'examples_first_10': examples,
        'note': '2023 development simulation only. No 2024/2025/2026 data read.'
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import json
from pathlib import Path

from fake_favorite_formation_sim_2023 import load_audited_gate, load_trio_odds, load_results, entrants_from_trio, norm_combo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'market_support_1_3_4_vs_1_3_6_2023.json'
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


def summarize(name, hits, payout, hit_flags, winning_odds, races):
    stake = races * STAKE
    win_sorted = sorted(winning_odds)
    median = None
    if win_sorted:
        n = len(win_sorted)
        median = win_sorted[n // 2] if n % 2 else (win_sorted[n // 2 - 1] + win_sorted[n // 2]) / 2
    return {
        'name': name,
        'bet_races': races,
        'tickets': races,
        'avg_tickets_per_race': 1.0,
        'hit_races': hits,
        'race_hit_rate_pct': 100 * hits / races if races else 0,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else 0,
        'max_losing_streak': max_losing_streak(hit_flags),
        'mean_winning_odds': sum(winning_odds) / len(winning_odds) if winning_odds else None,
        'median_winning_odds': median,
        'min_winning_odds': min(winning_odds) if winning_odds else None,
        'max_winning_odds': max(winning_odds) if winning_odds else None,
    }


def main():
    gate = load_audited_gate()
    trio = load_trio_odds()
    results = load_results()

    stats = {
        '1-3-4': {'hits': 0, 'payout': 0, 'flags': [], 'odds': []},
        '1-3-6': {'hits': 0, 'payout': 0, 'flags': [], 'odds': []},
    }
    races = 0
    examples = []

    for rid in sorted(gate):
        entrants = entrants_from_trio(trio[rid])
        support = rider_supports(trio[rid], entrants)
        ranked = sorted(entrants, key=lambda r: (-support[r], r))
        if len(ranked) < 7:
            continue
        r1, r3, r4, r6 = ranked[0], ranked[2], ranked[3], ranked[5]
        tickets = {
            '1-3-4': norm_combo(r1, r3, r4),
            '1-3-6': norm_combo(r1, r3, r6),
        }
        result = norm_combo(*results[rid])
        races += 1
        ex = {'race_id': rid, 'market_rank_1': r1, 'market_rank_3': r3, 'market_rank_4': r4, 'market_rank_6': r6, 'result': list(result)}
        for name, ticket in tickets.items():
            hit = result == ticket
            stats[name]['flags'].append(hit)
            if hit:
                odds = trio[rid][result]
                stats[name]['hits'] += 1
                stats[name]['odds'].append(odds)
                stats[name]['payout'] += round(STAKE * odds)
            ex[f'ticket_{name}'] = list(ticket)
            ex[f'hit_{name}'] = hit
        if len(examples) < 10:
            examples.append(ex)

    out = {
        'status': 'MARKET_SUPPORT_1_3_4_VS_1_3_6_2023',
        'year': 2023,
        'years_read': [2023],
        'population': 'Audited 208 fake-favorite races only.',
        'market_support_definition': 'For each rider, sum 1/trio_final_odds across every valid trio containing that rider; rank descending, lower car number only breaks exact ties.',
        'strategies': {
            '1-3-4': 'rank1-rank3-rank4; exactly 1 trio ticket/race.',
            '1-3-6': 'rank1-rank3-rank6; exactly 1 trio ticket/race.'
        },
        'payout_method': '100 yen times final trio odds for apples-to-apples comparison with prior development simulations; actual payout CSV not yet used.',
        'results': {name: summarize(name, s['hits'], s['payout'], s['flags'], s['odds'], races) for name, s in stats.items()},
        'examples_first_10': examples,
        'note': '2023 development simulation only. No 2024/2025/2026 data read.'
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

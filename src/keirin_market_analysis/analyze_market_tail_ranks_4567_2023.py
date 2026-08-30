from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from fake_favorite_formation_sim_2023 import (
    load_audited_gate,
    load_trio_odds,
    load_results,
    entrants_from_trio,
    norm_combo,
)
from analyze_market_support_123_4567_2023 import rider_supports

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'market_tail_ranks_4567_2023.json'
STAKE = 100
TAIL_RANKS = (4, 5, 6, 7)


def stats(xs):
    xs = list(xs)
    if not xs:
        return {'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
    return {
        'count': len(xs),
        'mean': mean(xs),
        'median': median(xs),
        'min': min(xs),
        'max': max(xs),
    }


def main():
    gate = load_audited_gate()
    trio = load_trio_odds()
    results = load_results()

    total_races = 0
    tail = {
        rank: {
            'top3_occurrences': 0,
            'formation_hits': 0,
            'winning_odds': [],
            'all_formation_ticket_odds': [],
            'payout_yen': 0,
            'tickets': 0,
        }
        for rank in TAIL_RANKS
    }

    for rid in sorted(gate):
        entrants = entrants_from_trio(trio[rid])
        support = rider_supports(trio[rid], entrants)
        ranked = sorted(entrants, key=lambda r: (-support[r], r))
        if len(ranked) < 7:
            continue
        total_races += 1
        result = norm_combo(*results[rid])
        result_set = set(result)
        r1 = ranked[0]
        mids = ranked[1:3]

        for rank in TAIL_RANKS:
            rider = ranked[rank - 1]
            if rider in result_set:
                tail[rank]['top3_occurrences'] += 1

            tickets = sorted({norm_combo(r1, m, rider) for m in mids})
            tail[rank]['tickets'] += len(tickets)
            for ticket in tickets:
                tail[rank]['all_formation_ticket_odds'].append(trio[rid][ticket])

            if result in tickets:
                tail[rank]['formation_hits'] += 1
                odds = trio[rid][result]
                tail[rank]['winning_odds'].append(odds)
                tail[rank]['payout_yen'] += round(STAKE * odds)

    output_ranks = {}
    for rank in TAIL_RANKS:
        d = tail[rank]
        stake = d['tickets'] * STAKE
        output_ranks[str(rank)] = {
            'top3_occurrences': d['top3_occurrences'],
            'top3_occurrence_pct': 100 * d['top3_occurrences'] / total_races if total_races else 0,
            'formation_hits': d['formation_hits'],
            'formation_hit_pct_of_all_races': 100 * d['formation_hits'] / total_races if total_races else 0,
            'share_of_1_23_4567_hits_pct': None,
            'winning_odds_stats': stats(d['winning_odds']),
            'all_candidate_ticket_odds_stats': stats(d['all_formation_ticket_odds']),
            'tickets': d['tickets'],
            'stake_yen': stake,
            'payout_yen': d['payout_yen'],
            'roi_pct_if_only_this_tail_rank': 100 * d['payout_yen'] / stake if stake else 0,
        }

    total_formation_hits = sum(tail[r]['formation_hits'] for r in TAIL_RANKS)
    for rank in TAIL_RANKS:
        output_ranks[str(rank)]['share_of_1_23_4567_hits_pct'] = (
            100 * tail[rank]['formation_hits'] / total_formation_hits if total_formation_hits else 0
        )

    out = {
        'status': 'MARKET_TAIL_RANKS_4567_2023',
        'year': 2023,
        'years_read': [2023],
        'population': 'Audited 208 fake-favorite races only.',
        'market_support_definition': 'For each rider, sum 1/trio_final_odds across every valid trio containing that rider; rank descending, lower car number only breaks exact ties.',
        'diagnostic': 'For support ranks 4-7, measure raw top-3 occurrence, contribution to the fixed 1-23-4567 formation, and trio final-odds distributions. Also show a diagnostic ROI for keeping only each tail rank (2 tickets/race).',
        'total_races': total_races,
        'total_formation_hits': total_formation_hits,
        'ranks': output_ranks,
        'note': '2023 development diagnostic only. No 2024/2025/2026 data read. Payout is 100 yen times final trio odds, consistent with prior development scripts; actual payout CSV not yet used.'
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

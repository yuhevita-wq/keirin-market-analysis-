from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from .analyze_middle_ticket_hit_anatomy_2023_validate_2024 import load_year

YEAR = 2024
OUT = Path('data/audits/fake_favorite_middle_plan_2024_frozen_from_2023.json')

# IMPORTANT: every threshold below is copied unchanged from the 2023 discovery/simulation.
# This file performs no search and no retuning on 2024.
FAKE_GATES = {
    'F1_low_fav_consensus': lambda r: r['fav_consensus'] <= 0.1884985310418076,
    'F2_close_1st_2nd': lambda r: r['second_to_fav_odds_ratio'] <= 1.2058823529411764,
    'F3_high_trifecta_set_entropy': lambda r: r['trifecta_set_entropy'] >= 2.7941491509245173,
    'F4_high_trio_entropy': lambda r: r['trio_entropy'] >= 2.856512307140793,
    'F12_consensus_AND_close': lambda r: (
        r['fav_consensus'] <= 0.1884985310418076
        and r['second_to_fav_odds_ratio'] <= 1.2058823529411764
    ),
    'F_any_consensus_OR_close': lambda r: (
        r['fav_consensus'] <= 0.1884985310418076
        or r['second_to_fav_odds_ratio'] <= 1.2058823529411764
    ),
}

MIDDLE_GATES = {
    'M0_all_selected_middle': lambda r: True,
    'M1_delta_ge_1.187pt': lambda r: r['delta'] >= 0.01186999093633212,
    'M2_scenario_score_ge_1.617pt': lambda r: r['scenario_score'] >= 0.016168991112891856,
    'M3_delta_rank_top2': lambda r: r['delta_rank'] <= 2,
    'M4_support_all3': lambda r: r['support_sets'] >= 3,
}


def stats(rows):
    if not rows:
        return {
            'races': 0, 'hit_races': 0, 'race_hit_rate_pct': 0.0,
            'tickets': 0, 'hits': 0, 'ticket_hit_rate_pct': 0.0,
            'avg_tickets_per_race': 0.0, 'avg_odds': None, 'median_odds': None,
            'stake_yen': 0, 'payout_yen': 0, 'profit_yen': 0, 'roi_pct': 0.0,
            'max_losing_race_streak': 0,
        }
    by_race = defaultdict(list)
    for r in rows:
        by_race[r['race_id']].append(r)
    race_ids = sorted(by_race)
    hit_races = sum(any(x['hit'] for x in by_race[rid]) for rid in race_ids)
    hits = sum(r['hit'] for r in rows)
    stake = 100 * len(rows)
    payout = sum(r['payout'] for r in rows)
    losing = 0
    max_losing = 0
    for rid in race_ids:
        if any(x['hit'] for x in by_race[rid]):
            losing = 0
        else:
            losing += 1
            max_losing = max(max_losing, losing)
    return {
        'races': len(race_ids),
        'hit_races': hit_races,
        'race_hit_rate_pct': 100 * hit_races / len(race_ids),
        'tickets': len(rows),
        'hits': hits,
        'ticket_hit_rate_pct': 100 * hits / len(rows),
        'avg_tickets_per_race': len(rows) / len(race_ids),
        'avg_odds': sum(r['odds'] for r in rows) / len(rows),
        'median_odds': median([r['odds'] for r in rows]),
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake,
        'max_losing_race_streak': max_losing,
    }


def main():
    rows, _ = load_year(YEAR)
    base = stats(rows)
    variants = []
    for fname, fgate in FAKE_GATES.items():
        race_pass = {}
        for r in rows:
            if r['race_id'] not in race_pass:
                race_pass[r['race_id']] = bool(fgate(r))
        for mname, mgate in MIDDLE_GATES.items():
            selected = [r for r in rows if race_pass.get(r['race_id'], False) and mgate(r)]
            if not selected:
                continue
            variants.append({'fake_gate': fname, 'middle_gate': mname, 'stats': stats(selected)})

    by_roi = sorted(variants, key=lambda x: (-x['stats']['roi_pct'], -x['stats']['races']))
    by_profit = sorted(variants, key=lambda x: (-x['stats']['profit_yen'], -x['stats']['races']))

    out = {
        'status': 'FAKE_FAVORITE_MIDDLE_PLAN_2024_FROZEN_FROM_2023',
        'year': YEAR,
        'years_read': [2024],
        'discovery_year': 2023,
        'thresholds_retuned_on_2024': False,
        'evaluation_year_2025_used': False,
        'evaluation_year_2026_used': False,
        'odds_phase': 'final',
        'staking': 'Flat 100 yen per selected middle ticket. No dutching.',
        'discipline': 'All fake-favorite and middle thresholds are copied unchanged from the completed 2023 analysis. 2024 is used only as frozen validation.',
        'baseline_all_selected_middle': base,
        'fake_gate_definitions': {
            'F1_low_fav_consensus': 'fav_consensus <= 0.1884985310418076',
            'F2_close_1st_2nd': 'second_to_fav_odds_ratio <= 1.2058823529411764',
            'F3_high_trifecta_set_entropy': 'trifecta_set_entropy >= 2.7941491509245173',
            'F4_high_trio_entropy': 'trio_entropy >= 2.856512307140793',
            'F12_consensus_AND_close': 'F1 AND F2',
            'F_any_consensus_OR_close': 'F1 OR F2',
        },
        'middle_gate_definitions': {
            'M0_all_selected_middle': 'all frozen-v1 selected middle tickets',
            'M1_delta_ge_1.187pt': 'delta >= 0.01186999093633212',
            'M2_scenario_score_ge_1.617pt': 'scenario_score >= 0.016168991112891856',
            'M3_delta_rank_top2': 'delta rank within all 35 trio sets <= 2',
            'M4_support_all3': 'all 3 one-car-replacement family sets have positive delta',
        },
        'variants': variants,
        'top10_by_roi': by_roi[:10],
        'top10_by_profit': by_profit[:10],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'baseline': base, 'top10_by_roi': by_roi[:10], 'top10_by_profit': by_profit[:10]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

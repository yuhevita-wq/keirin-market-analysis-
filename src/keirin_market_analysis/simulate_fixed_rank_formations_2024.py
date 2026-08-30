from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src' / 'keirin_market_analysis' / 'simulate_support_pair_formations_2023.py'
OUT = ROOT / 'data' / 'audits' / 'fixed_rank_formations_2024.json'

spec = importlib.util.spec_from_file_location('base2023', SRC)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)
base.DATA = ROOT / 'data' / '2024' / 's_class_yosen'

FORMATIONS = {
    '2023_champion_136_146': ((1, 3, 6), (1, 4, 6)),
    '2023_stable_123_136': ((1, 2, 3), (1, 3, 6)),
    '2023_temporal_robust_136_156': ((1, 3, 6), (1, 5, 6)),
}


def support_rank_map(r):
    inv = {c: 1.0 / r['odds'][c] for c in r['combos']}
    z = sum(inv.values())
    m = {c: v / z for c, v in inv.items()}
    support = {x: sum(v for c, v in m.items() if x in c) for x in r['riders']}
    ordered = sorted(r['riders'], key=lambda x: (-support[x], x))
    return {idx + 1: rider for idx, rider in enumerate(ordered)}


def actual_combo(rank_map, rank_pattern):
    return tuple(sorted(rank_map[i] for i in rank_pattern))


def score(races, patterns):
    stake = 0
    payout = 0
    hits = 0
    cur_loss = 0
    max_loss = 0
    for r in races:
        rm = support_rank_map(r)
        tickets = tuple(actual_combo(rm, p) for p in patterns)
        stake += 200
        if r['win'] in tickets:
            hits += 1
            payout += r['payout']
            cur_loss = 0
        else:
            cur_loss += 1
            max_loss = max(max_loss, cur_loss)
    return {
        'races': len(races),
        'tickets': len(races) * 2,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100.0 * payout / stake if stake else None,
        'hit_races': hits,
        'hit_rate_pct': 100.0 * hits / len(races) if races else None,
        'max_losing_streak_races': max_loss,
        'patterns': [''.join(map(str, p)) for p in patterns],
    }


def main():
    races, accounting = base.load_races()
    results = {name: score(races, pats) for name, pats in FORMATIONS.items()}
    out = {
        'status': 'FROZEN_FIXED_RANK_FORMATIONS_2024',
        'scope': '2024 out-of-sample check of formations selected from 2023 only',
        'frozen_context': {
            'must_enter_tv_threshold': base.MUST_TV,
            'beta': base.BETA,
            'support_rank_definition': 'Current-race trio-market marginal support S, descending; ties by rider number.',
            'stake_rule': '100 yen per ticket, exactly 2 tickets per MUST_ENTER race',
            'outcome_used_only_for_scoring': True,
            'odds_phase': 'final historical odds; not executable T-10 odds',
            'no_2024_tuning': True,
        },
        'accounting': accounting,
        'results': results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

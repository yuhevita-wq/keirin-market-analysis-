from __future__ import annotations

import itertools, json
from pathlib import Path

from simulate_support_pair_formations_2023 import load_races

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'support_rank_two_point_formations_2023.json'


def support_rank_map(r):
    inv = {c: 1.0 / r['odds'][c] for c in r['combos']}
    z = sum(inv.values())
    m = {c: v / z for c, v in inv.items()}
    support = {x: sum(v for c, v in m.items() if x in c) for x in r['riders']}
    ordered = sorted(r['riders'], key=lambda x: (-support[x], x))
    return {rank + 1: rider for rank, rider in enumerate(ordered)}


def specs():
    ranks = range(1, 8)
    out = []
    for pair in itertools.combinations(ranks, 2):
        rem = [x for x in ranks if x not in pair]
        for thirds in itertools.combinations(rem, 2):
            out.append((pair, thirds))
    assert len(out) == 210
    return out


def evaluate(races, pair_ranks, third_ranks):
    stake = payout = hits = 0
    cur_loss = max_loss = 0
    for r in races:
        rm = support_rank_map(r)
        a, b = rm[pair_ranks[0]], rm[pair_ranks[1]]
        tickets = {
            tuple(sorted((a, b, rm[third_ranks[0]]))),
            tuple(sorted((a, b, rm[third_ranks[1]]))),
        }
        assert len(tickets) == 2
        stake += 200
        if r['win'] in tickets:
            hits += 1
            payout += r['payout']
            cur_loss = 0
        else:
            cur_loss += 1
            max_loss = max(max_loss, cur_loss)
    max_loss = max(max_loss, cur_loss)
    return {
        'pair_support_ranks': ''.join(map(str, pair_ranks)),
        'third_support_ranks': ''.join(map(str, third_ranks)),
        'formation': f"{pair_ranks[0]}-{pair_ranks[1]}-{third_ranks[0]}{third_ranks[1]}",
        'tickets_as_rank_patterns': [
            ''.join(map(str, sorted((*pair_ranks, third_ranks[0])))),
            ''.join(map(str, sorted((*pair_ranks, third_ranks[1])))),
        ],
        'races': len(races),
        'tickets': 2 * len(races),
        'hit_races': hits,
        'hit_rate_pct': 100 * hits / len(races) if races else None,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else None,
        'max_losing_streak_races': max_loss,
    }


def main():
    races, accounting = load_races()
    rows = [evaluate(races, p, t) for p, t in specs()]
    rows.sort(key=lambda x: (-x['roi_pct'], -x['hit_rate_pct'], x['max_losing_streak_races'], x['formation']))
    out = {
        'status': 'SUPPORT_RANK_TWO_POINT_FORMATIONS_2023',
        'scope': '2023 MUST_ENTER races only; exhaustive 210 fixed support-rank 2-point formations',
        'accounting': accounting,
        'method': {
            'rank_definition': 'Rank riders 1-7 by current-race marginal support S, descending; ties by rider number.',
            'formation_definition': 'Choose a fixed pair of support ranks as the 2-rider core and exactly two other support ranks as third riders. Two 3連複 tickets, 100 yen each, every MUST_ENTER race.',
            'candidate_count': 210,
            'outcome_used_only_for_scoring': True,
            'odds_phase': 'final; historical diagnostic, not T-10 executable odds',
        },
        'ranked_by_realized_2023_roi': rows,
        'warning': 'Best formation is selected in-sample on 2023 and must be frozen before forward validation.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'accounting': accounting, 'top20': rows[:20]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

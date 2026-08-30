from __future__ import annotations

import itertools, json, math, statistics
from collections import defaultdict
from pathlib import Path

from simulate_support_pair_formations_2023 import load_races

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'audits' / 'best_two_point_formation_search_2023.json'

# Predefined before scoring. 2023 is development only.
SYN_MIN_GRID = (0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0)
POWER_GRID = (
    # name, coverage power, synthetic-odds power, uplift power, pair-mass power
    ('COV', 1.0, 0.0, 0.0, 0.0),
    ('EV', 1.0, 1.0, 0.0, 0.0),
    ('COVxSQRTODDS', 1.0, 0.5, 0.0, 0.0),
    ('COV2xODDS', 2.0, 1.0, 0.0, 0.0),
    ('SQRTOFCOVxODDS', 0.5, 1.0, 0.0, 0.0),
    ('EVxUPLIFT05', 1.0, 1.0, 0.5, 0.0),
    ('EVxUPLIFT1', 1.0, 1.0, 1.0, 0.0),
    ('EVxUPLIFT2', 1.0, 1.0, 2.0, 0.0),
    ('EVxPAIR05', 1.0, 1.0, 0.0, 0.5),
    ('EVxPAIR1', 1.0, 1.0, 0.0, 1.0),
    ('COVxPAIR1', 1.0, 0.0, 0.0, 1.0),
    ('UPLIFT', 0.0, 0.0, 1.0, 0.0),
)


def two_point_forms(r):
    inv = {c: 1.0 / r['odds'][c] for c in r['combos']}
    z = sum(inv.values())
    m = {c: v / z for c, v in inv.items()}
    out = []
    for pair in itertools.combinations(r['riders'], 2):
        all_pair_tickets = [c for c in r['combos'] if pair[0] in c and pair[1] in c]
        pair_mass = sum(r['p1'][c] for c in all_pair_tickets)
        thirds = [x for x in r['riders'] if x not in pair]
        for ts in itertools.combinations(thirds, 2):
            tickets = tuple(sorted(tuple(sorted((*pair, x))) for x in ts))
            coverage = sum(r['p1'][c] for c in tickets)
            market_coverage = sum(m[c] for c in tickets)
            uplift = coverage / market_coverage if market_coverage > 0 else 0.0
            invsum = sum(1.0 / r['odds'][c] for c in tickets)
            synthetic_odds = 1.0 / invsum
            model_ev = coverage * synthetic_odds
            out.append({
                'pair': pair,
                'tickets': tickets,
                'coverage': coverage,
                'market_coverage': market_coverage,
                'uplift': uplift,
                'pair_mass': pair_mass,
                'synthetic_odds': synthetic_odds,
                'model_ev': model_ev,
            })
    assert len(out) == 210
    return out


def score_value(f, spec):
    _, a, b, c, d = spec
    # all features are positive; log domain avoids silly under/overflow.
    return (
        a * math.log(max(f['coverage'], 1e-300))
        + b * math.log(max(f['synthetic_odds'], 1e-300))
        + c * math.log(max(f['uplift'], 1e-300))
        + d * math.log(max(f['pair_mass'], 1e-300))
    )


def choose(forms, spec, syn_min):
    cand = [f for f in forms if f['synthetic_odds'] >= syn_min]
    if not cand:
        return None
    return max(cand, key=lambda f: (score_value(f, spec), f['coverage'], f['model_ev'], tuple(-x for x in f['pair'])))


def evaluate(races, spec, syn_min):
    stake = payout = hits = bet_races = 0
    cur_loss = max_loss = 0
    pair_counts = defaultdict(int)
    syns = []; covs = []; evs = []; uplifts = []
    for r in races:
        f = choose(two_point_forms(r), spec, syn_min)
        if f is None:
            continue
        bet_races += 1
        stake += 200
        syns.append(f['synthetic_odds']); covs.append(f['coverage']); evs.append(f['model_ev']); uplifts.append(f['uplift'])
        pair_counts['-'.join(map(str, f['pair']))] += 1
        if r['win'] in f['tickets']:
            hits += 1
            payout += r['payout']
            cur_loss = 0
        else:
            cur_loss += 1
            max_loss = max(max_loss, cur_loss)
    max_loss = max(max_loss, cur_loss)
    return {
        'rule': spec[0],
        'coverage_power': spec[1],
        'synthetic_odds_power': spec[2],
        'uplift_power': spec[3],
        'pair_mass_power': spec[4],
        'synthetic_odds_min': syn_min,
        'must_enter_races': len(races),
        'bet_races': bet_races,
        'purchase_rate_pct': 100 * bet_races / len(races) if races else None,
        'tickets': 2 * bet_races,
        'hit_races': hits,
        'hit_rate_pct': 100 * hits / bet_races if bet_races else None,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100 * payout / stake if stake else None,
        'avg_synthetic_odds': statistics.mean(syns) if syns else None,
        'avg_coverage': statistics.mean(covs) if covs else None,
        'avg_model_ev': statistics.mean(evs) if evs else None,
        'avg_uplift': statistics.mean(uplifts) if uplifts else None,
        'max_losing_streak_bet_races': max_loss,
        'top_pair_cores': sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
    }


def main():
    races, accounting = load_races()
    rows = []
    for spec in POWER_GRID:
        for syn_min in SYN_MIN_GRID:
            rows.append(evaluate(races, spec, syn_min))
    rows.sort(key=lambda x: (
        -(x['roi_pct'] if x['roi_pct'] is not None else -1),
        -(x['hit_rate_pct'] if x['hit_rate_pct'] is not None else -1),
        -x['bet_races'],
        x['rule'], x['synthetic_odds_min'],
    ))
    out = {
        'status': 'BEST_TWO_POINT_FORMATION_SEARCH_2023',
        'scope': '2023 development only; in-sample exploratory search',
        'fixed_context': {
            'must_enter_rule': 'Frozen prior TV(P1,M) threshold; unchanged here.',
            'two_point_definition': 'Exactly two 3連複 tickets sharing the same 2-rider core.',
            'candidate_formations_per_race': 210,
            'outcome_used_only_for_final_scoring': True,
            'selection_features': ['P1 coverage', 'final-odds synthetic odds', 'P1/market coverage uplift', 'P1 pair mass'],
            'odds_phase': 'final; not executable T-10 data',
            'stake': '100 yen on each of the two tickets',
        },
        'accounting': accounting,
        'predefined_rule_count': len(POWER_GRID) * len(SYN_MIN_GRID),
        'rule_grid': {
            'score_specs': [list(x) for x in POWER_GRID],
            'synthetic_odds_min_grid': list(SYN_MIN_GRID),
        },
        'ranked_by_realized_2023_roi': rows,
        'warning': 'The top row is selected on 2023 outcomes and is not evidence of edge. It must be frozen and forward-tested on untouched years before adoption.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'accounting': accounting, 'rules': len(rows), 'top10': rows[:10]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

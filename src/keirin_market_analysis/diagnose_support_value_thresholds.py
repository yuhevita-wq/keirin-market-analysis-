from __future__ import annotations

import json
import statistics
from pathlib import Path

from develop_must_enter_formations_2023 import DATASETS, load_year

OUT = Path(__file__).resolve().parents[2] / 'data' / 'audits' / 'support_value_threshold_diagnostic.json'

# Economic, not outcome-optimized thresholds.
THRESHOLDS = (0.80, 0.90, 0.95, 1.00, 1.05)


def qtile(xs, q):
    if not xs:
        return None
    a = sorted(xs)
    p = (len(a)-1)*q
    lo = int(p)
    hi = min(lo+1, len(a)-1)
    f = p-lo
    return a[lo]*(1-f)+a[hi]*f


def score_threshold(races, threshold):
    bets = wins = stake = ret = bet_races = 0
    counts = []
    pair_formation_races = 0
    for r in races:
        chosen = [c for c in r['combos'] if r['p1'][c] * r['odds'][c] >= threshold]
        if not chosen:
            continue
        bet_races += 1
        counts.append(len(chosen))
        bets += len(chosen)
        stake += 100*len(chosen)
        if r['win'] in chosen:
            wins += 1
            ret += r['payout']
        # Can the selected set be expressed with at least one pair core and >=2 third riders?
        pair_counts = {}
        for c in chosen:
            for i in range(3):
                for j in range(i+1,3):
                    p = tuple(sorted((c[i],c[j])))
                    pair_counts[p] = pair_counts.get(p,0)+1
        if pair_counts and max(pair_counts.values()) >= 2:
            pair_formation_races += 1
    return {
        'threshold':threshold,
        'must_bet_races':bet_races,
        'tickets':bets,
        'avg_tickets_per_bet_race':statistics.mean(counts) if counts else None,
        'median_tickets_per_bet_race':statistics.median(counts) if counts else None,
        'hit_races':wins,
        'hit_rate_pct':100*wins/bet_races if bet_races else None,
        'stake_yen':stake,
        'payout_yen':ret,
        'profit_yen':ret-stake,
        'roi_pct':100*ret/stake if stake else None,
        'pair_formation_races':pair_formation_races,
        'pair_formation_share_pct':100*pair_formation_races/bet_races if bet_races else None,
    }


def analyze_year(year):
    races, accounting = load_year(year)
    max_values = []
    all_values = []
    for r in races:
        vals = [r['p1'][c] * r['odds'][c] for c in r['combos']]
        all_values.extend(vals)
        max_values.append(max(vals))
    return {
        'accounting':accounting,
        'must_enter_races':len(races),
        'value_definition':'V = support-augmented probability P1 * archived final trio odds',
        'max_V_per_race':{
            'mean':statistics.mean(max_values) if max_values else None,
            'median':statistics.median(max_values) if max_values else None,
            'q75':qtile(max_values,.75),
            'q90':qtile(max_values,.90),
            'q95':qtile(max_values,.95),
            'max':max(max_values) if max_values else None,
        },
        'all_combo_V':{
            'mean':statistics.mean(all_values) if all_values else None,
            'median':statistics.median(all_values) if all_values else None,
            'q90':qtile(all_values,.90),
            'q99':qtile(all_values,.99),
            'max':max(all_values) if all_values else None,
        },
        'threshold_diagnostics':[score_threshold(races,t) for t in THRESHOLDS],
    }


def main():
    years = {y:analyze_year(y) for y in DATASETS}
    out = {
        'status':'SUPPORT_VALUE_THRESHOLD_DIAGNOSTIC',
        'frozen_model':{
            'beta':0.02219612332210088,
            'must_enter_tv_threshold':0.004982810992042711,
        },
        'economic_rule':{
            'V':'P1 * odds',
            'V_ge_1_00':'model-implied gross break-even or better',
            'V_ge_1_05':'5% model-implied safety margin',
            'thresholds_below_1':'diagnostic only; never promoted to value bets by this audit',
        },
        'critical_limit':'All archived trio odds are KDreams FINAL odds. This audit is diagnostic only and is not an executable T-10 backtest. A deployable MUST-BET rule requires a pre-deadline odds snapshot captured at the same decision time as P1.',
        'years':years,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

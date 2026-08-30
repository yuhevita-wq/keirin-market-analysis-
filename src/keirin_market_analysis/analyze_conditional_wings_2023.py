from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src' / 'keirin_market_analysis' / 'develop_pair_flow_two_point_2023.py'
OUT = ROOT / 'data' / 'audits' / 'conditional_wings_2023.json'

spec = importlib.util.spec_from_file_location('pf', SRC)
pf = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(pf)

dyn = pf.dyn
ENTRY_THRESHOLD = 0.3179241680895883  # frozen from market-only 2023 q75 definition
WING_RULES = ('COND_RATIO', 'COND_DELTA', 'ANCHOR_DISCOUNT', 'QVALUE')


def pair_conditional(r, pair):
    cs = [c for c in r['combos'] if pair[0] in c and pair[1] in c]
    pm = sum(r['m'][c] for c in cs)
    pq = sum(r['q'][c] for c in cs)
    rows = []
    for c in cs:
        third = next(x for x in c if x not in pair)
        mc = r['m'][c] / pm
        qc = r['q'][c] / pq
        rows.append({
            'combo': c,
            'third': third,
            'm_cond': mc,
            'q_cond': qc,
            'ratio_discount': qc / mc if mc > 0 else 0.0,
            'delta_discount': qc - mc,
            'qvalue': (qc * qc / mc) if mc > 0 else 0.0,
        })
    return rows


def choose_wings(r, pair, rule):
    rows = pair_conditional(r, pair)
    if rule == 'COND_RATIO':
        rows.sort(key=lambda x: (-x['ratio_discount'], -x['q_cond'], x['third']))
        chosen = rows[:2]
    elif rule == 'COND_DELTA':
        rows.sort(key=lambda x: (-x['delta_discount'], -x['q_cond'], x['third']))
        chosen = rows[:2]
    elif rule == 'ANCHOR_DISCOUNT':
        anchor = max(rows, key=lambda x: (x['m_cond'], -x['third']))
        rest = [x for x in rows if x['combo'] != anchor['combo']]
        disc = max(rest, key=lambda x: (x['ratio_discount'], x['q_cond'], -x['third']))
        chosen = [anchor, disc]
    elif rule == 'QVALUE':
        rows.sort(key=lambda x: (-x['qvalue'], -x['q_cond'], x['third']))
        chosen = rows[:2]
    else:
        raise ValueError(rule)
    return tuple(sorted(x['combo'] for x in chosen))


def score(rows, rule):
    stake = payout = hits = 0
    cur = max_loss = 0
    for r in rows:
        pair, _ = pf.pair_stats(r)
        tickets = choose_wings(r, pair, rule)
        stake += 200
        if r['win'] in tickets:
            hits += 1
            payout += r['payout']
            cur = 0
        else:
            cur += 1
            max_loss = max(max_loss, cur)
    max_loss = max(max_loss, cur)
    return {
        'races': len(rows),
        'tickets': 2 * len(rows),
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi_pct': 100.0 * payout / stake if stake else None,
        'hit_races': hits,
        'hit_rate_pct': 100.0 * hits / len(rows) if rows else None,
        'max_losing_streak_races': max_loss,
    }


def main():
    old = dyn.MUST_TV
    dyn.MUST_TV = -1.0
    races, accounting = dyn.load_races()
    dyn.MUST_TV = old

    selected = [r for r in races if pf.pair_stats(r)[1] >= ENTRY_THRESHOLD]
    quarters = {f'Q{i}': [r for r in selected if (r['date'].month - 1) // 3 + 1 == i] for i in range(1, 5)}

    results = []
    for rule in WING_RULES:
        full = score(selected, rule)
        qs = {k: score(v, rule) for k, v in quarters.items()}
        q_rois = [qs[f'Q{i}']['roi_pct'] for i in range(1, 5)]
        results.append({
            'rule': rule,
            'full_year': full,
            'quarters': qs,
            'robustness': {
                'worst_quarter_roi_pct': min(q_rois),
                'profitable_quarters': sum(x >= 100 for x in q_rois),
                'quarters_ge_90': sum(x >= 90 for x in q_rois),
            },
        })

    results.sort(key=lambda x: (
        -x['robustness']['worst_quarter_roi_pct'],
        -x['robustness']['profitable_quarters'],
        -x['full_year']['roi_pct'],
    ))
    winner = results[0]['rule']

    out = {
        'status': 'CONDITIONAL_WING_LAWS_2023',
        'scope': '2023 development only; core and entry fixed from pair-flow model; four predefined wing laws',
        'entry_threshold': ENTRY_THRESHOLD,
        'core_rule': 'pair with maximum pair residual delta = sum(M-Q)',
        'wing_laws': {
            'COND_RATIO': 'choose two largest q_cond/m_cond inside selected pair',
            'COND_DELTA': 'choose two largest q_cond-m_cond inside selected pair',
            'ANCHOR_DISCOUNT': 'choose highest m_cond plus highest q_cond/m_cond among remaining thirds',
            'QVALUE': 'choose two largest q_cond^2/m_cond inside selected pair',
        },
        'outcome_used_for_entry_or_ticket_selection': False,
        'accounting': accounting,
        'must_enter_races': len(selected),
        'quarter_counts': {k: len(v) for k, v in quarters.items()},
        'ranked_by_temporal_robustness': results,
        'selected_for_2024_oos': winner,
        'warning': 'The selected wing law is chosen on 2023 realized outcomes and is development-only. Freeze it before 2024 scoring.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

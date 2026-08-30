from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src' / 'keirin_market_analysis' / 'develop_pair_flow_two_point_2023.py'
OUT = ROOT / 'data' / 'audits' / 'pair_flow_sign_diagnostic_2023.json'

spec = importlib.util.spec_from_file_location('pf', SRC)
pf = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(pf)
dyn = pf.dyn
ENTRY_THRESHOLD = 0.3179241680895883


def choose_with_signs(r):
    pair, concentration = pf.pair_stats(r)
    cs = [c for c in r['combos'] if pair[0] in c and pair[1] in c]
    delta = {c: r['m'][c] - r['q'][c] for c in cs}
    chosen = sorted(cs, key=lambda c: (-abs(delta[c]), -r['m'][c], c))[:2]
    signs = ''.join('+' if delta[c] > 0 else '-' if delta[c] < 0 else '0' for c in chosen)
    # normalize mixed sign category regardless of which abs residual ranks first
    if set(signs) == {'+', '-'}:
        signs = '+-'
    elif signs.count('+') == 2:
        signs = '++'
    elif signs.count('-') == 2:
        signs = '--'
    return pair, tuple(sorted(chosen)), signs, concentration, delta


def blank():
    return {'races': 0, 'tickets': 0, 'stake_yen': 0, 'payout_yen': 0, 'hit_races': 0}


def finish(x):
    x = dict(x)
    x['profit_yen'] = x['payout_yen'] - x['stake_yen']
    x['roi_pct'] = 100.0 * x['payout_yen'] / x['stake_yen'] if x['stake_yen'] else None
    x['hit_rate_pct'] = 100.0 * x['hit_races'] / x['races'] if x['races'] else None
    return x


def main():
    old = dyn.MUST_TV
    dyn.MUST_TV = -1.0
    races, accounting = dyn.load_races()
    dyn.MUST_TV = old
    selected = [r for r in races if pf.pair_stats(r)[1] >= ENTRY_THRESHOLD]

    groups = defaultdict(blank)
    quarters = {cat: {f'Q{i}': blank() for i in range(1, 5)} for cat in ('++', '+-', '--')}
    ticket_side = {'premium': {'tickets': 0, 'hit_tickets': 0, 'payout_yen': 0, 'stake_yen': 0},
                   'discount': {'tickets': 0, 'hit_tickets': 0, 'payout_yen': 0, 'stake_yen': 0}}

    for r in selected:
        pair, tickets, cat, concentration, delta = choose_with_signs(r)
        g = groups[cat]
        g['races'] += 1; g['tickets'] += 2; g['stake_yen'] += 200
        q = f"Q{(r['date'].month - 1)//3 + 1}"
        qg = quarters[cat][q]
        qg['races'] += 1; qg['tickets'] += 2; qg['stake_yen'] += 200
        if r['win'] in tickets:
            g['hit_races'] += 1; g['payout_yen'] += r['payout']
            qg['hit_races'] += 1; qg['payout_yen'] += r['payout']
        for c in tickets:
            side = 'premium' if delta[c] > 0 else 'discount'
            ticket_side[side]['tickets'] += 1
            ticket_side[side]['stake_yen'] += 100
            if r['win'] == c:
                ticket_side[side]['hit_tickets'] += 1
                ticket_side[side]['payout_yen'] += r['payout']

    final_groups = {k: finish(v) for k, v in groups.items()}
    final_quarters = {cat: {q: finish(v) for q, v in qs.items()} for cat, qs in quarters.items()}
    final_side = {}
    for side, x in ticket_side.items():
        y = dict(x)
        y['profit_yen'] = y['payout_yen'] - y['stake_yen']
        y['roi_pct'] = 100.0 * y['payout_yen'] / y['stake_yen'] if y['stake_yen'] else None
        y['ticket_hit_rate_pct'] = 100.0 * y['hit_tickets'] / y['tickets'] if y['tickets'] else None
        final_side[side] = y

    out = {
        'status': 'PAIR_FLOW_SIGN_DIAGNOSTIC_2023',
        'scope': '2023 exploratory diagnostic of the already-defined pair-flow two-point law; no ticket rule changed',
        'entry_threshold': ENTRY_THRESHOLD,
        'formation_rule': 'max positive pair delta core; two largest absolute trio residuals within core',
        'category_definition': {'++': 'both selected trio residuals M-Q positive', '+-': 'one positive and one negative', '--': 'both negative'},
        'accounting': accounting,
        'must_enter_races': len(selected),
        'by_formation_sign_category': final_groups,
        'quarters_by_category': final_quarters,
        'selected_ticket_side_diagnostic': final_side,
        'warning': 'Category performance uses 2023 outcomes and is exploratory. Any category filter derived from it must be frozen before 2024 testing.'
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

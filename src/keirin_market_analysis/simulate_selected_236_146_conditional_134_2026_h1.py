from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

BASE = Path('data/2026_h1/s_class_yosen')
OUT = Path('data/audits/selected_236_146_conditional_134_2026_h1.json')
STAKE = 100

# Frozen before reading this strategy's 2026 H1 result.
ENTROPY_MIN = 0.7598574338315534
TOP3_CONC_MAX = 0.5122247620383481
P123_MAX = 0.22900352400362795
RANK1_SHARE_MIN = 0.24054261443743438
BASE_PATTERNS = ((2, 3, 6), (1, 4, 6))
EXTRA_PATTERN = (1, 3, 4)
P134_ADD_MIN = 0.12502642493317004


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def comb(s):
    s = str(s).strip().replace('=', '-').replace(',', '-')
    parts = [p for p in s.split('-') if p.strip()]
    try:
        return tuple(sorted(int(x) for x in parts))
    except Exception:
        return ()


def max_losing_streak(hits):
    best = cur = 0
    for h in hits:
        if h:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return best


def pattern_combo(byrank, pat):
    return tuple(sorted(byrank[k] for k in pat))


def pattern_prob(p, byrank, pat):
    return p.get(pattern_combo(byrank, pat), 0.0)


def main():
    trio_rows = read_csv(BASE / 'trio_final_odds.csv')
    payout_rows = read_csv(BASE / 'payouts.csv')

    trios = defaultdict(list)
    meta = {}
    for r in trio_rows:
        if str(r.get('odds_status', 'available')) not in ('', 'available'):
            continue
        try:
            odd = float(r.get('odds') or r.get('final_odds') or r.get('trio_odds'))
        except Exception:
            continue
        c = comb(r.get('combination') or r.get('bet_code') or '')
        if len(c) == 3 and odd > 0:
            rid = str(r['race_id'])
            trios[rid].append((c, odd))
            meta.setdefault(rid, {
                'race_date': str(r.get('race_date', '')),
                'track': str(r.get('track', '')),
                'race_no': int(r.get('race_no') or 0),
            })

    paid = defaultdict(dict)
    for r in payout_rows:
        if (r.get('ticket_type') or '').strip() not in ('3連複', 'trio'):
            continue
        if str(r.get('status') or '').lower() not in ('', 'paid', 'success', '確定'):
            continue
        c = comb(r.get('combination') or r.get('bet_code') or '')
        if len(c) != 3:
            continue
        try:
            pay = int(float(r.get('payout_yen') or 0))
        except Exception:
            pay = 0
        paid[str(r['race_id'])][c] = pay

    ordered_rids = sorted(
        trios,
        key=lambda rid: (
            meta.get(rid, {}).get('race_date', ''),
            meta.get(rid, {}).get('track', ''),
            meta.get(rid, {}).get('race_no', 0),
            rid,
        ),
    )

    analyzable = 0
    selected = 0
    added = 0
    hit_races = 0
    base_hit_races = 0
    total_stake = 0
    total_payout = 0
    base_stake = 0
    base_payout = 0
    race_hits = []
    base_race_hits = []

    bypat = {
        '236': {'tickets': 0, 'hits': 0, 'stake_yen': 0, 'payout_yen': 0},
        '146': {'tickets': 0, 'hits': 0, 'stake_yen': 0, 'payout_yen': 0},
        '134_conditional': {'tickets': 0, 'hits': 0, 'stake_yen': 0, 'payout_yen': 0},
    }

    for rid in ordered_rids:
        rows = trios[rid]
        if rid not in paid or not paid[rid]:
            continue
        inv = [(c, 1.0 / odd) for c, odd in rows if odd > 0]
        z = sum(v for _, v in inv)
        if z <= 0:
            continue
        p = {c: v / z for c, v in inv}
        riders = sorted({x for c in p for x in c})
        sup = {x: sum(v for c, v in p.items() if x in c) for x in riders}
        ranked = sorted(sup, key=lambda x: (-sup[x], x))
        if len(ranked) < 6:
            continue

        analyzable += 1
        vals = sorted(sup.values(), reverse=True)
        n = len(p)
        entropy = -sum(v * math.log(v) for v in p.values()) / math.log(n) if n > 1 else 0.0
        top3_conc = sum(sorted(p.values(), reverse=True)[:3])
        rank1_share = vals[0] / 3 if vals else 0.0
        byrank = {i + 1: car for i, car in enumerate(ranked)}
        p123 = pattern_prob(p, byrank, (1, 2, 3))

        if not (
            entropy >= ENTROPY_MIN
            and top3_conc <= TOP3_CONC_MAX
            and p123 <= P123_MAX
            and rank1_share >= RANK1_SHARE_MIN
        ):
            continue

        selected += 1
        race_hit = False
        base_race_hit = False

        for label, pat in (('236', (2, 3, 6)), ('146', (1, 4, 6))):
            c = pattern_combo(byrank, pat)
            total_stake += STAKE
            base_stake += STAKE
            d = bypat[label]
            d['tickets'] += 1
            d['stake_yen'] += STAKE
            if c in paid[rid]:
                pay = paid[rid][c]
                total_payout += pay
                base_payout += pay
                d['hits'] += 1
                d['payout_yen'] += pay
                race_hit = True
                base_race_hit = True

        p134 = pattern_prob(p, byrank, EXTRA_PATTERN)
        if p134 >= P134_ADD_MIN:
            added += 1
            c = pattern_combo(byrank, EXTRA_PATTERN)
            total_stake += STAKE
            d = bypat['134_conditional']
            d['tickets'] += 1
            d['stake_yen'] += STAKE
            if c in paid[rid]:
                pay = paid[rid][c]
                total_payout += pay
                d['hits'] += 1
                d['payout_yen'] += pay
                race_hit = True

        if base_race_hit:
            base_hit_races += 1
        if race_hit:
            hit_races += 1
        base_race_hits.append(base_race_hit)
        race_hits.append(race_hit)

    for d in bypat.values():
        d['profit_yen'] = d['payout_yen'] - d['stake_yen']
        d['roi_pct'] = 100 * d['payout_yen'] / d['stake_yen'] if d['stake_yen'] else None

    base_result = {
        'selected_races': selected,
        'tickets': selected * 2,
        'hit_races': base_hit_races,
        'race_hit_rate_pct': 100 * base_hit_races / selected if selected else None,
        'stake_yen': base_stake,
        'payout_yen': base_payout,
        'profit_yen': base_payout - base_stake,
        'roi_pct': 100 * base_payout / base_stake if base_stake else None,
        'max_losing_streak': max_losing_streak(base_race_hits),
    }

    result = {
        'total_analyzable_races': analyzable,
        'selected_races': selected,
        'selection_rate_pct': 100 * selected / analyzable if analyzable else None,
        'base_tickets': selected * 2,
        'added_134_tickets': added,
        'add_rate_pct': 100 * added / selected if selected else None,
        'total_tickets': selected * 2 + added,
        'hit_races': hit_races,
        'race_hit_rate_pct': 100 * hit_races / selected if selected else None,
        'stake_yen': total_stake,
        'payout_yen': total_payout,
        'profit_yen': total_payout - total_stake,
        'roi_pct': 100 * total_payout / total_stake if total_stake else None,
        'max_losing_streak': max_losing_streak(race_hits),
        'by_pattern': bypat,
        'base_236_146_comparison': base_result,
    }

    out = {
        'status': 'FROZEN_236_146_CONDITIONAL_134_2026_H1_FORWARD_CHECK',
        'period': {'start_date': '2026-01-01', 'end_date': '2026-06-30'},
        'years_read': [2026],
        'gate': {
            'entropy_min': ENTROPY_MIN,
            'top3_conc_max': TOP3_CONC_MAX,
            'p123_max': P123_MAX,
            'rank1_share_min': RANK1_SHARE_MIN,
        },
        'strategy': {
            'base_patterns': BASE_PATTERNS,
            'base_stake_each_yen': STAKE,
            'conditional_extra_pattern': EXTRA_PATTERN,
            'conditional_p134_min': P134_ADD_MIN,
            'conditional_extra_stake_yen': STAKE,
        },
        'payout_method': 'Actual published 3連複 payout_yen from data/2026_h1/s_class_yosen/payouts.csv.',
        'odds_method': 'Archived KDreams final 3連複 odds; normalized inverse-odds mass used exactly as in 2023-2025 development.',
        'result': result,
        'warning': 'Rule frozen before this strategy result was read. However, 2026 H1 has been used elsewhere in this repository for other strategy evaluations, so this is a fixed-strategy forward check, not a globally pristine untouched OOS period.',
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

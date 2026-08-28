from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'early_candidate_v0'
THRESHOLD = 22.35


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def line_map(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get('line_id', '').isdigit() and e.get('line_position', '').isdigit():
            by[int(e['line_id'])].append(e)
    for lid in by:
        by[lid].sort(key=lambda e: int(e['line_position']))
    return by


def pair_strength(mem: list[dict[str, str]]) -> float:
    if len(mem) < 2:
        return float('-inf')
    return num(mem[0].get('score', '')) + num(mem[1].get('score', ''))


def strongest_rival(entries: list[dict[str, str]], main_id: int) -> list[dict[str, str]] | None:
    by = line_map(entries)
    cands = [(pair_strength(mem), -lid, mem) for lid, mem in by.items() if lid != main_id and len(mem) >= 2]
    if not cands:
        return None
    return max(cands, key=lambda x: (x[0], x[1]))[2]


def formation(a: int, b: int, r1l: int, r1b: int) -> list[str]:
    return [
        f'{r1l}-{r1b}-{a}',
        f'{r1l}-{r1b}-{b}',
        f'{r1b}-{r1l}-{a}',
        f'{r1b}-{r1l}-{b}',
    ]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    stake = sum(int(r['stake_yen']) for r in rows)
    payout = sum(int(r['payout_yen']) for r in rows)
    return {
        'races': len(rows),
        'hits': sum(int(r['hit']) for r in rows),
        'hit_rate': sum(int(r['hit']) for r in rows) / len(rows) if rows else 0.0,
        'stake_yen': stake,
        'payout_yen': payout,
        'profit_yen': payout - stake,
        'roi': payout / stake if stake else 0.0,
        'rival_pair_top2_races': sum(int(r['rival_pair_top2']) for r in rows),
        'rival_pair_top2_rate': sum(int(r['rival_pair_top2']) for r in rows) / len(rows) if rows else 0.0,
    }


def main() -> None:
    races = read_csv(DATA_DIR / 'races.csv')
    entries = read_csv(DATA_DIR / 'entries.csv')
    results = read_csv(DATA_DIR / 'results.csv')
    payouts = read_csv(DATA_DIR / 'payouts.csv')

    entries_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    results_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    trifecta: dict[str, dict[str, int]] = defaultdict(dict)

    for e in entries:
        entries_by[e['race_id']].append(e)
    for r in results:
        results_by[r['race_id']].append(r)
    for r in races:
        groups[(r['race_date'], r['track'])].append(r)
    for p in payouts:
        if p.get('ticket_type') == '3連単' and p.get('status') == 'paid' and p.get('combination'):
            try:
                trifecta[p['race_id']][p['combination']] = int(p['payout_yen'])
            except ValueError:
                pass

    segment: dict[str, str] = {}
    for g in groups.values():
        g.sort(key=lambda x: int(x['race_no']))
        n = len(g)
        for pos, r in enumerate(g, 1):
            segment[r['race_id']] = segment_for(pos, n)

    rows: list[dict[str, object]] = []
    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        if segment.get(rid) != '前半':
            continue
        main = choose_main_line(entries_by[rid])
        if not main:
            continue
        main_id, mem = main
        if len(mem) < 3:
            continue
        rival = strongest_rival(entries_by[rid], main_id)
        if not rival or len(rival) < 2:
            continue

        a = int(mem[0]['car_no'])
        b = int(mem[1]['car_no'])
        r1l = int(rival[0]['car_no'])
        r1b = int(rival[1]['car_no'])
        pair_win = (num(mem[0].get('win_rate', '')) if num(mem[0].get('win_rate', '')) != float('-inf') else 0.0) + (num(mem[1].get('win_rate', '')) if num(mem[1].get('win_rate', '')) != float('-inf') else 0.0)
        if pair_win > THRESHOLD:
            continue

        bets = formation(a, b, r1l, r1b)
        returned = 0
        hit_combos: list[str] = []
        for combo in bets:
            p = trifecta[rid].get(combo, 0)
            returned += p
            if p:
                hit_combos.append(combo)

        result_map = {int(r['car_no']): r.get('finish_position', '') for r in results_by[rid]}
        rival_pair_top2 = {str(result_map.get(r1l, '')), str(result_map.get(r1b, ''))} == {'1', '2'}
        stake = 100 * len(bets)
        rows.append({
            'race_id': rid,
            'race_date': race['race_date'],
            'half': 'H1' if race['race_date'] <= '2025-06-30' else 'H2',
            'track': race['track'],
            'race_no': race['race_no'],
            'main_line_id': main_id,
            'a_car': a,
            'b_car': b,
            'r1l_car': r1l,
            'r1b_car': r1b,
            'main_pair_win_sum': pair_win,
            'rival_pair_top2': 1 if rival_pair_top2 else 0,
            'bet_count': len(bets),
            'stake_yen': stake,
            'payout_yen': returned,
            'profit_yen': returned - stake,
            'hit': 1 if returned else 0,
            'hit_combinations': '/'.join(hit_combos),
        })

    h1 = [r for r in rows if r['half'] == 'H1']
    h2 = [r for r in rows if r['half'] == 'H2']
    summary = {
        'scope': '2025 exact S級予選, 前半, main line size >=3, strongest rival 2+ exists',
        'status': 'exploratory candidate only; not frozen V1',
        'pre_race_rule': f'main-line A+B win-rate sum <= {THRESHOLD}',
        'rule_origin': 'threshold selected from H1 structural classification of strongest-rival pair occupying 1st/2nd; payout not used to select the race condition',
        'bets': ['R1L-R1B-A', 'R1L-R1B-B', 'R1B-R1L-A', 'R1B-R1L-B'],
        'stake_per_race_yen': 400,
        'all': summarize(rows),
        'H1': summarize(h1),
        'H2': summarize(h2),
        'warning': 'Exploratory within 2025. Results are structural labels and payouts are evaluation only. Another year is required for true out-of-sample validation.',
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (OUT_DIR / 'races.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_branching_v1 import classify, mainline_bets
from .analyze_rough_pruning_v1 import CANDIDATES as ROUGH_CANDIDATES, combos as rough_combos

ROOT = Path('data')
OUT = ROOT / 'audits' / 'v2_middle_late_structure.json'
YEARS = (2024, 2025)


def read_csv(path: Path) -> list[dict[str, str]]:
    if 'payout' in path.name.lower():
        raise RuntimeError('Payout files are forbidden in structure-only V2 development.')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def metric(e: dict[str, str], key: str) -> float:
    v = num(e.get(key, ''))
    return 0.0 if v == float('-inf') else float(v)


def line_map(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get('line_id', '').isdigit() and e.get('line_position', '').isdigit():
            by[int(e['line_id'])].append(e)
    for lid in by:
        by[lid].sort(key=lambda e: int(e['line_position']))
    return by


def strongest_rival(entries: list[dict[str, str]], main_id: int) -> list[dict[str, str]] | None:
    cands = []
    for lid, members in line_map(entries).items():
        if lid == main_id or len(members) < 2:
            continue
        pair = metric(members[0], 'score') + metric(members[1], 'score')
        lead = metric(members[0], 'score')
        cands.append(((pair, lead, -lid), members))
    return max(cands, key=lambda x: x[0])[1] if cands else None


def normal_top3(results: list[dict[str, str]]) -> list[int] | None:
    by_pos: dict[int, list[int]] = defaultdict(list)
    for r in results:
        fp = r.get('finish_position', '')
        if fp in {'1', '2', '3'}:
            by_pos[int(fp)].append(int(r['car_no']))
    if any(len(by_pos[p]) != 1 for p in (1, 2, 3)):
        return None
    return [by_pos[1][0], by_pos[2][0], by_pos[3][0]]


def role_maps(entries: list[dict[str, str]], main_id: int, main: list[dict[str, str]], rival: list[dict[str, str]]) -> tuple[dict[str, int], dict[int, str]]:
    roles = {
        'A': int(main[0]['car_no']),
        'B': int(main[1]['car_no']),
        'M3': int(main[2]['car_no']),
        'R1L': int(rival[0]['car_no']),
        'R1B': int(rival[1]['car_no']),
    }
    car_role = {v: k for k, v in roles.items()}
    by = line_map(entries)
    for lid, members in by.items():
        for e in members:
            car = int(e['car_no'])
            if car in car_role:
                continue
            pos = int(e['line_position']) if e.get('line_position', '').isdigit() else 0
            if lid == main_id:
                car_role[car] = f'M{pos}'
            else:
                car_role[car] = f'O{pos}'
    return roles, car_role


def build_year(year: int) -> list[dict[str, object]]:
    d = ROOT / str(year) / 's_class_yosen'
    races = read_csv(d / 'races.csv')
    entries = read_csv(d / 'entries.csv')
    results = read_csv(d / 'results.csv')
    eb: dict[str, list[dict[str, str]]] = defaultdict(list)
    rb: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for e in entries: eb[e['race_id']].append(e)
    for r in results: rb[r['race_id']].append(r)
    for r in races: groups[(r['race_date'], r['track'])].append(r)
    seg: dict[str, str] = {}
    for g in groups.values():
        g.sort(key=lambda r: int(r['race_no']))
        for pos, r in enumerate(g, 1):
            seg[r['race_id']] = segment_for(pos, len(g))

    rows = []
    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        chosen = choose_main_line(eb[rid])
        if not chosen:
            continue
        main_id, main = chosen
        if len(main) < 3:
            continue
        rival = strongest_rival(eb[rid], main_id)
        if not rival or len(rival) < 2:
            continue
        top3 = normal_top3(rb[rid])
        if not top3:
            continue
        roles, car_role = role_maps(eb[rid], main_id, main, rival)
        seq = [car_role.get(c, 'OTHER') for c in top3]
        a, b, m3, r1l, r1b = main[0], main[1], main[2], rival[0], rival[1]
        row: dict[str, object] = {
            'year': year, 'race_id': rid, 'segment': seg[rid], 'race_date': race['race_date'],
            'roles': roles, 'role_seq': seq, 'top3': top3,
            'AB_top2': int(set(top3[:2]) == {roles['A'], roles['B']}),
            'R1_top2': int(set(top3[:2]) == {roles['R1L'], roles['R1B']}),
            'A_win': int(top3[0] == roles['A']), 'R1L_win': int(top3[0] == roles['R1L']),
            'A_score': metric(a, 'score'), 'B_score': metric(b, 'score'), 'M3_score': metric(m3, 'score'),
            'R1L_score': metric(r1l, 'score'), 'R1B_score': metric(r1b, 'score'),
            'pair_score_gap': metric(a, 'score') + metric(b, 'score') - metric(r1l, 'score') - metric(r1b, 'score'),
            'leader_score_gap': metric(a, 'score') - metric(r1l, 'score'),
            'second_score_gap': metric(b, 'score') - metric(r1b, 'score'),
            'pair_win': metric(a, 'win_rate') + metric(b, 'win_rate'),
            'pair_top2': metric(a, 'top2_rate') + metric(b, 'top2_rate'),
            'pair_top3': metric(a, 'top3_rate') + metric(b, 'top3_rate'),
            'A_top2': metric(a, 'top2_rate'), 'B_top2': metric(b, 'top2_rate'),
            'A_top3': metric(a, 'top3_rate'), 'B_top3': metric(b, 'top3_rate'), 'M3_top3': metric(m3, 'top3_rate'),
            'R1L_top3': metric(r1l, 'top3_rate'), 'R1B_top3': metric(r1b, 'top3_rate'),
        }
        if seg[rid] == '後半':
            branch, *_ = classify(main)
            row['late_branch'] = branch
            if branch == 'A_mainline':
                row['current_mainline_bets'] = mainline_bets(eb[rid], main)
        rows.append(row)
    return rows


FEATURES = [
    'A_score','B_score','R1L_score','R1B_score','pair_score_gap','leader_score_gap','second_score_gap',
    'pair_win','pair_top2','pair_top3','A_top2','B_top2','A_top3','B_top3','M3_top3','R1L_top3','R1B_top3'
]


def natural_thresholds(feature: str, rows: list[dict[str, object]]) -> list[float]:
    vals = [float(r[feature]) for r in rows]
    lo, hi = min(vals), max(vals)
    step = 1.0 if 'score' in feature else 5.0
    start = int(lo // step) * step
    t = start
    out = []
    while t <= hi:
        out.append(round(t, 2)); t += step
    return out


def split_stats(rows: list[dict[str, object]], feature: str, threshold: float, label: str) -> dict[str, object] | None:
    left = [r for r in rows if float(r[feature]) <= threshold]
    right = [r for r in rows if float(r[feature]) > threshold]
    if min(len(left), len(right)) < 30:
        return None
    lr = sum(int(r[label]) for r in left) / len(left)
    rr = sum(int(r[label]) for r in right) / len(right)
    return {'threshold': threshold, 'left_n': len(left), 'right_n': len(right), 'left_rate': lr, 'right_rate': rr, 'gap': rr-lr}


def stable_screen(rows_by_year: dict[int, list[dict[str, object]]], label: str) -> list[dict[str, object]]:
    all_rows = rows_by_year[2024] + rows_by_year[2025]
    cands = []
    for feature in FEATURES:
        for t in natural_thresholds(feature, all_rows):
            a = split_stats(rows_by_year[2024], feature, t, label)
            b = split_stats(rows_by_year[2025], feature, t, label)
            if not a or not b:
                continue
            if a['gap'] == 0 or b['gap'] == 0 or (a['gap'] > 0) != (b['gap'] > 0):
                continue
            min_gap = min(abs(float(a['gap'])), abs(float(b['gap'])))
            cands.append({'feature': feature, 'threshold': t, 'direction': 'high' if a['gap'] > 0 else 'low',
                          'min_abs_gap': min_gap, '2024': a, '2025': b})
    cands.sort(key=lambda x: (float(x['min_abs_gap']), min(int(x['2024']['left_n']), int(x['2024']['right_n']), int(x['2025']['left_n']), int(x['2025']['right_n']))), reverse=True)
    return cands[:20]


def role_sequence_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    c = Counter('-'.join(map(str, r['role_seq'])) for r in rows)
    return {'races': len(rows), 'top_sequences': [{'seq': k, 'count': v, 'rate': v/len(rows)} for k,v in c.most_common(20)] if rows else []}


def basic_rates(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    return {
        'races': n,
        'AB_top2_rate': sum(int(r['AB_top2']) for r in rows)/n if n else 0,
        'R1_top2_rate': sum(int(r['R1_top2']) for r in rows)/n if n else 0,
        'A_win_rate': sum(int(r['A_win']) for r in rows)/n if n else 0,
        'R1L_win_rate': sum(int(r['R1L_win']) for r in rows)/n if n else 0,
    }


def exact_order_hit(rows: list[dict[str, object]], p1: list[str], p2: list[str], p3: list[str]) -> dict[str, object]:
    hits = 0; total_points = 0
    for r in rows:
        roles = r['roles']
        s1 = [roles[x] for x in p1 if x in roles]
        s2 = [roles[x] for x in p2 if x in roles]
        s3 = [roles[x] for x in p3 if x in roles]
        cs = {(a,b,c) for a in s1 for b in s2 for c in s3 if len({a,b,c}) == 3}
        total_points += len(cs)
        if tuple(r['top3']) in cs:
            hits += 1
    return {'races': len(rows), 'hits': hits, 'hit_rate': hits/len(rows) if rows else 0,
            'avg_points': total_points/len(rows) if rows else 0}


def late_mainline_third(rows: list[dict[str, object]]) -> dict[str, object]:
    ab = [r for r in rows if int(r['AB_top2'])]
    c = Counter(str(r['role_seq'][2]) for r in ab)
    return {'AB_top2_races': len(ab), 'third_roles': [{'role': k, 'count': v, 'rate': v/len(ab)} for k,v in c.most_common()] if ab else []}


def current_mainline_structural_hit(rows: list[dict[str, object]]) -> dict[str, object]:
    hits = 0
    for r in rows:
        bets = set(str(x) for x in r.get('current_mainline_bets', []))
        combo = '-'.join(map(str, r['top3']))
        hits += int(combo in bets)
    return {'races': len(rows), 'hits': hits, 'hit_rate': hits/len(rows) if rows else 0, 'points': 4}


def rough_candidate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for name, (p1,p2,p3) in ROUGH_CANDIDATES.items():
        m = exact_order_hit(rows, p1,p2,p3)
        out.append({'name': name, **m, 'formation': [p1,p2,p3]})
    out.sort(key=lambda x: (float(x['hit_rate']), -float(x['avg_points'])), reverse=True)
    return out


def main() -> None:
    rows = {y: build_year(y) for y in YEARS}
    out: dict[str, object] = {
        'scope': 'V2 development structure audit using 2024+2025 only; selection uses results as structural labels, never payouts.',
        'payout_files_read': False,
        'validation_plan': 'After V2 is frozen, use 2023 exact-label S級予選 as fresh OOS. Do not alter V2 after reading 2023.',
        'middle': {}, 'late': {},
    }

    mid = {y: [r for r in rows[y] if r['segment'] == '中盤'] for y in YEARS}
    out['middle'] = {
        'basic': {str(y): basic_rates(mid[y]) for y in YEARS},
        'role_sequences': {str(y): role_sequence_summary(mid[y]) for y in YEARS},
        'stable_AB_top2_screens': stable_screen(mid, 'AB_top2'),
        'stable_R1_top2_screens': stable_screen(mid, 'R1_top2'),
        'note': 'Do not preserve B>105 by default. New filters must show the same structural direction in both 2024 and 2025 on coarse natural thresholds.',
    }

    late_all = {y: [r for r in rows[y] if r['segment'] == '後半'] for y in YEARS}
    late_a = {y: [r for r in late_all[y] if r.get('late_branch') == 'A_mainline'] for y in YEARS}
    late_b = {y: [r for r in late_all[y] if r.get('late_branch') == 'B_rough'] for y in YEARS}
    out['late'] = {
        'all_basic': {str(y): basic_rates(late_all[y]) for y in YEARS},
        'mainline_branch': {
            'basic': {str(y): basic_rates(late_a[y]) for y in YEARS},
            'third_after_AB_top2': {str(y): late_mainline_third(late_a[y]) for y in YEARS},
            'current_4pt_structural_hit': {str(y): current_mainline_structural_hit(late_a[y]) for y in YEARS},
        },
        'rough_branch': {
            'basic': {str(y): basic_rates(late_b[y]) for y in YEARS},
            'role_sequences': {str(y): role_sequence_summary(late_b[y]) for y in YEARS},
            'predeclared_pruning_candidates': {str(y): rough_candidate_summary(late_b[y]) for y in YEARS},
        },
        'routing_note': 'Frozen late routing is retained as a candidate because its AB-vs-other direction survived temporal H2 and 2024; ticket construction is re-developed separately.',
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

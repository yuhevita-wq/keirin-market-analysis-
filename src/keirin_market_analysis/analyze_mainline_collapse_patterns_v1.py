from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, segment_for

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'mainline_collapse_patterns_v1'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def line_maps(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get('line_id', '').isdigit() and e.get('line_position', '').isdigit():
            by_line[int(e['line_id'])].append(e)
    for lid in by_line:
        by_line[lid].sort(key=lambda e: int(e['line_position']))
    return by_line


def main() -> None:
    races = read_csv(DATA_DIR / 'races.csv')
    entries = read_csv(DATA_DIR / 'entries.csv')
    results = read_csv(DATA_DIR / 'results.csv')

    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    results_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    races_by_day_track: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        entries_by_race[e['race_id']].append(e)
    for r in results:
        results_by_race[r['race_id']].append(r)
    for r in races:
        races_by_day_track[(r['race_date'], r['track'])].append(r)

    segment_by_race: dict[str, str] = {}
    for group in races_by_day_track.values():
        group.sort(key=lambda r: int(r['race_no']))
        total = len(group)
        for pos, r in enumerate(group, 1):
            segment_by_race[r['race_id']] = segment_for(pos, total)

    main_survivors = Counter()
    ab_survival = Counter()
    first_second_relation = Counter()
    top3_line_shape = Counter()
    rival_top3_count = Counter()
    main3_presence = Counter()
    winner_role = Counter()
    second_role = Counter()
    third_role = Counter()
    branch_b = Counter()
    branch_b_ab_survival = Counter()
    branch_b_first_second_relation = Counter()
    branch_b_top3_line_shape = Counter()
    branch_b_rival_top3_count = Counter()

    rows = []
    total_scope = 0
    collapse = 0
    branch_b_total = 0
    branch_b_collapse = 0

    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        if segment_by_race.get(rid) != '後半':
            continue
        es = entries_by_race[rid]
        main = choose_main_line(es)
        if not main:
            continue
        main_id, main_members = main
        if len(main_members) < 3:
            continue
        total_scope += 1
        by_line = line_maps(es)
        a = int(main_members[0]['car_no'])
        b = int(main_members[1]['car_no'])
        c = int(main_members[2]['car_no'])

        # strongest rival 2+ line by pair score, same concept as prior analysis
        rival_candidates = []
        for lid, mem in by_line.items():
            if lid == main_id or len(mem) < 2:
                continue
            try:
                strength = float(mem[0]['score']) + float(mem[1]['score'])
            except Exception:
                strength = float('-inf')
            rival_candidates.append((strength, lid, mem))
        rival_id = max(rival_candidates, default=(float('-inf'), -1, []), key=lambda x: (x[0], -x[1]))[1]

        finish = {}
        for rr in results_by_race[rid]:
            fp = rr.get('finish_position', '')
            if fp in {'1','2','3'}:
                finish[int(rr['car_no'])] = int(fp)
        top3 = [car for car, pos in sorted(finish.items(), key=lambda kv: kv[1])]
        if len(top3) < 3:
            continue
        top3 = top3[:3]
        pos_by_car = {car: i+1 for i, car in enumerate(top3)}
        main_top2 = {pos_by_car.get(a), pos_by_car.get(b)} == {1,2}
        if main_top2:
            continue
        collapse += 1

        entry_by_car = {int(e['car_no']): e for e in es}
        top3_lines = [int(entry_by_car[x]['line_id']) if entry_by_car[x].get('line_id','').isdigit() else -1 for x in top3]
        main_count = sum(1 for lid in top3_lines if lid == main_id)
        rival_count = sum(1 for lid in top3_lines if lid == rival_id)
        main_survivors[main_count] += 1

        a_in = a in top3; b_in = b in top3
        if a_in and b_in: ab = 'A+B both top3'
        elif a_in: ab = 'A only top3'
        elif b_in: ab = 'B only top3'
        else: ab = 'A/B neither top3'
        ab_survival[ab] += 1

        if top3_lines[0] == top3_lines[1]: rel = '1-2 same line'
        else: rel = '1-2 different lines'
        first_second_relation[rel] += 1

        counts = Counter(top3_lines)
        shape = '+'.join(map(str, sorted(counts.values(), reverse=True)))
        top3_line_shape[shape] += 1
        rival_top3_count[rival_count] += 1
        main3_presence['main3 in top3' if c in top3 else 'main3 out'] += 1

        def role(car: int) -> str:
            e = entry_by_car[car]
            lid = int(e['line_id']) if e.get('line_id','').isdigit() else -1
            lp = int(e['line_position']) if e.get('line_position','').isdigit() else 0
            if lid == main_id:
                if car == a: return 'main leader'
                if car == b: return 'main second'
                if car == c: return 'main third'
                return 'main other'
            if lid == rival_id:
                return f'rival pos{lp}'
            return f'other pos{lp}'

        winner_role[role(top3[0])] += 1
        second_role[role(top3[1])] += 1
        third_role[role(top3[2])] += 1

        # Branch B rule from routing_v0
        # main_pair_top3_sum <= 106.1 and main_second_top3 <= 35.7 and not branch A
        try:
            leader_top3 = float(main_members[0].get('top3_rate','') or 0)
            second_top3 = float(main_members[1].get('top3_rate','') or 0)
            leader_win = float(main_members[0].get('win_rate','') or 0)
            second_win = float(main_members[1].get('win_rate','') or 0)
        except ValueError:
            leader_top3 = second_top3 = leader_win = second_win = 0.0
        pair_top3 = leader_top3 + second_top3
        pair_win = leader_win + second_win
        branch_a = pair_top3 > 106.1 and pair_win > 54.7
        is_branch_b = (not branch_a) and pair_top3 <= 106.1 and second_top3 <= 35.7
        if is_branch_b:
            branch_b_total += 1
            branch_b_collapse += 1
            branch_b_ab_survival[ab] += 1
            branch_b_first_second_relation[rel] += 1
            branch_b_top3_line_shape[shape] += 1
            branch_b_rival_top3_count[rival_count] += 1

        rows.append({
            'race_id': rid, 'race_date': race['race_date'], 'track': race['track'], 'race_no': race['race_no'],
            'a_car': a, 'b_car': b, 'main3_car': c, 'top3': '-'.join(map(str, top3)),
            'main_top3_count': main_count, 'ab_survival': ab, 'first_second_relation': rel,
            'top3_line_shape': shape, 'rival_top3_count': rival_count,
            'winner_role': role(top3[0]), 'second_role': role(top3[1]), 'third_role': role(top3[2]),
            'branch_b': 1 if is_branch_b else 0,
        })

    out = {
        'scope': '2025 exact S級予選, 後半, main line size >=3, mainline A/B NOT first-second',
        'scope_races_before_label_filter': total_scope,
        'collapse_races': collapse,
        'main_line_riders_in_top3': dict(sorted(main_survivors.items())),
        'A_B_survival': dict(ab_survival),
        'first_second_line_relation': dict(first_second_relation),
        'top3_line_count_shape': dict(top3_line_shape),
        'strongest_rival_riders_in_top3': dict(sorted(rival_top3_count.items())),
        'main3_presence': dict(main3_presence),
        'winner_role': dict(winner_role.most_common()),
        'second_role': dict(second_role.most_common()),
        'third_role': dict(third_role.most_common()),
        'branch_B_collapse_subset': {
            'collapse_races': branch_b_collapse,
            'A_B_survival': dict(branch_b_ab_survival),
            'first_second_line_relation': dict(branch_b_first_second_relation),
            'top3_line_count_shape': dict(branch_b_top3_line_shape),
            'strongest_rival_riders_in_top3': dict(sorted(branch_b_rival_top3_count.items())),
        },
        'note': 'Structural outcome audit only. Branch features are pre-race; finish data are labels and are not used to define the branch.'
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    fields = list(rows[0].keys()) if rows else []
    with (OUT_DIR / 'races.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

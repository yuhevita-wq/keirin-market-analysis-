from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'rough_branch_candidates_v1'


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


def branch_b(main_members: list[dict[str, str]]) -> bool:
    leader_top3 = num(main_members[0].get('top3_rate', ''))
    second_top3 = num(main_members[1].get('top3_rate', ''))
    leader_win = num(main_members[0].get('win_rate', ''))
    second_win = num(main_members[1].get('win_rate', ''))
    pair_top3 = leader_top3 + second_top3
    pair_win = leader_win + second_win
    branch_a = pair_top3 > 106.1 and pair_win > 54.7
    return (not branch_a) and pair_top3 <= 106.1 and second_top3 <= 35.7


def role_map(entries: list[dict[str, str]], main_id: int, main_members: list[dict[str, str]]) -> dict[str, int]:
    by_line = line_maps(entries)
    a = int(main_members[0]['car_no'])
    b = int(main_members[1]['car_no'])
    c = int(main_members[2]['car_no'])

    rivals: list[tuple[float, float, int, list[dict[str, str]]]] = []
    for lid, mem in by_line.items():
        if lid == main_id or len(mem) < 2:
            continue
        strength = num(mem[0].get('score', '')) + num(mem[1].get('score', ''))
        leader_score = num(mem[0].get('score', ''))
        rivals.append((strength, leader_score, -lid, mem))
    rivals.sort(key=lambda x: x[:3], reverse=True)

    out = {'A': a, 'B': b, 'M3': c}
    if rivals:
        out['R1L'] = int(rivals[0][3][0]['car_no'])
        out['R1B'] = int(rivals[0][3][1]['car_no'])
    if len(rivals) >= 2:
        out['R2L'] = int(rivals[1][3][0]['car_no'])
        out['R2B'] = int(rivals[1][3][1]['car_no'])
    return out


def normal_top3(results: list[dict[str, str]]) -> tuple[list[int], bool]:
    by_pos: dict[int, list[int]] = defaultdict(list)
    for r in results:
        fp = r.get('finish_position', '')
        if fp in {'1', '2', '3'}:
            by_pos[int(fp)].append(int(r['car_no']))
    if any(len(by_pos[p]) != 1 for p in (1, 2, 3)):
        return [], False
    return [by_pos[1][0], by_pos[2][0], by_pos[3][0]], True


def candidate_set(role_to_car: dict[str, int], roles: list[str]) -> set[int]:
    return {role_to_car[r] for r in roles if r in role_to_car}


def formation_combos(role_to_car: dict[str, int], p1: list[str], p2: list[str], p3: list[str]) -> set[tuple[int, int, int]]:
    s1 = candidate_set(role_to_car, p1)
    s2 = candidate_set(role_to_car, p2)
    s3 = candidate_set(role_to_car, p3)
    return {(a,b,c) for a in s1 for b in s2 for c in s3 if len({a,b,c}) == 3}


def main() -> None:
    races = read_csv(DATA_DIR / 'races.csv')
    entries = read_csv(DATA_DIR / 'entries.csv')
    results = read_csv(DATA_DIR / 'results.csv')

    e_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    r_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    day_track: dict[tuple[str,str], list[dict[str,str]]] = defaultdict(list)
    for e in entries: e_by_race[e['race_id']].append(e)
    for r in results: r_by_race[r['race_id']].append(r)
    for r in races: day_track[(r['race_date'], r['track'])].append(r)

    segment = {}
    for group in day_track.values():
        group.sort(key=lambda x: int(x['race_no']))
        total = len(group)
        for pos, race in enumerate(group,1): segment[race['race_id']] = segment_for(pos,total)

    set_defs = {
        'S3_A_R1': ['A','R1L','R1B'],
        'S4_plus_B': ['A','R1L','R1B','B'],
        'S4_plus_M3': ['A','R1L','R1B','M3'],
        'S4_plus_R2L': ['A','R1L','R1B','R2L'],
        'S5_plus_B_M3': ['A','R1L','R1B','B','M3'],
        'S5_plus_B_R2L': ['A','R1L','R1B','B','R2L'],
        'S5_plus_M3_R2L': ['A','R1L','R1B','M3','R2L'],
        'S6_all': ['A','R1L','R1B','B','M3','R2L'],
    }
    form_defs = {
        'F12_compact': (['A','R1L'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B']),
        'F18_add_R2L_3rd': (['A','R1L'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B','R2L']),
        'F_wider_heads': (['A','R1L','R1B'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B','R2L']),
    }

    records = []
    for race in sorted(races, key=lambda x: (x['race_date'],x['track'],int(x['race_no']))):
        rid = race['race_id']
        if segment.get(rid) != '後半': continue
        es = e_by_race[rid]
        main = choose_main_line(es)
        if not main: continue
        main_id, main_members = main
        if len(main_members) < 3 or not branch_b(main_members): continue
        roles = role_map(es, main_id, main_members)
        top3, normal = normal_top3(r_by_race[rid])
        if not normal:
            records.append({'race_id':rid,'half':'H1' if race['race_date'] <= '2025-06-30' else 'H2','normal':False})
            continue
        a,b = roles['A'],roles['B']
        collapse = set(top3[:2]) != {a,b}
        car_to_role = {v:k for k,v in roles.items()}
        rec = {
            'race_id':rid,'half':'H1' if race['race_date'] <= '2025-06-30' else 'H2','normal':True,
            'collapse':collapse,'top3':top3,'roles':roles,
            'top3_roles':[car_to_role.get(c,'OTHER') for c in top3],
        }
        records.append(rec)

    def summarize(sub: list[dict]) -> dict:
        normal = [r for r in sub if r.get('normal')]
        out = {'races':len(sub),'normal_races':len(normal),'collapse_races':sum(bool(r.get('collapse')) for r in normal)}
        role_by_pos = {'1':Counter(),'2':Counter(),'3':Counter()}
        for r in normal:
            for i,role in enumerate(r['top3_roles'],1): role_by_pos[str(i)][role]+=1
        out['role_by_position'] = {p:dict(c.most_common()) for p,c in role_by_pos.items()}
        sets = {}
        for name,roles in set_defs.items():
            usable=[]; captured=0; collapse_usable=0; collapse_captured=0
            for r in normal:
                s=candidate_set(r['roles'],roles)
                if len(s) < 3: continue
                usable.append(r)
                hit=set(r['top3']).issubset(s)
                captured += int(hit)
                if r['collapse']:
                    collapse_usable += 1; collapse_captured += int(hit)
            sets[name]={'roles':roles,'usable_races':len(usable),'captured':captured,'capture_rate':captured/len(usable) if usable else 0,
                        'collapse_usable':collapse_usable,'collapse_captured':collapse_captured,'collapse_capture_rate':collapse_captured/collapse_usable if collapse_usable else 0}
        out['candidate_sets']=sets
        forms={}
        for name,(p1,p2,p3) in form_defs.items():
            usable=hit=collapse_usable=collapse_hit=0; total_points=0
            for r in normal:
                combos=formation_combos(r['roles'],p1,p2,p3)
                if not combos: continue
                usable += 1; total_points += len(combos)
                actual=tuple(r['top3'])
                ok=actual in combos; hit += int(ok)
                if r['collapse']:
                    collapse_usable += 1; collapse_hit += int(ok)
            forms[name]={'p1':p1,'p2':p2,'p3':p3,'usable_races':usable,'avg_points':total_points/usable if usable else 0,
                         'hit_races':hit,'hit_rate':hit/usable if usable else 0,'collapse_hit_races':collapse_hit,
                         'collapse_hit_rate':collapse_hit/collapse_usable if collapse_usable else 0}
        out['formations']=forms
        return out

    summary={
        'scope':'2025 exact S級予選, 後半, main line size >=3, branch B rough-lean only',
        'definition':'Branch B is fixed from routing_v0; this analysis uses finish only as outcome labels.',
        'all':summarize(records),
        'H1':summarize([r for r in records if r['half']=='H1']),
        'H2':summarize([r for r in records if r['half']=='H2']),
        'warning':'Exploratory structural analysis. Prefer candidate rules that retain similar direction in H1 and H2; do not choose on payout here.'
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()

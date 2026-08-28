from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path('data/2025/s_class_yosen')
SIM_DIR = DATA_DIR / 'simulations' / 'mainline_v1'
OUT = SIM_DIR / 'third_slots_by_mainline_size_v1.json'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def val(x: object) -> float:
    if x is None:
        return float('-inf')
    m = re.search(r'-?\d+(?:\.\d+)?', str(x).replace(',', ''))
    return float(m.group()) if m else float('-inf')


def line_maps(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get('line_id', '').isdigit() and e.get('line_position', '').isdigit():
            by_line[int(e['line_id'])].append(e)
    for lid in by_line:
        by_line[lid].sort(key=lambda e: int(e['line_position']))
    return by_line


def strongest_rival(by_line: dict[int, list[dict[str, str]]], main_line_id: int):
    rivals = []
    for lid, members in by_line.items():
        if lid == main_line_id or not members:
            continue
        strength = val(members[0].get('score'))
        if len(members) >= 2:
            strength += val(members[1].get('score'))
        rivals.append((strength, -lid, members))
    return max(rivals, default=None, key=lambda x: (x[0], x[1]))


def top_by(rows: list[dict[str, str]], metric: str, n: int) -> list[int]:
    ranked = sorted(rows, key=lambda e: (-val(e.get(metric)), int(e['car_no'])))
    return [int(e['car_no']) for e in ranked[:n]]


def one_external_rules(external: list[dict[str, str]], by_line, main_line_id: int) -> dict[str, list[int]]:
    rules: dict[str, list[int]] = {}
    for metric in ('score','top3_rate','top2_rate','win_rate','third_count','mark_count','b_count'):
        rules[f'best_{metric}'] = top_by(external, metric, 1)
    rival = strongest_rival(by_line, main_line_id)
    if rival:
        members = [e for e in rival[2] if int(e['car_no']) in {int(x['car_no']) for x in external}]
        rules['strongest_rival_leader'] = [int(members[0]['car_no'])] if members else []
        rules['strongest_rival_best_score'] = top_by(members, 'score', 1) if members else []
        rules['strongest_rival_best_top3'] = top_by(members, 'top3_rate', 1) if members else []
    return rules


def two_external_rules(external: list[dict[str, str]], by_line, main_line_id: int) -> dict[str, list[int]]:
    rules: dict[str, list[int]] = {}
    for metric in ('score','top3_rate','top2_rate','win_rate','third_count','mark_count','b_count'):
        rules[f'top2_{metric}'] = top_by(external, metric, 2)
    rival = strongest_rival(by_line, main_line_id)
    if rival:
        members = [e for e in rival[2] if int(e['car_no']) in {int(x['car_no']) for x in external}]
        rival_cars = [int(e['car_no']) for e in members[:2]]
        rules['strongest_rival_first2'] = rival_cars
        best_rival = top_by(members, 'score', 1) if members else []
        remaining = [e for e in external if int(e['car_no']) not in set(best_rival)]
        rules['best_rival_plus_best_score_elsewhere'] = best_rival + top_by(remaining, 'score', 1)
    return rules


def main() -> None:
    entries = read_csv(DATA_DIR / 'entries.csv')
    sim = read_csv(SIM_DIR / 'race_simulation.csv')
    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        entries_by_race[e['race_id']].append(e)

    target = [r for r in sim if r['segment'] == '後半' and r['mainline_top2'] == '1']
    cases = Counter()
    one_rule_hits = Counter()
    two_rule_hits = Counter()
    one_rule_h1 = Counter(); one_rule_h2 = Counter()
    two_rule_h1 = Counter(); two_rule_h2 = Counter()
    one_den_h1 = one_den_h2 = two_den_h1 = two_den_h2 = 0
    external_actual_position = Counter()
    external_actual_score_rank = Counter()
    external_actual_top3_rank = Counter()

    for r in target:
        rid = r['race_id']
        es = entries_by_race[rid]
        by_line = line_maps(es)
        main_line_id = int(r['main_line_id'])
        main_members = by_line[main_line_id]
        a, b = int(r['a_car']), int(r['b_car'])
        actual = int(r['actual_third'].split('-')[0])
        external = [e for e in es if int(e['line_id']) != main_line_id]
        half = 'H1' if r['race_date'] <= '2025-06-30' else 'H2'

        if len(main_members) >= 3:
            cases['main3plus'] += 1
            main3 = int(main_members[2]['car_no'])
            if actual == main3:
                cases['main3_hit'] += 1
            else:
                cases['main3_miss_external_third'] += 1
                one_rules = one_external_rules(external, by_line, main_line_id)
                for name, cars in one_rules.items():
                    if actual in cars:
                        one_rule_hits[name] += 1
                        (one_rule_h1 if half == 'H1' else one_rule_h2)[name] += 1
                if half == 'H1': one_den_h1 += 1
                else: one_den_h2 += 1

                actual_e = next(e for e in external if int(e['car_no']) == actual)
                pos = int(actual_e['line_position']) if actual_e.get('line_position','').isdigit() else 0
                external_actual_position[pos] += 1
                score_ranked = sorted(external, key=lambda e: (-val(e.get('score')), int(e['car_no'])))
                top3_ranked = sorted(external, key=lambda e: (-val(e.get('top3_rate')), int(e['car_no'])))
                external_actual_score_rank[next(i for i,e in enumerate(score_ranked,1) if int(e['car_no'])==actual)] += 1
                external_actual_top3_rank[next(i for i,e in enumerate(top3_ranked,1) if int(e['car_no'])==actual)] += 1
        elif len(main_members) == 2:
            cases['main2'] += 1
            two_rules = two_external_rules(external, by_line, main_line_id)
            for name, cars in two_rules.items():
                if actual in cars:
                    two_rule_hits[name] += 1
                    (two_rule_h1 if half == 'H1' else two_rule_h2)[name] += 1
            if half == 'H1': two_den_h1 += 1
            else: two_den_h2 += 1

    one_den = cases['main3_miss_external_third']
    two_den = cases['main2']
    main3base = cases['main3_hit']

    one_summary = {}
    for name in sorted(set(one_rule_hits)|set(one_rule_h1)|set(one_rule_h2)):
        incremental = one_rule_hits[name]
        one_summary[name] = {
            'external_capture': incremental,
            'external_capture_rate': incremental / one_den if one_den else 0,
            'combined_main3_plus_external_capture': main3base + incremental,
            'combined_capture_rate_among_main3plus': (main3base + incremental) / cases['main3plus'] if cases['main3plus'] else 0,
            'H1_external_capture_rate': one_rule_h1[name] / one_den_h1 if one_den_h1 else 0,
            'H2_external_capture_rate': one_rule_h2[name] / one_den_h2 if one_den_h2 else 0,
        }

    two_summary = {}
    for name in sorted(set(two_rule_hits)|set(two_rule_h1)|set(two_rule_h2)):
        two_summary[name] = {
            'capture': two_rule_hits[name],
            'capture_rate': two_rule_hits[name] / two_den if two_den else 0,
            'H1_capture_rate': two_rule_h1[name] / two_den_h1 if two_den_h1 else 0,
            'H2_capture_rate': two_rule_h2[name] / two_den_h2 if two_den_h2 else 0,
        }

    out = {
        'scope': '2025 exact S級予選, 後半, mainline A/B occupied first-second; split by mainline size; capture analysis only',
        'case_counts': dict(cases),
        'main3plus_external_third_position_counts': dict(sorted(external_actual_position.items())),
        'main3plus_external_third_score_rank_counts': dict(sorted(external_actual_score_rank.items())),
        'main3plus_external_third_top3_rate_rank_counts': dict(sorted(external_actual_top3_rank.items())),
        'main3plus_one_external_slot_rules': one_summary,
        'main2_two_external_slot_rules': two_summary,
        'half_denominators': {'main3plus_external_H1': one_den_h1, 'main3plus_external_H2': one_den_h2, 'main2_H1': two_den_h1, 'main2_H2': two_den_h2},
        'warning': 'Exploratory descriptive comparison. No payout/ROI used to choose rules. Small mainline-2 sample should be treated cautiously.'
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path('data/2025/s_class_yosen')
SIM_DIR = DATA_DIR / 'simulations' / 'mainline_v1'
OUT = SIM_DIR / 'third_predictor_analysis.json'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def val(x: object) -> float:
    if x is None:
        return float('-inf')
    s = str(x).replace(',', '')
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    return float(m.group()) if m else float('-inf')


def rank_top2(rows: list[dict[str, str]], metric: str) -> list[int]:
    ranked = sorted(rows, key=lambda r: (-val(r.get(metric)), int(r['car_no'])))
    return [int(r['car_no']) for r in ranked[:2]]


def line_maps(entries: list[dict[str, str]]):
    by_line: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get('line_id', '').isdigit() and e.get('line_position', '').isdigit():
            by_line[int(e['line_id'])].append(e)
    for lid in by_line:
        by_line[lid].sort(key=lambda e: int(e['line_position']))
    return by_line


def main() -> None:
    entries = read_csv(DATA_DIR / 'entries.csv')
    sim = read_csv(SIM_DIR / 'race_simulation.csv')
    entries_by_race: dict[str, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        entries_by_race[e['race_id']].append(e)

    target = [r for r in sim if r['segment'] == '後半' and r['mainline_top2'] == '1']
    metrics = ['score', 'top3_rate', 'top2_rate', 'win_rate', 'third_count', 'second_count', 'first_count', 'mark_count', 'b_count']
    metric_rank_counts = {m: Counter() for m in metrics}
    metric_top2_capture = Counter()

    category_exposure = Counter()
    category_hits = Counter()
    race_category_available = Counter()
    race_category_hits = Counter()
    mainline_size_counts = Counter()
    main3_exists = 0
    main3_hits = 0
    strongest_rival_line_exists = 0
    strongest_rival_line_hits = 0
    strongest_rival_member_top2_capture = 0

    for r in target:
        rid = r['race_id']
        es = entries_by_race[rid]
        a, b = int(r['a_car']), int(r['b_car'])
        actual_third = int(r['actual_third'].split('-')[0])
        candidates = [e for e in es if int(e['car_no']) not in {a, b}]

        for m in metrics:
            ranked = sorted(candidates, key=lambda e: (-val(e.get(m)), int(e['car_no'])))
            rank = next(i for i, e in enumerate(ranked, 1) if int(e['car_no']) == actual_third)
            metric_rank_counts[m][rank] += 1
            if actual_third in [int(e['car_no']) for e in ranked[:2]]:
                metric_top2_capture[m] += 1

        by_line = line_maps(es)
        main_line_id = int(r['main_line_id'])
        main_members = by_line.get(main_line_id, [])
        mainline_size_counts[len(main_members)] += 1

        # Candidate-level category exposure: denominator is how many eligible candidate slots existed.
        categories_present = set()
        for e in candidates:
            lid = int(e['line_id']) if e.get('line_id', '').isdigit() else None
            pos = int(e['line_position']) if e.get('line_position', '').isdigit() else None
            cats = []
            if lid == main_line_id and pos == 3:
                cats.append('本線3番手')
            elif lid != main_line_id and pos == 1:
                cats.append('他ライン先頭')
            elif lid != main_line_id and pos == 2:
                cats.append('他ライン番手')
            else:
                cats.append('その他')
            for c in cats:
                category_exposure[c] += 1
                categories_present.add(c)
                if int(e['car_no']) == actual_third:
                    category_hits[c] += 1
        for c in categories_present:
            race_category_available[c] += 1
        # actual third category hit at race level
        for c in categories_present:
            hit = False
            for e in candidates:
                lid = int(e['line_id']) if e.get('line_id', '').isdigit() else None
                pos = int(e['line_position']) if e.get('line_position', '').isdigit() else None
                ec = '本線3番手' if lid == main_line_id and pos == 3 else '他ライン先頭' if lid != main_line_id and pos == 1 else '他ライン番手' if lid != main_line_id and pos == 2 else 'その他'
                if ec == c and int(e['car_no']) == actual_third:
                    hit = True
            if hit:
                race_category_hits[c] += 1

        if len(main_members) >= 3:
            main3_exists += 1
            if int(main_members[2]['car_no']) == actual_third:
                main3_hits += 1

        # Strongest rival line by first-two score sum. For singleton, its own score.
        rivals = []
        for lid, members in by_line.items():
            if lid == main_line_id:
                continue
            score = val(members[0].get('score'))
            if len(members) >= 2:
                score += val(members[1].get('score'))
            rivals.append((score, lid, members))
        if rivals:
            strongest_rival_line_exists += 1
            _, _, members = max(rivals, key=lambda x: (x[0], -x[1]))
            cars = [int(e['car_no']) for e in members]
            if actual_third in cars:
                strongest_rival_line_hits += 1
            top2_members = sorted(members, key=lambda e: (-val(e.get('score')), int(e['car_no'])))[:2]
            if actual_third in [int(e['car_no']) for e in top2_members]:
                strongest_rival_member_top2_capture += 1

    n = len(target)
    out = {
        'scope': '2025 exact S級予選, 後半, mainline A/B occupied first-second; descriptive pre-race predictor audit',
        'races': n,
        'mainline_size_counts': dict(sorted(mainline_size_counts.items())),
        'category_candidate_exposure': dict(category_exposure),
        'category_actual_third_hits': dict(category_hits),
        'category_hit_rate_per_candidate_exposure': {k: category_hits[k] / category_exposure[k] for k in category_exposure},
        'category_race_availability': dict(race_category_available),
        'category_race_hit_rate_when_available': {k: race_category_hits[k] / race_category_available[k] for k in race_category_available},
        'mainline_third_when_exists': {
            'exists_races': main3_exists,
            'third_hits': main3_hits,
            'hit_rate': main3_hits / main3_exists if main3_exists else 0,
        },
        'strongest_rival_line': {
            'exists_races': strongest_rival_line_exists,
            'actual_third_in_line': strongest_rival_line_hits,
            'rate': strongest_rival_line_hits / strongest_rival_line_exists if strongest_rival_line_exists else 0,
            'actual_third_in_top2_score_members': strongest_rival_member_top2_capture,
            'top2_member_rate': strongest_rival_member_top2_capture / strongest_rival_line_exists if strongest_rival_line_exists else 0,
        },
        'metric_actual_third_rank_counts_among_remaining': {m: dict(sorted(c.items())) for m, c in metric_rank_counts.items()},
        'metric_top2_capture_among_remaining': {m: {'capture': metric_top2_capture[m], 'rate': metric_top2_capture[m] / n} for m in metrics},
        'note': 'Descriptive only. Metrics are evaluated as pre-race ranking signals, not selected by payout/ROI.',
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, segment_for
from .simulate_early_candidate_v0 import strongest_rival


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def audit(data_dir: Path):
    races = read_csv(data_dir / 'races.csv')
    entries = read_csv(data_dir / 'entries.csv')
    eb = defaultdict(list)
    groups = defaultdict(list)
    for e in entries:
        eb[e['race_id']].append(e)
    for r in races:
        groups[(r['race_date'], r['track'])].append(r)

    segment = {}
    cards = Counter()
    for key, g in groups.items():
        g.sort(key=lambda r: int(r['race_no']))
        n = len(g)
        cards[n] += 1
        for pos, r in enumerate(g, 1):
            segment[r['race_id']] = segment_for(pos, n)

    stages = {
        s: {'all': 0, 'main_line_found': 0, 'main3plus': 0, 'main3plus_rival2plus': 0}
        for s in ('前半','中盤','後半')
    }
    no_main = []
    no_rival_after_main3 = []
    for r in races:
        rid = r['race_id']
        seg = segment[rid]
        stages[seg]['all'] += 1
        main = choose_main_line(eb[rid])
        if not main:
            no_main.append(rid)
            continue
        stages[seg]['main_line_found'] += 1
        main_id, mem = main
        if len(mem) < 3:
            continue
        stages[seg]['main3plus'] += 1
        rival = strongest_rival(eb[rid], main_id)
        if rival and len(rival) >= 2:
            stages[seg]['main3plus_rival2plus'] += 1
        else:
            no_rival_after_main3.append({'race_id':rid,'segment':seg,'race_date':r['race_date'],'track':r['track'],'race_no':r['race_no']})

    return {
        'dataset_races': len(races),
        'day_track_groups': len(groups),
        'card_size_distribution': dict(sorted(cards.items())),
        'stages': stages,
        'no_main_count': len(no_main),
        'no_rival_after_main3_count': len(no_rival_after_main3),
        'no_rival_after_main3': no_rival_after_main3,
    }


def main():
    out = {
        '2024': audit(Path('data/2024/s_class_yosen')),
        '2025': audit(Path('data/2025/s_class_yosen')),
    }
    Path('data/audits').mkdir(parents=True, exist_ok=True)
    Path('data/audits/oos_population_2024_vs_2025.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

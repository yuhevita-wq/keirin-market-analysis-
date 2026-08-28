from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .simulate_mainline_v1 import choose_main_line, num, segment_for

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'middle_structure_v1'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def metric(e: dict[str, str], name: str) -> float:
    v = num(e.get(name, ''))
    return 0.0 if v == float('-inf') else v


def line_map(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get('line_id', '').isdigit() and e.get('line_position', '').isdigit():
            by[int(e['line_id'])].append(e)
    for lid in by:
        by[lid].sort(key=lambda e: int(e['line_position']))
    return by


def line_pair_strength(members: list[dict[str, str]]) -> float:
    if len(members) < 2:
        return float('-inf')
    return metric(members[0], 'score') + metric(members[1], 'score')


FEATURES = [
    'pair_score_gap','main_pair_score_sum','leader_score_gap','second_score_gap',
    'main_leader_score','main_second_score','main_third_score',
    'main_pair_top2_sum','main_pair_top3_sum','main_pair_win_sum',
    'main_leader_top2','main_second_top2','main_leader_top3','main_second_top3','main_third_top3',
    'main_leader_b','main_second_mark','line_count','multi_line_count','main_line_size','race_no','segment_ordinal','s_yosen_count'
]


def gini(rows: list[dict[str, object]]) -> float:
    if not rows:
        return 0.0
    p = sum(int(r['mainline_top2']) for r in rows) / len(rows)
    return 2*p*(1-p)


def thresholds(rows: list[dict[str, object]], feature: str) -> list[float]:
    vals = sorted({float(r[feature]) for r in rows})
    if len(vals) < 2:
        return []
    mids = [(a+b)/2 for a,b in zip(vals, vals[1:])]
    if len(mids) <= 80:
        return mids
    return [mids[round(i*(len(mids)-1)/79)] for i in range(80)]


def stats(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    w = sum(int(r['mainline_top2']) for r in rows)
    return {'races': n, 'mainline_top2': w, 'rate': w/n if n else 0.0}


def best_split(rows: list[dict[str, object]], min_leaf: int = 20) -> dict[str, object] | None:
    base = gini(rows)
    best = None
    for f in FEATURES:
        for t in thresholds(rows, f):
            left = [r for r in rows if float(r[f]) <= t]
            right = [r for r in rows if float(r[f]) > t]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            weighted = (len(left)*gini(left) + len(right)*gini(right)) / len(rows)
            gain = base - weighted
            cand = {'feature': f, 'threshold': t, 'gain': gain, 'left': stats(left), 'right': stats(right)}
            if best is None or gain > float(best['gain']):
                best = cand
    return best


def build_tree(train: list[dict[str, object]]) -> dict[str, object]:
    root = best_split(train)
    if root is None:
        return {'leaf': stats(train)}
    f = str(root['feature']); t = float(root['threshold'])
    left = [r for r in train if float(r[f]) <= t]
    right = [r for r in train if float(r[f]) > t]
    out: dict[str, object] = {'split': root, 'left': {'train': stats(left)}, 'right': {'train': stats(right)}}
    for side, rows in [('left', left), ('right', right)]:
        child = best_split(rows, min_leaf=15)
        if child:
            cf = str(child['feature']); ct = float(child['threshold'])
            ll = [r for r in rows if float(r[cf]) <= ct]
            rr = [r for r in rows if float(r[cf]) > ct]
            out[side]['split'] = child
            out[side]['left'] = {'train': stats(ll)}
            out[side]['right'] = {'train': stats(rr)}
    return out


def apply_tree(tree: dict[str, object], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    root = tree.get('split')
    if not root:
        return [{'path': 'ALL', 'stats': stats(rows)}]
    rf = str(root['feature']); rt = float(root['threshold'])
    sides = [('L', tree['left'], [r for r in rows if float(r[rf]) <= rt]), ('R', tree['right'], [r for r in rows if float(r[rf]) > rt])]
    out = []
    for prefix, node, subset in sides:
        child = node.get('split')
        if child:
            cf = str(child['feature']); ct = float(child['threshold'])
            out.append({'path': prefix+'L', 'stats': stats([r for r in subset if float(r[cf]) <= ct])})
            out.append({'path': prefix+'R', 'stats': stats([r for r in subset if float(r[cf]) > ct])})
        else:
            out.append({'path': prefix, 'stats': stats(subset)})
    return out


def main() -> None:
    races = read_csv(DATA_DIR/'races.csv')
    entries = read_csv(DATA_DIR/'entries.csv')
    results = read_csv(DATA_DIR/'results.csv')
    entries_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    results_by: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups: dict[tuple[str,str], list[dict[str,str]]] = defaultdict(list)
    for e in entries: entries_by[e['race_id']].append(e)
    for r in results: results_by[r['race_id']].append(r)
    for r in races: groups[(r['race_date'], r['track'])].append(r)

    segment: dict[str,str] = {}; ordinal: dict[str,int] = {}; cardn: dict[str,int] = {}
    for g in groups.values():
        g.sort(key=lambda x: int(x['race_no']))
        n = len(g)
        for pos, r in enumerate(g, 1):
            segment[r['race_id']] = segment_for(pos, n)
            ordinal[r['race_id']] = pos
            cardn[r['race_id']] = n

    rows: list[dict[str, object]] = []
    collapse_rows: list[dict[str, object]] = []
    main_survivors = Counter(); ab_survival = Counter(); top3_shape = Counter(); rival_presence = Counter(); main3_presence = Counter()
    first_second_relation = Counter(); winner_role = Counter(); second_role = Counter(); third_role = Counter()

    for race in sorted(races, key=lambda r: (r['race_date'], r['track'], int(r['race_no']))):
        rid = race['race_id']
        if segment[rid] != '中盤':
            continue
        es = entries_by[rid]
        chosen = choose_main_line(es)
        if not chosen:
            continue
        main_id, main = chosen
        if len(main) < 3:
            continue
        by = line_map(es)
        rivals = [(line_pair_strength(m), lid, m) for lid,m in by.items() if lid != main_id and len(m) >= 2]
        rivals.sort(key=lambda x: (-x[0], x[1]))
        rival = rivals[0][2] if rivals else []
        rival_id = rivals[0][1] if rivals else -1
        a,b,c = main[0], main[1], main[2]
        result_map = {int(r['car_no']): r for r in results_by[rid]}
        fa = result_map.get(int(a['car_no']), {}).get('finish_position','')
        fb = result_map.get(int(b['car_no']), {}).get('finish_position','')
        y = 1 if {str(fa),str(fb)} == {'1','2'} else 0
        r1 = rival[0] if len(rival) >= 1 else {}
        r2 = rival[1] if len(rival) >= 2 else {}
        row: dict[str,object] = {
            'race_id': rid, 'race_date': race['race_date'], 'half': 'H1' if race['race_date'] <= '2025-06-30' else 'H2',
            'track': race['track'], 'race_no': int(race['race_no']), 'segment_ordinal': ordinal[rid], 's_yosen_count': cardn[rid],
            'main_line_id': main_id, 'main_line_size': len(main), 'mainline_top2': y,
            'main_pair_score_sum': metric(a,'score')+metric(b,'score'),
            'pair_score_gap': metric(a,'score')+metric(b,'score') - (line_pair_strength(rival) if rival else 0.0),
            'main_leader_score': metric(a,'score'), 'main_second_score': metric(b,'score'), 'main_third_score': metric(c,'score'),
            'leader_score_gap': metric(a,'score') - (metric(r1,'score') if r1 else 0.0),
            'second_score_gap': metric(b,'score') - (metric(r2,'score') if r2 else 0.0),
            'main_pair_top2_sum': metric(a,'top2_rate')+metric(b,'top2_rate'),
            'main_pair_top3_sum': metric(a,'top3_rate')+metric(b,'top3_rate'),
            'main_pair_win_sum': metric(a,'win_rate')+metric(b,'win_rate'),
            'main_leader_top2': metric(a,'top2_rate'), 'main_second_top2': metric(b,'top2_rate'),
            'main_leader_top3': metric(a,'top3_rate'), 'main_second_top3': metric(b,'top3_rate'), 'main_third_top3': metric(c,'top3_rate'),
            'main_leader_b': metric(a,'b_count'), 'main_second_mark': metric(b,'mark_count'),
            'line_count': len(by), 'multi_line_count': sum(1 for m in by.values() if len(m)>=2),
        }
        rows.append(row)

        finish = {}
        for rr in results_by[rid]:
            if rr.get('finish_position','') in {'1','2','3'}:
                finish[int(rr['car_no'])] = int(rr['finish_position'])
        top3 = [car for car,pos in sorted(finish.items(), key=lambda kv: kv[1])][:3]
        if y or len(top3) < 3:
            continue
        entry_by_car = {int(e['car_no']): e for e in es}
        lines = [int(entry_by_car[x]['line_id']) if entry_by_car[x].get('line_id','').isdigit() else -1 for x in top3]
        main_count = sum(1 for lid in lines if lid == main_id)
        rival_count = sum(1 for lid in lines if lid == rival_id)
        main_survivors[main_count] += 1; rival_presence[rival_count] += 1
        aa = int(a['car_no']); bb = int(b['car_no']); cc = int(c['car_no'])
        a_in = aa in top3; b_in = bb in top3
        ab = 'A+B both top3' if a_in and b_in else 'A only top3' if a_in else 'B only top3' if b_in else 'A/B neither top3'
        ab_survival[ab] += 1
        first_second_relation['1-2 same line' if lines[0] == lines[1] else '1-2 different lines'] += 1
        shape = '+'.join(map(str, sorted(Counter(lines).values(), reverse=True)))
        top3_shape[shape] += 1
        main3_presence['main3 in top3' if cc in top3 else 'main3 out'] += 1
        def role(car: int) -> str:
            e = entry_by_car[car]
            lid = int(e['line_id']) if e.get('line_id','').isdigit() else -1
            lp = int(e['line_position']) if e.get('line_position','').isdigit() else 0
            if lid == main_id:
                return 'A' if car == aa else 'B' if car == bb else 'M3' if car == cc else 'main_other'
            if lid == rival_id:
                return 'R1L' if lp == 1 else 'R1B' if lp == 2 else f'R1_{lp}'
            return f'other_{lp}'
        winner_role[role(top3[0])] += 1; second_role[role(top3[1])] += 1; third_role[role(top3[2])] += 1
        collapse_rows.append({'race_id':rid,'race_date':race['race_date'],'track':race['track'],'race_no':race['race_no'],'top3':'-'.join(map(str,top3)),'main_survivors':main_count,'ab_survival':ab,'top3_shape':shape,'rival_top3_count':rival_count,'winner_role':role(top3[0]),'second_role':role(top3[1]),'third_role':role(top3[2])})

    train = [r for r in rows if r['half']=='H1']; test = [r for r in rows if r['half']=='H2']
    tree = build_tree(train)
    train_leaf = {x['path']:x['stats'] for x in apply_tree(tree,train)}
    test_leaf = {x['path']:x['stats'] for x in apply_tree(tree,test)}
    paths = sorted(set(train_leaf)|set(test_leaf))

    feature_comp = {}
    yes = [r for r in rows if int(r['mainline_top2'])==1]; no = [r for r in rows if int(r['mainline_top2'])==0]
    for f in FEATURES:
        feature_comp[f] = {'mainline_mean':mean(float(r[f]) for r in yes),'collapse_mean':mean(float(r[f]) for r in no),'difference':mean(float(r[f]) for r in yes)-mean(float(r[f]) for r in no)}
    single = [best_split(train, min_leaf=20) for _f in [0]]
    # Rank each feature's best H1 split.
    ranked = []
    base = gini(train)
    for f in FEATURES:
        best = None
        for t in thresholds(train,f):
            l=[r for r in train if float(r[f])<=t]; rr=[r for r in train if float(r[f])>t]
            if len(l)<20 or len(rr)<20: continue
            gain=base-(len(l)*gini(l)+len(rr)*gini(rr))/len(train)
            cand={'feature':f,'threshold':t,'gain':gain,'left':stats(l),'right':stats(rr)}
            if best is None or gain>best['gain']: best=cand
        if best: ranked.append(best)
    ranked.sort(key=lambda x: float(x['gain']), reverse=True)

    out = {
        'scope':'2025 exact S級予選, 中盤, main line size >=3',
        'mainline_top2': {'all':stats(rows),'H1':stats(train),'H2':stats(test)},
        'collapse': {
            'races':len(collapse_rows),
            'main_line_riders_in_top3':dict(sorted(main_survivors.items())),
            'A_B_survival':dict(ab_survival),
            'first_second_line_relation':dict(first_second_relation),
            'top3_line_count_shape':dict(top3_shape),
            'strongest_rival_riders_in_top3':dict(sorted(rival_presence.items())),
            'main3_presence':dict(main3_presence),
            'winner_role':dict(winner_role.most_common()),
            'second_role':dict(second_role.most_common()),
            'third_role':dict(third_role.most_common()),
        },
        'feature_comparison':feature_comp,
        'top_single_splits_H1':ranked[:10],
        'depth2_tree_built_on_H1':tree,
        'leaf_validation':[{'path':p,'H1':train_leaf.get(p,{'races':0,'mainline_top2':0,'rate':0.0}),'H2':test_leaf.get(p,{'races':0,'mainline_top2':0,'rate':0.0})} for p in paths],
        'warning':'Exploratory. Thresholds are selected only on H1 and checked on H2. Results are labels only. No payout information is used to create branches.'
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if rows:
        with (OUT_DIR/'classification_dataset.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if collapse_rows:
        with (OUT_DIR/'collapse_races.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(collapse_rows[0].keys())); w.writeheader(); w.writerows(collapse_rows)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

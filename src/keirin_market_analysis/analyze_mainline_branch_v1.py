from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .simulate_mainline_v1 import choose_main_line, num, segment_for

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'mainline_branch_v1'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def finite(v: float) -> float:
    return 0.0 if v == float('-inf') else v


def metric(e: dict[str, str], name: str) -> float:
    return finite(num(e.get(name, '')))


def line_map(entries: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    by: dict[int, list[dict[str, str]]] = defaultdict(list)
    for e in entries:
        if e.get('line_id', '').isdigit() and e.get('line_position', '').isdigit():
            by[int(e['line_id'])].append(e)
    for lid in by:
        by[lid].sort(key=lambda e: int(e['line_position']))
    return by


def line_pair_strength(members: list[dict[str, str]]) -> float:
    if not members:
        return -999.0
    s = metric(members[0], 'score')
    if len(members) >= 2:
        s += metric(members[1], 'score')
    return s


def make_feature_row(race: dict[str, str], entries: list[dict[str, str]], result_rows: list[dict[str, str]], ordinal: int, card_n: int) -> dict[str, object] | None:
    chosen = choose_main_line(entries)
    if chosen is None:
        return None
    main_id, main_members = chosen
    if len(main_members) < 3:
        return None
    by = line_map(entries)
    rivals = [(line_pair_strength(m), lid, m) for lid, m in by.items() if lid != main_id and m]
    rivals.sort(key=lambda x: (-x[0], x[1]))
    rival_members = rivals[0][2] if rivals else []

    a, b, c = main_members[0], main_members[1], main_members[2]
    result_map = {int(r['car_no']): r for r in result_rows}
    fa = result_map.get(int(a['car_no']), {}).get('finish_position', '')
    fb = result_map.get(int(b['car_no']), {}).get('finish_position', '')
    y = 1 if {str(fa), str(fb)} == {'1','2'} else 0

    rival_leader = rival_members[0] if len(rival_members) >= 1 else {}
    rival_second = rival_members[1] if len(rival_members) >= 2 else {}
    main_pair_score = metric(a,'score') + metric(b,'score')
    rival_pair_score = line_pair_strength(rival_members) if rival_members else 0.0

    row: dict[str, object] = {
        'race_id': race['race_id'], 'race_date': race['race_date'], 'half': 'H1' if race['race_date'] <= '2025-06-30' else 'H2',
        'track': race['track'], 'race_no': int(race['race_no']), 'segment_ordinal': ordinal, 's_yosen_count': card_n,
        'main_line_id': main_id, 'main_line_size': len(main_members), 'mainline_top2': y,
        'main_pair_score_sum': main_pair_score,
        'rival_pair_score_sum': rival_pair_score,
        'pair_score_gap': main_pair_score - rival_pair_score,
        'main_leader_score': metric(a,'score'), 'main_second_score': metric(b,'score'), 'main_third_score': metric(c,'score'),
        'rival_leader_score': metric(rival_leader,'score') if rival_leader else 0.0,
        'rival_second_score': metric(rival_second,'score') if rival_second else 0.0,
        'leader_score_gap': metric(a,'score') - (metric(rival_leader,'score') if rival_leader else 0.0),
        'second_score_gap': metric(b,'score') - (metric(rival_second,'score') if rival_second else 0.0),
        'main_pair_top2_sum': metric(a,'top2_rate') + metric(b,'top2_rate'),
        'main_pair_top3_sum': metric(a,'top3_rate') + metric(b,'top3_rate'),
        'main_pair_win_sum': metric(a,'win_rate') + metric(b,'win_rate'),
        'main_leader_top2': metric(a,'top2_rate'), 'main_second_top2': metric(b,'top2_rate'),
        'main_leader_top3': metric(a,'top3_rate'), 'main_second_top3': metric(b,'top3_rate'), 'main_third_top3': metric(c,'top3_rate'),
        'main_leader_b': metric(a,'b_count'), 'main_second_mark': metric(b,'mark_count'),
        'line_count': len(by), 'multi_line_count': sum(1 for m in by.values() if len(m)>=2),
    }
    return row


FEATURES = [
    'pair_score_gap','main_pair_score_sum','leader_score_gap','second_score_gap','main_leader_score','main_second_score','main_third_score',
    'main_pair_top2_sum','main_pair_top3_sum','main_pair_win_sum','main_leader_top2','main_second_top2','main_leader_top3','main_second_top3','main_third_top3',
    'main_leader_b','main_second_mark','line_count','multi_line_count','main_line_size','race_no','segment_ordinal','s_yosen_count'
]


def gini(rows: list[dict[str, object]]) -> float:
    if not rows: return 0.0
    p = sum(int(r['mainline_top2']) for r in rows) / len(rows)
    return 2*p*(1-p)


def candidate_thresholds(rows: list[dict[str, object]], feature: str) -> list[float]:
    vals = sorted({float(r[feature]) for r in rows})
    if len(vals) < 2: return []
    mids = [(a+b)/2 for a,b in zip(vals, vals[1:])]
    if len(mids) <= 80: return mids
    return [mids[round(i*(len(mids)-1)/79)] for i in range(80)]


def best_split(rows: list[dict[str, object]], min_leaf: int = 20) -> dict[str, object] | None:
    base = gini(rows); best = None
    for f in FEATURES:
        for t in candidate_thresholds(rows, f):
            left = [r for r in rows if float(r[f]) <= t]
            right = [r for r in rows if float(r[f]) > t]
            if len(left) < min_leaf or len(right) < min_leaf: continue
            weighted = (len(left)*gini(left) + len(right)*gini(right))/len(rows)
            gain = base - weighted
            cand = {'feature':f,'threshold':t,'gain':gain,'left_n':len(left),'right_n':len(right)}
            if best is None or (gain, f, -t) > (float(best['gain']), str(best['feature']), -float(best['threshold'])):
                best = cand
    return best


def branch_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    n=len(rows); wins=sum(int(r['mainline_top2']) for r in rows)
    return {'races':n,'mainline_top2':wins,'rate':wins/n if n else 0.0}


def build_tree(train: list[dict[str, object]]) -> dict[str, object]:
    root = best_split(train)
    if root is None: return {'leaf': branch_stats(train)}
    f=str(root['feature']); t=float(root['threshold'])
    left=[r for r in train if float(r[f])<=t]; right=[r for r in train if float(r[f])>t]
    out={'split':root,'left':{'train':branch_stats(left)},'right':{'train':branch_stats(right)}}
    for side, rows in [('left',left),('right',right)]:
        child=best_split(rows, min_leaf=15)
        if child:
            cf=str(child['feature']); ct=float(child['threshold'])
            ll=[r for r in rows if float(r[cf])<=ct]; rr=[r for r in rows if float(r[cf])>ct]
            out[side]['split']=child
            out[side]['left']={'train':branch_stats(ll)}
            out[side]['right']={'train':branch_stats(rr)}
    return out


def apply_tree_leaves(tree: dict[str, object], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    leaves=[]
    root=tree.get('split')
    if not root:
        return [{'path':'ALL','rows':rows,'stats':branch_stats(rows)}]
    rf=str(root['feature']); rt=float(root['threshold'])
    sides=[('L',tree['left'],[r for r in rows if float(r[rf])<=rt]),('R',tree['right'],[r for r in rows if float(r[rf])>rt])]
    for prefix,node,subset in sides:
        child=node.get('split')
        if child:
            cf=str(child['feature']); ct=float(child['threshold'])
            for suffix,part in [('L',[r for r in subset if float(r[cf])<=ct]),('R',[r for r in subset if float(r[cf])>ct])]:
                leaves.append({'path':prefix+suffix,'rows':part,'stats':branch_stats(part)})
        else:
            leaves.append({'path':prefix,'rows':subset,'stats':branch_stats(subset)})
    return leaves


def feature_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    yes=[r for r in rows if int(r['mainline_top2'])==1]; no=[r for r in rows if int(r['mainline_top2'])==0]
    out={}
    for f in FEATURES:
        out[f]={'mainline_mean':mean(float(r[f]) for r in yes),'other_mean':mean(float(r[f]) for r in no),'difference':mean(float(r[f]) for r in yes)-mean(float(r[f]) for r in no)}
    return out


def main() -> None:
    races=read_csv(DATA_DIR/'races.csv'); entries=read_csv(DATA_DIR/'entries.csv'); results=read_csv(DATA_DIR/'results.csv')
    entries_by: dict[str,list[dict[str,str]]] = defaultdict(list); results_by: dict[str,list[dict[str,str]]] = defaultdict(list); groups: dict[tuple[str,str],list[dict[str,str]]] = defaultdict(list)
    for e in entries: entries_by[e['race_id']].append(e)
    for r in results: results_by[r['race_id']].append(r)
    for r in races: groups[(r['race_date'],r['track'])].append(r)
    segment={}; ordinal={}; cardn={}
    for g in groups.values():
        g.sort(key=lambda x:int(x['race_no'])); n=len(g)
        for pos,r in enumerate(g,1): segment[r['race_id']]=segment_for(pos,n); ordinal[r['race_id']]=pos; cardn[r['race_id']]=n

    rows=[]
    for race in sorted(races,key=lambda r:(r['race_date'],r['track'],int(r['race_no']))):
        rid=race['race_id']
        if segment[rid] != '後半': continue
        row=make_feature_row(race, entries_by[rid], results_by[rid], ordinal[rid], cardn[rid])
        if row is not None: rows.append(row)

    train=[r for r in rows if r['half']=='H1']; test=[r for r in rows if r['half']=='H2']
    tree=build_tree(train)
    train_leaves=apply_tree_leaves(tree,train); test_leaves=apply_tree_leaves(tree,test)
    leaf_map={x['path']:x['stats'] for x in test_leaves}
    leaf_report=[]
    for x in train_leaves:
        leaf_report.append({'path':x['path'],'H1':x['stats'],'H2':leaf_map.get(x['path'],{'races':0,'mainline_top2':0,'rate':0.0})})

    root_splits=[]
    for f in FEATURES:
        best=None
        for t in candidate_thresholds(train,f):
            l=[r for r in train if float(r[f])<=t]; rr=[r for r in train if float(r[f])>t]
            if len(l)<20 or len(rr)<20: continue
            gain=gini(train)-(len(l)*gini(l)+len(rr)*gini(rr))/len(train)
            cand={'feature':f,'threshold':t,'gain':gain,'left':branch_stats(l),'right':branch_stats(rr)}
            if best is None or gain>best['gain']: best=cand
        if best: root_splits.append(best)
    root_splits.sort(key=lambda x:float(x['gain']),reverse=True)

    OUT_DIR.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys())
    write_csv(OUT_DIR/'classification_dataset.csv',rows,fields)
    summary={
        'scope':'2025 exact S級予選, 後半, main line size >=3 only',
        'label':'mainline_top2 = main-line leader and second rider occupy 1st/2nd in either order',
        'all':branch_stats(rows),'H1':branch_stats(train),'H2':branch_stats(test),
        'feature_comparison':feature_summary(rows),
        'top_single_splits_H1':root_splits[:10],
        'depth2_tree_built_on_H1':tree,
        'leaf_validation':leaf_report,
        'warning':'Exploratory. Tree thresholds are selected only on H1 and reported on H2. Results/payouts are labels only and are never used as pre-race features.'
    }
    (OUT_DIR/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()

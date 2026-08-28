from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .simulate_branching_v1 import classify, read_csv, strongest_rival
from .simulate_mainline_v1 import choose_main_line, num, segment_for

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'rough_winner_split_v1'


def f(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def actual_winner(results: list[dict[str, str]]) -> int | None:
    winners = [int(r['car_no']) for r in results if r.get('finish_position') == '1']
    return winners[0] if len(winners) == 1 else None


def metrics(e: dict[str, str]) -> dict[str, float]:
    return {
        'score': f(e.get('score', '')),
        'win_rate': f(e.get('win_rate', '')),
        'top2_rate': f(e.get('top2_rate', '')),
        'top3_rate': f(e.get('top3_rate', '')),
        'first_count': f(e.get('first_count', '')),
        'second_count': f(e.get('second_count', '')),
        'third_count': f(e.get('third_count', '')),
        'b_count': f(e.get('b_count', '')),
        's_count': f(e.get('s_count', '')),
        'nige_count': f(e.get('nige_count', '')),
        'makuri_count': f(e.get('makuri_count', '')),
    }


def predict_higher(rec: dict, key: str) -> str:
    d = rec[f'd_{key}']
    return 'A' if d >= 0 else 'R1L'


def summarize_classifier(rows: list[dict], predictor) -> dict:
    eligible = [r for r in rows if r['winner_role'] in {'A','R1L'}]
    correct = sum(predictor(r) == r['winner_role'] for r in eligible)
    return {
        'eligible': len(eligible),
        'correct': correct,
        'accuracy': correct / len(eligible) if eligible else 0,
        'A_actual': sum(r['winner_role']=='A' for r in eligible),
        'R1L_actual': sum(r['winner_role']=='R1L' for r in eligible),
    }


def main() -> None:
    races = read_csv(DATA_DIR / 'races.csv')
    entries = read_csv(DATA_DIR / 'entries.csv')
    results = read_csv(DATA_DIR / 'results.csv')

    e_by_race: dict[str, list[dict[str,str]]] = defaultdict(list)
    r_by_race: dict[str, list[dict[str,str]]] = defaultdict(list)
    day_track: dict[tuple[str,str], list[dict[str,str]]] = defaultdict(list)
    for e in entries: e_by_race[e['race_id']].append(e)
    for r in results: r_by_race[r['race_id']].append(r)
    for r in races: day_track[(r['race_date'],r['track'])].append(r)

    segment = {}
    for group in day_track.values():
        group.sort(key=lambda x:int(x['race_no']))
        for pos,r in enumerate(group,1):
            segment[r['race_id']] = segment_for(pos,len(group))

    rows=[]
    for race in sorted(races,key=lambda x:(x['race_date'],x['track'],int(x['race_no']))):
        rid=race['race_id']
        if segment.get(rid)!='後半': continue
        es=e_by_race[rid]
        main=choose_main_line(es)
        if not main: continue
        main_id,members=main
        if len(members)<3: continue
        branch,*_=classify(members)
        if branch!='B_rough': continue
        rival=strongest_rival(es,main_id)
        if not rival or len(rival)<2: continue
        winner=actual_winner(r_by_race[rid])
        if winner is None: continue
        A,B=members[0],members[1]
        R1L,R1B=rival[0],rival[1]
        am,rm=metrics(A),metrics(R1L)
        rec={
            'race_id':rid,
            'race_date':race['race_date'],
            'half':'H1' if race['race_date']<='2025-06-30' else 'H2',
            'A':int(A['car_no']),'B':int(B['car_no']),'R1L':int(R1L['car_no']),'R1B':int(R1B['car_no']),
            'winner':winner,
            'winner_role':'A' if winner==int(A['car_no']) else ('R1L' if winner==int(R1L['car_no']) else 'OTHER'),
        }
        for k in am:
            rec[f'A_{k}']=am[k]; rec[f'R1L_{k}']=rm[k]; rec[f'd_{k}']=am[k]-rm[k]
        rec['main_pair_score']=num(A.get('score',''))+num(B.get('score',''))
        rec['rival_pair_score']=num(R1L.get('score',''))+num(R1B.get('score',''))
        rec['pair_score_gap']=rec['main_pair_score']-rec['rival_pair_score']
        rec['B_top3']=f(B.get('top3_rate',''))
        rec['R1B_top3']=f(R1B.get('top3_rate',''))
        rec['second_top3_gap']=rec['B_top3']-rec['R1B_top3']
        rows.append(rec)

    keys=['score','win_rate','top2_rate','top3_rate','first_count','second_count','third_count','b_count','s_count','nige_count','makuri_count']
    group_stats={}
    for role in ('A','R1L','OTHER'):
        grp=[r for r in rows if r['winner_role']==role]
        group_stats[role]={'count':len(grp)}
        for k in keys+['pair_score_gap','second_top3_gap']:
            field=f'd_{k}' if k in keys else k
            group_stats[role][f'avg_{field}']=mean([r[field] for r in grp]) if grp else 0

    classifiers={}
    for k in ['score','win_rate','top2_rate','top3_rate','first_count','b_count','nige_count','makuri_count']:
        classifiers[f'higher_{k}']={
            'H1':summarize_classifier([r for r in rows if r['half']=='H1'],lambda r,k=k: predict_higher(r,k)),
            'H2':summarize_classifier([r for r in rows if r['half']=='H2'],lambda r,k=k: predict_higher(r,k)),
            'ALL':summarize_classifier(rows,lambda r,k=k: predict_higher(r,k)),
        }

    # Predeclared majority vote among four intuitive leader-strength metrics.
    def majority4(r):
        votes=sum(r[f'd_{k}']>=0 for k in ('score','win_rate','top2_rate','top3_rate'))
        return 'A' if votes>=3 else 'R1L'
    classifiers['majority_score_win_top2_top3']={
        'H1':summarize_classifier([r for r in rows if r['half']=='H1'],majority4),
        'H2':summarize_classifier([r for r in rows if r['half']=='H2'],majority4),
        'ALL':summarize_classifier(rows,majority4),
    }

    # H1-only one-dimensional threshold search on leader score gap; H2 untouched.
    h1=[r for r in rows if r['half']=='H1' and r['winner_role'] in {'A','R1L'}]
    candidate_thresholds=[-5,-4,-3,-2,-1,0,1,2,3,4,5]
    scored=[]
    for t in candidate_thresholds:
        c=sum(('A' if r['d_score']>=t else 'R1L')==r['winner_role'] for r in h1)
        scored.append((c/len(h1) if h1 else 0,-abs(t),t))
    best_t=max(scored)[2] if scored else 0
    def score_threshold(r): return 'A' if r['d_score']>=best_t else 'R1L'
    classifiers['H1_selected_score_gap_threshold']={
        'threshold':best_t,
        'H1':summarize_classifier([r for r in rows if r['half']=='H1'],score_threshold),
        'H2':summarize_classifier([r for r in rows if r['half']=='H2'],score_threshold),
        'ALL':summarize_classifier(rows,score_threshold),
    }

    summary={
        'scope':'2025 exact S級予選, 後半, main line >=3, B rough branch',
        'races':len(rows),
        'winner_roles':{k:sum(r['winner_role']==k for r in rows) for k in ('A','R1L','OTHER')},
        'winner_roles_H1':{k:sum(r['winner_role']==k and r['half']=='H1' for r in rows) for k in ('A','R1L','OTHER')},
        'winner_roles_H2':{k:sum(r['winner_role']==k and r['half']=='H2' for r in rows) for k in ('A','R1L','OTHER')},
        'group_feature_differences':group_stats,
        'classifiers_A_vs_R1L_only':classifiers,
        'note':'Classifier accuracy is measured only where the actual winner is A or R1L. OTHER winners are irreducible misses if first place is narrowed to one of these two.',
        'warning':'Exploratory. Only the score-gap threshold is selected using H1; H2 is held untouched for that rule. Do not select a different rule because H2 looks better.'
    }
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUT_DIR/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

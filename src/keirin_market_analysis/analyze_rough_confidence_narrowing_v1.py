from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .analyze_rough_winner_split_v1 import actual_winner, f, metrics
from .simulate_branching_v1 import classify, read_csv, strongest_rival
from .simulate_mainline_v1 import choose_main_line, segment_for

DATA_DIR=Path('data/2025/s_class_yosen')
OUT_DIR=DATA_DIR/'simulations'/'rough_confidence_narrowing_v1'

ZONES={
 'score_ge_5': lambda r:r['d_score']>=5,
 'score_ge_8': lambda r:r['d_score']>=8,
 'score_ge_10':lambda r:r['d_score']>=10,
 'top3_ge_10':lambda r:r['d_top3_rate']>=10,
 'top3_ge_15':lambda r:r['d_top3_rate']>=15,
 'top3_ge_20':lambda r:r['d_top3_rate']>=20,
 'win_ge_10':lambda r:r['d_win_rate']>=10,
 'win_ge_15':lambda r:r['d_win_rate']>=15,
 'score5_top3_10':lambda r:r['d_score']>=5 and r['d_top3_rate']>=10,
 'score5_win10':lambda r:r['d_score']>=5 and r['d_win_rate']>=10,
 'top3_10_win10':lambda r:r['d_top3_rate']>=10 and r['d_win_rate']>=10,
}

def summarize(rows):
    a=sum(r['winner_role']=='A' for r in rows); q=sum(r['winner_role']=='R1L' for r in rows); o=sum(r['winner_role']=='OTHER' for r in rows)
    elig=a+q
    return {'races':len(rows),'A':a,'R1L':q,'OTHER':o,'A_share_vs_R1L':a/elig if elig else 0,'A_rate_all':a/len(rows) if rows else 0}

def main():
    races=read_csv(DATA_DIR/'races.csv'); entries=read_csv(DATA_DIR/'entries.csv'); results=read_csv(DATA_DIR/'results.csv')
    eb=defaultdict(list); rb=defaultdict(list); dt=defaultdict(list)
    for e in entries: eb[e['race_id']].append(e)
    for r in results: rb[r['race_id']].append(r)
    for r in races: dt[(r['race_date'],r['track'])].append(r)
    seg={}
    for g in dt.values():
        g.sort(key=lambda r:int(r['race_no']))
        for i,r in enumerate(g,1):seg[r['race_id']]=segment_for(i,len(g))
    rows=[]
    for race in races:
        rid=race['race_id']
        if seg.get(rid)!='後半':continue
        main=choose_main_line(eb[rid])
        if not main:continue
        mid,m=main
        if len(m)<3:continue
        if classify(m)[0]!='B_rough':continue
        rival=strongest_rival(eb[rid],mid)
        if not rival or len(rival)<2:continue
        w=actual_winner(rb[rid])
        if w is None:continue
        A=m[0]; R=rival[0]; am=metrics(A); rm=metrics(R)
        rec={'race_id':rid,'half':'H1' if race['race_date']<='2025-06-30' else 'H2','winner_role':'A' if w==int(A['car_no']) else ('R1L' if w==int(R['car_no']) else 'OTHER')}
        for k in am:rec[f'd_{k}']=am[k]-rm[k]
        rows.append(rec)
    out={'scope':'B rough branch; A-confidence zones only; no payout-based selection','overall':summarize(rows),'zones':{}}
    for name,fn in ZONES.items():
        z=[r for r in rows if fn(r)]
        out['zones'][name]={
          'ALL':summarize(z),
          'H1':summarize([r for r in z if r['half']=='H1']),
          'H2':summarize([r for r in z if r['half']=='H2']),
        }
    out['warning']='Exploratory predefined zones. Prefer zones with adequate sample and same direction in H1/H2. No zone is automatically adopted.'
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUT_DIR/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

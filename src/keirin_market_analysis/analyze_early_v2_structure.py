from __future__ import annotations

import csv, json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_early_candidate_v0 import strongest_rival

YEARS=(2024,2025)
OUT=Path('data/audits/early_v2_structure.json')


def read_csv(p):
    with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def f(v):
    x=num(v)
    return 0.0 if x==float('-inf') else float(x)

def rows_for(year:int):
    d=Path(f'data/{year}/s_class_yosen')
    races=read_csv(d/'races.csv'); entries=read_csv(d/'entries.csv'); results=read_csv(d/'results.csv')
    eb=defaultdict(list); rb=defaultdict(list); groups=defaultdict(list)
    for e in entries: eb[e['race_id']].append(e)
    for r in results: rb[r['race_id']].append(r)
    for r in races: groups[(r['race_date'],r['track'])].append(r)
    seg={}
    for g in groups.values():
        g.sort(key=lambda r:int(r['race_no']))
        for i,r in enumerate(g,1): seg[r['race_id']]=segment_for(i,len(g))
    out=[]
    for race in races:
        rid=race['race_id']
        if seg.get(rid)!='前半': continue
        main=choose_main_line(eb[rid])
        if not main: continue
        mid,m=main
        if len(m)<3: continue
        rival=strongest_rival(eb[rid],mid)
        if not rival or len(rival)<2: continue
        A,B=m[0],m[1]; R1L,R1B=rival[0],rival[1]
        finish={int(r['car_no']):r.get('finish_position','') for r in rb[rid]}
        out.append({
          'race_id':rid,
          'main_pair_win':f(A.get('win_rate'))+f(B.get('win_rate')),
          'main_pair_top2':f(A.get('top2_rate'))+f(B.get('top2_rate')),
          'main_pair_top3':f(A.get('top3_rate'))+f(B.get('top3_rate')),
          'rival_pair_win':f(R1L.get('win_rate'))+f(R1B.get('win_rate')),
          'rival_pair_top2':f(R1L.get('top2_rate'))+f(R1B.get('top2_rate')),
          'rival_pair_top3':f(R1L.get('top3_rate'))+f(R1B.get('top3_rate')),
          'main_pair_score':f(A.get('score'))+f(B.get('score')),
          'rival_pair_score':f(R1L.get('score'))+f(R1B.get('score')),
          'leader_score_gap':f(A.get('score'))-f(R1L.get('score')),
          'second_score_gap':f(B.get('score'))-f(R1B.get('score')),
          'pair_score_gap':f(A.get('score'))+f(B.get('score'))-f(R1L.get('score'))-f(R1B.get('score')),
          'pair_win_gap':f(A.get('win_rate'))+f(B.get('win_rate'))-f(R1L.get('win_rate'))-f(R1B.get('win_rate')),
          'pair_top2_gap':f(A.get('top2_rate'))+f(B.get('top2_rate'))-f(R1L.get('top2_rate'))-f(R1B.get('top2_rate')),
          'pair_top3_gap':f(A.get('top3_rate'))+f(B.get('top3_rate'))-f(R1L.get('top3_rate'))-f(R1B.get('top3_rate')),
          'target': int({finish.get(int(R1L['car_no'])),finish.get(int(R1B['car_no']))}=={'1','2'}),
        })
    return out

# Coarse, predeclared grids. These are structural screens, not payout optimization.
RULES=[]
for t in (18,20,22,24,26,28): RULES.append((f'main_win_le_{t}', lambda r,t=t:r['main_pair_win']<=t))
for t in (-4,-2,0,2,4,6): RULES.append((f'pair_score_gap_le_{t}', lambda r,t=t:r['pair_score_gap']<=t))
for t in (45,50,55,60,65,70): RULES.append((f'rival_top2_ge_{t}', lambda r,t=t:r['rival_pair_top2']>=t))
for t in (80,90,100,110,120): RULES.append((f'rival_top3_ge_{t}', lambda r,t=t:r['rival_pair_top3']>=t))
# Small, theory-driven conjunction set: weak main + competitive/strong rival.
for mw in (20,22,24):
  for sg in (0,2,4): RULES.append((f'main_win_le_{mw}__score_gap_le_{sg}', lambda r,mw=mw,sg=sg:r['main_pair_win']<=mw and r['pair_score_gap']<=sg))
for mw in (20,22,24):
  for rt in (50,60): RULES.append((f'main_win_le_{mw}__rival_top2_ge_{rt}', lambda r,mw=mw,rt=rt:r['main_pair_win']<=mw and r['rival_pair_top2']>=rt))

def stat(rows,fn):
    x=[r for r in rows if fn(r)]; n=len(x); h=sum(r['target'] for r in x)
    return {'races':n,'target_hits':h,'target_rate':h/n if n else 0.0}

def main():
    ys={y:rows_for(y) for y in YEARS}
    base={str(y):stat(ys[y],lambda r:True) for y in YEARS}
    candidates=[]
    for name,fn in RULES:
        s={str(y):stat(ys[y],fn) for y in YEARS}
        if min(s[str(y)]['races'] for y in YEARS)<20: continue
        min_rate=min(s[str(y)]['target_rate'] for y in YEARS)
        min_lift=min(s[str(y)]['target_rate']-base[str(y)]['target_rate'] for y in YEARS)
        candidates.append({'rule':name,**s,'min_year_rate':min_rate,'min_year_lift':min_lift})
    candidates.sort(key=lambda x:(x['min_year_rate'],x['min_year_lift']),reverse=True)
    v0=lambda r:r['main_pair_win']<=22.35
    out={
      'scope':'2024+2025 development only; early segment; main line 3+ and strongest rival 2+; structure labels only; no payouts read',
      'target':'R1L and R1B occupy 1st/2nd in either order',
      'base':base,
      'v0_22_35':{str(y):stat(ys[y],v0) for y in YEARS},
      'predeclared_rule_count':len(RULES),
      'selection_note':'Rank only by worst-year structural target rate, then worst-year lift; require >=20 races in each year. Financial evaluation comes later and cannot add rules.',
      'top_candidates':candidates[:15],
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

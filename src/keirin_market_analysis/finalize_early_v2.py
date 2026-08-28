from __future__ import annotations

import csv, json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_early_candidate_v0 import strongest_rival

YEARS=(2024,2025)
OUT=Path('data/audits/early_v2_finalization.json')


def read_csv(p: Path):
    # Guard: this finalizer is intentionally development-only.
    if '/2023/' in str(p).replace('\\','/'):
        raise RuntimeError('2023 OOS must not be read while finalizing V2')
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def f(v):
    x=num(v)
    return 0.0 if x==float('-inf') else float(x)


def dataset(year:int):
    d=Path(f'data/{year}/s_class_yosen')
    races=read_csv(d/'races.csv'); entries=read_csv(d/'entries.csv'); results=read_csv(d/'results.csv'); payouts=read_csv(d/'payouts.csv')
    eb=defaultdict(list); rb=defaultdict(list); groups=defaultdict(list); tri=defaultdict(dict)
    for e in entries: eb[e['race_id']].append(e)
    for r in results: rb[r['race_id']].append(r)
    for r in races: groups[(r['race_date'],r['track'])].append(r)
    for p in payouts:
        if p.get('ticket_type')=='3連単' and p.get('status')=='paid' and p.get('combination'):
            try: tri[p['race_id']][p['combination']]=int(p['payout_yen'])
            except ValueError: pass
    seg={}
    for g in groups.values():
        g.sort(key=lambda r:int(r['race_no']))
        for i,r in enumerate(g,1): seg[r['race_id']]=segment_for(i,len(g))
    return races,eb,rb,tri,seg


def rows_for(year:int):
    races,eb,rb,tri,seg=dataset(year)
    rows=[]
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
        # Frozen entrance candidate from structure audit: strongest rival pair top3-rate sum >= 90.
        if f(R1L.get('top3_rate'))+f(R1B.get('top3_rate')) < 90.0: continue
        a=int(A['car_no']); b=int(B['car_no']); l=int(R1L['car_no']); rbcar=int(R1B['car_no'])
        excluded={a,b,l,rbcar}
        remain=[e for e in eb[rid] if int(e['car_no']) not in excluded]
        # Deterministic pre-race extra third-slot candidate: highest competition score, tie -> lower car number.
        X=max(remain,key=lambda e:(f(e.get('score')),-int(e['car_no']))) if remain else None
        x=int(X['car_no']) if X else None
        finish={int(r['car_no']):r.get('finish_position','') for r in rb[rid]}
        target={finish.get(l),finish.get(rbcar)}=={'1','2'}
        third_car=next((c for c,pos in finish.items() if pos=='3'),None)
        f4=[f'{l}-{rbcar}-{a}',f'{l}-{rbcar}-{b}',f'{rbcar}-{l}-{a}',f'{rbcar}-{l}-{b}']
        f6=f4 + ([] if x is None else [f'{l}-{rbcar}-{x}',f'{rbcar}-{l}-{x}'])
        def ev(bets):
            ret=sum(tri[rid].get(combo,0) for combo in bets)
            return {'bets':bets,'stake':100*len(bets),'payout':ret,'hit':int(ret>0)}
        rows.append({
            'race_id':rid,'target':int(target),'third_is_A_or_B':int(target and third_car in {a,b}),
            'third_is_X':int(target and x is not None and third_car==x),
            'x_car':x,'f4':ev(f4),'f6x':ev(f6)
        })
    return rows


def sum_fin(rows,key):
    stake=sum(r[key]['stake'] for r in rows); payout=sum(r[key]['payout'] for r in rows); hits=sum(r[key]['hit'] for r in rows)
    return {'races':len(rows),'hits':hits,'hit_rate':hits/len(rows) if rows else 0.0,'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi':payout/stake if stake else 0.0}


def structural(rows):
    n=sum(r['target'] for r in rows)
    ab=sum(r['third_is_A_or_B'] for r in rows)
    x=sum(r['third_is_X'] for r in rows)
    covered=sum(1 for r in rows if r['target'] and (r['third_is_A_or_B'] or r['third_is_X']))
    return {'rival_pair_top2_races':n,'A_or_B_third':ab,'X_third':x,'A_or_B_or_X_covered':covered,'coverage_rate':covered/n if n else 0.0}


def main():
    ys={y:rows_for(y) for y in YEARS}
    out={
      'scope':'2024+2025 development only. 2023 is prohibited and not read.',
      'entrance_rule':'early segment; main line size >=3; strongest rival size >=2; R1L+R1B top3-rate sum >= 90.0',
      'formation_candidates':{
        'F4':['R1L-R1B-A','R1L-R1B-B','R1B-R1L-A','R1B-R1L-B'],
        'F6X':['F4 plus R1L-R1B-X and R1B-R1L-X'],
        'X':'highest competition-score rider excluding A,B,R1L,R1B; tie -> lower car number'
      },
      'structural_third_coverage':{str(y):structural(ys[y]) for y in YEARS},
      'financial':{k:{str(y):sum_fin(ys[y],k) for y in YEARS} for k in ('f4','f6x')},
      'selection_rule':'Choose the formation with higher worst-year ROI; tie -> higher combined ROI; no new conditions after this comparison.'
    }
    for k in ('f4','f6x'):
        vals=[out['financial'][k][str(y)]['roi'] for y in YEARS]
        stake=sum(out['financial'][k][str(y)]['stake_yen'] for y in YEARS)
        payout=sum(out['financial'][k][str(y)]['payout_yen'] for y in YEARS)
        out['financial'][k]['worst_year_roi']=min(vals)
        out['financial'][k]['combined']={'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi':payout/stake if stake else 0.0}
    ranked=sorted(('f4','f6x'),key=lambda k:(out['financial'][k]['worst_year_roi'],out['financial'][k]['combined']['roi']),reverse=True)
    out['selected_formation']=ranked[0]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

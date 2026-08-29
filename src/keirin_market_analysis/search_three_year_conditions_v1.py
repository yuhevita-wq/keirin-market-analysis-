from __future__ import annotations

import csv, json, itertools
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, num, segment_for
from .simulate_early_candidate_v0 import strongest_rival

YEARS=(2023,2024,2025)
OUT=Path('data/audits/three_year_condition_search_v1.json')


def read_csv(p:Path):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def f(v):
    x=num(v)
    return 0.0 if x==float('-inf') else float(x)

def load(year:int):
    d=Path(f'data/{year}/s_class_yosen')
    races=read_csv(d/'races.csv'); entries=read_csv(d/'entries.csv'); payouts=read_csv(d/'payouts.csv')
    eb=defaultdict(list); tri=defaultdict(dict); groups=defaultdict(list)
    for e in entries: eb[e['race_id']].append(e)
    for p in payouts:
        if p.get('ticket_type')=='3連単' and p.get('status')=='paid' and p.get('combination'):
            try: tri[p['race_id']][p['combination']]=int(p['payout_yen'])
            except ValueError: pass
    for r in races: groups[(r['race_date'],r['track'])].append(r)
    seg={}
    for g in groups.values():
        g.sort(key=lambda r:int(r['race_no']))
        for i,r in enumerate(g,1): seg[r['race_id']]=segment_for(i,len(g))
    return races,eb,tri,seg


def feature_row(race, es, main_id, main, rival):
    A,B,M3=main[0],main[1],main[2]
    R1L,R1B=rival[0],rival[1]
    def sm(a,b,k): return f(a.get(k))+f(b.get(k))
    return {
      'race_id':race['race_id'],'race_date':race['race_date'],'track':race['track'],'race_no':int(race['race_no']),
      'A':int(A['car_no']),'B':int(B['car_no']),'M3':int(M3['car_no']),'R1L':int(R1L['car_no']),'R1B':int(R1B['car_no']),
      'a_score':f(A.get('score')),'b_score':f(B.get('score')),'r1l_score':f(R1L.get('score')),'r1b_score':f(R1B.get('score')),
      'main_score':sm(A,B,'score'),'rival_score':sm(R1L,R1B,'score'),
      'main_win':sm(A,B,'win_rate'),'rival_win':sm(R1L,R1B,'win_rate'),
      'main_top2':sm(A,B,'top2_rate'),'rival_top2':sm(R1L,R1B,'top2_rate'),
      'main_top3':sm(A,B,'top3_rate'),'rival_top3':sm(R1L,R1B,'top3_rate'),
      'leader_score_gap':f(A.get('score'))-f(R1L.get('score')),
      'second_score_gap':f(B.get('score'))-f(R1B.get('score')),
      'pair_score_gap':sm(A,B,'score')-sm(R1L,R1B,'score'),
      'pair_win_gap':sm(A,B,'win_rate')-sm(R1L,R1B,'win_rate'),
      'pair_top2_gap':sm(A,B,'top2_rate')-sm(R1L,R1B,'top2_rate'),
      'pair_top3_gap':sm(A,B,'top3_rate')-sm(R1L,R1B,'top3_rate'),
    }


def formations(row, es):
    A,B,M3,R1L,R1B=[row[k] for k in ('A','B','M3','R1L','R1B')]
    excluded={A,B,M3}
    remain=[e for e in es if int(e['car_no']) not in excluded]
    X=max(remain,key=lambda e:(f(e.get('score')),-int(e['car_no']))) if remain else None
    x=int(X['car_no']) if X else None
    out={
      'RIVAL4':[f'{R1L}-{R1B}-{A}',f'{R1L}-{R1B}-{B}',f'{R1B}-{R1L}-{A}',f'{R1B}-{R1L}-{B}'],
      'MAIN4_RIVAL':[f'{A}-{B}-{R1L}',f'{A}-{B}-{R1B}',f'{B}-{A}-{R1L}',f'{B}-{A}-{R1B}'],
      'MAIN6':[f'{A}-{B}-{M3}',f'{B}-{A}-{M3}',f'{A}-{B}-{R1L}',f'{B}-{A}-{R1L}',f'{A}-{B}-{R1B}',f'{B}-{A}-{R1B}'],
      'MIX2':[f'{A}-{B}-{R1B}',f'{R1L}-{R1B}-{B}'],
    }
    if x is not None:
        out['MAIN4_X']=[f'{A}-{B}-{M3}',f'{B}-{A}-{M3}',f'{A}-{B}-{x}',f'{B}-{A}-{x}']
    return out

# Predeclared coarse atoms only. No data-derived threshold generation.
ATOM_SPECS={
 'a_score': [('ge',t) for t in (98,100,102,104,106,108,110)] + [('le',t) for t in (98,100,102,104,106,108,110)],
 'b_score': [('ge',t) for t in (98,100,102,104,106,108,110)] + [('le',t) for t in (98,100,102,104,106,108,110)],
 'r1l_score': [('ge',t) for t in (98,100,102,104,106,108,110)] + [('le',t) for t in (98,100,102,104,106,108,110)],
 'main_score': [('ge',t) for t in (200,204,208,212,216,220)] + [('le',t) for t in (200,204,208,212,216,220)],
 'rival_score': [('ge',t) for t in (196,200,204,208,212,216)] + [('le',t) for t in (196,200,204,208,212,216)],
 'main_win': [('ge',t) for t in (20,30,40,50,60,70)] + [('le',t) for t in (20,30,40,50,60,70)],
 'rival_win': [('ge',t) for t in (20,30,40,50,60)] + [('le',t) for t in (20,30,40,50,60)],
 'main_top2': [('ge',t) for t in (50,60,70,80,90,100)] + [('le',t) for t in (50,60,70,80,90,100)],
 'rival_top2': [('ge',t) for t in (40,50,60,70,80,90)] + [('le',t) for t in (40,50,60,70,80,90)],
 'main_top3': [('ge',t) for t in (70,80,90,100,110,120,130)] + [('le',t) for t in (70,80,90,100,110,120,130)],
 'rival_top3': [('ge',t) for t in (60,70,80,90,100,110,120)] + [('le',t) for t in (60,70,80,90,100,110,120)],
 'leader_score_gap': [('ge',t) for t in (-6,-4,-2,0,2,4,6)] + [('le',t) for t in (-6,-4,-2,0,2,4,6)],
 'second_score_gap': [('ge',t) for t in (-6,-4,-2,0,2,4,6)] + [('le',t) for t in (-6,-4,-2,0,2,4,6)],
 'pair_score_gap': [('ge',t) for t in (-10,-6,-2,2,6,10,14)] + [('le',t) for t in (-10,-6,-2,2,6,10,14)],
 'pair_win_gap': [('ge',t) for t in (-30,-20,-10,0,10,20,30)] + [('le',t) for t in (-30,-20,-10,0,10,20,30)],
 'pair_top2_gap': [('ge',t) for t in (-40,-25,-10,0,10,25,40)] + [('le',t) for t in (-40,-25,-10,0,10,25,40)],
 'pair_top3_gap': [('ge',t) for t in (-40,-25,-10,0,10,25,40)] + [('le',t) for t in (-40,-25,-10,0,10,25,40)],
}
ATOMS=[]
for feat,specs in ATOM_SPECS.items():
    for op,t in specs:
        ATOMS.append((f'{feat}_{op}_{t}',feat,op,float(t)))

def passes(r,atom):
    _,feat,op,t=atom; v=r[feat]
    return v>=t if op=='ge' else v<=t

def compatible(a,b):
    # avoid contradictory/redundant same-feature conjunctions
    if a[1]!=b[1]: return True
    return False

def fin(rows, form):
    n=len(rows); stake=payout=hits=0; hit_pays=[]
    halves=[{'n':0,'hits':0,'stake':0,'payout':0},{'n':0,'hits':0,'stake':0,'payout':0}]
    for r in rows:
        bets=r['forms'][form]; s=100*len(bets); p=sum(r['tri'].get(x,0) for x in bets)
        stake+=s; payout+=p; hits+=int(p>0)
        if p: hit_pays.append(p)
        h=0 if r['race_date']<=f"{r['year']}-06-30" else 1
        halves[h]['n']+=1; halves[h]['stake']+=s; halves[h]['payout']+=p; halves[h]['hits']+=int(p>0)
    for h in halves: h['roi']=h['payout']/h['stake'] if h['stake'] else 0.0
    return {
      'races':n,'hits':hits,'hit_rate':hits/n if n else 0.0,'stake_yen':stake,'payout_yen':payout,
      'profit_yen':payout-stake,'roi':payout/stake if stake else 0.0,
      'max_hit_payout_yen':max(hit_pays) if hit_pays else 0,
      'top1_payout_share':max(hit_pays)/payout if payout and hit_pays else 0.0,
      'halves':halves,
    }

def main():
    datasets={}
    segment_counts={}
    for year in YEARS:
        races,eb,tri,seg=load(year)
        ys=defaultdict(list)
        for race in races:
            rid=race['race_id']; s=seg.get(rid)
            if s not in ('前半','中盤','後半'): continue
            chosen=choose_main_line(eb[rid])
            if not chosen: continue
            mid,main=chosen
            if len(main)<3: continue
            rival=strongest_rival(eb[rid],mid)
            if not rival or len(rival)<2: continue
            row=feature_row(race,eb[rid],mid,main,rival)
            row['year']=year; row['tri']=tri[rid]; row['forms']=formations(row,eb[rid])
            ys[s].append(row)
        datasets[year]=ys
        segment_counts[str(year)]={s:len(ys[s]) for s in ('前半','中盤','後半')}

    # segment-specific predeclared formations
    FORMS={
      '前半':['RIVAL4','MAIN4_RIVAL','MIX2'],
      '中盤':['MAIN6','MAIN4_RIVAL','RIVAL4','MIX2'],
      '後半':['MAIN4_X','MAIN6','MAIN4_RIVAL','RIVAL4'],
    }
    out={
      'scope':'2023+2024+2025 development search. This is NOT OOS; any selected strategy needs fresh 2022 OOS.',
      'guardrails':{
        'conditions':'1 or 2 coarse predeclared atoms only','min_races_each_year':10,'min_hits_each_year':3,
        'roi_each_year':'>1.0','max_top1_payout_share_each_year':0.70,
        'half_year_requirement':'at least 1 hit in each half of each year',
        'ranking':'worst-year ROI, then median-year ROI, then combined ROI, then lower complexity',
      },
      'segment_counts':segment_counts,'atom_count':len(ATOMS),'segments':{}
    }
    for segment in ('前半','中盤','後半'):
        rules=[(a[0],[a]) for a in ATOMS]
        rules += [(a[0]+'__AND__'+b[0],[a,b]) for i,a in enumerate(ATOMS) for b in ATOMS[i+1:] if compatible(a,b)]
        candidates=[]; evaluated=0
        for rule_name,atoms in rules:
            selected={y:[r for r in datasets[y][segment] if all(passes(r,a) for a in atoms)] for y in YEARS}
            if min(len(selected[y]) for y in YEARS)<10: continue
            for form in FORMS[segment]:
                evaluated+=1
                stats={y:fin(selected[y],form) for y in YEARS}
                if min(stats[y]['hits'] for y in YEARS)<3: continue
                if min(stats[y]['roi'] for y in YEARS)<=1.0: continue
                if max(stats[y]['top1_payout_share'] for y in YEARS)>0.70: continue
                if any(min(h['hits'] for h in stats[y]['halves'])<1 for y in YEARS): continue
                st=sum(stats[y]['stake_yen'] for y in YEARS); pay=sum(stats[y]['payout_yen'] for y in YEARS)
                rois=sorted(stats[y]['roi'] for y in YEARS)
                c={'rule':rule_name,'complexity':len(atoms),'formation':form,
                   **{str(y):stats[y] for y in YEARS},'worst_year_roi':rois[0],'median_year_roi':rois[1],
                   'combined':{'races':sum(stats[y]['races'] for y in YEARS),'hits':sum(stats[y]['hits'] for y in YEARS),
                               'stake_yen':st,'payout_yen':pay,'profit_yen':pay-st,'roi':pay/st if st else 0.0}}
                candidates.append(c)
        candidates.sort(key=lambda c:(c['worst_year_roi'],c['median_year_roi'],c['combined']['roi'],-c['complexity']),reverse=True)
        # leave-one-year-out stability: rank candidates using only other 2 years and report heldout performance
        for c in candidates[:100]:
            loo={}
            for held in YEARS:
                train=[y for y in YEARS if y!=held]
                loo[str(held)]={'train_worst_roi':min(c[str(y)]['roi'] for y in train),'train_combined_roi':sum(c[str(y)]['payout_yen'] for y in train)/sum(c[str(y)]['stake_yen'] for y in train),'heldout_roi':c[str(held)]['roi'],'heldout_races':c[str(held)]['races'],'heldout_hits':c[str(held)]['hits']}
            c['loo']=loo
        out['segments'][segment]={'formations':FORMS[segment],'evaluated_rule_formation_count':evaluated,'qualified_count':len(candidates),'top_candidates':candidates[:30]}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({s:{'qualified':out['segments'][s]['qualified_count'],'top':out['segments'][s]['top_candidates'][:3]} for s in out['segments']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

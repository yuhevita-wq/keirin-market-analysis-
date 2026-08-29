from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from .research_high_payout_structures_v1 import load_year, parse_combination
from .research_weakest_line_2023_2025 import weakest_line
from .simulate_early_candidate_v0 import pair_strength
from .simulate_mainline_v1 import num
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival
from .audit_weakest_line_formations_2023_2025 import FORMS

YEARS=(2023,2024,2025)
SEGMENTS=('前半','中盤','後半')
OUT=Path('data/audits/weakest_line_buy_condition_search_2023_2025.json')

# Only weakest-line-oriented formations already audited before this search.
FORMATIONS=(
    'WLEAD_MAINPAIR_6',
    'WSECOND_MAINPAIR_6',
    'WEAK1_MAINPAIR_12',
    'WEAKPAIR_MAIN_18',
    'WEAKPAIR_RIVAL_12',
    'WHEAD_AB_4',
    'WPAIR_TOP2_MAIN_6',
)

def f(v):
    x=num(v)
    return 0.0 if x==float('-inf') else float(x)

def sm(a,b,key):
    return f(a.get(key))+f(b.get(key))

def make_features(main,rival,weak_mem):
    A,B=main[0],main[1]
    R1L,R1B=rival[0],rival[1]
    W1L,W1B=weak_mem[0],weak_mem[1]
    weak_score=sm(W1L,W1B,'score')
    weak_win=sm(W1L,W1B,'win_rate')
    weak_top2=sm(W1L,W1B,'top2_rate')
    weak_top3=sm(W1L,W1B,'top3_rate')
    main_score=sm(A,B,'score'); rival_score=sm(R1L,R1B,'score')
    main_top2=sm(A,B,'top2_rate'); rival_top2=sm(R1L,R1B,'top2_rate')
    main_top3=sm(A,B,'top3_rate'); rival_top3=sm(R1L,R1B,'top3_rate')
    return {
        'weak_lead_score':f(W1L.get('score')),
        'weak_second_score':f(W1B.get('score')),
        'weak_score':weak_score,
        'weak_win':weak_win,
        'weak_top2':weak_top2,
        'weak_top3':weak_top3,
        'weak_size':float(len(weak_mem)),
        'main_minus_weak_score':main_score-weak_score,
        'rival_minus_weak_score':rival_score-weak_score,
        'main_minus_weak_top2':main_top2-weak_top2,
        'rival_minus_weak_top2':rival_top2-weak_top2,
        'main_minus_weak_top3':main_top3-weak_top3,
        'rival_minus_weak_top3':rival_top3-weak_top3,
        'lead_score_gap_main_weak':f(A.get('score'))-f(W1L.get('score')),
        'second_score_gap_main_weak':f(B.get('score'))-f(W1B.get('score')),
    }

# Coarse, predeclared thresholds only. No data-derived threshold generation.
ATOM_SPECS={
    'weak_lead_score':[('ge',t) for t in (94,96,98,100,102,104)]+[('le',t) for t in (94,96,98,100,102,104)],
    'weak_second_score':[('ge',t) for t in (94,96,98,100,102,104)]+[('le',t) for t in (94,96,98,100,102,104)],
    'weak_score':[('ge',t) for t in (190,194,198,202,206)]+[('le',t) for t in (190,194,198,202,206)],
    'weak_win':[('ge',t) for t in (10,20,30,40,50)]+[('le',t) for t in (10,20,30,40,50)],
    'weak_top2':[('ge',t) for t in (30,40,50,60,70,80)]+[('le',t) for t in (30,40,50,60,70,80)],
    'weak_top3':[('ge',t) for t in (50,60,70,80,90,100)]+[('le',t) for t in (50,60,70,80,90,100)],
    'weak_size':[('ge',3),('le',2)],
    'main_minus_weak_score':[('ge',t) for t in (4,8,12,16,20)]+[('le',t) for t in (4,8,12,16,20)],
    'rival_minus_weak_score':[('ge',t) for t in (2,4,6,8,10,12)]+[('le',t) for t in (2,4,6,8,10,12)],
    'main_minus_weak_top2':[('ge',t) for t in (10,20,30,40)]+[('le',t) for t in (10,20,30,40)],
    'rival_minus_weak_top2':[('ge',t) for t in (0,10,20,30)]+[('le',t) for t in (0,10,20,30)],
    'main_minus_weak_top3':[('ge',t) for t in (10,20,30,40)]+[('le',t) for t in (10,20,30,40)],
    'rival_minus_weak_top3':[('ge',t) for t in (0,10,20,30)]+[('le',t) for t in (0,10,20,30)],
    'lead_score_gap_main_weak':[('ge',t) for t in (0,2,4,6,8)]+[('le',t) for t in (0,2,4,6,8)],
    'second_score_gap_main_weak':[('ge',t) for t in (0,2,4,6,8)]+[('le',t) for t in (0,2,4,6,8)],
}
ATOMS=[]
for feat,specs in ATOM_SPECS.items():
    for op,t in specs:
        ATOMS.append((f'{feat}_{op}_{t}',feat,op,float(t)))

def passes(row, atom):
    _,feat,op,t=atom
    v=row[feat]
    return v>=t if op=='ge' else v<=t

def compatible(a,b):
    return a[1]!=b[1]

def family(atoms,form):
    if not atoms: return f'{form}|ALL'
    return form+'|'+'&'.join(sorted(f'{a[1]}:{a[2]}' for a in atoms))

def half_key(date):
    return 'H1' if date[5:7] <= '06' else 'H2'

def stats(rows,form):
    orders=FORMS[form]
    n=len(rows); stake=100*len(orders)*n; payout=0; hits=0; hitp=[]; halves={'H1':0,'H2':0}
    cur=mx=0
    for r in sorted(rows,key=lambda z:z['race_date']):
        if r['role_order'] in orders:
            hits+=1; payout+=r['payout_yen']; hitp.append(r['payout_yen']); halves[half_key(r['race_date'])]+=1; cur=0
        else:
            cur+=1; mx=max(mx,cur)
    hitp.sort()
    return {
        'races':n,'points':len(orders),'hits':hits,'hit_rate':hits/n if n else 0.0,
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi':payout/stake if stake else 0.0,
        'median_hit_payout_yen':hitp[len(hitp)//2] if hitp else 0,
        'mean_hit_payout_yen':sum(hitp)/len(hitp) if hitp else 0.0,
        'high5000_hits':sum(x>=5000 for x in hitp),'high10000_hits':sum(x>=10000 for x in hitp),'high20000_hits':sum(x>=20000 for x in hitp),
        'top1_payout_share':max(hitp)/payout if payout and hitp else 0.0,
        'max_losing_streak':mx,'half_hits':halves,
    }

def build_data():
    ds={}; counts={}
    for y in YEARS:
        races,eb,win3,seg=load_year(y); ys=defaultdict(list)
        for race in races:
            rid=race['race_id']; s=seg.get(rid)
            if s not in SEGMENTS or rid not in win3: continue
            chosen=choose_main_line(eb[rid])
            if not chosen: continue
            mid,main=chosen
            if len(main)<3: continue
            rival=strongest_rival(eb[rid],mid)
            if not rival or len(rival)<2: continue
            weak,reason=weakest_line(eb[rid],mid)
            if not weak: continue
            wid,wmem,wscore=weak
            rival_id=int(rival[0]['line_id'])
            if wid==rival_id: continue
            roles={int(e['car_no']):name for name,e in [('A',main[0]),('B',main[1]),('M3',main[2]),('R1L',rival[0]),('R1B',rival[1]),('W1L',wmem[0]),('W1B',wmem[1])]}
            combo,pay=win3[rid]; cars=parse_combination(combo)
            row={'year':y,'segment':s,'race_id':rid,'race_date':race['race_date'],'role_order':tuple(roles.get(c,'OTHER') for c in cars),'payout_yen':pay}
            row.update(make_features(main,rival,wmem))
            ys[s].append(row)
        ds[y]=ys; counts[str(y)]={s:len(ys[s]) for s in SEGMENTS}
    return ds,counts

def qualifies(st, strict=True):
    min_races=15 if strict else 20
    if min(x['races'] for x in st.values())<min_races: return False
    if min(x['hits'] for x in st.values())<2: return False
    if min(x['roi'] for x in st.values())<=1.0: return False
    if min(x['high5000_hits'] for x in st.values())<1: return False
    if sum(x['high10000_hits'] for x in st.values())<3: return False
    if max(x['top1_payout_share'] for x in st.values())>(0.70 if strict else 0.80): return False
    if min(x['median_hit_payout_yen']/(100*x['points']) if x['points'] else 0 for x in st.values())<2.0: return False
    if strict and any(min(x['half_hits'].values())<1 for x in st.values()): return False
    return True

def main():
    ds,counts=build_data()
    rules=[(a[0],[a]) for a in ATOMS]
    rules += [(a[0]+'__AND__'+b[0],[a,b]) for i,a in enumerate(ATOMS) for b in ATOMS[i+1:] if compatible(a,b)]
    out={
        'status':'WEAKEST_LINE_BUY_CONDITION_SEARCH_2023_2025_ONLY',
        'years_read':list(YEARS),'evaluation_year_2026_used':False,
        'definition':'Search pre-race conditions under which weakest-line-oriented formations are buyable.',
        'guardrails':{
            'conditions':'1 or 2 coarse predeclared weakest-line feature atoms only',
            'strict':'min 15 races/year; 2 hits/year; ROI>100% every year; >=1 >=5000 hit/year; >=3 >=10000 hits total; top1 share<=70% each year; median recovery>=2x each year; >=1 hit in each half of each year',
            'exploratory':'min 20 races/year; 2 hits/year; ROI>100% every year; >=1 >=5000 hit/year; >=3 >=10000 hits total; top1 share<=80% each year; median recovery>=2x each year',
            'family_support':'same formation + feature/direction family must have >=2 qualified thresholds for robust shortlist',
            'ranking':'worst-year ROI, then worst-year median recovery multiple, then combined ROI, then lower complexity, then fewer points',
        },
        'structural_counts':counts,'atom_count':len(ATOMS),'formations':{f:len(FORMS[f]) for f in FORMATIONS},'segments':{}
    }
    for seg in SEGMENTS:
        raw={'strict':[],'exploratory':[]}
        for rname,atoms in rules:
            selected={y:[r for r in ds[y][seg] if all(passes(r,a) for a in atoms)] for y in YEARS}
            if min(len(selected[y]) for y in YEARS)<15: continue
            for form in FORMATIONS:
                st={y:stats(selected[y],form) for y in YEARS}
                for tier in ('strict','exploratory'):
                    if not qualifies(st, strict=(tier=='strict')): continue
                    total_stake=sum(st[y]['stake_yen'] for y in YEARS); total_pay=sum(st[y]['payout_yen'] for y in YEARS)
                    medrec=min(st[y]['median_hit_payout_yen']/(100*st[y]['points']) for y in YEARS)
                    raw[tier].append({
                        'rule':rname,'atoms':[a[0] for a in atoms],'complexity':len(atoms),'formation':form,'points':len(FORMS[form]),'family':family(atoms,form),
                        'periods':{str(y):st[y] for y in YEARS},'worst_year_roi':min(st[y]['roi'] for y in YEARS),'worst_year_median_recovery_multiple':medrec,
                        'combined':{'races':sum(st[y]['races'] for y in YEARS),'hits':sum(st[y]['hits'] for y in YEARS),'stake_yen':total_stake,'payout_yen':total_pay,'profit_yen':total_pay-total_stake,'roi':total_pay/total_stake if total_stake else 0.0,'high10000_hits':sum(st[y]['high10000_hits'] for y in YEARS),'high20000_hits':sum(st[y]['high20000_hits'] for y in YEARS)}
                    })
        outseg={}
        for tier,cands in raw.items():
            fam=defaultdict(int)
            for c in cands: fam[c['family']]+=1
            robust=[c for c in cands if fam[c['family']]>=2]
            robust.sort(key=lambda c:(c['worst_year_roi'],c['worst_year_median_recovery_multiple'],c['combined']['roi'],-c['complexity'],-c['points']),reverse=True)
            for c in robust: c['family_qualified_count']=fam[c['family']]
            outseg[tier]={'qualified_before_family':len(cands),'qualified_robust':len(robust),'recommended':robust[0] if robust else None,'shortlist':robust[:20]}
        out['segments'][seg]=outseg
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({s:{t:{'qualified':out['segments'][s][t]['qualified_robust'],'recommended':out['segments'][s][t]['recommended']} for t in ('strict','exploratory')} for s in SEGMENTS},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())

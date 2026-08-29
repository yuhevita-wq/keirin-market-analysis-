from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import search_three_year_conditions_v1 as s

YEARS=(2023,2024,2025)
OUT=Path('data/audits/three_year_selection_stability.json')


def build():
    datasets={}
    for year in YEARS:
        races,eb,tri,seg=s.load(year)
        ys=defaultdict(list)
        for race in races:
            rid=race['race_id']; section=seg.get(rid)
            if section not in ('前半','中盤','後半'): continue
            chosen=s.choose_main_line(eb[rid])
            if not chosen: continue
            mid,main=chosen
            if len(main)<3: continue
            rival=s.strongest_rival(eb[rid],mid)
            if not rival or len(rival)<2: continue
            row=s.feature_row(race,eb[rid],mid,main,rival)
            row['year']=year; row['tri']=tri[rid]; row['forms']=s.formations(row,eb[rid])
            ys[section].append(row)
        datasets[year]=ys
    return datasets

FORMS={'前半':['RIVAL4','MAIN4_RIVAL','MIX2'],'中盤':['MAIN6','MAIN4_RIVAL','RIVAL4','MIX2'],'後半':['MAIN4_X','MAIN6','MAIN4_RIVAL','RIVAL4']}
RULES=None

def rules():
    global RULES
    if RULES is not None:return RULES
    r=[(a[0],[a]) for a in s.ATOMS]
    r += [(a[0]+'__AND__'+b[0],[a,b]) for i,a in enumerate(s.ATOMS) for b in s.ATOMS[i+1:] if s.compatible(a,b)]
    RULES=r; return r

def qualifies(st):
    return st['races']>=10 and st['hits']>=3 and st['roi']>1.0 and st['top1_payout_share']<=0.70 and min(h['hits'] for h in st['halves'])>=1

def candidate_stats(datasets,segment,rule_name,atoms,form,years):
    stats={}
    for y in years:
        rows=[r for r in datasets[y][segment] if all(s.passes(r,a) for a in atoms)]
        stats[y]=s.fin(rows,form)
    return stats

def rank_tuple(stats,years,complexity):
    rois=sorted(stats[y]['roi'] for y in years)
    st=sum(stats[y]['stake_yen'] for y in years); pay=sum(stats[y]['payout_yen'] for y in years)
    return (rois[0],pay/st if st else 0.0,-complexity)

def pack(rule_name,atoms,form,stats,years):
    st=sum(stats[y]['stake_yen'] for y in years); pay=sum(stats[y]['payout_yen'] for y in years)
    return {'rule':rule_name,'complexity':len(atoms),'formation':form,**{str(y):stats[y] for y in years},
            'worst_year_roi':min(stats[y]['roi'] for y in years),'combined_roi':pay/st if st else 0.0,
            'combined_races':sum(stats[y]['races'] for y in years),'combined_hits':sum(stats[y]['hits'] for y in years)}

def main():
    d=build(); out={'scope':'selection-stability audit on 2023-2025 development data; fresh 2022 still required','segments':{}}
    for segment in ('前半','中盤','後半'):
        all3=[]; best_simple=None; best_large=None
        # all-three-qualified summaries
        for rule_name,atoms in rules():
            selected={y:[r for r in d[y][segment] if all(s.passes(r,a) for a in atoms)] for y in YEARS}
            if min(len(selected[y]) for y in YEARS)<10: continue
            for form in FORMS[segment]:
                stats={y:s.fin(selected[y],form) for y in YEARS}
                if not all(qualifies(stats[y]) for y in YEARS): continue
                c=pack(rule_name,atoms,form,stats,YEARS); all3.append(c)
                if len(atoms)==1 and (best_simple is None or rank_tuple(stats,YEARS,1)>rank_tuple({y:best_simple[str(y)] for y in YEARS},YEARS,1)): best_simple=c
                if min(stats[y]['races'] for y in YEARS)>=30 and (best_large is None or rank_tuple(stats,YEARS,len(atoms))>rank_tuple({y:best_large[str(y)] for y in YEARS},YEARS,best_large['complexity'])): best_large=c
        all3.sort(key=lambda c:(c['worst_year_roi'],c['combined_roi'],-c['complexity']),reverse=True)

        # True leave-one-year-out: select using training years ONLY, then evaluate untouched held year.
        loo={}
        for held in YEARS:
            train=[y for y in YEARS if y!=held]; best=None; best_key=None; best_atoms=None
            for rule_name,atoms in rules():
                selected={y:[r for r in d[y][segment] if all(s.passes(r,a) for a in atoms)] for y in train}
                if min(len(selected[y]) for y in train)<10: continue
                for form in FORMS[segment]:
                    stats={y:s.fin(selected[y],form) for y in train}
                    if not all(qualifies(stats[y]) for y in train): continue
                    key=rank_tuple(stats,train,len(atoms))
                    if best is None or key>best_key:
                        best=pack(rule_name,atoms,form,stats,train); best_key=key; best_atoms=atoms
            if best is None:
                loo[str(held)]={'selected':None}; continue
            hrows=[r for r in d[held][segment] if all(s.passes(r,a) for a in best_atoms)]
            hstat=s.fin(hrows,best['formation'])
            loo[str(held)]={'selected':best,'heldout':hstat,'heldout_profitable':hstat['roi']>1.0}
        out['segments'][segment]={
          'all3_qualified_count':len(all3),'best_overall':all3[0] if all3 else None,'best_simple_one_condition':best_simple,
          'best_large_min30_each_year':best_large,'true_leave_one_year_out':loo
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

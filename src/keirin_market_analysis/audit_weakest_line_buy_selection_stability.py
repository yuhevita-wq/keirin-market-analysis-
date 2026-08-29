from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .search_weakest_line_buy_conditions_2023_2025 import (
    YEARS, ATOMS, FORMATIONS, FORMS, build_data, passes, compatible, stats, family
)

OUT=Path('data/audits/weakest_line_buy_selection_stability_2023_2025.json')
SEGMENT='中盤'

def train_qualifies(st):
    if min(x['races'] for x in st.values())<15: return False
    if min(x['hits'] for x in st.values())<2: return False
    if min(x['roi'] for x in st.values())<=1.0: return False
    if min(x['high5000_hits'] for x in st.values())<1: return False
    if max(x['top1_payout_share'] for x in st.values())>0.70: return False
    if min(x['median_hit_payout_yen']/(100*x['points']) if x['points'] else 0 for x in st.values())<2.0: return False
    if any(min(x['half_hits'].values())<1 for x in st.values()): return False
    return True

def candidate_rank(c):
    return (c['worst_train_roi'],c['worst_train_median_recovery_multiple'],c['combined_train_roi'],-c['complexity'],-c['points'])

def main():
    ds,counts=build_data()
    rules=[(a[0],[a]) for a in ATOMS]
    rules += [(a[0]+'__AND__'+b[0],[a,b]) for i,a in enumerate(ATOMS) for b in ATOMS[i+1:] if compatible(a,b)]
    folds={}
    for held in YEARS:
        train=[y for y in YEARS if y!=held]
        cands=[]
        for rname,atoms in rules:
            selected={y:[r for r in ds[y][SEGMENT] if all(passes(r,a) for a in atoms)] for y in train}
            if min(len(selected[y]) for y in train)<15: continue
            for form in FORMATIONS:
                st={y:stats(selected[y],form) for y in train}
                if not train_qualifies(st): continue
                fam=family(atoms,form)
                total_stake=sum(st[y]['stake_yen'] for y in train); total_pay=sum(st[y]['payout_yen'] for y in train)
                cands.append({
                    'rule':rname,'atoms':[a[0] for a in atoms],'complexity':len(atoms),'formation':form,'points':len(FORMS[form]),'family':fam,
                    'train':{str(y):st[y] for y in train},
                    'worst_train_roi':min(st[y]['roi'] for y in train),
                    'worst_train_median_recovery_multiple':min(st[y]['median_hit_payout_yen']/(100*st[y]['points']) for y in train),
                    'combined_train_roi':total_pay/total_stake if total_stake else 0.0,
                    '_atoms':atoms,
                })
        famcount=defaultdict(int)
        for c in cands: famcount[c['family']]+=1
        robust=[c for c in cands if famcount[c['family']]>=2]
        robust.sort(key=candidate_rank,reverse=True)
        if robust:
            sel=robust[0]
            hold_rows=[r for r in ds[held][SEGMENT] if all(passes(r,a) for a in sel['_atoms'])]
            h=stats(hold_rows,sel['formation'])
            clean={k:v for k,v in sel.items() if k!='_atoms'}
            clean['family_qualified_count_train']=famcount[sel['family']]
            clean['heldout_year']=held
            clean['heldout']=h
            clean['heldout_roi_gt_1']=h['roi']>1.0
            clean['heldout_min_2_hits']=h['hits']>=2
            folds[str(held)]={'training_years':train,'qualified_before_family':len(cands),'qualified_robust':len(robust),'selected':clean}
        else:
            folds[str(held)]={'training_years':train,'qualified_before_family':len(cands),'qualified_robust':0,'selected':None}

    # Local grid around the full-sample strict recommendation.
    grid=[]
    def atom_by(name):
        return next(a for a in ATOMS if a[0]==name)
    for wl in (96,98,100):
        for gap in (4,6,8):
            atoms=[atom_by(f'weak_lead_score_le_{wl}'),atom_by(f'rival_minus_weak_score_le_{gap}')]
            per={}
            for y in YEARS:
                rows=[r for r in ds[y][SEGMENT] if all(passes(r,a) for a in atoms)]
                per[str(y)]=stats(rows,'WEAK1_MAINPAIR_12')
            grid.append({'weak_lead_score_le':wl,'rival_minus_weak_score_le':gap,'periods':per,'all3_roi_gt_1':all(per[str(y)]['roi']>1.0 for y in YEARS),'all3_min2_hits':all(per[str(y)]['hits']>=2 for y in YEARS)})

    out={
        'status':'WEAKEST_LINE_BUY_SELECTION_STABILITY_2023_2025_ONLY',
        'years_read':list(YEARS),'evaluation_year_2026_used':False,
        'segment':SEGMENT,
        'selection_rule':'For each held-out year, search the same 1-2 coarse weakest-line atoms and formations using only the other two years; require training strict guardrails and family support>=2; rank by worst training ROI, recovery multiple, combined ROI, simplicity, fewer points.',
        'folds':folds,
        'heldout_positive_count':sum(1 for x in folds.values() if x['selected'] and x['selected']['heldout_roi_gt_1']),
        'heldout_positive_and_2hits_count':sum(1 for x in folds.values() if x['selected'] and x['selected']['heldout_roi_gt_1'] and x['selected']['heldout_min_2_hits']),
        'recommended_family_neighborhood':grid,
        'neighborhood_all3_positive_count':sum(1 for x in grid if x['all3_roi_gt_1']),
        'neighborhood_all3_positive_and_2hits_count':sum(1 for x in grid if x['all3_roi_gt_1'] and x['all3_min2_hits']),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())

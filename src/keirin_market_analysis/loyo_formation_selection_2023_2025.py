from __future__ import annotations

import json
from pathlib import Path

from .simulate_formation_slots_2023_2025 import ATOMS, calc, make_rows, masks
from .analyze_multifeature_signals_2023_2025 import YEARS, SEGMENTS, quantile_cut

OUT=Path('data/audits/formation_slot_loyo_2023_2025.json')

NUM_GATES={
    'WEAK_TOP2_HIGH':('weak_top2_rate','HIGH',.75),
    'WEAK_TOP3_HIGH':('weak_top3_rate','HIGH',.75),
    'W1L_ATTACK_HIGH':('w1l_attack','HIGH',.75),
    'WEAK_B_HIGH':('weak_b_count','HIGH',.75),
    'RW_ATTACK_GAP_LOW':('rival_weak_attack_gap','LOW',.25),
    'W1L_TOP3_HIGH':('w1l_top3_rate','HIGH',.75),
    'W1B_TOP3_HIGH':('w1b_top3_rate','HIGH',.75),
}
BOOL_GATES=('HIDDEN_ATTACK','HIDDEN_TOP2','HIDDEN_TOP3')


def build_gate(train,segment,name):
    sr=[r for r in train if r['segment']==segment]
    if name=='ALL': return (lambda r: True),None
    if name=='HIDDEN_ATTACK': return (lambda r: bool(r['weak_hidden_attack_vs_rival'])),None
    if name=='HIDDEN_TOP2': return (lambda r: bool(r['weak_hidden_top2_vs_rival'])),None
    if name=='HIDDEN_TOP3': return (lambda r: bool(r['weak_hidden_top3_vs_rival'])),None
    feat,direction,q=NUM_GATES[name]
    cut=quantile_cut([float(r[feat]) for r in sr],q)
    if direction=='HIGH': return (lambda r,feat=feat,cut=cut: r[feat]>=cut),cut
    return (lambda r,feat=feat,cut=cut: r[feat]<=cut),cut


def metric(rows,orders,points):
    x=calc(rows,orders,points)
    return x


def main():
    rows=make_rows(); ms=masks(); gate_names=['ALL',*BOOL_GATES,*NUM_GATES.keys()]
    results=[]
    for hold in YEARS:
        train_years=[y for y in YEARS if y!=hold]
        train=[r for r in rows if r['year'] in train_years]
        test=[r for r in rows if r['year']==hold]
        for s in SEGMENTS:
            cands=[]
            for gname in gate_names:
                gfn,cut=build_gate(train,s,gname)
                tr=[r for r in train if r['segment']==s and gfn(r)]
                te=[r for r in test if r['segment']==s and gfn(r)]
                for m in ms:
                    per={y:metric([r for r in tr if r['year']==y],m['orders'],m['points']) for y in train_years}
                    if min(per[y]['races'] for y in train_years)<15: continue
                    if min(per[y]['race_hits'] for y in train_years)<2: continue
                    if not all(per[y]['roi']>1 for y in train_years): continue
                    comb=metric(tr,m['orders'],m['points'])
                    if comb['top1_payout_share']>0.6: continue
                    cands.append({'gate':gname,'cut':cut,'atoms':list(m['atoms']),'points':m['points'],'orders':m['orders'],
                                  'train_by_year':per,'train_combined':comb,
                                  'worst_train_roi':min(per[y]['roi'] for y in train_years),
                                  'min_train_hits':min(per[y]['race_hits'] for y in train_years),'test_rows':te})
            cands.sort(key=lambda c:(c['worst_train_roi'],c['min_train_hits'],c['train_combined']['roi'],c['train_combined']['race_hits']),reverse=True)
            if not cands:
                results.append({'holdout_year':hold,'segment':s,'status':'NO_TRAIN_QUALIFIER'})
                continue
            b=cands[0]; testm=metric(b['test_rows'],b['orders'],b['points'])
            results.append({
                'holdout_year':hold,'segment':s,'status':'SELECTED_AND_TESTED','train_years':train_years,
                'selected':{'gate':b['gate'],'cut_from_train_only':b['cut'],'atoms':b['atoms'],'points':b['points'],
                            'worst_train_roi':b['worst_train_roi'],'min_train_hits':b['min_train_hits'],
                            'train_combined_roi':b['train_combined']['roi']},
                'holdout':{k:testm[k] for k in ('races','points','race_hits','hit_rate','stake_yen','payout_yen','profit_yen','roi','median_hit_payout_yen','ge10000_hits','ge20000_hits','max_losing_streak','top1_payout_share')}
            })
    tested=[r for r in results if r['status']=='SELECTED_AND_TESTED']
    out={'status':'FORMATION_SELECTION_LOYO_2023_2025','years_read':list(YEARS),'evaluation_year_2026_used':False,
         'selection_protocol':'For each held-out year and segment, derive numeric gate thresholds from the other two years only; require both training years ROI>1, >=15 races and >=2 hits each, train combined top1 payout share<=0.6; rank by worst training-year ROI then min hits then combined ROI.',
         'results':results,
         'summary':{'tests':len(tested),'profitable_holdouts':sum(r['holdout']['roi']>1 for r in tested),
                    'by_segment':{s:{'tests':sum(r['segment']==s for r in tested),'profitable':sum(r['segment']==s and r['holdout']['roi']>1 for r in tested)} for s in SEGMENTS}}
         }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())

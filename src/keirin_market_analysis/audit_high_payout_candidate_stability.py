from __future__ import annotations
import json
from pathlib import Path
from .search_high_payout_strategy_v1 import YEARS, build_data, stats

OUT=Path('data/audits/high_payout_candidate_stability_2023_2025.json')

FAMILIES={
 'early_o1o2':{
   'segment':'前半','formation':'O1O2_KEY_18',
   'variants':[(b,t) for b in (102,104,106) for t in (60,70,80)],
   'name':lambda v:f'b_score_ge_{v[0]}__main_top2_ge_{v[1]}',
   'pred':lambda r,v:r['b_score']>=v[0] and r['main_top2']>=v[1],
 },
 'middle_o1_mainpair':{
   'segment':'中盤','formation':'O1_MAINPAIR_6',
   'variants':[(b,t) for b in (102,104,106) for t in (60,70,80)],
   'name':lambda v:f'b_score_ge_{v[0]}__main_top2_le_{v[1]}',
   'pred':lambda r,v:r['b_score']>=v[0] and r['main_top2']<=v[1],
 },
 'late_o1_mainpair':{
   'segment':'後半','formation':'O1_MAINPAIR_6',
   'variants':[(rs,t) for rs in (196,200,204) for t in (100,110,120)],
   'name':lambda v:f'rival_score_le_{v[0]}__main_top3_ge_{v[1]}',
   'pred':lambda r,v:r['rival_score']<=v[0] and r['main_top3']>=v[1],
 },
}

def train_ok(st,years):
    return all(st[y]['races']>=15 and st[y]['hits']>=2 and st[y]['roi']>1.0 and st[y]['high5_hits']>=1 and st[y]['median_recovery_multiple']>=2.0 and st[y]['top1_payout_share']<=0.8 for y in years)

def main():
    ds,_=build_data()
    out={'status':'HIGH_PAYOUT_CANDIDATE_STABILITY_2023_2025_ONLY','years_read':list(YEARS),'evaluation_year_2026_used':False,'families':{}}
    for key,fam in FAMILIES.items():
        variants=[]
        for v in fam['variants']:
            selected={y:[r for r in ds[y][fam['segment']] if fam['pred'](r,v)] for y in YEARS}
            st={y:stats(selected[y],fam['formation']) for y in YEARS}
            all3=train_ok(st,YEARS)
            total_stake=sum(st[y]['stake_yen'] for y in YEARS); total_pay=sum(st[y]['payout_yen'] for y in YEARS)
            variants.append({'rule':fam['name'](v),'params':list(v),'all3_guardrails':all3,'periods':{str(y):st[y] for y in YEARS},'worst_year_roi':min(st[y]['roi'] for y in YEARS),'combined_roi':total_pay/total_stake if total_stake else 0.0})
        variants.sort(key=lambda z:(z['all3_guardrails'],z['worst_year_roi'],z['combined_roi']),reverse=True)
        loo={}
        for held in YEARS:
            train=[y for y in YEARS if y!=held]
            eligible=[]
            for z in variants:
                st={y:z['periods'][str(y)] for y in YEARS}
                if train_ok(st,train):
                    train_stake=sum(st[y]['stake_yen'] for y in train); train_pay=sum(st[y]['payout_yen'] for y in train)
                    eligible.append((min(st[y]['roi'] for y in train),min(st[y]['median_recovery_multiple'] for y in train),train_pay/train_stake if train_stake else 0,z))
            eligible.sort(key=lambda x:(x[0],x[1],x[2]),reverse=True)
            if eligible:
                z=eligible[0][3]; hs=z['periods'][str(held)]
                loo[str(held)]={'selected_rule':z['rule'],'train_worst_roi':eligible[0][0],'heldout':{k:hs[k] for k in ('races','hits','roi','high5_hits','high10_hits','median_recovery_multiple','top1_payout_share','max_losing_streak')},'heldout_profitable':hs['roi']>1.0}
            else: loo[str(held)]={'selected_rule':None}
        out['families'][key]={'segment':fam['segment'],'formation':fam['formation'],'all3_count':sum(v['all3_guardrails'] for v in variants),'variants':variants,'leave_one_year_out':loo,'loo_profitable_count':sum(x.get('heldout_profitable',False) for x in loo.values())}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{'all3_count':v['all3_count'],'loo_profitable_count':v['loo_profitable_count'],'top':v['variants'][0],'loo':v['leave_one_year_out']} for k,v in out['families'].items()},ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from .search_high_payout_strategy_v1 import YEARS, FORMS, build_data, stats
from .search_three_year_conditions_v1 import ATOMS, passes

OUT=Path('data/audits/high_payout_strategy_simple_2023_2025_v1.json')

def main():
    ds,counts=build_data()
    rules=[('ALL',None)]+[(a[0],a) for a in ATOMS]
    out={'status':'SIMPLE_HIGH_PAYOUT_SEARCH_2023_2025_ONLY','years_read':list(YEARS),'evaluation_year_2026_used':False,'guardrails':{'conditions':'0 or 1 coarse atom only','min_races_each_year':20,'min_hits_each_year':2,'roi_each_year':'>1.0','min_high5000_hits_each_year':1,'min_total_high10000_hits':3,'max_top1_share_each_year':0.8,'min_median_recovery_multiple_each_year':2.0},'formations':{k:len(v) for k,v in FORMS.items()},'segments':{}}
    for seg in ('前半','中盤','後半'):
        cand=[]
        for name,atom in rules:
            selected={y:[r for r in ds[y][seg] if atom is None or passes(r,atom)] for y in YEARS}
            if min(len(selected[y]) for y in YEARS)<20: continue
            for form in FORMS:
                st={y:stats(selected[y],form) for y in YEARS}
                if min(st[y]['hits'] for y in YEARS)<2: continue
                if min(st[y]['roi'] for y in YEARS)<=1.0: continue
                if min(st[y]['high5_hits'] for y in YEARS)<1: continue
                if sum(st[y]['high10_hits'] for y in YEARS)<3: continue
                if max(st[y]['top1_payout_share'] for y in YEARS)>0.8: continue
                if min(st[y]['median_recovery_multiple'] for y in YEARS)<2.0: continue
                ss=sum(st[y]['stake_yen'] for y in YEARS); pp=sum(st[y]['payout_yen'] for y in YEARS)
                cand.append({'rule':name,'formation':form,'points':len(FORMS[form]),'periods':{str(y):st[y] for y in YEARS},'worst_year_roi':min(st[y]['roi'] for y in YEARS),'worst_year_median_recovery_multiple':min(st[y]['median_recovery_multiple'] for y in YEARS),'combined':{'races':sum(st[y]['races'] for y in YEARS),'hits':sum(st[y]['hits'] for y in YEARS),'stake_yen':ss,'payout_yen':pp,'profit_yen':pp-ss,'roi':pp/ss if ss else 0,'high10_hits':sum(st[y]['high10_hits'] for y in YEARS),'high20_hits':sum(st[y]['high20_hits'] for y in YEARS)}})
        cand.sort(key=lambda c:(c['worst_year_roi'],c['worst_year_median_recovery_multiple'],c['combined']['roi'],-c['points']),reverse=True)
        out['segments'][seg]={'qualified':len(cand),'recommended':cand[0] if cand else None,'shortlist':cand[:15]}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({s:{'qualified':out['segments'][s]['qualified'],'recommended':out['segments'][s]['recommended']} for s in out['segments']},ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())

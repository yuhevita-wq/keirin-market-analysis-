from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .analyze_rough_pruning_v1 import CANDIDATES, combos, role_map
from .simulate_branching_v1 import classify, read_csv
from .simulate_mainline_v1 import choose_main_line, segment_for

DATA_DIR=Path('data/2025/s_class_yosen')
OUT_DIR=DATA_DIR/'simulations'/'rough_pruning_robustness'
TARGETS=['P27_base','P18_first_A_R1L','P18_drop_R1B_second']


def metric(rows,name):
    p1,p2,p3=CANDIDATES[name]
    stake=payout=hits=0; hit_payouts=[]
    for r in rows:
        cs=combos(r['roles'],p1,p2,p3)
        stake += len(cs)*100
        ret=sum(r['payouts'].get(c,0) for c in cs)
        payout += ret
        if ret:
            hits += 1; hit_payouts.append(ret)
    hit_payouts.sort(reverse=True)
    return {
        'races':len(rows),'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,
        'roi':payout/stake if stake else 0,'hits':hits,'hit_rate':hits/len(rows) if rows else 0,
        'largest_hit_yen':hit_payouts[0] if hit_payouts else 0,
        'top3_payout_share':sum(hit_payouts[:3])/payout if payout else 0,
    }


def main():
    races=read_csv(DATA_DIR/'races.csv'); entries=read_csv(DATA_DIR/'entries.csv'); payouts=read_csv(DATA_DIR/'payouts.csv')
    e_by=defaultdict(list); p_by=defaultdict(dict); day=defaultdict(list)
    for e in entries:e_by[e['race_id']].append(e)
    for p in payouts:
        if p.get('ticket_type')=='3連単' and p.get('status')=='paid' and p.get('combination'):
            try:p_by[p['race_id']][p['combination']]=int(p['payout_yen'])
            except ValueError:pass
    for r in races:day[(r['race_date'],r['track'])].append(r)
    seg={}
    for g in day.values():
        g.sort(key=lambda r:int(r['race_no']))
        for pos,r in enumerate(g,1):seg[r['race_id']]=segment_for(pos,len(g))
    rows=[]
    for race in sorted(races,key=lambda r:(r['race_date'],r['track'],int(r['race_no']))):
        rid=race['race_id']
        if seg.get(rid)!='後半':continue
        main=choose_main_line(e_by[rid])
        if not main:continue
        mid,members=main
        if len(members)<3:continue
        branch,*_=classify(members)
        if branch!='B_rough':continue
        roles=role_map(e_by[rid],mid,members)
        if roles:
            rows.append({'date':race['race_date'],'month':race['race_date'][:7],'quarter':f"Q{(int(race['race_date'][5:7])-1)//3+1}",'roles':roles,'payouts':p_by[rid]})
    out={'scope':'B rough branch robustness for fixed pruning candidates','targets':{}}
    for name in TARGETS:
        months={}
        for m in sorted({r['month'] for r in rows}):months[m]=metric([r for r in rows if r['month']==m],name)
        quarters={}
        for q in ('Q1','Q2','Q3','Q4'):quarters[q]=metric([r for r in rows if r['quarter']==q],name)
        out['targets'][name]={'all':metric(rows,name),'months':months,'quarters':quarters}
    out['warning']='Robustness check only. P18_first_A_R1L was noticed after H2 was already observed, so this is not a fresh holdout validation.'
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUT_DIR/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

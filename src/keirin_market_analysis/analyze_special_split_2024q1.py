from __future__ import annotations
import json
from collections import Counter, defaultdict
from statistics import median
from simulate_v8_1_f02_2024q1 import load, pi, pl
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_8_f09_market_cliff import build_v8_8_f09
from v8_11_f12_race_type_adaptive import build_v8_11_f12

STAKE=100


def subtype(rt:str)->str:
    t=(rt or '').strip()
    if '初特選' in t:
        return 'INITIAL_SPECIAL'
    if '特選' in t:
        return 'SPECIAL_SELECTION'
    if '選抜' in t:
        return 'SELECTION'
    return 'OTHER_SPECIAL'


def summ(rows, key='base'):
    rs=[r for r in rows if r.get(key+'_buy')]
    if not rs:return {'races':0}
    hits=sum(r[key+'_hit'] for r in rs);tickets=sum(r[key+'_tickets'] for r in rs);payout=sum(r[key+'_payout'] for r in rs);stake=tickets*STAKE
    hit_pays=[r[key+'_payout'] for r in rs if r[key+'_hit']]
    return {
        'races':len(rs),'hits':hits,'hit_rate_pct':100*hits/len(rs),'tickets':tickets,
        'avg_tickets':tickets/len(rs),'stake_yen':stake,'payout_yen':payout,
        'profit_yen':payout-stake,'roi_pct':100*payout/stake if stake else None,
        'median_hit_payout':median(hit_pays) if hit_pays else None,
        'hits_ge_5000':sum(p>=5000 for p in hit_pays),'hits_ge_10000':sum(p>=10000 for p in hit_pays),
    }


def main():
    races,trio,tf,pay=load(); rows=[]; populations=Counter(); exact=Counter()
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'):continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210:continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]:continue
        rt=r.get('race_type') or ''
        if classify_race_type(rt)!='SPECIAL':continue
        st=subtype(rt); populations[st]+=1; exact[rt]+=1
        base=build_v8_8_f09(trio[rid],tf[rid],r.get('predicted_line_formation') or '')
        price=build_v8_11_f12(trio[rid],tf[rid],r.get('predicted_line_formation') or '',rt)
        row={'race_id':rid,'race_date':r.get('race_date'),'race_type':rt,'subtype':st}
        for name,d in [('base',base),('price',price)]:
            buy=bool(d.get('buy')); ts=tuple(d.get('tickets') or ()); wins=[t for t in ts if t in pay[rid]]; payout=sum(pay[rid][t] for t in wins)
            row.update({name+'_buy':buy,name+'_hit':int(bool(wins)) if buy else 0,name+'_payout':payout if buy else 0,name+'_tickets':int(d.get('ticket_count') or 0) if buy else 0,name+'_formation':d.get('formation')})
        eg=base.get('entry_gate') or {}
        if base.get('buy'):
            conc=not bool(eg.get('H_RATIO')); hab=bool(eg.get('H_AB'))
            row.update({'H_CONCENTRATED':conc,'H_AB':hab,'H_state':f'HAB{int(hab)}_CONC{int(conc)}','H1_ratio':(float(eg.get('H1_top'))/float(eg.get('H1_second'))) if eg.get('H1_second') else None,'first_size':len(base.get('first') or ()),'second_size':len(base.get('second') or ()),'third_size':len(base.get('third') or ()), 'q_mass':base.get('q_mass')})
        rows.append(row)

    out={'analysis':'SPECIAL_SPLIT_DIAGNOSTIC','dataset':'2024Q1','population_by_subtype':dict(populations),'exact_race_types':dict(exact),'subtypes':{}}
    for st in sorted(populations):
        rr=[r for r in rows if r['subtype']==st]
        ps=[r for r in rr if r.get('base_buy')]
        hs={}
        for state in ('HAB0_CONC0','HAB0_CONC1','HAB1_CONC0','HAB1_CONC1'):
            z=[r for r in ps if r.get('H_state')==state]; hs[state]=summ(z,'base')
        fs={}
        for label,fn in [('F1',lambda n:n==1),('F2',lambda n:n==2),('F3PLUS',lambda n:n>=3)]:
            z=[r for r in ps if fn(r.get('first_size',0))]; fs[label]=summ(z,'base')
        out['subtypes'][st]={
            'population':populations[st],
            'PS_AB_market_cliff':summ(rr,'base'),
            'PS_AB_then_price_knee':summ(rr,'price'),
            'H_states_on_PS_AB':hs,
            'first_pool_size_on_PS_AB':fs,
            'sample_rows':[{k:r.get(k) for k in ('race_id','race_date','race_type','H_state','H1_ratio','first_size','second_size','third_size','base_formation','base_tickets','base_hit','base_payout','price_formation','price_tickets','price_hit','price_payout')} for r in ps[:8]],
        }
    out['notes']=[
        'Diagnostic only: no cutoff is selected from outcomes.',
        'base = PS_AB + unchanged F09 market-cliff formation.',
        'price = old SPECIAL PS_AB-only type gate followed by structural whole-rider price knee.',
        'H_CONCENTRATED means H1_top >= 2*H1_second; the 2.0 boundary predates this split.'
    ]
    print('SPECIAL_SPLIT_DIAGNOSTIC_BEGIN');print(json.dumps(out,ensure_ascii=False,indent=2));print('SPECIAL_SPLIT_DIAGNOSTIC_END')

if __name__=='__main__':main()

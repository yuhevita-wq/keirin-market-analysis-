from __future__ import annotations
import json
from collections import Counter, defaultdict
from simulate_v8_1_f02_2024q1 import load, pi, pl
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_15_f16_special_split_rebuild import build_v8_15_f16, special_subtype

STAKE = 100
GROUPS = ('QUALIFYING','GENERAL','SEMIFINAL','SPECIAL','FINAL','OTHER')
SPECIAL_SUBTYPES = ('SELECTION','SPECIAL','INITIAL_SPECIAL','OTHER_SPECIAL')


def summary(rows):
    n=len(rows);h=sum(r['hit'] for r in rows);t=sum(r['tickets'] for r in rows);p=sum(r['payout'] for r in rows);s=t*STAKE
    return {
        'bet_races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,
        'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,
        'profit_yen':p-s,'roi_pct':100*p/s if s else None,
        'compressed_races':sum(r['compressed'] for r in rows),
    }


def main():
    races,trio,tf,pay=load();rows=[];by=defaultdict(list);by_special=defaultdict(list)
    pop=Counter();special_pop=Counter();fail=Counter();population=0
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'):continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210:continue
        cars=sorted({v for c in trio[rid] for v in c});lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]:continue
        population+=1;rt=r.get('race_type') or '';g=classify_race_type(rt);pop[g]+=1
        st=special_subtype(rt) if g=='SPECIAL' else None
        if st:special_pop[st]+=1
        d=build_v8_15_f16(trio[rid],tf[rid],r.get('predicted_line_formation') or '',rt)
        if not d.get('buy'):
            fail[f'{g}:{st or "-"}:{d.get("reason")}']+=1;continue
        wins=[t for t in d['tickets'] if t in pay[rid]];payout=sum(pay[rid][t] for t in wins);n=int(d['ticket_count'])
        before=int(d.get('ticket_count_before_price') or n)
        row={
            'race_id':rid,'race_date':r.get('race_date'),'race_type':rt,'group':g,'special_subtype':st,
            'hit':int(bool(wins)),'payout':payout,'tickets':n,'compressed':int(n<before),
            'formation':d.get('formation'),'branch_policy':d.get('branch_policy'),
            'H_state':(d.get('entry_gate') or {}).get('H_state'),
        }
        rows.append(row);by[g].append(row)
        if st:by_special[st].append(row)
    result={
        'scheme':'v8.15-F16','dataset':'2024Q1','status':'DEVELOPMENT_SPECIAL_SPLIT_REBUILD',
        'population':population,'overall':summary(rows),
        'by_group':{g:{**summary(by[g]),'population':pop[g]} for g in GROUPS},
        'special_split':{st:{**summary(by_special[st]),'population':special_pop[st]} for st in SPECIAL_SUBTYPES if special_pop[st]},
        'entry_fail_reasons':dict(fail),
        'rules':{
            'SELECTION':'skip / quarantined pending a distinct market model',
            'SPECIAL':'PS_AB + H1_top >= 2*H1_second; unchanged F09 market-cliff formation',
            'INITIAL_SPECIAL':'PS_AB + (H_AB == H_CONCENTRATED); unchanged F09 market-cliff formation',
            'non_SPECIAL':'preserve v8.14-F15 branches unchanged',
            'individual_ticket_pruning':False,
            'fixed_place_counts':False,
        },
    }
    result['acceptance']='PASS_Q1_PROFIT' if result['overall']['profit_yen']>0 else 'REJECT_Q1_NOT_PROFITABLE'
    print('V8_15_F16_Q1_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('V8_15_F16_Q1_END')
    print('V8_15_F16_SPECIAL_ROWS_BEGIN');print(json.dumps({st:by_special[st] for st in SPECIAL_SUBTYPES if by_special[st]},ensure_ascii=False,indent=2));print('V8_15_F16_SPECIAL_ROWS_END')

if __name__=='__main__':main()

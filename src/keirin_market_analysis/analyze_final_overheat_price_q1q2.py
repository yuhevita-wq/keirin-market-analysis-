from __future__ import annotations

import json

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import _state
from v8_10_f11_structural_price_compression import _price_state
from v8_20_f21_initial_special_unified import build_v8_20_f21

STAKE=100


def summary(rows):
    n=len(rows); h=sum(r['hit'] for r in rows); t=sum(r['tickets'] for r in rows); p=sum(r['payout'] for r in rows); s=t*STAKE
    return {'bet_races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None}


def evaluate(label,data):
    races,trio,tf,pay=data
    rows=[]
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or (r.get('race_type') or '')!='Ｓ級決勝': continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]: continue
        d=build_v8_20_f21(trio[rid],tf[rid],r.get('predicted_line_formation') or '','Ｓ級決勝')
        if not d.get('buy'): continue
        q=implied_probabilities(tf[rid]); st=_state(d['first'],d['second'],d['third'],q)
        if st is None: continue
        ps=_price_state(st,st.q_mass,q,tf[rid])
        wins=[t for t in d['tickets'] if t in pay[rid]]; payout=sum(pay[rid][t] for t in wins)
        n=int(d['ticket_count'])
        row={'race_id':rid,'race_date':r.get('race_date'),'track':r.get('track'),'hit':int(bool(wins)),'payout':payout,'tickets':n,'formation':d.get('formation'),'weighted_gm_odds':ps.weighted_gm_odds,'gm_return_multiple':ps.gm_return_multiple,'profitable_q_share':ps.profitable_q_share,'q_mass':st.q_mass}
        row['gm_break_even']=int(ps.gm_return_multiple>=1.0)
        row['profitable_majority']=int(ps.profitable_q_share>=0.5)
        row['both_price_viable']=int(row['gm_break_even'] and row['profitable_majority'])
        row['overheated_both_fail']=int(not row['gm_break_even'] and not row['profitable_majority'])
        rows.append(row)
    states={'ALL':summary(rows)}
    for f in ('gm_break_even','profitable_majority','both_price_viable','overheated_both_fail'):
        states[f.upper()]=summary([r for r in rows if r[f]])
    states['NOT_GM_BREAK_EVEN']=summary([r for r in rows if not r['gm_break_even']])
    states['NOT_PROFITABLE_MAJORITY']=summary([r for r in rows if not r['profitable_majority']])
    return {'dataset':label,'states':states,'rows':rows}


def main():
    q1=evaluate('2024Q1_DEVELOPMENT',load()); q2=evaluate('2024Q2_DEVELOPMENT_AFTER_OBSERVED',load_q2())
    result={'analysis':'FINAL_OVERHEAT_PRICE_Q1Q2','concept':'Formation is built first; then reject only whole races whose selected rectangular formation is market-overheated at semantic break-even boundaries.','boundaries':{'gm_return_multiple':'>=1.0','profitable_q_share':'>=0.5'},'Q1':q1,'Q2':q2}
    print('FINAL_OVERHEAT_PRICE_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('FINAL_OVERHEAT_PRICE_END')

if __name__=='__main__':main()

from __future__ import annotations
import json
from collections import defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v8_8_f09_market_cliff import build_v8_8_f09
from v8_11_f12_race_type_adaptive import classify_race_type, build_v8_11_f12
from v8_17_f18_market_semantics import selection_divergence_diagnostics

STAKE=100


def summ(rows):
    n=len(rows); h=sum(r['hit'] for r in rows); t=sum(r['tickets'] for r in rows); p=sum(r['payout'] for r in rows); s=t*STAKE
    return {
        'races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,
        'tickets':t,'avg_tickets':t/n if n else None,
        'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None,
        'avg_payout_per_hit':p/h if h else None,
    }


def evaluate(label,data):
    races,trio,tf,pay=data
    states=defaultdict(list); current=[]; eligible=0
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or classify_race_type(r.get('race_type') or '')!='QUALIFYING': continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]: continue
        base=build_v8_8_f09(trio[rid],tf[rid],r.get('predicted_line_formation') or '')
        if not base.get('buy'): continue
        eligible+=1
        eg=dict(base.get('entry_gate') or {})
        hconc=not bool(eg.get('H_RATIO')); hab=bool(eg.get('H_AB'))
        div=selection_divergence_diagnostics(trio[rid],tf[rid])
        wins=[t for t in base['tickets'] if t in pay[rid]]
        row={'race_id':rid,'hit':int(bool(wins)),'payout':sum(pay[rid][t] for t in wins),'tickets':int(base['ticket_count'])}
        states['H_CONCENTRATED' if hconc else 'H_BALANCED'].append(row)
        states[f"H_{'CONC' if hconc else 'BAL'}__HAB_{int(hab)}"].append(row)
        states['TOP_SET_AGREE' if div['top_set_same'] else 'TOP_SET_DISAGREE'].append(row)
        if hconc:
            states['CONC__TOP_AGREE' if div['top_set_same'] else 'CONC__TOP_DISAGREE'].append(row)
        else:
            states['BAL__TOP_AGREE' if div['top_set_same'] else 'BAL__TOP_DISAGREE'].append(row)

        cur=build_v8_11_f12(trio[rid],tf[rid],r.get('predicted_line_formation') or '',r.get('race_type') or '')
        if cur.get('buy'):
            cw=[t for t in cur['tickets'] if t in pay[rid]]
            current.append({'race_id':rid,'hit':int(bool(cw)),'payout':sum(pay[rid][t] for t in cw),'tickets':int(cur['ticket_count'])})
    return {'dataset':label,'ps_ab_qualifying':eligible,'current_hard_branch':summ(current),'f09_semantic_splits':{k:summ(v) for k,v in sorted(states.items())}}


def main():
    out={'analysis':'QUALIFYING_TWO_PATHS_Q1Q2','status':'DEVELOPMENT_DIAGNOSTIC','note':'No fitted numeric cutoffs. Split only by pre-existing H concentration/alignment and exact trio-vs-collapsed-trifecta top-set agreement. F09 formation used for neutral split diagnostics; current branch shown separately.','Q1':evaluate('2024Q1',load()),'Q2':evaluate('2024Q2',load_q2())}
    print('QUALIFYING_TWO_PATHS_BEGIN'); print(json.dumps(out,ensure_ascii=False,indent=2)); print('QUALIFYING_TWO_PATHS_END')

if __name__=='__main__': main()

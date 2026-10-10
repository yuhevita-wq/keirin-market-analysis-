from __future__ import annotations

import json
from math import log
from statistics import mean, median

from simulate_v8_1_f02_2024q1 import load, pi, pl
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v7_0_f01_market_hierarchy import implied_probabilities, positional_support

STAKE=100


def ent(d, cars):
    vals=[float(d[c]) for c in cars]
    s=sum(vals)
    if s<=0:return 0.0
    ps=[v/s for v in vals if v>0]
    return -sum(p*log(p) for p in ps)/log(len(cars))


def summary(rows):
    if not rows:return {'races':0}
    hits=sum(r['hit'] for r in rows)
    tickets=sum(r['tickets'] for r in rows)
    payout=sum(r['payout'] for r in rows)
    stake=tickets*STAKE
    return {
        'races':len(rows),'hits':hits,'hit_rate_pct':100*hits/len(rows),
        'tickets':tickets,'avg_tickets':tickets/len(rows),
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,
        'roi_pct':100*payout/stake if stake else None,
        'avg_anchor_spread':mean(r['anchor_spread'] for r in rows),
        'median_anchor_spread':median(r['anchor_spread'] for r in rows),
    }


def main():
    races,trio,tf,pay=load()
    rows=[]
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or r.get('race_type')!='Ｓ級一般':continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210:continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for L in lines for v in L)!=set(cars) or rid not in pay:continue
        base=build_v8_8_f09(trio[rid],tf[rid],r.get('predicted_line_formation') or '')
        if not base.get('buy'):continue
        eg=base.get('entry_gate') or {}
        # Preserve the old GENERAL high-hit entry state only.
        if bool(eg.get('H_RATIO')) or not bool(eg.get('H_AB')):continue
        q=implied_probabilities(tf[rid]); h1,h2,h3=positional_support(q,cars)
        e1,e2,e3=ent(h1,cars),ent(h2,cars),ent(h3,cars)
        asp=(e2+e3)/2-e1
        st=_state(base['first'],base['second'],base['third'],q)
        wins=[t for t in st.tickets if t in pay[rid]] if st else []
        payout=sum(pay[rid][t] for t in wins)
        tickets=st.ticket_count if st else 0
        row={
            'race_id':rid,'race_date':r.get('race_date'),'formation':st.display if st else None,
            'tickets':tickets,'H1_ratio':float(eg.get('H1_top'))/float(eg.get('H1_second')) if eg.get('H1_second') else None,
            'E1':e1,'E2':e2,'E3':e3,'anchor_spread':asp,
            'rear_both_more_disperse':bool(e2>e1 and e3>e1),
            'rear_avg_more_disperse':bool(asp>0),
            'first_pool_size':len(st.first) if st else None,'second_pool_size':len(st.second) if st else None,'third_pool_size':len(st.third) if st else None,
            'q_mass':st.q_mass if st else None,'hit':int(bool(wins)),'payout':payout,'race_profit':payout-tickets*STAKE,
        }
        # One-rider removal audit from the completed rectangle, no selection/tuning.
        removals=[]
        for place,xs in ((1,st.first),(2,st.second),(3,st.third)):
            if len(xs)<=1:continue
            for rider in xs:
                f1,f2,f3=st.first,st.second,st.third
                if place==1:f1=tuple(x for x in f1 if x!=rider)
                elif place==2:f2=tuple(x for x in f2 if x!=rider)
                else:f3=tuple(x for x in f3 if x!=rider)
                nxt=_state(f1,f2,f3,q)
                if nxt is None:continue
                saved=tickets-nxt.ticket_count
                qloss=st.q_mass-nxt.q_mass
                removals.append({
                    'place':place,'rider':rider,'tickets_saved':saved,
                    'ticket_share_saved':saved/tickets if tickets else 0,
                    'q_loss':qloss,'q_loss_share':qloss/st.q_mass if st.q_mass else 0,
                    'support_to_cost_ratio':(qloss/st.q_mass)/(saved/tickets) if saved and st.q_mass else None,
                    'new_formation':nxt.display,
                })
        row['removals']=removals
        rows.append(row)

    groups={
        'ALL_OLD_GENERAL_ENTRY':summary(rows),
        'ANCHOR_SPREAD_GT0':summary([r for r in rows if r['rear_avg_more_disperse']]),
        'ANCHOR_SPREAD_LE0':summary([r for r in rows if not r['rear_avg_more_disperse']]),
        'BOTH_REAR_MORE_DISPERSE':summary([r for r in rows if r['rear_both_more_disperse']]),
        'NOT_BOTH_REAR_MORE_DISPERSE':summary([r for r in rows if not r['rear_both_more_disperse']]),
    }
    result={
        'analysis':'GENERAL_ANCHOR_SPREAD_AND_BRANCH_AUDIT','dataset':'2024Q1','entry':'PS_AB + H_CONCENTRATED + H_AB',
        'groups':groups,
        'rows':rows,
        'notes':[
            'Diagnostic only. No fitted cutoff is selected.',
            'AnchorSpread=((E2+E3)/2)-E1 where position entropies are normalized to [0,1].',
            'The zero boundary means rear-position markets are, on average, more dispersed than the first-position market.',
            'One-rider removals are audit counterfactuals only and never create irregular ticket sets.',
        ]
    }
    print('GENERAL_ANCHOR_SPREAD_Q1_BEGIN')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('GENERAL_ANCHOR_SPREAD_Q1_END')

if __name__=='__main__':main()

from __future__ import annotations

"""Selection ultra-ROI stage 3.

Stage 2 result: removing global H1 is a dead end. The reproducible positive state
is H_BALANCED while targeting the most trifecta-discounted set among trio top3.
This stage asks how to translate that state into an even sparser order bet without
fitted numeric thresholds.

Target set:
- among trio top3, choose the set with the most negative D_log, if any.
Entry semantics tested:
- H_BALANCED (H1 < 2*H2),
- target spans exactly two predicted lines,
- target includes global H1.
Order construction:
- head = highest H rider inside target;
- two possible orders of remaining riders;
- BOTH_2T = buy both;
- LONGER_1T = buy the higher-odds (less-supported) tail order only;
- SHORTER_1T = buy the lower-odds tail order only.
All choices are pre-race and parameter-free.
"""

import json
from collections import defaultdict
from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v7_0_f01_market_hierarchy import implied_probabilities, trio_implied_probabilities
from v8_17_f18_market_semantics import selection_divergence_diagnostics

STAKE=100


def hs(tf):
    q=implied_probabilities(tf); h=defaultdict(float)
    for t,p in q.items(): h[t[0]]+=p
    return dict(h)

def summ(rows):
    n=len(rows); t=sum(x['n'] for x in rows); hits=sum(x['hit'] for x in rows); p=sum(x['payout'] for x in rows); s=t*STAKE; wins=[x for x in rows if x['hit']]
    return {'races':n,'hits':hits,'hit_rate_pct':100*hits/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None,'avg_payout_per_hit':sum(x['payout'] for x in wins)/len(wins) if wins else None,'max_payout':max((x['payout'] for x in rows),default=0)}

def put(b,key,rid,tickets,pay):
    po=sum(pay.get(t,0) for t in tickets); b[key].append({'race_id':rid,'n':len(tickets),'hit':int(po>0),'payout':po})

def evaluate(label,data):
    races,trio,tf,payouts=data; b=defaultdict(list); pop=0
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or (r.get('race_type') or '').strip()!='Ｓ級選抜': continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in payouts or not payouts[rid]: continue
        pop+=1
        p=trio_implied_probabilities(trio[rid]); pr=tuple(sorted(p,key=lambda c:(-p[c],c))); diag=selection_divergence_diagnostics(trio[rid],tf[rid]); d={tuple(x['set']):x['D_log'] for x in diag['set_rows']}; h=hs(tf[rid]); hr=tuple(sorted(h,key=lambda x:(-h[x],x))); h1,h2=hr[:2]
        neg=[c for c in pr[:3] if d[c] is not None and d[c]<0]
        if not neg: continue
        target=min(neg,key=lambda c:(d[c],pr.index(c),c)); head=max(target,key=lambda x:(h.get(x,0.0),-x)); tails=tuple(sorted(x for x in target if x!=head)); t1=(head,tails[0],tails[1]); t2=(head,tails[1],tails[0]); o1=float(tf[rid][t1]); o2=float(tf[rid][t2]); longer=t1 if o1>=o2 else t2; shorter=t2 if o1>=o2 else t1
        rider_line={v:i for i,line in enumerate(lines) for v in line}; span=len({rider_line[v] for v in target}); hbal=h[h1]<2*h[h2]; h1in=h1 in target
        states=[]
        if hbal: states.append('H_BAL')
        if hbal and h1in: states.append('H_BAL_H1_IN')
        if hbal and span==2: states.append('H_BAL_2LINES')
        if hbal and h1in and span==2: states.append('H_BAL_H1_IN_2LINES')
        for st in states:
            put(b,st+'__BOTH_2T',rid,(t1,t2),payouts[rid]); put(b,st+'__LONGER_1T',rid,(longer,),payouts[rid]); put(b,st+'__SHORTER_1T',rid,(shorter,),payouts[rid])
    return {'dataset':label,'population':pop,'strategies':{k:summ(v) for k,v in sorted(b.items())}}

def main():
    print('SELECTION_ULTRA_ROI_STAGE3_BEGIN')
    print(json.dumps({'analysis':'SELECTION_ULTRA_ROI_STAGE3_Q1Q2','status':'DEVELOPMENT_NO_FITTED_CUTOFFS','Q1':evaluate('2024Q1',load()),'Q2':evaluate('2024Q2',load_q2())},ensure_ascii=False,indent=2))
    print('SELECTION_ULTRA_ROI_STAGE3_END')
if __name__=='__main__': main()

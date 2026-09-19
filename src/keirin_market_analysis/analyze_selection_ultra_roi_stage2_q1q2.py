from __future__ import annotations

"""Selection ultra-ROI stage 2: contrarian discounted-set structures.

Starting point from stage 1:
- 2-ticket targeting successfully increases payout tail but broad entry is too frequent.
- The most coherent high-ROI question is whether the trio market preserves a
  supported three-rider set that the trifecta head market has over-discounted
  because money is concentrated on a different global H1 rider.

No fitted numeric cutoffs are used. All splits are semantic or parameter-free:
- existing H_CONCENTRATED boundary H1 >= 2*H2;
- whether the target set includes global H1;
- whether the most-discounted D among trio-top3 is a single D outlier, defined
  by the first adjacent D gap exceeding the second adjacent gap;
- number of distinct predicted lines represented by the target set.
"""

import json
from collections import defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v7_0_f01_market_hierarchy import implied_probabilities, trio_implied_probabilities
from v8_17_f18_market_semantics import selection_divergence_diagnostics

STAKE=100


def h_support(tf):
    q=implied_probabilities(tf); h=defaultdict(float)
    for t,p in q.items(): h[t[0]]+=p
    return dict(h)


def two_tickets(target,h):
    target=tuple(sorted(target)); head=max(target,key=lambda x:(h.get(x,0.0),-x)); tails=tuple(sorted(x for x in target if x!=head))
    return head,((head,tails[0],tails[1]),(head,tails[1],tails[0]))


def summ(rows):
    n=len(rows); hits=sum(x['hit'] for x in rows); t=2*n; p=sum(x['payout'] for x in rows); s=t*STAKE
    win=[x for x in rows if x['hit']]
    return {'races':n,'hits':hits,'hit_rate_pct':100*hits/n if n else None,'tickets':t,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None,'avg_payout_per_hit':sum(x['payout'] for x in win)/len(win) if win else None,'max_payout':max((x['payout'] for x in rows),default=0)}


def add(b,key,row): b[key].append(row)


def evaluate(label,data):
    races,trio,tf,pay=data; b=defaultdict(list); pop=0
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or (r.get('race_type') or '').strip()!='Ｓ級選抜': continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]: continue
        pop+=1
        p=trio_implied_probabilities(trio[rid]); pr=tuple(sorted(p,key=lambda c:(-p[c],c))); diag=selection_divergence_diagnostics(trio[rid],tf[rid]); dr={tuple(x['set']):x['D_log'] for x in diag['set_rows']}
        h=h_support(tf[rid]); hr=tuple(sorted(h,key=lambda x:(-h[x],x))); h1,h2=hr[:2]; hconc=h[h1]>=2*h[h2]
        neg=[c for c in pr[:3] if dr[c] is not None and dr[c]<0]
        if not neg: continue
        target=min(neg,key=lambda c:(dr[c],pr.index(c),c)); head,tickets=two_tickets(target,h); payout=sum(pay[rid].get(t,0) for t in tickets)
        rider_line={v:i for i,line in enumerate(lines) for v in line}; line_span=len({rider_line[v] for v in target})
        row={'race_id':rid,'hit':int(payout>0),'payout':payout,'target':target,'D':dr[target],'global_H1':h1,'target_head':head,'h_concentrated':hconc,'h1_in_target':h1 in target,'line_span':line_span}
        add(b,'ALL_TOP3_DISCOUNTED',row)
        add(b,'TARGET_H1_IN' if h1 in target else 'TARGET_H1_OUT',row)
        add(b,'H_CONC' if hconc else 'H_BAL',row)
        if hconc and h1 not in target: add(b,'H_CONC__H1_OUT',row)
        if (not hconc) and h1 not in target: add(b,'H_BAL__H1_OUT',row)
        add(b,f'LINE_SPAN_{line_span}',row)
        if h1 not in target and line_span==3: add(b,'H1_OUT__3_LINES',row)
        if hconc and h1 not in target and line_span==3: add(b,'H_CONC__H1_OUT__3_LINES',row)

        # Parameter-free single-discount-outlier state within trio top3.
        d3=sorted((dr[c],c) for c in pr[:3] if dr[c] is not None)
        if len(d3)==3 and d3[0][0] < 0:
            gap1=d3[1][0]-d3[0][0]; gap2=d3[2][0]-d3[1][0]
            if gap1>gap2 and d3[0][1]==target:
                add(b,'D_SINGLE_OUTLIER',row)
                if h1 not in target: add(b,'D_SINGLE_OUTLIER__H1_OUT',row)
                if hconc and h1 not in target: add(b,'D_SINGLE_OUTLIER__H_CONC__H1_OUT',row)
    return {'dataset':label,'population':pop,'states':{k:summ(v) for k,v in sorted(b.items())}}


def main():
    print('SELECTION_ULTRA_ROI_STAGE2_BEGIN')
    print(json.dumps({'analysis':'SELECTION_ULTRA_ROI_STAGE2_Q1Q2','status':'DEVELOPMENT_NO_FITTED_CUTOFFS','Q1':evaluate('2024Q1',load()),'Q2':evaluate('2024Q2',load_q2())},ensure_ascii=False,indent=2))
    print('SELECTION_ULTRA_ROI_STAGE2_END')

if __name__=='__main__': main()

from __future__ import annotations

"""Selection ultra-ROI development diagnostic on observed Q1+Q2.

Goal is deliberately different from qualifying:
- low bet frequency and low hit rate are acceptable;
- prioritize return multiple and sparse clean structures;
- exploit cross-market disagreement instead of stronger consensus.

No fitted numeric cutoff is used. Candidate states are semantic/rank based.
The core construction is a 2-ticket clean formation from one unordered trio set:
- the 3連複 market chooses the set;
- the 3連単 H market chooses the head within that set;
- buy both orders of the other two riders.

Primary value hypothesis:
a set strongly supported by 3連複 but relatively discounted by collapsed 3連単
(D_log < 0) may offer higher 3連単 return potential.
"""

import json
from collections import defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v7_0_f01_market_hierarchy import implied_probabilities, trio_implied_probabilities
from v8_17_f18_market_semantics import selection_divergence_diagnostics

STAKE=100


def head_support(tf_odds):
    q=implied_probabilities(tf_odds)
    h=defaultdict(float)
    for order,p in q.items(): h[order[0]]+=p
    return dict(h)


def two_ticket_target(target,h):
    target=tuple(sorted(target))
    head=max(target,key=lambda x:(h.get(x,0.0),-x))
    tails=tuple(sorted(x for x in target if x!=head))
    tickets=((head,tails[0],tails[1]),(head,tails[1],tails[0]))
    return head,tickets


def summarize(rows):
    n=len(rows); hits=sum(r['hit'] for r in rows); tickets=sum(r['tickets'] for r in rows); payout=sum(r['payout'] for r in rows); stake=tickets*STAKE
    winning=[r for r in rows if r['hit']]
    return {
        'races':n,'hits':hits,'hit_rate_pct':100*hits/n if n else None,
        'tickets':tickets,'avg_tickets':tickets/n if n else None,
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,
        'roi_pct':100*payout/stake if stake else None,
        'avg_payout_per_hit':sum(r['payout'] for r in winning)/len(winning) if winning else None,
        'max_payout':max((r['payout'] for r in rows),default=0),
    }


def evaluate(label,data):
    races,trio,tf,pay=data
    buckets=defaultdict(list); population=0
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or (r.get('race_type') or '').strip()!='Ｓ級選抜': continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]: continue
        population+=1
        diag=selection_divergence_diagnostics(trio[rid],tf[rid])
        p=trio_implied_probabilities(trio[rid]); h=head_support(tf[rid])
        p_rank=tuple(sorted(p,key=lambda c:(-p[c],c)))
        top_p=p_rank[0]; top_q=tuple(diag['tf_set_top_sets'][0]); rows={tuple(x['set']):x for x in diag['set_rows']}
        d_top=rows[top_p]['D_log']
        global_h1=max(h,key=lambda x:(h[x],-x))

        # Baseline sparse structure: top trio set, regardless of disagreement.
        head,tickets=two_ticket_target(top_p,h)
        payout=sum(pay[rid].get(t,0) for t in tickets); hit=int(payout>0)
        base={'race_id':rid,'hit':hit,'tickets':2,'payout':payout,'target':top_p,'head':head,'D':d_top}
        buckets['TOP_TRIO_2T_BASELINE'].append(base)

        if top_p != top_q:
            buckets['TOP_SET_DISAGREE_2T'].append(base)
        if d_top is not None and d_top < 0:
            buckets['TOP_TRIO_D_NEG_2T'].append(base)
        if top_p != top_q and d_top is not None and d_top < 0:
            buckets['TOP_DISAGREE_AND_D_NEG_2T'].append(base)
            if global_h1 in top_p:
                buckets['TOP_DISAGREE_D_NEG_GLOBAL_H1_IN_SET_2T'].append(base)

        # Parameter-free alternative: among the trio market's top 3 sets,
        # choose the one most discounted by collapsed trifecta market.
        top3=p_rank[:3]
        neg=[c for c in top3 if rows[c]['D_log'] is not None and rows[c]['D_log']<0]
        if neg:
            target=min(neg,key=lambda c:(rows[c]['D_log'],p_rank.index(c),c))
            hd,tk=two_ticket_target(target,h); po=sum(pay[rid].get(t,0) for t in tk)
            rr={'race_id':rid,'hit':int(po>0),'tickets':2,'payout':po,'target':target,'head':hd,'D':rows[target]['D_log']}
            buckets['TOP3_MOST_DISCOUNTED_2T'].append(rr)
            if target != top_q:
                buckets['TOP3_MOST_DISCOUNTED_NOT_TF_TOP_2T'].append(rr)

    return {'dataset':label,'population':population,'strategies':{k:summarize(v) for k,v in sorted(buckets.items())}}


def main():
    q1=evaluate('2024Q1_DEVELOPMENT',load()); q2=evaluate('2024Q2_DEVELOPMENT_AFTER_OBSERVED',load_q2())
    print('SELECTION_ULTRA_ROI_BEGIN')
    print(json.dumps({'analysis':'SELECTION_ULTRA_ROI_Q1Q2','status':'DEVELOPMENT_DIAGNOSTIC_NO_FITTED_CUTOFFS','Q1':q1,'Q2':q2},ensure_ascii=False,indent=2))
    print('SELECTION_ULTRA_ROI_END')

if __name__=='__main__':main()

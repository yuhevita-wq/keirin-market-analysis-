from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import log

from simulate_v8_1_f02_2024q1 import load, pi, pl
from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _compression_path, _select_structural_knee

STAKE = 100


def summary(rows):
    n=len(rows); hits=sum(r['hit'] for r in rows); tickets=sum(r['tickets_after'] for r in rows)
    payout=sum(r['payout_yen'] for r in rows); stake=tickets*STAKE
    ph=sum(1 for r in rows if r['hit'] and r['payout_yen'] > r['tickets_after']*STAKE)
    return {
        'races':n,'hits':hits,'hit_rate_pct':100*hits/n if n else None,
        'tickets':tickets,'avg_tickets':tickets/n if n else None,
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,
        'roi_pct':100*payout/stake if stake else None,
        'profitable_hit_share_pct':100*ph/hits if hits else None,
        'compressed_races':sum(r['compressed'] for r in rows),
        'avg_q_mass':sum(r['q_mass'] for r in rows)/n if n else None,
        'avg_gm_return_multiple':sum(r['gm_mult'] for r in rows)/n if n else None,
        'avg_profitable_q_share':sum(r['profitable_q_share'] for r in rows)/n if n else None,
    }


def price_compress(base, tf_odds):
    q=implied_probabilities(tf_odds)
    s=_state(base['first'],base['second'],base['third'],q)
    p0=_price_state(s,s.q_mass,q,tf_odds)
    states,moves=_compression_path(p0,q,tf_odds)
    i=_select_structural_knee(states)
    c=states[i]
    return c,i


def tband(n):
    if n <= 6: return '01_<=6'
    if n <= 12: return '02_7-12'
    if n <= 18: return '03_13-18'
    return '04_19+'


def qband(x):
    if x < 0.25: return '01_<0.25'
    if x < 0.40: return '02_0.25-0.40'
    if x < 0.55: return '03_0.40-0.55'
    return '04_>=0.55'


def gmband(x):
    if x < 1.0: return '01_<1.0'
    if x < 1.5: return '02_1.0-1.5'
    if x < 2.5: return '03_1.5-2.5'
    return '04_>=2.5'


def pqband(x):
    if x < 0.40: return '01_<0.40'
    if x < 0.60: return '02_0.40-0.60'
    if x < 0.80: return '03_0.60-0.80'
    return '04_>=0.80'


def main():
    races,trio,tf,pay=load(); rows=[]; fail=Counter(); population=0
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or (r.get('race_type') or '')!='Ｓ級一般': continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]: continue
        population += 1
        b=build_v8_8_f09(trio[rid],tf[rid],r.get('predicted_line_formation') or '')
        if not b.get('buy'):
            fail[str(b.get('reason'))]+=1; continue
        c,steps=price_compress(b,tf[rid])
        wins=[t for t in c.state.tickets if t in pay[rid]]
        payout=sum(pay[rid][t] for t in wins)
        eg=b.get('entry_gate') or {}
        h_ab=int(bool(eg.get('H_AB'))); h_ratio=int(bool(eg.get('H_RATIO')))
        n=c.state.ticket_count
        rows.append({
            'race_id':rid,'race_date':r.get('race_date') or '',
            'H_AB':h_ab,'H_RATIO':h_ratio,'h_state':f'HAB{h_ab}_HRATIO{h_ratio}',
            'formation_before':b['formation'],'formation_after':c.display,
            'tickets_before':b['ticket_count'],'tickets_after':n,'compressed':int(n<b['ticket_count']),
            'q_mass':c.state.q_mass,'q_retention':c.q_retention,'gm_mult':c.gm_return_multiple,
            'profitable_q_share':c.profitable_q_share,
            'ticket_band':tband(n),'q_mass_band':qband(c.state.q_mass),
            'gm_band':gmband(c.gm_return_multiple),'pq_band':pqband(c.profitable_q_share),
            'hit':int(bool(wins)),'payout_yen':payout,'race_profit_yen':payout-n*STAKE,
            'compression_steps':steps,
        })

    def grouped(key):
        d=defaultdict(list)
        for r in rows:d[r[key]].append(r)
        return {k:summary(v) for k,v in sorted(d.items())}

    result={
        'analysis':'GENERAL_BRANCH_REBUILD_DIAGNOSTIC','dataset':'2024Q1','race_type':'Ｓ級一般',
        'population':population,'ps_ab_pass':len(rows),'ps_ab_fail_reasons':dict(fail),
        'all_ps_ab_after_structural_price':summary(rows),
        'by_h_state':grouped('h_state'),'by_ticket_band':grouped('ticket_band'),
        'by_q_mass_band':grouped('q_mass_band'),'by_gm_return_band':grouped('gm_band'),
        'by_profitable_q_share_band':grouped('pq_band'),
        'notes':[
            'This is diagnostic only; no cutoff is selected from Q1.',
            'All rows pass PS_AB. H state is classified, not used as a hard gate.',
            'Formation is completed before structural whole-rider price compression.',
            'Individual ticket pruning is never used.'
        ]
    }
    print('GENERAL_REBUILD_Q1_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('GENERAL_REBUILD_Q1_END')
    audit=sorted(rows,key=lambda r:(r['race_profit_yen'],r['race_id']))
    samples=(audit[:5]+audit[-5:]) if audit else []
    print('GENERAL_REBUILD_Q1_AUDIT_BEGIN');print(json.dumps(samples,ensure_ascii=False,indent=2));print('GENERAL_REBUILD_Q1_AUDIT_END')

if __name__=='__main__': main()

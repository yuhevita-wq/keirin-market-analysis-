from __future__ import annotations

"""Second-stage semifinal rebuild diagnostic on observed Q1+Q2 development data.

Concept: translate the trifecta first-place market H directly into the first-place
formation shape instead of using H only as an entry gate.
- If H is concentrated (H1 >= 2*H2): first pool = H1 only.
- If H is balanced: first pool = top two H riders.
Second/third pools remain the F09 market-cliff pools.
Price contraction remains clean whole-rider contraction, but the H-derived first
pool is protected: only second/third pools may contract after formation completion.
No result-derived numeric threshold is searched.
"""

from dataclasses import dataclass
from math import log
import json

from simulate_v8_1_f02_2024q1 import load
from evaluate_v8_17_q2_oos import load_q2
from v8_8_f09_market_cliff import build_v8_8_f09, _state
from v8_10_f11_structural_price_compression import _price_state, _select_structural_knee
from v7_0_f01_market_hierarchy import implied_probabilities
from analyze_semifinal_rebuild_q1q2 import valid_semifinals, summary


@dataclass(frozen=True)
class Move:
    place: int
    rider: int
    q_loss: float
    gain: float
    efficiency: float
    next_state: object


def protected_path(p0, q, tf_odds):
    states=[p0]; moves=[]; cur=p0; base_mass=p0.state.q_mass
    while True:
        cand=[]
        s=cur.state
        for place, pool in ((2,s.second),(3,s.third)):
            if len(pool)<=1: continue
            for rider in pool:
                f2=s.second; f3=s.third
                if place==2: f2=tuple(x for x in f2 if x!=rider)
                else: f3=tuple(x for x in f3 if x!=rider)
                ns=_state(s.first,f2,f3,q)
                if ns is None: continue
                pn=_price_state(ns,base_mass,q,tf_odds)
                qloss=s.q_mass-ns.q_mass
                if qloss<=0: continue
                gain=log(pn.gm_return_multiple)-log(cur.gm_return_multiple)
                if gain<=0: continue
                cand.append(Move(place,rider,qloss,gain,gain/qloss,pn))
        if not cand: break
        m=max(cand,key=lambda x:(x.efficiency,-x.q_loss,-x.place,-x.rider))
        moves.append(m); cur=m.next_state; states.append(cur)
    return states,moves


def build_candidate(trio, tf, line, mode):
    base=build_v8_8_f09(trio,tf,line)
    if not base.get('buy'): return {'buy':False,'reason':base.get('reason')}
    eg=dict(base.get('entry_gate') or {})
    h_conc=not bool(eg.get('H_RATIO'))
    h_ab=bool(eg.get('H_AB'))
    h_consistent=(h_ab==h_conc)
    if mode=='H_ADAPTIVE_HAB' and not h_ab: return {'buy':False,'reason':'H_NOT_AB'}
    if mode=='H_ADAPTIVE_CONSISTENT' and not h_consistent: return {'buy':False,'reason':'H_STATE_MISMATCH'}
    if mode=='H_BALANCED_ADAPTIVE' and h_conc: return {'buy':False,'reason':'H_CONCENTRATED'}
    if mode=='H_BALANCED_HAB_ADAPTIVE' and (h_conc or not h_ab): return {'buy':False,'reason':'NOT_BALANCED_HAB'}
    if mode=='H_CONCENTRATED_ADAPTIVE' and not h_conc: return {'buy':False,'reason':'H_NOT_CONCENTRATED'}

    htop=tuple(int(x) for x in eg['top2_H'])
    first=(htop[0],) if h_conc else tuple(sorted(htop[:2]))
    q=implied_probabilities(tf)
    st=_state(first,base['second'],base['third'],q)
    if st is None: return {'buy':False,'reason':'INVALID_H_ADAPTIVE_RECTANGLE'}
    p0=_price_state(st,st.q_mass,q,tf)
    states,moves=protected_path(p0,q,tf)
    k=_select_structural_knee(states); ch=states[k]
    return {
        'buy':True,'formation':ch.display,'tickets':ch.state.tickets,'ticket_count':ch.state.ticket_count,
        'H_CONCENTRATED':h_conc,'H_AB':h_ab,'H_STATE_CONSISTENT':h_consistent,
        'H_first_pool':first,'price_steps':k,'gm':ch.gm_return_multiple,'pqs':ch.profitable_q_share,'qret':ch.q_retention,
    }


def eval_one(label,data,modes):
    rows={m:[] for m in modes}
    hrows={k:[] for k in ['HAB0_HCONC0','HAB0_HCONC1','HAB1_HCONC0','HAB1_HCONC1']}
    pop=0; ps=0
    for rid,r,trio,tf,pay in valid_semifinals(data):
        pop+=1
        base=build_v8_8_f09(trio,tf,r.get('predicted_line_formation') or '')
        if base.get('buy'):
            ps+=1
            eg=base.get('entry_gate') or {}
            hs=f"HAB{int(bool(eg.get('H_AB')))}_HCONC{int(not bool(eg.get('H_RATIO')))}"
            d=build_candidate(trio,tf,r.get('predicted_line_formation') or '','H_ADAPTIVE_ALL')
            if d.get('buy'):
                wins=[t for t in d['tickets'] if t in pay]; payout=sum(pay[t] for t in wins)
                hrows[hs].append({'hit':int(bool(wins)),'payout':payout,'tickets':d['ticket_count']})
        for m in modes:
            d=build_candidate(trio,tf,r.get('predicted_line_formation') or '',m)
            if not d.get('buy'): continue
            wins=[t for t in d['tickets'] if t in pay]; payout=sum(pay[t] for t in wins)
            rows[m].append({'hit':int(bool(wins)),'payout':payout,'tickets':d['ticket_count'],'gm':d['gm'],'pqs':d['pqs']})
    return {'dataset':label,'population':pop,'ps_ab_pass':ps,'candidates':{m:summary(rows[m]) for m in modes},'adaptive_by_h_state':{k:summary(v) for k,v in hrows.items()},'_rows':rows}


def main():
    modes=['H_ADAPTIVE_ALL','H_ADAPTIVE_HAB','H_ADAPTIVE_CONSISTENT','H_BALANCED_ADAPTIVE','H_BALANCED_HAB_ADAPTIVE','H_CONCENTRATED_ADAPTIVE']
    q1=eval_one('2024Q1',load(),modes); q2=eval_one('2024Q2',load_q2(),modes)
    combined={m:summary(q1['_rows'][m]+q2['_rows'][m]) for m in modes}
    q1.pop('_rows',None); q2.pop('_rows',None)
    out={'analysis':'SEMIFINAL_H_ADAPTIVE_Q1Q2_DEVELOPMENT','rule':'H concentrated => first=H1; H balanced => first=top2 H; F09 second/third; first pool protected during whole-rider price contraction','Q1':q1,'Q2':q2,'Q1Q2_COMBINED':combined,'q3_untouched':True}
    print('SEMIFINAL_H_ADAPTIVE_BEGIN');print(json.dumps(out,ensure_ascii=False,indent=2));print('SEMIFINAL_H_ADAPTIVE_END')

if __name__=='__main__': main()

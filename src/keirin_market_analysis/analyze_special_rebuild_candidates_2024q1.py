from __future__ import annotations
import json
from collections import Counter,defaultdict
from simulate_v8_1_f02_2024q1 import load,pi,pl
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_8_f09_market_cliff import build_v8_8_f09,_state
from v8_10_f11_structural_price_compression import _price_state,_compression_path,_select_structural_knee
from v8_13_f14_general_price_viability import _general_price_select
from v7_0_f01_market_hierarchy import implied_probabilities

STAKE=100

def subtype(rt:str)->str:
    t=(rt or '').strip()
    if '初特選' in t or '初日特選' in t:return 'INITIAL_SPECIAL'
    if t=='Ｓ級特選' or ('特選' in t and '初' not in t):return 'SPECIAL'
    if '選抜' in t:return 'SELECTION'
    return 'OTHER_SPECIAL'

def eval_state(state,payrid):
    if state is None:return None
    wins=[t for t in state.tickets if t in payrid]
    return {'buy':True,'tickets':state.ticket_count,'hit':int(bool(wins)),'payout':sum(payrid[t] for t in wins),'formation':state.display}

def sumrows(rows,key):
    rs=[r[key] for r in rows if r.get(key) and r[key].get('buy')]
    if not rs:return {'races':0}
    n=len(rs);h=sum(x['hit'] for x in rs);t=sum(x['tickets'] for x in rs);p=sum(x['payout'] for x in rs);s=t*STAKE
    return {'races':n,'hits':h,'hit_rate_pct':100*h/n,'tickets':t,'avg_tickets':t/n,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None}

def knee(state,q,tf):
    p0=_price_state(state,state.q_mass,q,tf);states,moves=_compression_path(p0,q,tf);return states[_select_structural_knee(states)].state

def main():
    races,trio,tf,pay=load();rows=[];pop=Counter()
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'):continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210:continue
        cars=sorted({v for c in trio[rid] for v in c});lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]:continue
        rt=r.get('race_type') or ''
        if classify_race_type(rt)!='SPECIAL':continue
        st=subtype(rt);pop[st]+=1
        base=build_v8_8_f09(trio[rid],tf[rid],r.get('predicted_line_formation') or '')
        row={'race_id':rid,'race_type':rt,'subtype':st}
        if not base.get('buy'):
            rows.append(row);continue
        eg=base.get('entry_gate') or {}; conc=not bool(eg.get('H_RATIO'));hab=bool(eg.get('H_AB'));q=implied_probabilities(tf[rid])
        bs=_state(base['first'],base['second'],base['third'],q)
        row['BASE']=eval_state(bs,pay[rid])
        # Selection: price viability only, using semantic break-even/majority-support conditions.
        if st=='SELECTION':
            sel,states=_general_price_select(base,tf[rid])
            if sel is not None: row['SELECTION_PRICE_VIABLE']=eval_state(sel[1].state,pay[rid])
        # Ordinary special: pre-existing H concentration gate; test anchor consistency and price knee.
        if st=='SPECIAL' and conc:
            row['SPECIAL_CONC_BASE']=eval_state(bs,pay[rid])
            anchor=int(eg['top2_H'][0]);anch=_state((anchor,),base['second'],base['third'],q)
            if anch is not None:
                row['SPECIAL_CONC_ANCHOR']=eval_state(anch,pay[rid])
                row['SPECIAL_CONC_ANCHOR_KNEE']=eval_state(knee(anch,q,tf[rid]),pay[rid])
        # Initial special: H alignment state must agree with concentration state.
        if st=='INITIAL_SPECIAL' and (hab==conc):
            row['INITIAL_MATCH_BASE']=eval_state(bs,pay[rid])
            struct=bs
            if conc:
                anchor=int(eg['top2_H'][0]);anch=_state((anchor,),base['second'],base['third'],q)
                if anch is not None:struct=anch
            row['INITIAL_MATCH_STRUCTURE']=eval_state(struct,pay[rid])
            row['INITIAL_MATCH_STRUCTURE_KNEE']=eval_state(knee(struct,q,tf[rid]),pay[rid])
        row.update({'H_CONCENTRATED':conc,'H_AB':hab,'H1_ratio':float(eg.get('H1_top'))/float(eg.get('H1_second')) if eg.get('H1_second') else None})
        rows.append(row)
    keys=['BASE','SELECTION_PRICE_VIABLE','SPECIAL_CONC_BASE','SPECIAL_CONC_ANCHOR','SPECIAL_CONC_ANCHOR_KNEE','INITIAL_MATCH_BASE','INITIAL_MATCH_STRUCTURE','INITIAL_MATCH_STRUCTURE_KNEE']
    out={'analysis':'SPECIAL_REBUILD_CANDIDATES','dataset':'2024Q1','population_by_subtype':dict(pop),'by_subtype':{}}
    for st in ('SELECTION','SPECIAL','INITIAL_SPECIAL'):
        rr=[r for r in rows if r['subtype']==st]
        out['by_subtype'][st]={k:sumrows(rr,k) for k in keys if any(r.get(k) for r in rr)}
    out['notes']=[
        'Development diagnostics; all candidate inputs are pre-race market structure only.',
        'SELECTION_PRICE_VIABLE uses semantic GM return >=1.0 and profitable-q-share >=0.5 on clean whole-rider rectangles.',
        'SPECIAL_CONC uses the pre-existing H1_top >= 2*H1_second boundary, not a newly fitted threshold.',
        'INITIAL_MATCH uses H_AB == H_CONCENTRATED as a structural state-consistency hypothesis suggested by the split diagnostic.',
        'Anchor means preserve the H1 cliff already present at entry; it is not a global fixed-one-rider rule.'
    ]
    print('SPECIAL_REBUILD_CANDIDATES_BEGIN');print(json.dumps(out,ensure_ascii=False,indent=2));print('SPECIAL_REBUILD_CANDIDATES_END')

if __name__=='__main__':main()

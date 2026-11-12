from __future__ import annotations
import json

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v7_0_f01_market_hierarchy import implied_probabilities
from v8_8_f09_market_cliff import _state
from v8_10_f11_structural_price_compression import _price_state,_compression_path,_select_structural_knee
from v8_17_f18_market_semantics import selection_divergence_diagnostics
from v8_20_f21_initial_special_unified import build_v8_20_f21

STAKE=100

def summary(rows):
    n=len(rows);h=sum(r['hit'] for r in rows);t=sum(r['tickets'] for r in rows);p=sum(r['payout'] for r in rows);s=t*STAKE
    return {'bet_races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None}

def contains_set(state,c):
    c=set(c)
    return any(set(t)==c for t in state.state.tickets)

def row_from_state(rid,r,ps,pay):
    wins=[t for t in ps.state.tickets if t in pay]
    return {'race_id':rid,'race_date':r.get('race_date'),'track':r.get('track'),'hit':int(bool(wins)),'payout':sum(pay[t] for t in wins),'tickets':ps.state.ticket_count,'formation':ps.state.display}

def evaluate(label,data):
    races,trio,tf,pay=data
    base_rows=[];knee_rows=[];core_rows=[];line4_core=[];not3_core=[]
    traces=[]
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or (r.get('race_type') or '')!='Ｓ級決勝':continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210:continue
        lines=pl(r.get('predicted_line_formation')); cars=sorted({v for c in trio[rid] for v in c})
        if lines is None or len(cars)!=7 or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]:continue
        d=build_v8_20_f21(trio[rid],tf[rid],r.get('predicted_line_formation') or '','Ｓ級決勝')
        if not d.get('buy'):continue
        q=implied_probabilities(tf[rid]); st=_state(d['first'],d['second'],d['third'],q)
        if st is None:continue
        p0=_price_state(st,st.q_mass,q,tf[rid]); states,moves=_compression_path(p0,q,tf[rid])
        ki=_select_structural_knee(states); knee=states[ki]
        diag=selection_divergence_diagnostics(trio[rid],tf[rid]); c1=diag['trio_top_sets'][0]; c2=diag['tf_set_top_sets'][0]
        prefix=[]
        for ps in states:
            if contains_set(ps,c1) and contains_set(ps,c2): prefix.append(ps)
            else: break
        ci=_select_structural_knee(prefix) if prefix else 0
        core=prefix[ci] if prefix else p0
        br=row_from_state(rid,r,p0,pay[rid]); kr=row_from_state(rid,r,knee,pay[rid]); cr=row_from_state(rid,r,core,pay[rid])
        base_rows.append(br);knee_rows.append(kr);core_rows.append(cr)
        if len(lines)==4: line4_core.append(cr)
        if len(lines)!=3: not3_core.append(cr)
        traces.append({'race_id':rid,'line_count':len(lines),'base':br,'knee':kr,'core':cr,'path_states':len(states),'core_prefix_states':len(prefix),'knee_step':ki,'core_step':ci})
    return {'dataset':label,'states':{'BASE':summary(base_rows),'UNCONSTRAINED_KNEE':summary(knee_rows),'MARKET_CORE_PROTECTED_KNEE':summary(core_rows),'LINE4_PLUS_CORE_KNEE':summary(line4_core),'NOT_3LINE_PLUS_CORE_KNEE':summary(not3_core)},'traces':traces}

def main():
    result={'analysis':'FINAL_STRUCTURAL_COMPRESSION_Q1Q2','status':'DEVELOPMENT_DIAGNOSTIC','concept':'After current FINAL entry and F09 formation, whole-rider price compression only. Market-core variant stops compression before losing representation of either trio-top or trifecta-collapsed-top three-rider set.','no_individual_ticket_pruning':True,'Q1':evaluate('2024Q1_DEVELOPMENT',load()),'Q2':evaluate('2024Q2_DEVELOPMENT_AFTER_OBSERVED',load_q2())}
    print('FINAL_STRUCTURAL_COMPRESSION_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('FINAL_STRUCTURAL_COMPRESSION_END')
if __name__=='__main__':main()

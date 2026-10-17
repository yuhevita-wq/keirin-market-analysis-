from __future__ import annotations
import json
from collections import Counter, defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_20_f21_initial_special_unified import INITIAL_SPECIAL_LABELS
from v8_22_f23_qualifying_single_pole import build_v8_22_f23

STAKE=100
GROUPS=('QUALIFYING','GENERAL','SEMIFINAL','SPECIAL','FINAL','OTHER')


def summary(rows):
    n=len(rows); h=sum(r['hit'] for r in rows); t=sum(r['tickets'] for r in rows); p=sum(r['payout'] for r in rows); s=t*STAKE
    return {'bet_races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None}


def evaluate(label,data):
    races,trio,tf,pay=data
    rows=[]; by_group=defaultdict(list); by_type=defaultdict(list); initial_rows=[]
    pop_group=Counter(); pop_type=Counter(); fail=Counter(); population=0; selection_diag=0; final_quarantined=0
    qualifying_anchor=[]
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'): continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]: continue
        population+=1; rt=r.get('race_type') or ''; g=classify_race_type(rt); pop_group[g]+=1; pop_type[rt]+=1
        d=build_v8_22_f23(trio[rid],tf[rid],r.get('predicted_line_formation') or '',rt)
        if d.get('selection_diagnostics') is not None: selection_diag+=1
        if rt=='Ｓ級決勝' and d.get('branch_policy')=='FINAL_HEAT_QUARANTINE_DIAGNOSTIC_ONLY': final_quarantined+=1
        if not d.get('buy'):
            fail[f'{rt}:{d.get("reason")}']+=1; continue
        wins=[t for t in d['tickets'] if t in pay[rid]]; payout=sum(pay[rid][t] for t in wins); n=int(d['ticket_count'])
        row={'race_id':rid,'race_date':r.get('race_date'),'race_type':rt,'group':g,'hit':int(bool(wins)),'payout':payout,'tickets':n,'formation':d.get('formation'),'branch_policy':d.get('branch_policy')}
        rows.append(row); by_group[g].append(row); by_type[rt].append(row)
        if rt in INITIAL_SPECIAL_LABELS: initial_rows.append(row)
        if g=='QUALIFYING': qualifying_anchor.append(row)
    return {
        'dataset':label,'population':population,'overall':summary(rows),
        'by_group':{g:{**summary(by_group[g]),'population':pop_group[g]} for g in GROUPS},
        'by_race_type':{rt:{**summary(by_type[rt]),'population':pop_type[rt]} for rt in sorted(pop_type)},
        'INITIAL_SPECIAL_UNIFIED':{**summary(initial_rows),'population':sum(pop_type[x] for x in INITIAL_SPECIAL_LABELS)},
        'QUALIFYING_SINGLE_POLE':summary(qualifying_anchor),
        'selection_diagnostic_races':selection_diag,
        'final_quarantined_population':final_quarantined,
        'entry_fail_reasons':dict(fail)
    }


def main():
    q1=evaluate('2024Q1_DEVELOPMENT',load()); q2=evaluate('2024Q2_DEVELOPMENT_AFTER_OBSERVED',load_q2())
    result={
        'scheme':'v8.22-F23',
        'status':'DEVELOPMENT_Q1Q2_QUALIFYING_SINGLE_POLE',
        'change_from_v8_21':'QUALIFYING only: PS_AB + H_CONCENTRATED + H_AB=0; protect H1 in first place; price contraction only in second/third. All other branches unchanged.',
        'Q1':q1,'Q2':q2,
        'acceptance':{'Q1_profit_positive':q1['overall']['profit_yen']>0,'Q2_profit_positive':q2['overall']['profit_yen']>0},
        'next_validation':'Q3_UNTOUCHED_ONLY_IF_V8_22_RULE_IS_FROZEN'
    }
    print('V8_22_F23_Q1Q2_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('V8_22_F23_Q1Q2_END')

if __name__=='__main__':main()

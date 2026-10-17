from __future__ import annotations
import json
from collections import Counter, defaultdict

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_20_f21_initial_special_unified import INITIAL_SPECIAL_LABELS
from v8_25_f26_final_set_lock_only import build_v8_25_f26

STAKE=100
GROUPS=('QUALIFYING','GENERAL','SEMIFINAL','SPECIAL','FINAL','OTHER')

def summary(rows):
    n=len(rows);h=sum(r['hit'] for r in rows);t=sum(r['tickets'] for r in rows);p=sum(r['payout'] for r in rows);s=t*STAKE
    return {'bet_races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None}

def evaluate(label,data):
    races,trio,tf,pay=data
    rows=[];by_group=defaultdict(list);by_type=defaultdict(list);initial=[];selection=[];final=[];pop_group=Counter();pop_type=Counter();fail=Counter();population=0;head_diag=0
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'):continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210:continue
        cars=sorted({v for c in trio[rid] for v in c});lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]:continue
        population+=1;rt=r.get('race_type') or '';g=classify_race_type(rt);pop_group[g]+=1;pop_type[rt]+=1
        d=build_v8_25_f26(trio[rid],tf[rid],r.get('predicted_line_formation') or '',rt)
        if rt=='Ｓ級決勝' and d.get('reason')=='FINAL_HEAD_LOCK_DIAGNOSTIC_ONLY_Q1_FAIL':head_diag+=1
        if not d.get('buy'):
            fail[f'{rt}:{d.get("reason")}']+=1;continue
        wins=[t for t in d['tickets'] if t in pay[rid]];payout=sum(pay[rid][t] for t in wins);n=int(d['ticket_count'])
        row={'race_id':rid,'race_date':r.get('race_date'),'race_type':rt,'group':g,'hit':int(bool(wins)),'payout':payout,'tickets':n,'formation':d.get('formation'),'branch_policy':d.get('branch_policy')}
        rows.append(row);by_group[g].append(row);by_type[rt].append(row)
        if rt in INITIAL_SPECIAL_LABELS:initial.append(row)
        if rt=='Ｓ級選抜':selection.append(row)
        if rt=='Ｓ級決勝':final.append(row)
    return {
        'dataset':label,'population':population,'overall':summary(rows),
        'by_group':{g:{**summary(by_group[g]),'population':pop_group[g]} for g in GROUPS},
        'by_race_type':{rt:{**summary(by_type[rt]),'population':pop_type[rt]} for rt in sorted(pop_type)},
        'INITIAL_SPECIAL_UNIFIED':{**summary(initial),'population':sum(pop_type[x] for x in INITIAL_SPECIAL_LABELS)},
        'SELECTION_ULTRA_ROI':{**summary(selection),'population':pop_type['Ｓ級選抜']},
        'FINAL_SET_LOCK_ORDER_SPLIT':{**summary(final),'population':pop_type['Ｓ級決勝'],'head_lock_diagnostic_only_count':head_diag},
        'entry_fail_reasons':dict(fail)
    }

def main():
    q1=evaluate('2024Q1_DEVELOPMENT',load());q2=evaluate('2024Q2_DEVELOPMENT_AFTER_OBSERVED',load_q2())
    result={'scheme':'v8.25-F26','status':'DEVELOPMENT_Q1Q2_FINAL_SET_LOCK_ONLY','change_from_v8_24':'FINAL HEAD_LOCK x TAIL_SPLIT moved to diagnostic-only because Q1 ROI <100. SET_LOCK x ORDER_SPLIT remains active unchanged. All non-FINAL branches unchanged from v8.23.','Q1':q1,'Q2':q2,'acceptance':{'Q1_profit_positive':q1['overall']['profit_yen']>0,'Q2_profit_positive':q2['overall']['profit_yen']>0,'FINAL_Q1_profit_positive':q1['FINAL_SET_LOCK_ORDER_SPLIT']['profit_yen']>0,'FINAL_Q2_profit_positive':q2['FINAL_SET_LOCK_ORDER_SPLIT']['profit_yen']>0},'q3_status':'UNTOUCHED_BY_V8_25'}
    print('V8_25_F26_Q1Q2_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('V8_25_F26_Q1Q2_END')
if __name__=='__main__':main()

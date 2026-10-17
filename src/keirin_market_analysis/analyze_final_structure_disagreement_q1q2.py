from __future__ import annotations
import json
from collections import Counter

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2
from v8_17_f18_market_semantics import selection_divergence_diagnostics
from v8_20_f21_initial_special_unified import build_v8_20_f21

STAKE=100

def summary(rows):
    n=len(rows);h=sum(r['hit'] for r in rows);t=sum(r['tickets'] for r in rows);p=sum(r['payout'] for r in rows);s=t*STAKE
    return {'bet_races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None}

def evaluate(label,data):
    races,trio,tf,pay=data;rows=[]
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or (r.get('race_type') or '')!='Ｓ級決勝':continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210:continue
        lines=pl(r.get('predicted_line_formation'))
        cars=sorted({v for c in trio[rid] for v in c})
        if lines is None or len(cars)!=7 or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]:continue
        d=build_v8_20_f21(trio[rid],tf[rid],r.get('predicted_line_formation') or '','Ｓ級決勝')
        if not d.get('buy'):continue
        wins=[t for t in d['tickets'] if t in pay[rid]];payout=sum(pay[rid][t] for t in wins)
        diag=selection_divergence_diagnostics(trio[rid],tf[rid])
        sizes=tuple(sorted((len(x) for x in lines),reverse=True))
        singleton_count=sum(1 for x in lines if len(x)==1)
        top_trio=tuple(diag['trio_top_sets'][0]); top_tf=tuple(diag['tf_set_top_sets'][0])
        first=set(d.get('first') or ()); second=set(d.get('second') or ()); third=set(d.get('third') or ())
        trio_top_in_rect=any((a in first and b in second and c in third) for a in top_trio for b in top_trio for c in top_trio if len({a,b,c})==3)
        tf_top_in_rect=any((a in first and b in second and c in third) for a in top_tf for b in top_tf for c in top_tf if len({a,b,c})==3)
        row={'race_id':rid,'race_date':r.get('race_date'),'track':r.get('track'),'hit':int(bool(wins)),'payout':payout,'tickets':int(d['ticket_count']),'formation':d.get('formation'),'line_count':len(lines),'line_sizes':'-'.join(map(str,sizes)),'singleton_count':singleton_count,'top_set_same':int(bool(diag['top_set_same'])),'top3_overlap':int(diag['top3_overlap']),'top5_overlap':int(diag['top5_overlap']),'tv_distance':float(diag['tv_distance']),'trio_top_set_in_rect':int(trio_top_in_rect),'tf_top_set_in_rect':int(tf_top_in_rect)}
        row['both_top_sets_in_rect']=int(trio_top_in_rect and tf_top_in_rect)
        row['top_set_disagree']=int(not diag['top_set_same'])
        row['top3_full_agree']=int(diag['top3_overlap']==3)
        row['top3_not_full']=int(diag['top3_overlap']<3)
        rows.append(row)
    states={'ALL':summary(rows)}
    for k in (2,3,4,5): states[f'LINE_COUNT_{k}']=summary([r for r in rows if r['line_count']==k])
    for f in ('top_set_same','top_set_disagree','top3_full_agree','top3_not_full','trio_top_set_in_rect','tf_top_set_in_rect','both_top_sets_in_rect'):
        states[f.upper()]=summary([r for r in rows if r[f]])
    for s in sorted(set(r['line_sizes'] for r in rows)):
        states[f'LINE_SIZES_{s}']=summary([r for r in rows if r['line_sizes']==s])
    return {'dataset':label,'states':states,'rows':rows,'line_size_counts':dict(Counter(r['line_sizes'] for r in rows))}

def main():
    result={'analysis':'FINAL_STRUCTURE_DISAGREEMENT_Q1Q2','status':'DEVELOPMENT_DIAGNOSTIC','rules_tested':'semantic line-count/line-size buckets and trio-vs-trifecta set-ranking agreement; no fitted numeric cutoff','Q1':evaluate('2024Q1_DEVELOPMENT',load()),'Q2':evaluate('2024Q2_DEVELOPMENT_AFTER_OBSERVED',load_q2())}
    print('FINAL_STRUCTURE_DISAGREEMENT_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('FINAL_STRUCTURE_DISAGREEMENT_END')
if __name__=='__main__':main()

from collections import Counter
import json
from simulate_v8_1_f02_2024q1 import load, pi, pl
from v8_10_f11_structural_price_compression import build_v8_10_f11


def main():
    races,trio,tf,pay=load(); rows=[]; fail=Counter()
    for rid,r in sorted(races.items(),key=lambda x:(x[1].get('race_date',''),x[0])):
        if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'): continue
        if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
        cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
        if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]: continue
        d=build_v8_10_f11(trio[rid],tf[rid],r.get('predicted_line_formation') or '')
        if not d.get('buy'):
            fail[str(d.get('reason'))]+=1; continue
        wins=[t for t in d['tickets'] if t in pay[rid]]
        rows.append((rid,d,sum(pay[rid][t] for t in wins),bool(wins)))
    n=len(rows); hits=sum(x[3] for x in rows); tb=sum(int(x[1].get('ticket_count_before_price') or x[1]['ticket_count']) for x in rows); ta=sum(x[1]['ticket_count'] for x in rows); payout=sum(x[2] for x in rows); stake=ta*100
    result={
      'scheme':'v8.10-F11','dataset':'2024Q1','bet_races':n,'hits':hits,'hit_rate_pct':100*hits/n if n else None,
      'tickets_before_price':tb,'tickets_after_price':ta,'avg_before':tb/n if n else None,'avg_after':ta/n if n else None,
      'ticket_reduction_pct':100*(tb-ta)/tb if tb else None,'compressed_races':sum(x[1]['ticket_count']<int(x[1].get('ticket_count_before_price') or x[1]['ticket_count']) for x in rows),
      'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi_pct':100*payout/stake if stake else None,
      'entry_fail_reasons':dict(fail),'acceptance':'PASS_Q1_PROFIT' if payout>stake else 'REJECT_Q1_NOT_PROFITABLE'}
    print('V8_10_F11_Q1_RESULT_BEGIN'); print(json.dumps(result,ensure_ascii=False,indent=2)); print('V8_10_F11_Q1_RESULT_END')
    audit=[]
    for rid,d,p,h in sorted(rows,key=lambda x:(-(int(x[1].get('ticket_count_before_price') or x[1]['ticket_count'])-x[1]['ticket_count']),x[0]))[:10]:
        audit.append({'race_id':rid,'before':d.get('formation_before_price'),'after':d['formation'],'points_before':d.get('ticket_count_before_price'),'points_after':d['ticket_count'],'q_retention':d.get('price_q_retention'),'hit':int(h),'payout_yen':p})
    print('V8_10_F11_Q1_AUDIT_BEGIN'); print(json.dumps(audit,ensure_ascii=False,indent=2)); print('V8_10_F11_Q1_AUDIT_END')

if __name__=='__main__': main()

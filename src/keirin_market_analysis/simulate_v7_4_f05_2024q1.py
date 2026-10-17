from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
from v7_4_f05_cumulative_elbow import build_v7_4_f05
SCHEME="v7.4-F05";STAKE=100;ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/"data/2024/s_class_f1_all_parts/2024_q1"
def rr(n):
 with (DATA/n).open("r",encoding="utf-8-sig",newline="") as f:yield from csv.DictReader(f)
def pi(x):
 try:return int(float(x))
 except:return None
def pf(x):
 try:return float(x)
 except:return None
def pt(s):
 x=tuple(sorted(int(v) for v in (s or '').replace('=','-').replace(',','-').split('-') if v.strip().isdigit()));return x if len(x)==3 and len(set(x))==3 else None
def p3(s):
 x=tuple(int(v) for v in (s or '').replace('=','-').replace(',','-').split('-') if v.strip().isdigit());return x if len(x)==3 and len(set(x))==3 else None
def pl(s):
 z=[]
 for raw in (s or '').split('/'):
  if not raw.strip():continue
  a=raw.strip().split('-')
  if not all(v.strip().isdigit() for v in a):return None
  z.append(tuple(int(v) for v in a))
 flat=[v for a in z for v in a];return tuple(z) if z and len(flat)==len(set(flat)) else None
def load():
 races={r['race_id']:r for r in rr('races.csv')};trio=defaultdict(dict);tf=defaultdict(dict);pay=defaultdict(dict)
 for r in rr('trio_final_odds.csv'):
  c,o=pt(r.get('combination')),pf(r.get('odds'))
  if c and o and o>0 and r.get('odds_status')=='available':trio[r['race_id']][c]=o
 for r in rr('trifecta_final_odds.csv'):
  c,o=p3(r.get('combination')),pf(r.get('odds'))
  if c and o and o>0 and r.get('odds_status')=='available':tf[r['race_id']][c]=o
 for r in rr('payouts.csv'):
  if r.get('bet_code')!='trifecta' and r.get('ticket_type')!='3連単':continue
  if r.get('status')!='paid':continue
  c,y=p3(r.get('combination')),pi(r.get('payout_yen'))
  if c and y is not None:pay[r['race_id']][c]=y
 return races,trio,tf,pay
def streak(rs):
 cur=best=0
 for r in sorted(rs,key=lambda x:(x['race_date'],x['race_id'])):
  if r['hit']:cur=0
  else:cur+=1;best=max(best,cur)
 return best
def main():
 races,trio,tf,pay=load();out=[];fail=Counter();sizes=Counter();attr=defaultdict(int);attr['races_csv']=len(races)
 for rid,r in sorted(races.items(),key=lambda kv:(kv[1].get('race_date',''),kv[0])):
  if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'):continue
  attr['f1_s']+=1
  if pi(r.get('entry_count'))!=7:continue
  attr['seven_car']+=1
  if len(trio.get(rid,{}))!=35:continue
  attr['complete_trio35']+=1
  if len(tf.get(rid,{}))!=210:continue
  attr['complete_tf210']+=1
  cars=sorted({v for c in trio[rid] for v in c});lines=pl(r.get('predicted_line_formation'))
  if len(cars)!=7 or lines is None or set(v for a in lines for v in a)!=set(cars):continue
  attr['complete_line']+=1
  if rid not in pay or not pay[rid]:continue
  attr['has_tf_payout']+=1;attr['population']+=1
  d=build_v7_4_f05(trio[rid],tf[rid],r.get('predicted_line_formation') or '')
  if not d.get('buy'):fail[str(d.get('reason'))]+=1;continue
  attr['entry_pass']+=1;ts=tuple(d['tickets']);win=[t for t in ts if t in pay[rid]];payout=sum(pay[rid][t] for t in win);size=f"{len(d['first'])}-{len(d['second'])}-{len(d['third'])}";sizes[size]+=1
  out.append({'race_id':rid,'race_date':r.get('race_date'),'formation':d['formation'],'size_pattern':size,'ticket_count':d['ticket_count'],'market_mass':d['market_mass'],'hit':int(bool(win)),'payout_yen':payout})
 h=sum(r['hit'] for r in out);n=sum(r['ticket_count'] for r in out);p=sum(r['payout_yen'] for r in out);s=n*STAKE
 result={'scheme_version':SCHEME,'dataset':'2024Q1','entry_rule':'unchanged v6.1 pre-formation gate','fixed_place_counts':False,'price_cut':False,'attrition':dict(attr),'entry_fail_reasons':dict(fail),'summary':{'bet_races':len(out),'hit_races':h,'miss_races':len(out)-h,'hit_rate_pct':100*h/len(out) if out else None,'tickets':n,'avg_tickets_per_race':n/len(out) if out else None,'min_tickets_per_race':min((r['ticket_count'] for r in out),default=None),'max_tickets_per_race':max((r['ticket_count'] for r in out),default=None),'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None,'max_losing_streak':streak(out)},'size_pattern_distribution':dict(sorted(sizes.items(),key=lambda kv:(-kv[1],kv[0])))}
 print('V7_4_F05_Q1_RESULT_BEGIN');print(json.dumps(result,ensure_ascii=False,indent=2));print('V7_4_F05_Q1_RESULT_END')
if __name__=='__main__':main()

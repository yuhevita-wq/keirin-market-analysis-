from __future__ import annotations
import json
from collections import defaultdict
from simulate_v8_1_f02_2024q1 import load, pi, pl
from v8_8_f09_market_cliff import build_v8_8_f09

STAKE=100
TARGETS={'SELECTION':'Ｓ級選抜','FINAL':'Ｓ級決勝'}

def summ(rows):
 n=len(rows); h=sum(r['hit'] for r in rows); t=sum(r['n'] for r in rows); p=sum(r['payout'] for r in rows); s=t*STAKE
 ph=sum(1 for r in rows if r['hit'] and r['payout']>r['n']*STAKE)
 hl=sum(1 for r in rows if r['hit'] and r['payout']<=r['n']*STAKE)
 return {'races':n,'hits':h,'hit_rate':100*h/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake':s,'payout':p,'profit':p-s,'roi':100*p/s if s else None,'profitable_hits':ph,'hit_loss_or_even':hl}

def main():
 races,trio,tf,pay=load(); out={}
 for label,rtarget in TARGETS.items():
  pop=0; passrows=[]; psfail=0
  for rid,r in races.items():
   if r.get('meeting_grade')!='F1' or r.get('race_type')!=rtarget: continue
   if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: continue
   cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
   if len(cars)!=7 or lines is None or set(v for line in lines for v in line)!=set(cars) or rid not in pay or not pay[rid]: continue
   pop+=1; d=build_v8_8_f09(trio[rid],tf[rid],r.get('predicted_line_formation') or '')
   if not d.get('buy'): psfail+=1; continue
   wins=[x for x in d['tickets'] if x in pay[rid]]; payout=sum(pay[rid][x] for x in wins); n=d['ticket_count']; eg=d['entry_gate']
   hc=not bool(eg['H_RATIO']); hab=bool(eg['H_AB']); gm=(d.get('weighted_gm_selected_odds') or 0)/n if n else 0
   pqs=(d.get('profit_mass_1x') or 0)/(d.get('q_mass') or 1)
   passrows.append({'rid':rid,'hit':int(bool(wins)),'payout':payout,'n':n,'hstate':eg['H_state'],'hc':hc,'hab':hab,'consistent':hab==hc,'first1':len(d['first'])==1,'gmbe':gm>=1,'pqs50':pqs>=.5,'gm':gm,'pqs':pqs,'formation':d['formation']})
  groups={}
  selectors={
   'ALL_PS_AB':lambda r:True,
   'H_CONCENTRATED':lambda r:r['hc'],
   'H_NOT_CONCENTRATED':lambda r:not r['hc'],
   'H_AB':lambda r:r['hab'],
   'H_NOT_AB':lambda r:not r['hab'],
   'H_STATE_CONSISTENT':lambda r:r['consistent'],
   'H_STATE_MISMATCH':lambda r:not r['consistent'],
   'FIRST_POOL_1':lambda r:r['first1'],
   'FIRST_POOL_GE2':lambda r:not r['first1'],
   'GM_BREAK_EVEN':lambda r:r['gmbe'],
   'GM_BELOW_BREAK_EVEN':lambda r:not r['gmbe'],
   'PROFITABLE_Q_MAJORITY':lambda r:r['pqs50'],
   'PROFITABLE_Q_MINORITY':lambda r:not r['pqs50'],
  }
  for k,f in selectors.items(): groups[k]=summ([r for r in passrows if f(r)])
  bystate={s:summ([r for r in passrows if r['hstate']==s]) for s in sorted({r['hstate'] for r in passrows})}
  buckets={'N_1_6':summ([r for r in passrows if r['n']<=6]),'N_7_12':summ([r for r in passrows if 7<=r['n']<=12]),'N_13_PLUS':summ([r for r in passrows if r['n']>=13])}
  out[label]={'population':pop,'ps_ab_fail':psfail,'ps_ab_pass':len(passrows),'groups':groups,'by_h_state':bystate,'ticket_buckets':buckets,'rows':passrows}
 print('SELECTION_FINAL_Q1_DIAGNOSTIC_BEGIN'); print(json.dumps(out,ensure_ascii=False,indent=2)); print('SELECTION_FINAL_Q1_DIAGNOSTIC_END')
if __name__=='__main__':main()

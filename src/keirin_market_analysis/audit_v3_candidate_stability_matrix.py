from __future__ import annotations

import json
from pathlib import Path

from .audit_v3_failure_decomposition import PERIODS, load_period
from .simulate_mainline_v1 import choose_main_line
from .simulate_early_candidate_v0 import strongest_rival
from .search_three_year_conditions_v1 import feature_row, formations

OUT=Path('data/audits/v3_candidate_stability_matrix.json')

FAMILIES={
 'early_main_balance':{
  'segment':'前半','form':'MIX2',
  'rules':[(f'main_top2_ge_{t}__leader_gap_le_{g}',lambda r,t=t,g=g:float(r['main_top2'])>=t and float(r['leader_score_gap'])<=g) for t in (50,60,70) for g in (0,2,4)]},
 'early_score_win':{
  'segment':'前半','form':'MIX2',
  'rules':[(f'a_score_le_{a}__pair_win_gap_le_{w}',lambda r,a=a,w=w:float(r['a_score'])<=a and float(r['pair_win_gap'])<=w) for a in (102,104,106) for w in (-10,0,10)]},
 'middle_rival_band_main4':{
  'segment':'中盤','form':'MAIN4_RIVAL',
  'rules':[(f'rival_top2_ge_{t2}__rival_top3_le_{t3}',lambda r,t2=t2,t3=t3:float(r['rival_top2'])>=t2 and float(r['rival_top3'])<=t3) for t2 in (50,60,70) for t3 in (80,90,100)]},
 'middle_rival_band_main6':{
  'segment':'中盤','form':'MAIN6',
  'rules':[(f'rival_top2_ge_{t2}__rival_top3_le_{t3}',lambda r,t2=t2,t3=t3:float(r['rival_top2'])>=t2 and float(r['rival_top3'])<=t3) for t2 in (50,60,70) for t3 in (80,90,100)]},
 'late_pair_win':{
  'segment':'後半','form':'MAIN4_X',
  'rules':[(f'pair_win_gap_le_{w}',lambda r,w=w:float(r['pair_win_gap'])<=w) for w in (-20,-10,0)]},
 'late_pair_win_top3':{
  'segment':'後半','form':'MAIN4_X',
  'rules':[(f'pair_win_gap_le_{w}__pair_top3_gap_le_{g}',lambda r,w=w,g=g:float(r['pair_win_gap'])<=w and float(r['pair_top3_gap'])<=g) for w in (-20,-10,0) for g in (0,10,25,40)]},
}


def fin(rows):
 n=len(rows); st=sum(x[0] for x in rows); py=sum(x[1] for x in rows); hp=[x[1] for x in rows if x[1]>0]
 return {'races':n,'hits':len(hp),'stake_yen':st,'payout_yen':py,'profit_yen':py-st,'roi':py/st if st else 0.0,'top1_payout_share':max(hp)/py if hp and py else 0.0}

def build(segment,form):
 out={p:[] for p in PERIODS}
 for period,path in PERIODS.items():
  races,eb,tri,_rb,seg=load_period(path)
  for race in races:
   rid=race['race_id']
   if seg.get(rid)!=segment: continue
   es=eb[rid]; chosen=choose_main_line(es)
   if not chosen: continue
   mid,main=chosen
   if len(main)<3: continue
   rival=strongest_rival(es,mid)
   if not rival or len(rival)<2: continue
   row=feature_row(race,es,mid,main,rival); bets=formations(row,es)[form]
   out[period].append((row,100*len(bets),sum(tri[rid].get(b,0) for b in bets)))
 return out

def main():
 out={'status':'V3_CANDIDATE_STABILITY_MATRIX_COMPLETE','families':{}}
 cache={}
 for fname,spec in FAMILIES.items():
  key=(spec['segment'],spec['form']); data=cache.setdefault(key,build(*key))
  rows=[]
  for name,pred in spec['rules']:
   ps={p:fin([(st,py) for r,st,py in data[p] if pred(r)]) for p in PERIODS}
   rois=[ps[p]['roi'] for p in PERIODS]
   rows.append({'rule':name,'periods':ps,'all4_profitable':all(x>1 for x in rois),'profitable_periods':sum(x>1 for x in rois),'worst_roi':min(rois),'min_races':min(ps[p]['races'] for p in PERIODS),'combined_roi':sum(ps[p]['payout_yen'] for p in PERIODS)/sum(ps[p]['stake_yen'] for p in PERIODS) if sum(ps[p]['stake_yen'] for p in PERIODS) else 0})
  rows.sort(key=lambda x:(x['all4_profitable'],x['worst_roi'],x['min_races'],x['combined_roi']),reverse=True)
  out['families'][fname]={'segment':spec['segment'],'formation':spec['form'],'all4_count':sum(x['all4_profitable'] for x in rows),'variants':rows}
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:{'all4_count':v['all4_count'],'top':v['variants'][:5]} for k,v in out['families'].items()},ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())

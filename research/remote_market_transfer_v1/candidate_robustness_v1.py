#!/usr/bin/env python3
"""Robustness audit of post-hoc odds-only positive ROI candidate.
Do NOT market this discovery as established profit: thousands of signatures were searched.
"""
import itertools,json,math,sys
from datetime import date,timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'/'nonlocal_break_even_v1'))
from analyze import load
sys.path.insert(0,str(ROOT/'research'/'remote_market_transfer_v1'))
from analyze_adaptive_v2 import keys_for_race
OUT=ROOT/'data/audits/remote_market_candidate_robustness_v1.json'
BINS={
  'mass_neighbors':[(.0246583,.0307585),(.0307585,.0366855),(.0366855,.0428778),(.0428778,.051597),(.051597,1.)],
  'rank_neighbors':[(1,6),(7,12),(13,24),(25,48),(49,100)],
  'gap_neighbors':[(-2.,-.15),(-.15,-.05),(-.05,.05),(.05,.15),(.15,5.)],
  'outside_share':[(0.,.0632254),(.0632254,.1196454),(.1196454,.1755241),(.1755241,.23835),(.23835,1.1)],
}
def row(y,month,field,source_mass,rank,gap,share,odds,market_q,win):
 return {'year':y,'month':month,'field':field,'donor_mass':source_mass,
         'rank':rank,'gap':gap,'outside_share':share,'odds':odds,'q':market_q,'winner':win}
def actual_month(rid):
 try:
  dt=date(int(rid[2:6]),int(rid[6:8]),int(rid[8:10]))+timedelta(days=int(rid[10:12])-1)
  return dt.month
 except (ValueError,IndexError):
  return 0
def expand():
 by_year,audit,labels=load()
 groups=defaultdict(list)
 for y,rs in by_year.items():
  for r in rs:
   values,ordered,_=keys_for_race(r)
   source_mass=float(values['donor_mass'][0])
   month=actual_month(r.rid)
   field=len(set(itertools.chain.from_iterable(r.sets)))
   for j in range(len(r.sets)):
    rank=float(values['rank'][j])
    gap=float(values['market_gap'][j])
    share=float(values['outside_share'][j])
    odds=float(r.trio_odds[j])
    q=float(r.qt[j])
    is_hit=int(j==r.y_set)
    base=row(y,month,field,source_mass,rank,gap,share,odds,q,is_hit)
    if gap>.15 and 12<rank<=24:
     groups[f'Y{y}_all_donor_mass'].append(base)
     for lo,hi in BINS['mass_neighbors']:
      if lo<source_mass<=hi:
       groups[f'Y{y}_donor_mass({lo},{hi}]'].append(base)
    if gap>.15 and .0366855<source_mass<=.0428778:
     for lo,hi in BINS['rank_neighbors']:
      if lo<=rank<=hi:
       groups[f'Y{y}_rank({lo},{hi})'].append(base)
    if 12<rank<=24 and .0366855<source_mass<=.0428778:
     for lo,hi in BINS['gap_neighbors']:
      if lo<gap<=hi:
       groups[f'Y{y}_gap({lo},{hi}]'].append(base)
    if gap>.15 and 12<rank<=24 and .0366855<source_mass<=.0428778:
     groups[f'Y{y}_selected'].append(base)
     if field<=7: groups[f'Y{y}_field<=7'].append(base)
     if field>=8: groups[f'Y{y}_field>=8'].append(base)
     if month<=6: groups[f'Y{y}_H1'].append(base)
     else: groups[f'Y{y}_H2'].append(base)
     for lo,hi in BINS['outside_share']:
      if lo<=share<=hi:
       groups[f'Y{y}_outside_share({lo},{hi})'].append(base)
  print('EXTRACT',y,'total',len(rs),flush=True)
 return groups,audit
def stats(rows):
 if not rows:return None
 n=len(rows)
 hit=[r for r in rows if r['winner']]
 payout=float(sum(r['odds'] for r in hit))
 expected=float(sum(r['q'] for r in rows))
 maxhit=max(hit,key=lambda r:r['odds']) if hit else None
 rno=payout/max(n,1)
 r_minus=max(0,payout-(maxhit['odds'] if maxhit else 0))/max(n-1,1)
 # Each group has <= one winner per race; confidence below is a conservative
 # independent-ticket approximation, not a multiplicity-corrected CI.
 v=np.array([r['odds'] if r['winner'] else 0 for r in rows],float)
 stderr=np.std(v,ddof=1)/math.sqrt(n) if n>1 else None
 return {'tickets':n,'hits':len(hit),'expected_hits':round(expected,3),
         'hit_to_market':round(len(hit)/expected,4) if expected else None,
         'roi_pct':round(100*rno,3),
         'payout_yen':round(100*payout),
         'profit_yen':round(100*(payout-n)),
         'max_win_odds':round(maxhit['odds'],1) if maxhit else None,
         'largest_win_payout_fraction_pct':round(100*maxhit['odds']/payout,2) if maxhit else 0,
         'roi_after_largest_win_removed_pct':round(100*r_minus,3),
         'approx95pct_ci':[round(100*(rno-1.96*stderr),2),
                          round(100*(rno+1.96*stderr),2)] if stderr else None}
def main():
 groups,audit=expand()
 out={
  'status':'POST_HOC_CANDIDATE_SENSITIVITY_NOT_A_VALIDATED_EDGE',
  'candidate':'3連複, q3/qtrio log gap > .15, trio market rank 13-24, race donor mass in (0.0366855,0.0428778]',
  'source_bin':'calibrated from 2023 odds only, without outcomes',
  'selected_after_wide_search':'5,446 correlated market signature groups; no multiple-testing adjustment',
  'no_line_rider_card_data':True,
  'no_prestart_15min_validation':True,
  'warning':'2025 and 2026 data were already inspected before this post-hoc candidate robustness. 2021/22 is earlier-time check, not forward heldout.',
  'audit':audit,
  'summary':{k:stats(v) for k,v in groups.items()},
 }
 OUT.parent.mkdir(parents=True,exist_ok=True)
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('SELECTED',[{'key':k,'result':stats(v)} for k,v in groups.items() if k.endswith('_selected')],flush=True)
 print('SAVED',OUT,flush=True)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Race-cluster bootstrap of a post-hoc odds signature, with normalized-market null."""
import itertools,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'/'nonlocal_break_even_v1'))
from analyze import load
sys.path.insert(0,str(ROOT/'research'/'remote_market_transfer_v1'))
from analyze_adaptive_v2 import keys_for_race
OUT=ROOT/'data/audits/remote_market_candidate_cluster_ci.json'
def bootstrap(stakes,returns,n_boot=3000):
 rng=np.random.default_rng(20261017)
 n=len(stakes)
 samples=np.empty(n_boot)
 for i in range(n_boot):
  idx=rng.integers(0,n,size=n)
  denom=stakes[idx].sum()
  samples[i]=returns[idx].sum()/denom if denom else np.nan
 return {'ci_95pct_roi':[round(100*v,2) for v in np.nanquantile(samples,[.025,.975])],
         'replicates':n_boot,
         'fraction_bootstrap_roi_above100':round(float(np.nanmean(samples>1)),4)}
def per_year(recs):
 stakes=[];returns=[];expected=[];variance=[];hit_count=0;bet_races=0;winners=[]
 for rec in recs:
  feats,_,m=keys_for_race(rec)
  if not (.0366855<m<=.0428778):
   stakes.append(0);returns.append(0.);expected.append(0.);variance.append(0.)
   continue
  sel=(feats['market_gap']>.15)&(feats['rank']>12)&(feats['rank']<=24)
  num=int(sel.sum())
  price=np.asarray(rec.trio_odds,dtype=float)
  q=np.asarray(rec.qt,dtype=float)
  win=rec.y_set
  payout=float(price[win]) if sel[win] else 0.
  avg=float((q[sel]*price[sel]).sum())
  sq=float((q[sel]*(price[sel]**2)).sum())
  stakes.append(num);returns.append(payout)
  expected.append(avg);variance.append(max(0,sq-avg**2))
  bet_races+=bool(num)
  hit_count+=int(payout>0)
  if payout>0:winners.append(payout)
 stakes=np.array(stakes,float)
 returns=np.array(returns,float)
 expected=np.array(expected,float)
 variance=np.array(variance,float)
 n_tickets=stakes.sum()
 actual=returns.sum()
 base=expected.sum()
 sd=math.sqrt(max(1e-12,variance.sum()))
 b=bootstrap(stakes,returns)
 return {'races':len(recs),'bet_races':bet_races,'tickets':int(n_tickets),
     'hits':hit_count,'actual_payout_yen':int(round(actual*100)),
     'expected_payout_yen_under_market':int(round(base*100)),
     'actual_roi_pct':round(100*actual/n_tickets,3) if n_tickets else None,
     'market_baseline_roi_pct':round(100*base/n_tickets,3) if n_tickets else None,
     'market_null_payout_z':round((actual-base)/sd,3),
     'market_null_approx_one_sided_tail_p':round(.5*math.erfc((actual-base)/sd/math.sqrt(2)),7),
     'largest_winner_odds':max(winners) if winners else 0,
     'race_cluster_bootstrap':b,
     '_stake_vec':stakes,'_return_vec':returns,'_expected_vec':expected,
     '_variance_vec':variance}
def main():
 years,audit,labels=load()
 stats={str(y):per_year(years[y]) for y in range(2021,2027)}
 d=stats['2025'];e=stats['2026']
 stakes=np.concatenate((d['_stake_vec'],e['_stake_vec']))
 returns=np.concatenate((d['_return_vec'],e['_return_vec']))
 expected=np.concatenate((d['_expected_vec'],e['_expected_vec']))
 variance=np.concatenate((d['_variance_vec'],e['_variance_vec']))
 total_stake=stakes.sum();actual=returns.sum();base=expected.sum();std=math.sqrt(variance.sum())
 combined={'tickets':int(total_stake),'payout_yen':int(round(actual*100)),
   'profit_yen':int(round(100*(actual-total_stake))),
   'roi_pct':round(100*actual/total_stake,3),
   'market_baseline_roi_pct':round(100*base/total_stake,3),
   'market_null_payout_z':round((actual-base)/std,3),
   'market_null_approx_one_sided_tail_p':round(.5*math.erfc((actual-base)/std/math.sqrt(2)),7),
   'race_cluster_bootstrap':bootstrap(stakes,returns)}
 def clean(x):
  return {k:v for k,v in x.items() if not k.startswith('_')}
 report={'schema':'POST_HOC_HISTORIC_MARKET_ODDS_SENSITIVITY_V1',
   'candidate':'3連複, own cross log gap >0.15, own trio rank 13..24, race-wide overpriced support gap in (0.0366855,0.0428778]',
   'input':'3連複 and 3連単 full final odds, no other covariates',
   'warning':['POST-HOC selection from 5446 correlated groups. This is NOT an unselected test.',
     'Normalized-market null assumes independent races and q proportional to inverse odds.',
     'Race cluster bootstrap CI is NOT adjusted for searching 5446 signatures.',
     'Pre-start 15-minute odds and the other five pools remain untested.'],
   'by_year':{y:clean(v) for y,v in stats.items()},
   '2025_2026_combined':combined}
 OUT.parent.mkdir(parents=True,exist_ok=True)
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('ROBUSTNESS',json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

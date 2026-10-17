#!/usr/bin/env python3
"""Control cross-market mistake for market-rank and offered-price confounding.
Everything except the true result used as the evaluation label is odds-only.
"""
import json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'/'nonlocal_break_even_v1'))
from analyze import load
OUT=ROOT/'data/audits/market_error_census_v1_rank_controls.json'
BINS={'rank':[1,3,6,12,24,48,96],'odds':[5,10,20,50,100,300,1000],
      'cross':[-.15,-.05,.05,.15]}
def i(v,cuts):return np.searchsorted(cuts,v,side='left').astype(int)
def main():
  years,audit,labels=load()
  out={'inputs':'trifecta and trio final odds only','result_only':'true top-three actual',
       'method':'every trio ticket stratified by offered price OR market rank and cross market discrepancy','year':{}}
  for year in [2023,2024,2025,2026]:
    data={}
    for key in ['rank_cross','odds_cross']:
      data[key]={k:np.zeros((8,5)) for k in ('ticket','expected','hits','payout','q2')}
    for r in years[year]:
      qt=np.asarray(r.qt,np.float64)
      q3=np.asarray(r.q3,np.float64)
      n=len(qt)
      rank=np.empty(n,int);rank[np.argsort(-qt)]=np.arange(1,n+1)
      cells={
        'rank_cross':(i(rank,BINS['rank']),i(np.log(q3/qt),BINS['cross'])),
        'odds_cross':(i(r.trio_odds,BINS['odds']),i(np.log(q3/qt),BINS['cross']))
      }
      for kind,(a,b) in cells.items():
        g=a*5+b
        sums=np.bincount(g,minlength=40).reshape((8,5))
        exp=np.bincount(g,weights=qt,minlength=40).reshape((8,5))
        stats=data[kind]
        stats['ticket']+=sums;stats['expected']+=exp;stats['q2']+=exp*exp
        x,y=a[r.y_set],b[r.y_set]
        stats['hits'][x,y]+=1;stats['payout'][x,y]+=float(r.trio_odds[r.y_set])
    out['year'][str(year)]={}
    for kind,arr in data.items():
      cells=[]
      for a in range(8):
        for b in range(5):
          num=arr['ticket'][a,b]
          if not num:continue
          exp=arr['expected'][a,b];hit=arr['hits'][a,b]
          pay=arr['payout'][a,b]
          z=(hit-exp)/math.sqrt(max(exp-arr['q2'][a,b],1e-9))
          cells.append({'base_bin':a,'cross_bin':b,'tickets':int(num),
            'expected_hits':round(float(exp),3),'actual_hits':int(hit),
            'actual_over_market':round(float(hit/exp),4) if exp else None,
            'roi_pct':round(float(100*pay/num),3),'deviation_z':round(z,3)})
      out['year'][str(year)][kind]=cells
    print('DONE',year,'races',len(years[year]),flush=True)
  OUT.parent.mkdir(parents=True,exist_ok=True)
  OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  print('SAVED',OUT,flush=True)
if __name__=='__main__':main()

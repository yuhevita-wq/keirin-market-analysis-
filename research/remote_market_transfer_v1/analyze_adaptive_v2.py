#!/usr/bin/env python3
"""Outcome-first donor->remote-receiver study using odds-only markets.

Target: identify from actual results whether 3連複 contracts overpriced RELATIVE
to complete 3連単 can forecast other distinct combination winners, and whether
the resulting price errors are large enough to beat 1/odds.

No lineups, rider features, grades, course, handicaps or race dates as predictors.
2023/24 discovery; 2025/26 temporal inspection; no future data for selection.
"""
import json, math, sys, itertools
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'/'nonlocal_break_even_v1'))
from analyze import load,get_masks

OUT=ROOT/'data/audits/remote_market_transfer_adaptive_v2.json'
# Breaks are fixed before inspecting this experiment's results.
BINS={
 'market_gap':[-.15,-.05,.05,.15],
 'price':[10,20,50,100,300,1000],
 'rank':[3,6,12,24,48],
 'donor_mass':[.10,.20,.35,.50],
 'outside_mass':[.08,.15,.25,.40],
 'outside_share':[.20,.40,.60,.80],
 'receiver_overlap':[.5,1.5,2.5],
 'n_car':[7.5],
 'order_rank':[1.5,2.5,3.5,4.5,5.5],
}
DIMENSIONS=[
 ('market_gap','outside_share'),('market_gap','outside_mass'),
 ('market_gap','donor_mass'),('market_gap','rank','outside_share'),
 ('market_gap','price','outside_share'),
 ('market_gap','rank','outside_mass'),('market_gap','price','outside_mass'),
 ('market_gap','rank','donor_mass'),
 ('market_gap','outside_share','n_car'),
 ('market_gap','price','donor_mass'),
 ('market_gap','outside_mass','n_car'),
 ('market_gap','price','n_car'),
 ('market_gap','rank','n_car'),
 ('market_gap','outside_share','donor_mass'),
 ('market_gap','outside_mass','donor_mass'),
]
ORDER_EXTRA=[
 ('market_gap','outside_share','order_rank'),
 ('market_gap','outside_mass','order_rank'),
 ('market_gap','donor_mass','order_rank'),
 ('market_gap','price','order_rank'),
 ('market_gap','rank','order_rank'),
]
FEATURES=['n','hits','expect','q2','payout','payout2','stake2','paystake']
def arrstat():
 return np.zeros(8,dtype=np.float64)
def keys_for_race(rec):
 sets=rec.sets
 q=np.asarray(rec.qt,dtype=float)
 q3=np.asarray(rec.q3,dtype=float)
 odds=np.asarray(rec.trio_odds,dtype=float)
 n=len(sets)
 lp=np.log(q3/q)
 donor=(lp < -.15)
 excess=np.maximum(q-q3,0)*donor
 donor_total=float(excess.sum())
 _,disjoint,overlap_one=get_masks(sets)
 # Excluded self from every "external information" calculation:
 other_mass=disjoint@excess
 other_share=other_mass/max(donor_total,1e-15)
 ranked=np.empty(n,dtype=float)
 ranked[np.argsort(-q)]=np.arange(1,n+1)
 vals={
 'market_gap':lp,
 'price':odds,
 'rank':ranked,
 'donor_mass':np.full(n,donor_total),
 'outside_mass':other_mass,
 'outside_share':other_share,
 'n_car':np.full(n,len(set(itertools.chain.from_iterable(sets)))),
 }
 ids=rec.ids
 p3=np.asarray(rec.q3[ids]*rec.cond,dtype=float)
 rank6=np.zeros(len(ids),dtype=float)
 conditional=np.asarray(rec.cond,dtype=float)
 for j in range(n):
  indices=np.flatnonzero(ids==j)
  order=np.argsort(-conditional[indices])
  rank6[indices[order]]=np.arange(1,len(indices)+1)
 ordered={k:v[ids] for k,v in vals.items()}
 ordered['price']=np.asarray(rec.tri3_odds,dtype=float)
 ordered['rank']=np.empty(len(ids),dtype=float)
 ordered['rank'][np.argsort(-p3)]=np.arange(1,len(ids)+1)
 ordered['order_rank']=rank6
 return vals,ordered,donor_total

def accumulate(tbl,rec,pool,features,values, year):
 if pool=='3連複':
  q=np.asarray(rec.qt,dtype=float)
  odds=np.asarray(rec.trio_odds,dtype=float)
  win=rec.y_set
 else:
  q=np.asarray(rec.q3[rec.ids]*rec.cond,dtype=float)
  odds=np.asarray(rec.tri3_odds,dtype=float)
  win=rec.y_order
 cats={k:np.searchsorted(BINS[k],v,side='left') for k,v in values.items()}
 for feat in features:
  levels=tuple(len(BINS[x])+1 for x in feat)
  ids=np.ravel_multi_index(tuple(cats[x] for x in feat),levels)
  count=math.prod(levels)
  key=(pool,feat,year)
  if key not in tbl:tbl[key]=np.zeros((8,count),dtype=float)
  m=tbl[key]
  n=np.bincount(ids,minlength=count)
  exp=np.bincount(ids,weights=q,minlength=count)
  expected2=exp*exp
  m[0]+=n;m[2]+=exp;m[3]+=expected2
  m[1,int(ids[win])]+=1
  m[4,int(ids[win])]+=float(odds[win])
  m[5,int(ids[win])]+=float(odds[win])**2
  m[6]+=n*n
  m[7,int(ids[win])]+=float(odds[win]*n[int(ids[win])])

def metrics(m,index):
 n=float(m[0,index])
 if n<=0:return None
 h=float(m[1,index]);ex=float(m[2,index])
 pay=float(m[4,index])
 roi=pay/n
 hit_z=(h-ex)/max(math.sqrt(max(ex-m[3,index],1e-8)),1e-4)
 err2=max(0,float(m[5,index]-2*roi*m[7,index]+roi*roi*m[6,index]))
 se=math.sqrt(err2)/n
 return {'n':round(n),'hits':round(h),'market_expected_hits':round(ex,3),
  'hit_to_market_ratio':round(h/ex,4) if ex>0 else None,
  'hit_z':round(hit_z,3),'roi_pct':round(100*roi,3),
  'net_yen':round(100*(pay-n)),
  'roi_approx95pct_ci':[round(100*(roi-1.96*se),2),round(100*(roi+1.96*se),2)]}


def odds_only_cutpoints(records):
 # Cutpoints computed only from 2023 raw odds, never labels or later outcomes.
 # This permits fine resolution where donor excess mass is actually concentrated.
 arr=defaultdict(list)
 for rec in records:
  vals,_,_=keys_for_race(rec)
  for key in ('donor_mass','outside_mass','outside_share'):
   if key=='donor_mass':
    arr[key].append(float(vals[key][0]))
   else:
    arr[key].extend(np.asarray(vals[key],dtype=float).tolist())
 ret={}
 for key,points in arr.items():
  choices=np.quantile(np.asarray(points,float),[.10,.25,.40,.55,.70,.85])
  uniq=sorted({round(float(x),7) for x in choices})
  if len(uniq)<3:
   # Strictly odds-only adaptive fallback. Empty bins allowed but no labels involved.
   uniq=sorted({round(float(x),7) for x in np.quantile(np.asarray(points,float),[.01,.20,.40,.60,.80,.99])})
  ret[key]=uniq
 return ret

def main():
 years,audit,labels=load()
 BINS.update(odds_only_cutpoints(years[2023]))
 print('2023_ONLY_ODDS_CUTS',{k:BINS[k] for k in ('donor_mass','outside_mass','outside_share')},flush=True)
 data={}
 profiles={}
 for year in range(2021,2027):
  profiles[str(year)]={'races':len(years[year]),'mean_donor_excess_mass':0.,'no_donor_races':0,
    'market_hit_in_donor':0,'market_hit_in_other':0,'target_coverage':0}
  for j,r in enumerate(years[year]):
   sv,ov,total=keys_for_race(r)
   z=profiles[str(year)];z['mean_donor_excess_mass']+=total
   z['no_donor_races']+=int(total==0)
   z['market_hit_in_donor']+=int(sv['market_gap'][r.y_set]<-.15)
   z['market_hit_in_other']+=int(sv['market_gap'][r.y_set]>=-.15)
   z['target_coverage']+=1
   accumulate(data,r,'3連複',DIMENSIONS,sv,year)
   accumulate(data,r,'3連単',DIMENSIONS+ORDER_EXTRA,ov,year)
   if (j+1)%2500==0: print('YEAR_PROGRESS',year,j+1,flush=True)
  z=profiles[str(year)]
  z['mean_donor_excess_mass']=round(z['mean_donor_excess_mass']/max(z['races'],1),5)
  print('YEAR_DONE',year,'races',z['races'],'source_mass',z['mean_donor_excess_mass'],flush=True)
 allsig=defaultdict(dict)
 for (pool,feat,year),m in data.items():
  levels=tuple(len(BINS[f])+1 for f in feat)
  for idx in np.flatnonzero(m[0]>0):
   t=tuple(np.unravel_index(idx,levels))
   signature='|'.join((pool,'+'.join(feat),','.join(map(str,t))))
   allsig[signature][str(year)]=metrics(m,idx)
 def annotate(sig, d):
  return {'signature':sig,
          '2023':d.get('2023'),'2024':d.get('2024'),
          '2025':d.get('2025'),'2026':d.get('2026')}
 viable=[]
 for sig,d in allsig.items():
  a=d.get('2023');b=d.get('2024')
  if a is None or b is None:continue
  if a['n']<400 or b['n']<400 or a['hits']<5 or b['hits']<5:continue
  if a['roi_pct']>95 and b['roi_pct']>95:
   x=annotate(sig,d)
   x['dev_return_pct']=round(100+ (a['net_yen']+b['net_yen'])/(a['n']+b['n']),3)
   x['dev_hits']=a['hits']+b['hits']
   viable.append(x)
 viable.sort(key=lambda s:(-s['dev_return_pct'],-s['dev_hits']))
 # Freeze top candidates based only on development 2023+24.
 good=[x for x in viable if all(x[str(y)] and x[str(y)]['roi_pct']>100 for y in (2025,2026))]
 convincing=[x for x in good if all(x[str(y)]['roi_approx95pct_ci'][0]>100 for y in (2025,2026))]
 # Also top reliable "underpriced" ratios regardless of ROI but normalized
 # by offered price. This is not an advantage unless ROI>100.
 negative=[]
 for sig,d in allsig.items():
  a=d.get('2023');b=d.get('2024')
  if not a or not b or a['n']<3000 or b['n']<3000:continue
  if a['hit_z']<-2 and b['hit_z']<-2:
   x=annotate(sig,d)
   x['dev_hit_z']=round(a['hit_z']+b['hit_z'],3)
   negative.append(x)
 negative.sort(key=lambda x:x['dev_hit_z'])
 # Directly answer user's remote source->destination mapping by keeping
 # fixed cross x donor pressure partitions with all four years.
 view={}
 for s,d in allsig.items():
  p,axes,code=s.split('|')
  if p=='3連複' and axes in ('market_gap+outside_share',
      'market_gap+outside_mass','market_gap+donor_mass'):
   if all(str(y) in d for y in (2023,2024,2025,2026)):
    view[s]=d
 report={
  'schema':'ODDS_ONLY_REMOTE_DONOR_RECEIVER_ADAPTIVE_V2',
  'training':'2023 and 2024, winners only as labels',
  'forward_periods':['2025','2026H1'],
  'only_odds_features':['3連単 full price distribution','3連複 full price distribution'],
  'no_line_or_rider_information':True,
  'predicted_source':'D_s=log(q3trifecta_aggregate(s)/qtrio(s))<-0.15, identified without observing result',
  'remote_destination':'source overprice excess qtrio(s)-qtrifecta(s) on sets disjoint from target',
  'donor_relation':'remote0 disjoint means no car numbers shared; no prior restriction on winner outcomes',
  'note':'This discovers conditional associations, not causal movement of probability from one ticket to another.',
  'not_using_other_5_pools':'other 5 market types remain encrypted; final odds not T-15',
  'audit':audit,
  'year':profiles,
  'cuts':BINS,
  'cutpoint_calibration':'quantiles from 2023 odds-only features; no outcomes, 2024/25/26 not used',
  'search':{'num_signatures':len(allsig),
      'eligible_2023_2024_positive':len(viable),
      'same_2025_2026_profit_positive':len(good),
      'both_forward_ci_lower_above_100':len(convincing),
      'development_negative_with_min_3000':len(negative)},
  'development_positive_top50':viable[:50],
  'forward_both_profitable_top50':good[:50],
  'forward_both_ci_lower_above100':convincing[:50],
  'development_overpriced_top40':negative[:40],
  'full_source_destination_matrix':view,
  'cautions':['Many correlated subgroups, raw CI does not adjust for multiple hypotheses.',
  '2025/2026 were previously seen in other exploratory work; this is not truly pristine forward test.',
  'A recipient with hit/market ratio >1 may still be negative ROI.',
  'Market normalized q has about 25% takeout; use 1/offered odds for actual break-even.'],
 }
 OUT.parent.mkdir(parents=True,exist_ok=True)
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('COMPLETE',report['search'],flush=True)
 print('BEST_DEV',[{'sig':x['signature'],'dev':x['dev_return_pct'],
  '2025':x['2025']['roi_pct'] if x['2025'] else None,
  '2026':x['2026']['roi_pct'] if x['2026'] else None} for x in viable[:12]],flush=True)
 print('BEST_FORWARD',[x['signature'] for x in good[:10]],flush=True)
if __name__=='__main__':main()

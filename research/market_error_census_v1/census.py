#!/usr/bin/env python3
"""Outcome-conditioned market error census: all available full trio and trifecta odds.

No riders, lineups, scores or favorite-winner narratives. For every ticket,
measure its actual hit, its market-implied expected hit, and the real 100-yen
return. Search price/rank/cross-market/nonlocal odds signature patterns across
2023-24, then apply the frozen candidate signatures to 2025 and 2026H1.
A positive backtest is NOT by itself evidence of a repeatable edge.
"""
from __future__ import annotations
import json, math, sys, itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'/'nonlocal_break_even_v1'))
from analyze import load
OUTPUT=ROOT/'data/audits/market_error_census_v1.json'
YEARS=(2023,2024,2025,2026)

# Predeclared, nonoptimized bins. No result data used to define these boundaries.
CUTS={
 'price': [5,10,20,50,100,300,1000],
 'rank': [1,3,6,12,24,48,96],
 'cross': [-.15,-.05,.05,.15],
 'remote0': [-.16,-.04,.04,.16],
 'remote1': [-.16,-.04,.04,.16],
 'overlap': [.5,1.5,2.5],
 'fave_mass': [.22,.35,.50,.65],
 'divergence': [.025,.05,.10,.20],
 'num_cars': [5.5,6.5,7.5,8.5],
 'order_rank': [1,2,3,4,5],
 'order_cond': [.02,.05,.10,.20,.40],
}
GROUPS=[
 ('price',),('rank',),('cross',),('remote0',),('remote1',),
 ('overlap',),('fave_mass',),('divergence',),('num_cars',),
 ('rank','cross'),('rank','remote0'),('rank','remote1'),
 ('price','cross'),('price','remote0'),('price','remote1'),
 ('overlap','cross'),('overlap','remote0'),('overlap','remote1'),
 ('divergence','cross'),('divergence','remote0'),
 ('fave_mass','remote0'),('fave_mass','cross'),
 ('remote0','remote1'),('cross','remote0'),('cross','remote1'),
 ('price','rank'),('price','overlap'),
 ('rank','cross','remote0'),('price','cross','remote0'),
 ('rank','remote0','remote1'),('overlap','cross','remote0'),
]
ORDER_ADDITIONAL=[
 ('order_rank',),('order_cond',),
 ('order_rank','cross'),('order_rank','remote0'),
 ('order_cond','remote0'),('order_cond','cross'),
 ('price','order_rank'),('rank','order_rank'),
 ('order_rank','remote0','cross'),
]
METRICS=('ticket_count','actual_hits','market_expected_hits','market_expected_hitsq',
 'realized_payout_units','market_expected_payout_units',
 'realized_payoutsq','ticketsq','payout_times_tickets')
IDX={x:i for i,x in enumerate(METRICS)}
def mass_rec(rec):
    n=len(rec.sets)
    qt=np.asarray(rec.qt,np.float64)
    q3=np.asarray(rec.q3,np.float64)
    price=np.asarray(rec.trio_odds,np.float64)
    rank=np.zeros(n,dtype=np.int32)
    rank[np.argsort(-qt)]=np.arange(1,n+1)
    fset=set(rec.sets[int(np.argmax(qt))])
    overlap=np.array([len(fset.intersection(t)) for t in rec.sets],np.float64)
    r0=np.asarray(rec.x[:,2],np.float64)/4
    r1=np.asarray(rec.x[:,3],np.float64)/4
    cross=np.log(np.maximum(q3,1e-30)/np.maximum(qt,1e-30))
    fm=float(np.sort(qt)[-3:].sum())
    divergence=float(np.abs(q3-qt).sum()/2)
    values={
      'price':price,'rank':rank,'cross':cross,
      'remote0':r0,'remote1':r1,'overlap':overlap,
      'fave_mass':np.full(n,fm),'divergence':np.full(n,divergence),
      'num_cars':np.full(n,len(set(itertools.chain.from_iterable(rec.sets)))),
    }
    # Each 3連単 ordered ticket belongs to an unordered set with a different
    # market probability/remote signature.
    order=np.asarray(rec.cond,np.float64)
    norder=len(rec.orders)
    o_rank=np.zeros(norder,dtype=np.int32)
    for sid in range(n):
        ix=np.flatnonzero(rec.ids==sid)
        o_rank[ix[np.argsort(-order[ix])]]=np.arange(1,len(ix)+1)
    ordered_values={k:np.asarray(v)[rec.ids] for k,v in values.items() if k not in ('price','rank')}
    ordered_values['price']=np.asarray(rec.tri3_odds,np.float64)
    qord=q3[rec.ids]*order
    ord_rank=np.zeros(norder,dtype=np.int32)
    ord_rank[np.argsort(-qord)]=np.arange(1,norder+1)
    ordered_values['rank']=ord_rank
    ordered_values['order_rank']=o_rank
    ordered_values['order_cond']=order
    return values,ordered_values

def categories(src):
    return {k:np.searchsorted(CUTS[k],v,side='left').astype(np.int32)
            for k,v in src.items()}

def label(feature, index):
    cut=CUTS[feature]
    if index==0:return f'{feature}<= {cut[0]}'
    if index==len(cut):return f'{feature}> {cut[-1]}'
    return f'{feature}({cut[index-1]}, {cut[index]}]'

def serialize_groups(counts,level,yr,grp):
    names=list(grp)
    levels=[len(CUTS[s])+1 for s in names]
    c=counts
    present=np.flatnonzero(c[0]>0)
    for z in present:
        idx=np.unravel_index(z,tuple(levels))
        idkey=';'.join(f'{n}:{x}' for n,x in zip(names,idx))
        vals={k:float(c[i,z]) for k,i in IDX.items()}
        yield (f'{level}|{"-".join(names)}|{idkey}',{'labels':[label(n,j) for n,j in zip(names,idx)],'levels':names,**vals})

def aggregate_one(r,pool,groups,stat,yr,audit):
    v_set,v_order=mass_rec(r)
    if pool=='3連複':
        odds=np.asarray(r.trio_odds,np.float64)
        probs=np.asarray(r.qt,np.float64)
        win=r.y_set
        vals=v_set
    else:
        odds=np.asarray(r.tri3_odds,np.float64)
        probs=np.asarray(r.q3[r.ids]*r.cond,np.float64)
        win=r.y_order
        vals=v_order
    m=len(odds)
    audit[f'{yr}_{pool}_races']+=1
    audit[f'{yr}_{pool}_tickets']+=m
    audit[f'{yr}_{pool}_overround_sum']+=float(np.sum(1/odds))
    audit[f'{yr}_{pool}_normalization_factor']+=float(1/np.sum(1/odds))
    audit[f'{yr}_{pool}_winner_odds_sum']+=float(odds[win])
    audit[f'{yr}_{pool}_favorite_hit']+=int(np.argmax(probs)==win)
    c=categories(vals)
    for features in groups:
        ranks=[len(CUTS[f])+1 for f in features]
        encoded=np.ravel_multi_index(tuple(c[f] for f in features),tuple(ranks))
        total=math.prod(ranks)
        key=(yr,pool,features)
        if key not in stat:
            stat[key]=np.zeros((len(METRICS),total),np.float64)
        z=stat[key]
        tickets=np.bincount(encoded,minlength=total)
        qgroup=np.bincount(encoded,weights=probs,minlength=total)
        expected_pay=np.bincount(encoded,weights=probs*odds,minlength=total)
        z[IDX['ticket_count']]+=tickets
        z[IDX['actual_hits'],int(encoded[win])]+=1
        z[IDX['market_expected_hits']]+=qgroup
        z[IDX['market_expected_hitsq']]+=qgroup*qgroup
        z[IDX['realized_payout_units'],int(encoded[win])]+=odds[win]
        z[IDX['market_expected_payout_units']]+=expected_pay
        z[IDX['realized_payoutsq'],int(encoded[win])]+=odds[win]*odds[win]
        z[IDX['ticketsq']]+=tickets*tickets
        z[IDX['payout_times_tickets'],int(encoded[win])]+=odds[win]*tickets[int(encoded[win])]

def metrics(s):
    n=s['ticket_count']
    if n<=0:return None
    h=s['actual_hits'];ex=s['market_expected_hits']
    pay=s['realized_payout_units']
    mexp=s['market_expected_payout_units']
    actual=pay/n
    expected=mexp/n
    # conservative race-level normal uncertainty; large payouts can still dominate
    residual2=s['realized_payoutsq'] - 2*actual*s['payout_times_tickets']+actual**2*s['ticketsq']
    se=math.sqrt(max(residual2,0))/n
    var_hit=max(ex-s['market_expected_hitsq'],1e-9)
    return {
        'tickets':int(n),'hits':int(h),'expected_hits':round(ex,2),
        'hit_surprise_z':round((h-ex)/math.sqrt(var_hit),3),
        'hit_to_market_ratio':round(h/ex,4) if ex else None,
        'actual_roi_pct':round(100*actual,3),
        'baseline_market_roi_pct':round(100*expected,3),
        'payout_units':round(pay,2),
        'profit_yen':round(100*(pay-n)),
        'roi_se_pct':round(100*se,3),
        'roi_approx95pct_ci':[round(100*(actual-1.96*se),2),round(100*(actual+1.96*se),2)],
    }

def main():
    records,audit,labels=load()
    stats={}
    audit=Counter(audit)
    for y in YEARS:
        for j,r in enumerate(records[y]):
            aggregate_one(r,'3連複',GROUPS,stats,y,audit)
            aggregate_one(r,'3連単',GROUPS+ORDER_ADDITIONAL,stats,y,audit)
            if (j+1)%2000==0:print('SCAN',y,j+1,'of',len(records[y]),flush=True)
    groupdata=defaultdict(dict)
    for (yr,pool,fs),matrix in stats.items():
        for key,s in serialize_groups(matrix,pool,yr,fs):
            groupdata[key][str(yr)]=metrics(s)
    signatures=[]
    base=[]
    for key,ys in groupdata.items():
        if all(z in ys for z in ('2023','2024')):
            a,b=ys['2023'],ys['2024']
            # Both development years must have enough exposures and hits.
            # Do not use 2025 or 2026 to select these signatures.
            if a['tickets']<500 or b['tickets']<500:continue
            if a['hits']<8 or b['hits']<8:continue
            # Keep strongest actual deviations; future years are locked tests.
            zscore=a['hit_surprise_z']+b['hit_surprise_z']
            weights=(a['tickets']+b['tickets'])
            net=(a['profit_yen']+b['profit_yen'])
            combined_roi=100+100*net/(100*weights)
            infos={'signature':key,'labels':next(iter([k for k in key.split('|')[2].split(';')])) if False else None,
                'development':{'2023':a,'2024':b,
                    'combined_actual_roi_pct':round(combined_roi,3),
                    'combined_hit_surprise_z':round(zscore,3)},
                'validation':{y:ys.get(y) for y in ('2025','2026')}}
            if a['actual_roi_pct']>95 and b['actual_roi_pct']>95:
                infos['type']='high_price_candidate'
                infos['pretest_rank_score']=round(zscore,4)
                signatures.append(infos)
            if a['hit_surprise_z'] < -1 and b['hit_surprise_z']< -1:
                infos['type']='overestimated_candidate'
                infos['pretest_rank_score']=round(-zscore,4)
                base.append(infos)
    # Rank and select entirely from 2023/2024 training/validation.
    signatures.sort(key=lambda z:(-z['pretest_rank_score'],-z['development']['combined_actual_roi_pct']))
    base.sort(key=lambda z:(-z['pretest_rank_score'],z['development']['combined_actual_roi_pct']))
    # Individual group comparisons in all years, plus all cross-market groups for reference
    display=[]
    for k,years in groupdata.items():
        if '|' not in k or '-' in k.split('|')[1]:continue
        if k.split('|')[1] not in ('price','rank','cross','remote0','remote1','overlap','order_rank'):continue
        if not all(y in years for y in ('2023','2024','2025','2026')):continue
        display.append({'signature':k,'years':years})
    display.sort(key=lambda z:z['signature'])
    stable_positive=[s for s in signatures if all(s['validation'].get(y) and s['validation'][y]['actual_roi_pct']>100 for y in ('2025','2026'))]
    stable_negative=[s for s in base if all(s['validation'].get(y) and s['validation'][y]['hit_surprise_z']<0 for y in ('2025','2026'))]
    confidence_positive=[s for s in stable_positive if all(s['validation'][y]['roi_approx95pct_ci'][0]>100 for y in ('2025','2026'))]
    report={
        'schema':'OUTCOME_CONDITIONED_ODDS_MARKET_ERROR_CENSUS_V1',
        'workflow':'search deviations 2023+2024 only, freeze signature, inspect 2025+2026 heldout',
        'inputs':'complete finalized 3連複 and 3連単 odds ONLY',
        'labels':'true result for backtest ONLY',
        'no_rider_line_score_racecard':True,
        'rows_by_year':{str(y):len(records[y]) for y in YEARS},
        'audit':dict(audit),'cuts_fixed_before_results':CUTS,
        'groups_explored':{
            'trio':len(GROUPS),'trifecta':len(GROUPS+ORDER_ADDITIONAL),
            'matched_group_signatures':len(groupdata),
            'eligible_positive_development_signatures':len(signatures),
            'eligible_negative_development_signatures':len(base),
            'positive_both_forward_periods':len(stable_positive),
            'negative_both_forward_periods':len(stable_negative),
            'positive_both_95pct_lower_ci_over_100':len(confidence_positive),
        },
        'definition':'Market wrong requires systematic observed vs expected candidate hit discrepancy; buying edge requires actual payout/stake > 1 out-of-time, confidence/profit robustness.',
        'cautions':['Multi-feature searches create multiple-testing bias; no causal claim.',
          'Historical finalized odds not obtainable before betting closes.',
          'Only 2 odds pools decrypted; seven pool encrypted universe not included.',
          'Overestimated result segments can be exclusion rules but do not automatically imply profitable bets elsewhere.',
          'Approx confidence is based on per-race independent payout contribution, not track/calendar cluster.'],
        'single_dimension_calibration':display,
        'top_40_2023_2024_high_price_signatures':signatures[:40],
        'top_40_2023_2024_overestimated_signatures':base[:40],
        'forward_stable_positive':stable_positive[:50],
        'forward_stable_negative':stable_negative[:50],
        'forward_significant_positive':confidence_positive[:50],
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SUCCESS',json.dumps(report['groups_explored']),flush=True)
    print('AUDIT_TOTAL',{k:round(v,2) for k,v in audit.items() if 'overround' in k},flush=True)
    print('TOP5_POSITIVE',[{'sig':s['signature'],'dev':s['development']['combined_actual_roi_pct'],
        'forward':{y:s['validation'][y]['actual_roi_pct'] if s['validation'][y] else None for y in ('2025','2026')}} for s in signatures[:5]],flush=True)

if __name__=='__main__':
    main()

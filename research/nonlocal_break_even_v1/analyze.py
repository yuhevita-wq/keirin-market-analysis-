#!/usr/bin/env python3
"""Non-local odds residuals -> full unordered/ordered probabilities -> executable ticket equations.
Only two archived odds books (3連複, 3連単) and independently stored finish labels.
No lineup, race card, rating, rider identity, prediction line, selection-size restriction.
"""
from __future__ import annotations
import gzip
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
ODDS=ROOT/'data/odds_only'
LABELS=ROOT/'data/results_only'
OUT=ROOT/'data/audits/nonlocal_break_even_v1.json'
COEFF=ROOT/'data/models/nonlocal_break_even_v1.json'
SEED=20261017
LAMS=(0.002,0.015,0.08,0.3)
THRESHOLDS=(1.0,1.03,1.05,1.10,1.20)
FEATS=('self_cross_market_log_ratio','market_popularity_temperature',
'disjoint_signed_cross_gap','one_shared_signed_cross_gap',
'disjoint_positive_gap','disjoint_negative_gap',
'one_shared_positive_gap','one_shared_negative_gap',
'disjoint_gap_x_local_ratio','one_shared_gap_x_local_ratio',
'disjoint_gap_x_popularity','disjoint_gap_x_one_shared_gap')
LOCAL=(0,1)
FULL=tuple(range(len(FEATS)))
MASKS={}

@dataclass
class Record:
    rid:str
    year:int
    x:np.ndarray
    qt:np.ndarray
    q3:np.ndarray
    sets:tuple
    orders:tuple
    ids:np.ndarray
    cond:np.ndarray
    trio_odds:np.ndarray
    tri3_odds:np.ndarray
    y_set:int
    y_order:int

def lookup():
    out={}
    counts=Counter()
    for y in range(2021,2027):
        p=LABELS/f'results_{y}.jsonl.gz'
        if not p.exists():continue
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for ln in f:
                row=json.loads(ln)
                rid=str(row['race_id'])
                v=tuple(int(z) for z in row['finish_order'])
                if len(v)!=3 or len(set(v))!=3:continue
                if rid in out and out[rid]!=v:raise RuntimeError('result conflict '+rid)
                out[rid]=v
                counts[str(y)]+=1
    return out,dict(counts)

def get_masks(sets):
    cars=tuple(sorted(set(itertools.chain.from_iterable(sets))))
    if cars not in MASKS:
        ss=tuple(itertools.combinations(cars,3))
        ix={c:i for i,c in enumerate(cars)}
        b=np.array([sum(1<<ix[z] for z in t) for t in ss])
        ov=np.array([[int(int(a&c).bit_count()) for c in b] for a in b],dtype=np.int8)
        MASKS[cars]=(ss,(ov==0).astype(np.float64),(ov==1).astype(np.float64))
    return MASKS[cars]

def parse(row,winner):
    odds=row['odds']
    if set(odds)!={'3連単','3連複'}:return None
    tr={tuple(int(z) for z in k.split('-')):float(v) for k,v in odds['3連単'].items()}
    trio={tuple(sorted(int(z) for z in k.split('='))):float(v) for k,v in odds['3連複'].items()}
    if any(not math.isfinite(z) or z<1 for z in itertools.chain(tr.values(),trio.values())):return None
    sets=tuple(sorted(trio))
    ordered=tuple(sorted(tr))
    cars=tuple(sorted({z for s in sets for z in s}))
    n=len(cars)
    if n<5 or n>9 or len(sets)!=math.comb(n,3) or len(ordered)!=n*(n-1)*(n-2):return None
    canonical,d0,d1=get_masks(sets)
    if sets!=canonical or tuple(sorted(ordered))!=tuple(itertools.permutations(cars,3)):return None
    wset=tuple(sorted(winner))
    if wset not in trio or winner not in tr:return None
    st={s:i for i,s in enumerate(sets)}
    ids=np.array([st[tuple(sorted(t))] for t in ordered],dtype=np.int16)
    inv3=np.array([1/tr[t] for t in ordered])
    invt=np.array([1/trio[t] for t in sets])
    q3o=inv3/inv3.sum()
    qt=invt/invt.sum()
    q3=np.bincount(ids,weights=q3o,minlength=len(sets))
    delta=q3-qt
    signed0=d0@delta/(0.03+d0@qt)
    signed1=d1@delta/(0.03+d1@qt)
    pos=np.maximum(delta,0)
    neg=np.maximum(-delta,0)
    p0=d0@pos/(0.03+d0@qt); n0=d0@neg/(0.03+d0@qt)
    p1=d1@pos/(0.03+d1@qt); n1=d1@neg/(0.03+d1@qt)
    logratio=np.clip(np.log(q3/qt),-2,2)
    popularity=np.clip((np.log(qt)+math.log(len(sets)))/3,-2,2)
    feats=np.column_stack((logratio,popularity,
        signed0,signed1,p0,n0,p1,n1,
        signed0*logratio,signed1*logratio,
        signed0*popularity,signed0*signed1))
    feats[:,2:]=np.clip(feats[:,2:]*4,-3,3)
    cond=q3o/q3[ids]
    return Record(str(row['race_id']),int(str(row['race_id'])[2:6]),
        feats.astype(np.float32),qt.astype(np.float32),q3.astype(np.float32),
        sets,ordered,ids,cond.astype(np.float32),
        np.array([trio[k] for k in sets],dtype=np.float32),
        np.array([tr[k] for k in ordered],dtype=np.float32),
        st[wset],ordered.index(winner))

def load():
    truth,counts=lookup()
    by_year=defaultdict(list)
    audit=Counter()
    for y in range(2021,2027):
        p=ODDS/f'odds_{y}.jsonl.gz'
        if not p.exists():continue
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for ln in f:
                audit['odds_rows']+=1
                row=json.loads(ln)
                w=truth.get(str(row['race_id']))
                if w is None:
                    audit['missing_result']+=1
                    continue
                race=parse(row,w)
                if race is None:
                    audit['invalid_book_or_result']+=1
                    continue
                by_year[y].append(race)
                audit['valid_joined']+=1
    return by_year,dict(audit),counts

def packed(rows):
    n=len(rows);maxs=max(len(r.sets) for r in rows)
    x=np.zeros((n,maxs,len(FEATS)),dtype=np.float32)
    logq=np.full((n,maxs),-10000,dtype=np.float32)
    idx=np.zeros(n,dtype=np.int32)
    for i,r in enumerate(rows):
        m=len(r.sets)
        x[i,:m]=r.x
        logq[i,:m]=np.log(r.qt)
        idx[i]=r.y_set
    return x,logq,idx

def modelprobs(x,logq,active,w):
    if len(active):
        z=logq+np.einsum('nkd,d->nk',x[:,:,active],w,optimize=True)
    else:z=logq
    z=z-np.max(z,axis=1,keepdims=True)
    p=np.exp(z)
    return p/p.sum(axis=1,keepdims=True)

def fit(rows,active,lam):
    x,logq,y=packed(rows);n=len(rows); ar=np.arange(n)
    if not active:return np.zeros(0,dtype=np.float32)
    def objective(w):
        p=modelprobs(x,logq,active,w.astype(np.float32))
        losses=-np.log(np.maximum(p[ar,y],1e-30))
        weights=p.copy();weights[ar,y]-=1
        grad=np.einsum('nk,nkd->d',weights,x[:,:,active],optimize=True)/n + lam*w
        return float(losses.mean()+lam/2*np.dot(w,w)),grad.astype(np.float64)
    sol=minimize(objective,np.zeros(len(active)),method='L-BFGS-B',jac=True,
          bounds=[(-5,5)]*len(active),options={'maxiter':90,'ftol':1e-9})
    if not math.isfinite(sol.fun):raise RuntimeError('fit failed')
    return np.asarray(sol.x,dtype=np.float32)

def set_loss(rows,active,w):
    x,l,y=packed(rows)
    p=modelprobs(x,l,active,w)
    return float(np.mean(-np.log(np.maximum(p[np.arange(len(y)),y],1e-30))))

def orderloss(records,k):
    s=0.
    for r in records:
        a=np.power(r.cond.astype(np.float64),k)
        z=np.bincount(r.ids,weights=a,minlength=len(r.sets))
        winner=r.y_order
        s-=math.log(max(float(a[winner]/z[r.ids[winner]]),1e-30))
    return s/len(records)

def psets(r,active,w):
    z=np.log(r.qt)+r.x[:,active]@w if len(active) else np.log(r.qt)
    z=z-np.max(z)
    p=np.exp(z);return p/p.sum()

def porder(r,p_set,k):
    a=np.power(r.cond,k)
    denom=np.bincount(r.ids,weights=a,minlength=len(r.sets))
    return p_set[r.ids]*a/denom[r.ids]

def tickets(rows,active,w,k,threshold,ticket_type):
    count=0;stake=0.;payout=0.;positive_races=0
    profit_by_race=[];stake_by_race=[]
    max_ev_before=0
    cheapest_min=1e99;largest_max=0.
    hits=0
    for r in rows:
        ps=psets(r,active,w)
        if ticket_type=='3連複':
            probs=ps;odds=r.trio_odds;win=r.y_set
        else:
            probs=porder(r,ps,k);odds=r.tri3_odds;win=r.y_order
        ev=probs*odds
        mask=(ev>threshold+1e-10)
        num=int(mask.sum())
        max_ev_before+=float(np.max(ev))
        count+=num;stake+=100*num
        if num:positive_races+=1
        payout_i=100*float(odds[win]) if mask[win] else 0.
        if mask[win]:hits+=1
        payout+=payout_i
        profit_by_race.append(payout_i-100*num)
        stake_by_race.append(100*num)
        cheapest_min=min(cheapest_min,float(np.min(ev)))
        largest_max=max(largest_max,float(np.max(ev)))
    roi=payout/stake if stake else None
    # Approx race-cluster sandwich CI for payout/stake ratio; blank when no tickets.
    if stake:
        residual=np.array(profit_by_race,dtype=np.float64) - ((payout-stake)/stake)*np.array(stake_by_race)
        se=float(np.std(residual,ddof=1)*math.sqrt(len(rows))/stake) if len(rows)>1 else None
        ci=[max(0,roi-1.96*se),roi+1.96*se] if se is not None else None
    else:ci=None
    return {'races':len(rows),'bet_races':positive_races,'ticket_count':count,
            'stake_yen':int(stake),'payout_yen':round(payout),
            'net_yen':round(payout-stake),'roi_pct':round(100*roi,4) if roi is not None else None,
            'roi_approx95pct_ci': [round(100*v,3) for v in ci] if ci else None,
            'hits':hits,'mean_max_model_EV':round(max_ev_before/len(rows),5),
            'maximum_model_EV':round(largest_max,4)}

def quality(rows,active,w,k):
    log_set=0.;log_order=0.;brier_set=0.
    for r in rows:
        ps=psets(r,active,w)
        po=porder(r,ps,k)
        log_set-=math.log(max(ps[r.y_set],1e-30))
        log_order-=math.log(max(po[r.y_order],1e-30))
        brier_set+=1-2*ps[r.y_set]+ps@ps
    n=len(rows)
    return {'sets_logloss':round(log_set/n,7),'orders_logloss':round(log_order/n,7),
            'sets_brier':round(brier_set/n,7)}

def main():
    by_year,audit,label_counts=load()
    print('AUDIT',audit,'YEARS',{y:len(v) for y,v in by_year.items()},flush=True)
    if len(by_year[2023])<1000 or len(by_year[2024])<1000:raise RuntimeError('insufficient odds/result joins')
    t2023=by_year[2023]
    tuned={}
    validation={}
    for name,active in [('local',LOCAL),('nonlocal',FULL)]:
        attempts=[]
        for lam in LAMS:
            w=fit(t2023,active,lam)
            loss=set_loss(by_year[2024],active,w)
            attempts.append({'lambda':lam,'logloss_2024':loss})
            print('VALIDATE',name,lam,loss,flush=True)
        chosen=min(attempts,key=lambda z:z['logloss_2024'])
        tuned[name]=chosen['lambda'];validation[name]=attempts
    kopt=minimize_scalar(lambda k:orderloss(t2023,k),bounds=(0.45,1.8),method='bounded',
                  options={'xatol':1e-4})
    k=float(kopt.x)
    print('ORDER_TEMPERATURE',k,flush=True)
    # 2024 is used only for lambda tuning, then training includes it, and 2025-26 frozen.
    dev=by_year[2023]+by_year[2024]
    models={
      'market':([],np.zeros(0,dtype=np.float32)),
      'local':(LOCAL,fit(dev,LOCAL,tuned['local'])),
      'nonlocal':(FULL,fit(dev,FULL,tuned['nonlocal']))
    }
    print('FITTED',{name:w.tolist() for name,(_,w) in models.items()},flush=True)
    # Keep order temperature trained on 2023: 2025-26 truly frozen within this experiment.
    report={'status':'REAL_BREAK_EVEN_TEST_NO_PROFIT_PROMISE',
        'scope':'odds-only male 2021 to 2026-H1; 3連複 and 3連単 complete final books',
        'training_2023_races':len(t2023),'validation_2024_races':len(by_year[2024]),
        'testing_2025_2026_races':len(by_year[2025])+len(by_year[2026]),
        'disallowed_inputs':['predicted lines','score','riders','race cards','actual results in inputs','payout in inputs'],
        'result_labels':['actual finish order only used for fitting/evaluation'],
        'market_probability':'q3set from 3連単 and qt from 3連複; normalized inverse odds',
        'target':'p(trio set) proportional qt(set) times exp(local and all nonlocal bias features)',
        'order':'p(order)=p(set)*cond(order)^kappa / sum_order_in_set cond^kappa',
        'nonlocal_definition':'for EACH target set aggregate all disjoint and one-shared source sets cross-market odds residuals; use signed and positive/negative components plus interactions',
        'ticket_solution':'for either ticket kind buy exactly those combinations with p_model(ticket)*final_odds(ticket)>threshold, no forced cap, abstain if empty',
        'break_even':'100 yen expected profit =100*(p*odds-1)',
        'risk_note':'one ticket 100 yen flat; all selected tickets purchased; no ex-post ticket pruning; interdependent outcomes make naive ticketwise Kelly unsafe',
        'price_note':'final archived odds were not available at T-15; real-money profitability not validated',
        'exposure_note':'2 markets only; other five betting pools archived separately as encrypted and not joined',
        'audit':audit,'label_counts':label_counts,'races_by_year':{str(y):len(v) for y,v in by_year.items()},
        'features':list(FEATS),'lambda_tuning_2024':validation,'chosen_lambdas':tuned,
        'conditional_order_temperature_2023':k,
        'coefficients':{name:{feat:float(v) for feat,v in zip((FEATS[i] for i in active),w)} for name,(active,w) in models.items()},
        'results':{}}
    for year in [2023,2024,2025,2026]:
        rows=by_year[year]
        if not rows:continue
        stats={'races':len(rows),'probability_score':{},'ticket_roi':{}}
        for name,(active,w) in models.items():
            stats['probability_score'][name]=quality(rows,active,w,k if name!='market' else 1.0)
            stats['ticket_roi'][name]={}
            for kind in ('3連複','3連単'):
                stats['ticket_roi'][name][kind]={
                    str(th):tickets(rows,active,w,k if name!='market' else 1.0,th,kind) for th in THRESHOLDS}
        report['results'][str(year)]=stats
        print('EVAL',year,'RACES',len(rows),'SET_LOSS',
              {a:z['sets_logloss'] for a,z in stats['probability_score'].items()},
              'ROI', {a:{k:{th:{field:v[field] for field in ('ticket_count','roi_pct','net_yen')}
                 for th,v in kinds[k].items()} for k in kinds}
                 for a,kinds in stats['ticket_roi'].items()},flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    COEFF.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    COEFF.write_text(json.dumps({'status':'RESEARCH_FINAL_ODDS_ONLY','coefficients':report['coefficients'],
        'kappa':k,'features':report['features'],'lambda':tuned,
        'not_t15_executable':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SAVED',OUT,COEFF,flush=True)

if __name__=='__main__':
    main()

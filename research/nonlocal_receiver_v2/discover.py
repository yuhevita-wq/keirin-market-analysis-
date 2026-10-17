#!/usr/bin/env python3
"""OUTCOME->ODDS residual discovery. Learn which odds combinations can and cannot win.
Exact-ID actual results are labels ONLY. No riders/lines/race cards.
2023 train; 2024 calibrate; freeze both for 2025 and 2026 H1.
Outputs 3 things: receiver residual anatomy, exclusion coverage, actual tickets and ROI.
"""
import json, math, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import lightgbm as lgb
from scipy.stats import entropy

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research'/'nonlocal_break_even_v1'))
from analyze import load, get_masks

AUDIT=ROOT/'data/audits/nonlocal_receiver_v2.json'
PARAM=ROOT/'data/models/nonlocal_receiver_v2.json'
THRESH=(0.9,0.95,1.0,1.03,1.07,1.12)
RISK=(0.0,0.001,0.005)
BLENDS=(0.,.2,.4,.6,.8,1.)
FEATURES=(
 'log_qtrio','log_q3aggregate','log_market_odds','own_cross_log_ratio',
 'own_q3_minus_qtrio','trio_popularity_rank','q3_popularity_rank','rank_disagreement',
 'remote_zero_signed','remote_one_signed','remote_zero_pos','remote_zero_neg',
 'remote_one_pos','remote_one_neg','interaction_remote_zero_one',
 'interaction_own_remote_zero','interaction_own_remote_one',
 'race_trio_entropy','race_three_concentration','race_market_disagreement',
 'per_set_conditional_order_entropy','race_number_cars')
def feats(r):
    qt=np.asarray(r.qt,np.float64)
    q3=np.asarray(r.q3,np.float64)
    n=len(qt)
    rk=np.empty(n);rk[np.argsort(-qt)]=np.arange(n)/max(n-1,1)
    r3=np.empty(n);r3[np.argsort(-q3)]=np.arange(n)/max(n-1,1)
    cond_ent=np.bincount(r.ids,weights=-r.cond*np.log(np.maximum(r.cond,1e-15)),minlength=n)
    x=r.x.astype(np.float64)
    qlog=np.log(np.maximum(qt,1e-20))
    values=np.column_stack((
      qlog,np.log(np.maximum(q3,1e-20)),np.log(r.trio_odds),
      x[:,0],q3-qt,rk,r3,rk-r3,
      x[:,2],x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],
      x[:,2]*x[:,3],x[:,0]*x[:,2],x[:,0]*x[:,3],
      np.full(n,entropy(qt)/math.log(n)),
      np.full(n,float(sum(sorted(qt,reverse=True)[:3]))),
      np.full(n,float(abs(q3-qt).sum()/2)),
      cond_ent,np.full(n,len(set(v for t in r.sets for v in t)))
    ))
    return np.asarray(values,np.float32)

def matrix(rs):
    mat=[];labels=[]
    for r in rs:
        f=feats(r)
        y=np.zeros(len(f),np.uint8)
        y[r.y_set]=1
        mat.append(f); labels.append(y)
    return np.vstack(mat),np.concatenate(labels)

def trained(rs):
    x,y=matrix(rs)
    print('FIT_ROWS',len(y),'WINNERS',int(y.sum()),flush=True)
    clf=lgb.LGBMClassifier(n_estimators=145,num_leaves=13,learning_rate=.055,
             min_child_samples=1000,reg_lambda=5.0,reg_alpha=.25,
             subsample=.85,subsample_freq=1,colsample_bytree=.95,
             random_state=2781,n_jobs=4,verbosity=-1)
    clf.fit(x,y,feature_name=list(FEATURES))
    return clf

def infer_records(clf,rs):
    # all candidate ticket scores from all odds only; labels not touched here
    x,_=matrix(rs)
    z=np.asarray(clf.predict_proba(x)[:,1],dtype=np.float64)
    preds=[];off=0
    for r in rs:
        raw=z[off:off+len(r.sets)]
        off+=len(r.sets)
        raw=np.maximum(raw,1e-12);raw/=raw.sum()
        preds.append(raw)
    return preds

def combine(r,pr,alpha):
    q=np.asarray(r.qt,np.float64)
    if alpha==0:return q
    z=(1-alpha)*np.log(q)+alpha*np.log(np.maximum(pr,1e-30))
    z-=z.max()
    p=np.exp(z)
    return p/p.sum()

def logloss(rs,preds,alpha):
    n=0;s=0
    for r,z in zip(rs,preds):
        p=combine(r,z,alpha)
        s-=math.log(max(float(p[r.y_set]),1e-30));n+=1
    return s/n

def measure_roi(rs,ps,market,threshold,risk=0):
    # *No arbitrary point count.* Each ticket stake fixed 100 yen.
    # Risk penalty acts on posterior probability of losing, not just EV.
    n_bets=0;n_races=0;hits=0;stake=0.;payment=0.;profits=[]
    top5=[];per_race=[]
    for r,p in zip(rs,ps):
        if market=='trio':
            prob=p;odds=r.trio_odds;winning=r.y_set
        else:
            prob=p[r.ids]*r.cond
            odds=r.tri3_odds;winning=r.y_order
        val=prob*odds
        # Risk-corrected expected surplus: p*(odds-1)- (1-p) - risk*sqrt(p*(1-p))*odds.
        rr=val-1-risk*odds*np.sqrt(np.maximum(prob*(1-prob),0.))
        selected=np.flatnonzero(rr>threshold-1)
        # stop automatic dominance by highly uncertain extreme odds:
        # selection above remains probabilistic, no manually capped number.
        count=len(selected)
        n_bets+=count
        stake_i=100*count
        payout_i=100*float(odds[winning]) if np.any(selected==winning) else 0
        n_races+=bool(count);hits+=bool(payout_i)
        stake+=stake_i;payment+=payout_i
        profits.append(payout_i-stake_i)
        per_race.append((stake_i,payout_i))
        top5.append(float(np.max(val)))
    roi=payment/stake if stake else None
    # cluster by race: ratio CI using influence function, not naive ticket Bernoulli
    lo=hi=None
    if stake>0 and len(rs)>1:
        m=np.array(per_race,dtype=float)
        mu=payment/stake
        v=(m[:,1]-mu*m[:,0])
        se=math.sqrt(len(rs))*float(np.std(v,ddof=1))/stake
        lo=mu-1.96*se;hi=mu+1.96*se
    return {'bet_races':n_races,'tickets':n_bets,'winners':hits,
       'stake_yen':round(stake),'payout_yen':round(payment),'profit_yen':round(payment-stake),
       'roi_pct':round(100*roi,3) if roi is not None else None,
       'roi_normal_approx_95pct_ci': [round(100*lo,2),round(100*hi,2)] if lo is not None else None,
       'mean_max_ev':round(float(np.mean(top5)),4)}
def losses_from_exclusions(rs,ps,retained_fractions=(.1,.25,.5,.75)):
    # How many realized outcomes are excluded when suppressing the lowest predicted
    # probability per race, and how much probability mass is retained?
    d={}
    for frac in retained_fractions:
        wins=0;baseline_mass=0;model_mass=0;total_kept=0
        for r,p in zip(rs,ps):
            m=max(1,math.ceil(len(p)*frac))
            ix=np.argsort(-p)[:m]
            wins+=int(r.y_set in ix)
            baseline_mass+=float(sum(r.qt[ix]))
            model_mass+=float(sum(p[ix]))
            total_kept+=m
        d[str(frac)]={'survivor_ticket_count':total_kept,
              'realized_winner_coverage_pct':round(100*wins/len(rs),3),
              'market_prob_covered_pct':round(100*baseline_mass/len(rs),3),
              'model_prob_covered_pct':round(100*model_mass/len(rs),3),
              'false_exclusion_races':len(rs)-wins}
    return d

def remote_anatomy(rs,groups):
    # 2023 odds-defined boundaries are used for subsequent outcome-conditioned audits;
    # all candidates counted rather than only selected tickets.
    out=defaultdict(lambda:{'actual':0,'expected':0.,'rows':0,'races':0})
    for r in rs:
        b=np.asarray(r.x[:,2],dtype=float)
        bucket=np.searchsorted(groups,b,side='right')
        q=np.asarray(r.qt,float)
        for i in range(5):
            ok=(bucket==i)
            if not np.any(ok):continue
            z=out[str(i+1)]
            z['actual']+=int(ok[r.y_set])
            z['expected']+=float(np.sum(q[ok]))
            z['rows']+=int(sum(ok))
            z['races']+=1
    return {k:{'actual':v['actual'],'expected':round(v['expected'],3),
                 'observed_vs_market':round(v['actual']/v['expected'],5) if v['expected'] else None,
                 'candidate_rows':v['rows']}
             for k,v in sorted(out.items())}

def main():
    by_year,audit,label_counts=load()
    for y in [2023,2024,2025,2026]:
        print('RACES',y,len(by_year[y]),flush=True)
    if any(len(by_year[y])<1000 for y in (2023,2024,2025,2026)):raise RuntimeError('bad data')
    train=by_year[2023];val=by_year[2024];test=by_year[2025]+by_year[2026]
    model=trained(train)
    pred_val=infer_records(model,val)
    tune=[{'alpha':a,'logloss':logloss(val,pred_val,a)} for a in BLENDS]
    alpha=min(tune,key=lambda d:d['logloss'])['alpha']
    print('TUNE_2024',tune,'CHOSEN',alpha,flush=True)
    # Freeze feature architecture, alpha, risk and selection; refit model 2023+2024 for 2025/26.
    final=trained(train+val)
    rs={y:by_year[y] for y in (2025,2026)}
    predictions={y:infer_records(final,items) for y,items in rs.items()}
    # 2023-derived odds-only quintiles, no labels
    cuts=np.quantile(np.concatenate([r.x[:,2] for r in train]), [.2,.4,.6,.8])
    result={'schema':'RESULT_TO_ODDS_DEVIATION_AND_SURVIVOR_v2',
       'dataset':'all non-girls completed markets 2021..2026-H1 3連単+3連複',
       'scope_note':'2 available pools; 7-pool archive is encrypted and not joined. These are FINAL odds only.',
       'audit':audit,'result_counts':label_counts,
       'inputs_only':['complete trifecta odds','complete trio odds'],'label_only':'true finishing order',
       'models':{'train_2023_n':len(train),'validate_2024_n':len(val),
                 'test_2025_n':len(rs[2025]),'test_2026_n':len(rs[2026]),
                 'features':list(FEATURES),'blend_lambda_2024':tune,
                 'chosen_blend_alpha':alpha,
                 'feature_importance_gain':{k:round(float(v),3) for k,v in zip(FEATURES,final.booster_.feature_importance(importance_type='gain'))}},
       'note_selection':'Risk uses posterior downside uncertainty; all scores calculated before reading result. No cap on ticket count.',
       'limitations':['2025/2026 previously viewed in other project analyses, not wholly untouched OOS.',
                      'EV from archived final odds not actionable T-15.',
                      'Risk coefficients are illustrative NOT ex-ante tuned; multiple thresholds examined.',
                      'Positive ROI in a small selected sample does not prove repeatable edge.'],
       'years':{}}
    for year,rows in rs.items():
        pr=predictions[year]
        ps=[combine(r,z,alpha) for r,z in zip(rows,pr)]
        print('TEST_MODEL',year,'alpha',alpha,
              'logloss',logloss(rows,pr,alpha),
              'market_logloss',logloss(rows,pr,0),flush=True)
        rr={'races':len(rows),'local_remote_actual_vs_expected':remote_anatomy(rows,cuts),
           'survivor_exclusion':losses_from_exclusions(rows,ps),
           'selection':{}}
        for typ in ('trio','trifecta'):
            rr['selection'][typ]={}
            for risk in RISK:
                rr['selection'][typ][str(risk)]={}
                for th in THRESH:
                    stats=measure_roi(rows,ps,typ,th,risk)
                    rr['selection'][typ][str(risk)][str(th)]=stats
                    print('ROI',year,typ,'risk',risk,'min_EV',th,
                          'bets',stats['tickets'],'roi',stats['roi_pct'],
                          'profit',stats['profit_yen'],flush=True)
        result['years'][str(year)]=rr
    AUDIT.parent.mkdir(parents=True,exist_ok=True)
    AUDIT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    PARAM.parent.mkdir(parents=True,exist_ok=True)
    final.booster_.save_model(str(PARAM.with_suffix('.txt')))
    PARAM.write_text(json.dumps({'feature_names':FEATURES,'alpha':alpha,
          'trained_on':'2023+2024 only','test_periods':['2025','2026H1'],
          'model_file':str(PARAM.with_suffix('.txt').relative_to(ROOT))},indent=2)+'\n')
    print('COMPLETE',AUDIT,PARAM,flush=True)

if __name__=='__main__':main()

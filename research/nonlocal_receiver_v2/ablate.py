#!/usr/bin/env python3
"""Ablation of nonlocal odds features, judged on actual heldout result calibration."""
import json, math, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from discover import load, matrix, feats, combine, logloss, measure_roi, BLENDS, FEATURES
from scipy.optimize import minimize_scalar

OUT=ROOT/'data/audits/nonlocal_receiver_v2_ablation.json'
LOCAL=tuple(list(range(8))+list(range(17,22)))
FULL=tuple(range(len(FEATURES)))
def train(rs,sel):
    X,Y=matrix(rs)
    cl=lgb.LGBMClassifier(n_estimators=145,num_leaves=13,learning_rate=.055,
             min_child_samples=1000,reg_lambda=5.0,reg_alpha=.25,
             subsample=.85,subsample_freq=1,colsample_bytree=.95,
             random_state=2781,n_jobs=4,verbosity=-1)
    cl.fit(X[:,sel],Y,feature_name=[FEATURES[i] for i in sel])
    return cl
def inference(cl,rs,sel):
    X,_=matrix(rs)
    scores=np.asarray(cl.predict_proba(X[:,sel])[:,1],np.float64)
    out=[];pos=0
    for r in rs:
        s=scores[pos:pos+len(r.qt)]
        pos+=len(s)
        s=np.maximum(s,1e-12);s/=s.sum()
        out.append(s)
    return out
def main():
    years,audit,labels=load()
    results={'scope':'Complete trio+trifecta final odds, real result labels only, chronological tests',
       'audit':audit,'compared':{},'years':{}}
    for label,sel in [('LOCAL',LOCAL),('WITH_REMOTE',FULL)]:
        m=train(years[2023],sel)
        val_preds=inference(m,years[2024],sel)
        trials=[{'alpha':a,'logloss':logloss(years[2024],val_preds,a)} for a in BLENDS]
        alpha=min(trials,key=lambda x:x['logloss'])['alpha']
        final=train(years[2023]+years[2024],sel)
        results['compared'][label]={'features':[FEATURES[i] for i in sel],
             'validation':trials,'chosen_alpha':alpha,'forward':{}}
        print('MODEL',label,'alpha',alpha,flush=True)
        for y in [2025,2026]:
            rs=years[y]
            preds=inference(final,rs,sel)
            p=[combine(r,z,alpha) for r,z in zip(rs,preds)]
            ll=logloss(rs,preds,alpha)
            ll_market=logloss(rs,preds,0)
            roi={kind:{str(th):measure_roi(rs,p,kind,th,0) for th in (.9,.95,1.,1.03)}
                  for kind in ('trio','trifecta')}
            results['compared'][label]['forward'][str(y)]={'logloss':ll,'market_logloss':ll_market,
                                                   'roi':roi}
            print('EVAL',label,y,'loss',ll,'market',ll_market, '3t0.9',roi['trifecta']['0.9']['roi_pct'],
                  'trio0.9',roi['trio']['0.9']['roi_pct'],flush=True)
    for y in ('2025','2026'):
        q=results['compared']
        diff=q['LOCAL']['forward'][y]['logloss']-q['WITH_REMOTE']['forward'][y]['logloss']
        results['years'][y]={'remote_incremental_logloss_gain':diff,
          'remote_improves_logloss':diff>0}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('SAVE',OUT,flush=True)
if __name__=='__main__':main()

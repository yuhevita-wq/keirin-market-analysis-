#!/usr/bin/env python3
"""Buy only after rejecting low predicted-probability outcomes, then risk-aware value check.
Tree model and blend alpha frozen from 2023-24; evaluate 2025, 2026 only.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from discover import load, matrix, combine, FEATURES
OUT=ROOT/'data/audits/nonlocal_receiver_v2_exclusion_purchase.json'

class Wrapper:
    def __init__(self,bst):self.bst=bst
    def predict_proba(self,X):
        z=np.asarray(self.bst.predict(X,num_threads=4),np.float64)
        return np.stack((1-z,z),axis=1)

def infer(bs,rows):
    X,Y=matrix(rows)
    z=bs.predict_proba(X)[:,1];result=[];off=0
    for r in rows:
        q=np.maximum(z[off:off+len(r.sets)],1e-12)
        off+=len(r.sets)
        result.append(q/q.sum())
    return result

def test(rows,ps,kind,keep_fraction,minimum_ev,risk):
    bought=hit=events=0
    stake=payout=0.
    for r,p in zip(rows,ps):
        m=max(1,math.ceil(len(r.sets)*keep_fraction))
        keep=np.zeros(len(r.sets),dtype=bool)
        keep[np.argsort(-p)[:m]]=True
        if kind=='trio':
            odds=np.asarray(r.trio_odds,float)
            pp=np.asarray(p,float)
            win=r.y_set
            allow=keep
        else:
            odds=np.asarray(r.tri3_odds,float)
            pp=np.asarray(p[r.ids]*r.cond,float)
            win=r.y_order
            allow=keep[r.ids]
        val=pp*odds
        score=val-risk*odds*np.sqrt(np.maximum(pp*(1-pp),0))
        mask=allow & (score > minimum_ev)
        n=int(mask.sum());bought+=n;events+=bool(n)
        stake+=100*n
        if mask[win]:
            hit+=1
            payout+=100*float(odds[win])
    return {'tickets':bought,'bet_races':events,'hits':hit,
            'stake_yen':round(stake),'payout_yen':round(payout),
            'net_yen':round(payout-stake),
            'roi_pct':round(100*payout/stake,3) if stake else None}

def main():
    by_year,audit,_=load()
    model=Wrapper(lgb.Booster(model_file=str(ROOT/'data/models/nonlocal_receiver_v2.txt')))
    params=json.loads((ROOT/'data/models/nonlocal_receiver_v2.json').read_text())
    alpha=float(params['alpha'])
    report={'experiment':'AFTER_ODDS_ONLY_REJECTION_BUYING_TEST','scope':'2025/2026-H1 evaluated with frozen 2023/24 model',
            'description':'remove lower relative-to-race true-probability candidates first, then choose value positives and risk-penalized portfolio without fixed ticket cap',
            'note':'Retaining 25/50/75% are sensitivity examples; if any wins, not untouched model-selection evidence',
            'years':{}}
    for year in [2025,2026]:
        rows=by_year[year]
        q=infer(model,rows)
        probs=[combine(r,p,alpha) for r,p in zip(rows,q)]
        report['years'][str(year)]={}
        for kind in ['trio','trifecta']:
            out={}
            for frac in [0.25,0.5,0.75,1.]:
                for th in [0.9,1.,1.03]:
                    for risk in [0.,0.005]:
                        key=f'keep={frac};ev>{th};risk={risk}'
                        out[key]=test(rows,probs,kind,frac,th,risk)
            report['years'][str(year)][kind]=out
            summary={k:v for k,v in out.items() if 'risk=0.0' in k and ('ev>1.0;' in k or 'ev>0.9;' in k)}
            print('EVAL',year,kind,json.dumps(summary),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('SAVED',OUT,flush=True)
if __name__=='__main__':main()

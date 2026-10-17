#!/usr/bin/env python3
"""Read one future race's full odds books and solve tickets from saved frozen model.

Example:
python research/nonlocal_break_even_v1/predict.py \
  --input /path/to/full_odds.json --threshold 1.03 --output /path/to/decision.json
The file must have race_id and odds with full 3連単 and 3連複 prices;
never accepts result labels, lineups, rider form or outcome information.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze as model

DEFAULT_MODEL=model.ROOT/'data/models/nonlocal_break_even_v1.json'

def solve(payload,parameters,threshold=1.0):
    if not isinstance(payload,dict) or not {'race_id','odds'}.issubset(payload):
        raise ValueError('Expected race_id and odds')
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or threshold<1:
        raise ValueError('Threshold must be >= 1')
    if set(payload['odds']) != {'3連単','3連複'}:
        raise ValueError('Exactly two complete price books are required: 3連単 and 3連複')
    # Winner=None is mandatory. No actual outcome, fake labels or riders are supplied.
    r=model.parse(payload,None)
    if r is None:
        raise ValueError('Odds book is invalid/incomplete or insufficient entrants')
    if parameters.get('status')!='RESEARCH_FINAL_ODDS_ONLY':
        raise ValueError('Unknown model status')
    if tuple(parameters.get('features',[]))!=tuple(model.FEATS):
        raise ValueError('Feature mismatch: cannot solve from this model')
    weights=np.array([parameters['coefficients']['nonlocal'][feat] for feat in model.FEATS],dtype=np.float32)
    kappa=float(parameters['kappa'])
    ps=model.psets(r,model.FULL,weights)
    po=model.porder(r,ps,kappa)
    answer={'race_id':r.rid,'model':'nonlocal_break_even_v1',
            'based_on':'2023-2024 archived final odds training; not confirmed T-15 executable',
            'threshold_product_p_times_odds':threshold,
            'no_point_limit':True,'result_was_not_read':True,'markets':{}}
    for kind,combinations,odds,probs in (
        ('3連複',r.sets,r.trio_odds,ps),('3連単',r.orders,r.tri3_odds,po)):
        sep='=' if kind=='3連複' else '-'
        candidates=[]
        for ticket,odd,p in zip(combinations,odds,probs):
            ev=float(odd*p)
            candidates.append({'ticket':sep.join(str(v) for v in ticket),
                               'odds':round(float(odd),5),
                               'probability':round(float(p),9),
                               'expected_return_ratio':round(ev,6),
                               'expected_profit_per_100yen':round(100*(ev-1),4)})
        candidates.sort(key=lambda a:(-a['expected_return_ratio'],a['ticket']))
        chosen=[v for v in candidates if v['expected_return_ratio']>threshold]
        answer['markets'][kind]={
            'decision':'BUY' if chosen else 'PASS',
            'best_theoretical_ticket':candidates[0] if candidates else None,
            'bet_count':len(chosen),
            'stake_yen_at_100_per_ticket':100*len(chosen),
            'buy_tickets':chosen}
    return answer

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--model',default=DEFAULT_MODEL,type=Path)
    p.add_argument('--threshold',type=float,default=1.0,
                   help='Buy when estimated p*odds exceeds this value (1.0=positive expected value)')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    payload=json.loads(a.input.read_text(encoding='utf-8'))
    params=json.loads(a.model.read_text(encoding='utf-8'))
    out=solve(payload,params,a.threshold)
    rendered=json.dumps(out,ensure_ascii=False,indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(rendered,encoding='utf-8')
    print(rendered,end='')

if __name__=='__main__':
    main()

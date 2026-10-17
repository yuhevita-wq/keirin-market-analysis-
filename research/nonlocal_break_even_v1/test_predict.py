#!/usr/bin/env python3
"""Smoke test: mathematical input to buy/pass output, with no result labels."""
import itertools
import json
from pathlib import Path
import numpy as np
from research.nonlocal_break_even_v1 import predict

cars=range(1,6)
tr={'-'.join(map(str,k)):10.0+((k[0]*3+k[1]*5+k[2]*7)%37)
    for k in itertools.permutations(cars,3)}
to={'='.join(map(str,k)):4.0+((k[0]*3+k[1]*5+k[2]*7)%17)
    for k in itertools.combinations(cars,3)}
payload={'race_id':'1120230101010001','odds':{'3連単':tr,'3連複':to}}
f=Path('data/models/nonlocal_break_even_v1.json')
params=json.loads(f.read_text(encoding='utf-8'))
out=predict.solve(payload,params,threshold=1.0)
assert out['result_was_not_read']
assert out['no_point_limit']
for kind in ('3連単','3連複'):
    market=out['markets'][kind]
    assert market['best_theoretical_ticket'] is not None
    assert market['bet_count']==len(market['buy_tickets'])
    assert market['stake_yen_at_100_per_ticket']==100*market['bet_count']
    assert all(t['expected_return_ratio']>1 for t in market['buy_tickets'])
print('PURE_ODDS_SOLVER_PASS',[(k,out['markets'][k]['bet_count']) for k in out['markets']])

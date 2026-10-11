"""Verify probability mass, full coverage, aggregation, EV and frozen examples."""
import csv
import json
from pathlib import Path
import numpy as np
from model import calculate, expected_value, fit, probabilities

ROOT = Path(__file__).parent
read = lambda name: json.loads((ROOT/name).read_text())
model = read('frozen_model.json')
cards = read('cards_20261011.json')
odds = {r['race']:r for r in read('odds_20261011.json')}
saved = {(int(r['race']),r['type'],r['code']):r
         for r in csv.DictReader((ROOT/'all_outcomes_20261011.csv').open())}
for card in cards:
    p,t = probabilities(card,model)
    assert np.isclose(p.sum(),1) and np.isclose(t.sum(),1)
    rows = calculate(card,model,odds[card['race']])
    assert len(rows) == 245 and len({(r['type'],r['code']) for r in rows}) == 245
    for r in rows:
        s = saved[card['race'],r['type'],r['code']]
        for key in ['prob','odds','expected_return_ratio','expected_net_yen','break_even_probability']:
            assert np.isclose(r[key],float(s[key]),rtol=1e-10,atol=1e-10), (r['code'],key)
    trifecta = [r for r in rows if r['type']=='3連単']
    for r in [r for r in rows if r['type']=='3連複']:
        members = set(r['code'].split('='))
        matching = [s for s in trifecta if set(s['code'].split('-')) == members]
        assert len(matching)==6 and np.isclose(sum(s['prob'] for s in matching),r['prob'])
assert expected_value(.1,10)['expected_net_yen']==0
assert np.isclose(expected_value(.1,12)['expected_return_ratio'],1.2)
training=read('training.json')
assert len(training)==model['training_count']==199
assert all(r['card']['date']<'2026-10-11' for r in training)
assert np.allclose(fit(training,model['ridge']),model['weights'],atol=1e-7,rtol=1e-7)
for race,code,price in [(8,'1-4-7',40.1),(11,'1-2-7',27.0)]:
    r=saved[race,'3連単',code]
    assert float(r['odds'])==price and 0<float(r['prob'])<1
history=read('snapshot_history.json')
for snapshot in history['snapshots']:
    assert len(snapshot['rows'])==245
    for values in snapshot['rows']:
        row=dict(zip(history['columns'],values))
        assert np.isclose(row['profit100'],100*(row['prob']*row['odds']-1))
print('PASS: 199-race refit; 6 races / 1470 outcomes; normalization, aggregation, EV, audit examples')

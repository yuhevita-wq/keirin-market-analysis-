from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from simulate_v8_1_f02_2024q1 import load, pi, pf, pl
from evaluate_v8_17_q2_oos import load_q2, ensure_q2_data
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_20_f21_initial_special_unified import build_v8_20_f21

ROOT = Path(__file__).resolve().parents[2]
Q1 = ROOT / 'data/2024/s_class_f1_all_parts/2024_q1'
Q2 = ROOT / 'data/2024/s_class_f1_all_parts/2024_q2'
STAKE = 100


def summary(rows):
    n = len(rows)
    h = sum(r['hit'] for r in rows)
    t = sum(r['tickets'] for r in rows)
    p = sum(r['payout'] for r in rows)
    s = t * STAKE
    return {
        'bet_races': n,
        'hits': h,
        'hit_rate_pct': 100*h/n if n else None,
        'tickets': t,
        'avg_tickets': t/n if n else None,
        'stake_yen': s,
        'payout_yen': p,
        'profit_yen': p-s,
        'roi_pct': 100*p/s if s else None,
    }


def load_votes(data_dir: Path):
    out = {'trio': {}, 'tf': {}}
    for market, fn in [('trio','trio_final_odds.csv'),('tf','trifecta_final_odds.csv')]:
        with (data_dir/fn).open('r', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if r.get('odds_status') != 'available':
                    continue
                v = pi(r.get('total_votes'))
                if v is not None and v > 0:
                    out[market].setdefault(r['race_id'], v)
    return out


def eligible(rid, r, trio, tf, pay):
    if r.get('meeting_grade') != 'F1' or not (r.get('race_type') or '').startswith('Ｓ級'):
        return False
    if pi(r.get('entry_count')) != 7 or len(trio.get(rid,{})) != 35 or len(tf.get(rid,{})) != 210:
        return False
    cars = sorted({v for c in trio[rid] for v in c})
    lines = pl(r.get('predicted_line_formation'))
    return len(cars)==7 and lines is not None and set(v for line in lines for v in line)==set(cars) and rid in pay and bool(pay[rid])


def evaluate(label, data, data_dir):
    races,trio,tf,pay = data
    votes = load_votes(data_dir)

    # Same track/date eligible S-class baseline. Exclude the target race itself when computing its heat premium.
    peer_ids = defaultdict(list)
    valid_ids = []
    for rid,r in races.items():
        if eligible(rid,r,trio,tf,pay):
            valid_ids.append(rid)
            peer_ids[(r.get('race_date'), r.get('track'))].append(rid)

    qual_current=[]; qual_hab=[]
    final_current=[]

    for rid in sorted(valid_ids, key=lambda x:(races[x].get('race_date',''),x)):
        r=races[rid]; rt=r.get('race_type') or ''; g=classify_race_type(rt)
        d=build_v8_20_f21(trio[rid],tf[rid],r.get('predicted_line_formation') or '',rt)
        if not d.get('buy'):
            continue
        wins=[t for t in d['tickets'] if t in pay[rid]]
        row={'race_id':rid,'race_date':r.get('race_date'),'track':r.get('track'),'race_type':rt,'hit':int(bool(wins)),'payout':sum(pay[rid][t] for t in wins),'tickets':int(d['ticket_count']),'formation':d.get('formation')}

        if g=='QUALIFYING':
            eg=d.get('entry_gate') or {}
            row['H_AB']=bool(eg.get('H_AB'))
            qual_current.append(row)
            if row['H_AB']:
                qual_hab.append(row)

        if g=='FINAL':
            peers=[x for x in peer_ids[(r.get('race_date'),r.get('track'))] if x!=rid]
            trio_peer=[votes['trio'].get(x) for x in peers if votes['trio'].get(x)]
            tf_peer=[votes['tf'].get(x) for x in peers if votes['tf'].get(x)]
            trio_v=votes['trio'].get(rid); tf_v=votes['tf'].get(rid)
            trio_med=statistics.median(trio_peer) if trio_peer else None
            tf_med=statistics.median(tf_peer) if tf_peer else None
            trio_heat=(trio_v/trio_med) if trio_v and trio_med else None
            tf_heat=(tf_v/tf_med) if tf_v and tf_med else None
            row.update({
                'trio_votes':trio_v,'tf_votes':tf_v,
                'trio_heat_vs_same_day_median':trio_heat,
                'tf_heat_vs_same_day_median':tf_heat,
                'trio_hot': bool(trio_heat is not None and trio_heat>=1.0),
                'tf_hot': bool(tf_heat is not None and tf_heat>=1.0),
                'both_hot': bool(trio_heat is not None and tf_heat is not None and trio_heat>=1.0 and tf_heat>=1.0),
                'heat_geomean': math.sqrt(trio_heat*tf_heat) if trio_heat and tf_heat else None,
                'peer_count':len(peers),
            })
            final_current.append(row)

    heat_states={}
    for name,pred in {
        'ALL_CURRENT_FINAL': lambda r: True,
        'BOTH_HOT_GE_1': lambda r: r['both_hot'],
        'TF_HOT_GE_1': lambda r: r['tf_hot'],
        'TRIO_HOT_GE_1': lambda r: r['trio_hot'],
        'NOT_BOTH_HOT': lambda r: not r['both_hot'],
    }.items():
        xs=[r for r in final_current if pred(r)]
        heat_states[name]=summary(xs)

    return {
        'dataset':label,
        'qualifying':{
            'current':summary(qual_current),
            'plus_H_AB':summary(qual_hab),
            'removed':summary([r for r in qual_current if not r['H_AB']]),
        },
        'final_heat_definition':{
            'trio_heat':'final trio total_votes / median trio total_votes of other eligible S-class races at same track/date',
            'tf_heat':'final trifecta total_votes / median trifecta total_votes of other eligible S-class races at same track/date',
            'hot_boundary':'1.0 = at least same-day peer median; semantic boundary, not fitted to outcome',
        },
        'final_heat_states':heat_states,
        'final_rows':final_current,
    }


def main():
    ensure_q2_data()
    q1=evaluate('2024Q1_DEVELOPMENT',load(),Q1)
    q2=evaluate('2024Q2_DEVELOPMENT_AFTER_OBSERVED',load_q2(),Q2)
    out={'analysis':'QUALIFYING_HAB_AND_FINAL_MARKET_HEAT','Q1':q1,'Q2':q2}
    print('QUAL_FINAL_HEAT_BEGIN')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    print('QUAL_FINAL_HEAT_END')

if __name__=='__main__':
    main()

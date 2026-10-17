from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from simulate_v8_1_f02_2024q1 import load, pi, pl
from evaluate_v8_17_q2_oos import load_q2, ensure_q2_data
from v8_11_f12_race_type_adaptive import classify_race_type
from v8_20_f21_initial_special_unified import build_v8_20_f21

ROOT=Path(__file__).resolve().parents[2]
Q1=ROOT/'data/2024/s_class_f1_all_parts/2024_q1'
Q2=ROOT/'data/2024/s_class_f1_all_parts/2024_q2'
STAKE=100


def summary(rows):
    n=len(rows);h=sum(r['hit'] for r in rows);t=sum(r['tickets'] for r in rows);p=sum(r['payout'] for r in rows);s=t*STAKE
    return {'bet_races':n,'hits':h,'hit_rate_pct':100*h/n if n else None,'tickets':t,'avg_tickets':t/n if n else None,'stake_yen':s,'payout_yen':p,'profit_yen':p-s,'roi_pct':100*p/s if s else None}


def load_votes(data_dir):
    out={'trio':{},'tf':{}}
    for market,fn in [('trio','trio_final_odds.csv'),('tf','trifecta_final_odds.csv')]:
        with (data_dir/fn).open('r',encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if r.get('odds_status')!='available': continue
                v=pi(r.get('total_votes'))
                if v and v>0: out[market].setdefault(r['race_id'],v)
    return out


def eligible(rid,r,trio,tf,pay):
    if r.get('meeting_grade')!='F1' or not (r.get('race_type') or '').startswith('Ｓ級'): return False
    if pi(r.get('entry_count'))!=7 or len(trio.get(rid,{}))!=35 or len(tf.get(rid,{}))!=210: return False
    cars=sorted({v for c in trio[rid] for v in c}); lines=pl(r.get('predicted_line_formation'))
    return len(cars)==7 and lines is not None and set(v for line in lines for v in line)==set(cars) and rid in pay and bool(pay[rid])


def evaluate(label,data,data_dir):
    races,trio,tf,pay=data; votes=load_votes(data_dir)
    day=defaultdict(list); valid=[]
    for rid,r in races.items():
        if eligible(rid,r,trio,tf,pay):
            valid.append(rid); day[(r.get('race_date'),r.get('track'))].append(rid)
    rows=[]
    for rid in valid:
        r=races[rid]
        if classify_race_type(r.get('race_type') or '')!='FINAL': continue
        d=build_v8_20_f21(trio[rid],tf[rid],r.get('predicted_line_formation') or '',r.get('race_type') or '')
        if not d.get('buy'): continue
        wins=[t for t in d['tickets'] if t in pay[rid]]
        peers=[x for x in day[(r.get('race_date'),r.get('track'))] if x!=rid]
        tv=votes['trio'].get(rid); fv=votes['tf'].get(rid)
        tpeer=[votes['trio'].get(x) for x in peers if votes['trio'].get(x)]
        fpeer=[votes['tf'].get(x) for x in peers if votes['tf'].get(x)]
        tsum=(tv or 0)+sum(tpeer); fsum=(fv or 0)+sum(fpeer)
        tmax=max(tpeer) if tpeer else None; fmax=max(fpeer) if fpeer else None
        row={
            'race_id':rid,'race_date':r.get('race_date'),'track':r.get('track'),'hit':int(bool(wins)),
            'payout':sum(pay[rid][t] for t in wins),'tickets':int(d['ticket_count']),'formation':d.get('formation'),
            'trio_votes':tv,'tf_votes':fv,
            'trio_vs_peer_max': tv/tmax if tv and tmax else None,
            'tf_vs_peer_max': fv/fmax if fv and fmax else None,
            'trio_day_share': tv/tsum if tv and tsum else None,
            'tf_day_share': fv/fsum if fv and fsum else None,
            'tf_heat_amplification_vs_trio': (fv/fmax)/(tv/tmax) if fv and fmax and tv and tmax else None,
        }
        rows.append(row)
    states={
        'ALL':lambda r:True,
        'BOTH_DOMINATE_PEER_MAX':lambda r:(r['trio_vs_peer_max'] or 0)>=1 and (r['tf_vs_peer_max'] or 0)>=1,
        'TF_DOMINATES_PEER_MAX':lambda r:(r['tf_vs_peer_max'] or 0)>=1,
        'TRIO_DOMINATES_PEER_MAX':lambda r:(r['trio_vs_peer_max'] or 0)>=1,
        'TF_DAY_MAJORITY':lambda r:(r['tf_day_share'] or 0)>=0.5,
        'TRIO_DAY_MAJORITY':lambda r:(r['trio_day_share'] or 0)>=0.5,
        'BOTH_DAY_MAJORITY':lambda r:(r['tf_day_share'] or 0)>=0.5 and (r['trio_day_share'] or 0)>=0.5,
        'TF_RELATIVE_HEAT_GT_TRIO':lambda r:(r['tf_heat_amplification_vs_trio'] or 0)>=1,
        'TF_RELATIVE_HEAT_LT_TRIO':lambda r:(r['tf_heat_amplification_vs_trio'] or 0)<1,
    }
    return {'dataset':label,'states':{k:summary([r for r in rows if pred(r)]) for k,pred in states.items()},'rows':rows}


def main():
    ensure_q2_data()
    out={'analysis':'FINAL_HEAT_DEPTH','Q1':evaluate('2024Q1',load(),Q1),'Q2':evaluate('2024Q2',load_q2(),Q2)}
    print('FINAL_HEAT_DEPTH_BEGIN');print(json.dumps(out,ensure_ascii=False,indent=2));print('FINAL_HEAT_DEPTH_END')

if __name__=='__main__':main()

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from .search_middle_race_filters_2023_validate_2024 import load_year
from .simulate_market_scenario_portfolio_2023_v1 import build_race

YEAR=2023
DATA=Path('data/2023/s_class_yosen')
OUT=Path('data/audits/middle_filter_hit_combined_2023.json')


def read_csv(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def load_market_odds():
    tri=defaultdict(list); trio=defaultdict(list)
    for r in read_csv(DATA/'trifecta_final_odds.csv'): tri[r['race_id']].append(r)
    for r in read_csv(DATA/'trio_final_odds.csv'): trio[r['race_id']].append(r)
    out={}
    for rid in set(tri)&set(trio):
        m=build_race(tri[rid],trio[rid])
        if m: out[rid]=m
    return out


def parse_middle_cars():
    import re
    out={}
    for r in read_csv('data/audits/market_scenario_portfolio_2023_v1_decisions.csv'):
        mids=[]
        for part in (r.get('tickets') or '').split(' | '):
            m=re.match(r'^middle:([0-9-]+)@',part.strip())
            if m:
                cars=tuple(sorted(int(x) for x in m.group(1).split('-')))
                mids.append(cars)
        if mids: out[r['race_id']]=sorted(set(mids))
    return out


def match(row, feature, op, threshold):
    v=float(row[feature]); t=float(threshold)
    return v<=t if op=='<=' else v>=t


def summarize(rows, market_odds, mids_by_race):
    rs=[]
    for r in rows:
        rid=r['race_id']; mids=mids_by_race.get(rid,[]); market=market_odds.get(rid,{})
        odds=[float(market[c]['odds']) for c in mids if c in market and float(market[c]['odds'])>1]
        if not odds: continue
        inv=sum(1/o for o in odds)
        combined=1/inv if inv>0 else None
        hit=1 if float(r['payout'])>0 else 0
        stake=float(r['stake']); payout=float(r['payout'])
        rs.append({'rid':rid,'hit':hit,'stake':stake,'payout':payout,'points':len(mids),'combined':combined})
    stake=sum(x['stake'] for x in rs); payout=sum(x['payout'] for x in rs)
    hit_races=sum(x['hit'] for x in rs)
    hit_stake=sum(x['stake'] for x in rs if x['hit'])
    hit_payout=sum(x['payout'] for x in rs if x['hit'])
    comb=[x['combined'] for x in rs if x['combined'] is not None]
    return {
        'races':len(rs),
        'hit_races':hit_races,
        'hit_rate_pct':100*hit_races/len(rs) if rs else 0.0,
        'tickets':sum(x['points'] for x in rs),
        'avg_points_per_race':sum(x['points'] for x in rs)/len(rs) if rs else 0.0,
        'median_points_per_race':median([x['points'] for x in rs]) if rs else None,
        'median_combined_odds':median(comb) if comb else None,
        'mean_combined_odds':sum(comb)/len(comb) if comb else None,
        'p25_combined_odds':sorted(comb)[int((len(comb)-1)*.25)] if comb else None,
        'p75_combined_odds':sorted(comb)[int((len(comb)-1)*.75)] if comb else None,
        'stake_yen':round(stake),
        'payout_yen':round(payout),
        'profit_yen':round(payout-stake),
        'roi_pct':100*payout/stake if stake else 0.0,
        'hit_race_stake_yen':round(hit_stake),
        'hit_race_payout_yen':round(hit_payout),
        'realized_hit_race_roi_pct':100*hit_payout/hit_stake if hit_stake else None,
    }


def main():
    rows=load_year(2023)
    market_odds=load_market_odds(); mids=parse_middle_cars()
    conditions=[
        {'name':'全中間','feature':None,'op':None,'threshold':None},
        {'name':'本線3連複オッズ<=3.3','feature':'main_odds','op':'<=','threshold':3.3},
        {'name':'中間delta最大>=0.0110519','feature':'middle_delta_max','op':'>=','threshold':0.011051904329873324},
        {'name':'中間delta最大>=0.0169712','feature':'middle_delta_max','op':'>=','threshold':0.016971156145507316},
        {'name':'本線consensus>=0.2333405','feature':'main_consensus','op':'>=','threshold':0.23334047120286375},
        {'name':'中間delta合計>=0.0251466','feature':'middle_delta_sum','op':'>=','threshold':0.0251466280060078},
    ]
    out={'status':'MIDDLE_FILTER_HIT_COMBINED_AUDIT_2023','year':2023,'years_read':[2023],
         'evaluation_year_2024_used':False,'evaluation_year_2025_used':False,'evaluation_year_2026_used':False,
         'combined_odds_definition':'For each race, 1/sum(1/final_trio_odds) across all selected middle 3連複 tickets in that race. This is the standard dutch-equivalent combined odds and is descriptive only; the simulated staking here remains 100 yen per selected middle ticket.',
         'conditions':[]}
    for c in conditions:
        subset=rows if c['feature'] is None else [r for r in rows if match(r,c['feature'],c['op'],c['threshold'])]
        out['conditions'].append({**c,'stats':summarize(subset,market_odds,mids)})
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

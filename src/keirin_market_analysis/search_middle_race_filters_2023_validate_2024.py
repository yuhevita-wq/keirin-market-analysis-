from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

from .simulate_market_scenario_portfolio_2023_v1 import build_race

YEARS=(2023,2024)
OUT=Path('data/audits/middle_race_filters_2023_validate_2024.json')


def read_csv(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def parse_tickets(row):
    out=[]
    for part in (row.get('tickets') or '').split(' | '):
        m=re.match(r'^(main|middle|hole):([0-9-]+)@([0-9.]+)x([0-9]+)u$',part.strip())
        if not m: continue
        kind,cars_s,odds_s,units_s=m.groups()
        cars=tuple(sorted(int(x) for x in cars_s.split('-')))
        out.append({'kind':kind,'cars':cars,'odds':float(odds_s),'units':int(units_s)})
    return out


def load_year(year):
    data=Path(f'data/{year}/s_class_yosen')
    decisions=read_csv(f'data/audits/market_scenario_portfolio_{year}_v1_decisions.csv')
    tri=defaultdict(list); trio=defaultdict(list)
    for r in read_csv(data/'trifecta_final_odds.csv'): tri[r['race_id']].append(r)
    for r in read_csv(data/'trio_final_odds.csv'): trio[r['race_id']].append(r)

    payout={}
    for p in read_csv(data/'payouts.csv'):
        if p.get('ticket_type')!='3連複' or p.get('status')!='paid': continue
        nums=tuple(sorted(int(x) for x in re.findall(r'\d+',p.get('combination',''))))
        if len(nums)!=3: continue
        try: py=int(float(p.get('payout_yen') or 0))
        except Exception: continue
        if py>0: payout[(p['race_id'],nums)]=py

    rows=[]
    for d in decisions:
        ts=parse_tickets(d)
        mids=[t for t in ts if t['kind']=='middle']
        mains=[t for t in ts if t['kind']=='main']
        if not mids or not mains: continue
        rid=d['race_id']
        market=build_race(tri.get(rid,[]),trio.get(rid,[]))
        if not market: continue
        main=mains[0]
        if main['cars'] not in market or any(t['cars'] not in market for t in mids): continue
        mvals=[market[t['cars']] for t in mids]
        allvals=list(market.values())
        trio_probs=[x['p_trio'] for x in allvals if x['p_trio']>0]
        tri_probs=[x['p_tri_set'] for x in allvals if x['p_tri_set']>0]
        def entropy(ps):
            s=sum(ps)
            if s<=0:return 0.0
            qs=[x/s for x in ps]
            return -sum(q*math.log(q) for q in qs if q>0)
        mainv=market[main['cars']]
        mid_odds=[t['odds'] for t in mids]
        deltas=[x['delta'] for x in mvals]
        ratios=[x['ratio'] for x in mvals if x['ratio'] is not None]
        mid_cons=[x['consensus'] for x in mvals]
        total_cons=sum(x['consensus'] for x in allvals)
        stake=100*len(mids)
        pay=sum(payout.get((rid,t['cars']),0) for t in mids)
        rows.append({
            'race_id':rid,
            'middle_count':len(mids),
            'main_odds':main['odds'],
            'middle_odds_min':min(mid_odds),
            'middle_odds_median':median(mid_odds),
            'middle_odds_max':max(mid_odds),
            'middle_delta_sum':sum(deltas),
            'middle_delta_mean':sum(deltas)/len(deltas),
            'middle_delta_max':max(deltas),
            'middle_ratio_mean':sum(ratios)/len(ratios) if ratios else 0.0,
            'middle_ratio_max':max(ratios) if ratios else 0.0,
            'middle_consensus_sum':sum(mid_cons),
            'main_consensus':mainv['consensus'],
            'main_p_trio':mainv['p_trio'],
            'main_p_tri_set':mainv['p_tri_set'],
            'main_abs_delta':abs(mainv['delta']),
            'main_to_mid_consensus_ratio':mainv['consensus']/(max(mid_cons) or 1e-12),
            'middle_tri_share':sum(x['p_tri_set'] for x in mvals),
            'middle_trio_share':sum(x['p_trio'] for x in mvals),
            'middle_cross_market_shift':sum(x['p_tri_set']-x['p_trio'] for x in mvals),
            'trio_entropy':entropy(trio_probs),
            'trifecta_set_entropy':entropy(tri_probs),
            'consensus_top1_share':mainv['consensus']/total_cons if total_cons>0 else 0.0,
            'stake':stake,
            'payout':pay,
        })
    return rows


def stats(rows):
    stake=sum(r['stake'] for r in rows); pay=sum(r['payout'] for r in rows)
    return {
        'races':len(rows),'tickets':sum(r['middle_count'] for r in rows),
        'stake_yen':stake,'payout_yen':pay,'profit_yen':pay-stake,
        'roi_pct':100*pay/stake if stake else 0.0,
    }


def q(vals,p):
    xs=sorted(vals)
    if not xs:return None
    pos=(len(xs)-1)*p; lo=int(pos); hi=min(lo+1,len(xs)-1); f=pos-lo
    return xs[lo]+(xs[hi]-xs[lo])*f


def match(row,c):
    v=float(row[c['feature']]); t=float(c['threshold'])
    return v<=t if c['op']=='<=' else v>=t


def main():
    rows23=load_year(2023); rows24=load_year(2024)
    features=[
        'middle_count','main_odds','middle_odds_min','middle_odds_median','middle_odds_max',
        'middle_delta_sum','middle_delta_mean','middle_delta_max','middle_ratio_mean','middle_ratio_max',
        'middle_consensus_sum','main_consensus','main_abs_delta','main_to_mid_consensus_ratio',
        'middle_tri_share','middle_trio_share','middle_cross_market_shift','trio_entropy',
        'trifecta_set_entropy','consensus_top1_share'
    ]
    candidates=[]
    for f in features:
        vals=[float(r[f]) for r in rows23]
        thresholds=sorted(set(q(vals,p) for p in (.25,.50,.75)))
        for t in thresholds:
            for op in ('<=','>='):
                c={'feature':f,'op':op,'threshold':t}
                s23=[r for r in rows23 if match(r,c)]
                if len(s23)<100: continue
                s24=[r for r in rows24 if match(r,c)]
                candidates.append({**c,'train_2023':stats(s23),'validation_2024':stats(s24)})
    # Add simple categorical race-count cuts because middle_count is discrete and central to the user's hypothesis.
    for op,t in [('<=',1),('<=',2),('>=',2),('>=',3)]:
        c={'feature':'middle_count','op':op,'threshold':t}
        s23=[r for r in rows23 if match(r,c)]
        s24=[r for r in rows24 if match(r,c)]
        if len(s23)>=100:
            candidates.append({**c,'train_2023':stats(s23),'validation_2024':stats(s24)})

    # Deduplicate identical conditions.
    uniq={}
    for c in candidates:
        k=(c['feature'],c['op'],round(float(c['threshold']),12))
        uniq[k]=c
    candidates=list(uniq.values())
    candidates.sort(key=lambda c:(-c['train_2023']['roi_pct'],-c['train_2023']['races']))
    top=candidates[:20]
    out={
        'status':'MIDDLE_RACE_FILTER_SEARCH_2023_VALIDATE_2024',
        'years_read':[2023,2024],
        'evaluation_year_2025_used':False,
        'evaluation_year_2026_used':False,
        'principle':'Race-selection features are pre-race market-only. 2023 discovers simple one-feature thresholds; the exact thresholds are applied unchanged to 2024. Within selected races all selected middle 3連複 tickets are bought at 100 yen each, so this tests buy-race vs skip-race, not ticket pruning or dutching.',
        'baseline':{'2023':stats(rows23),'2024':stats(rows24)},
        'candidate_count':len(candidates),
        'top20_by_2023_roi_with_frozen_2024':top,
        'all_candidates':candidates,
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'baseline':out['baseline'],'top20':top},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .analyze_odds_contradictions_2023_2025 import (
    YEARS, load_results, load_odds, normalize_market, safe_log_ratio
)

OUT = Path('data/audits/odds_contradiction_rank_control_2023_2025.json')


def new_bet():
    return {'bets':0,'wins':0,'expected_trio':0.0,'expected_trifecta':0.0,'gross_odds_return':0.0}


def finish_bet(a):
    return {**a,
        'hit_rate_pct': a['wins']/a['bets']*100 if a['bets'] else None,
        'roi_pct_final_odds': a['gross_odds_return']/a['bets']*100 if a['bets'] else None,
        'observed_to_trio_expected': a['wins']/a['expected_trio'] if a['expected_trio'] else None,
        'observed_to_trifecta_expected': a['wins']/a['expected_trifecta'] if a['expected_trifecta'] else None,
    }


def add_bet(a,r):
    a['bets'] += 1; a['wins'] += int(r['win'])
    a['expected_trio'] += r['p_trio']; a['expected_trifecta'] += r['p_tri3']
    if r['win']: a['gross_odds_return'] += r['odds']


def new_car():
    return {'cars':0,'top3':0,'expected_trio':0.0,'expected_trifecta':0.0}


def finish_car(a):
    return {**a,
        'top3_rate_pct': a['top3']/a['cars']*100 if a['cars'] else None,
        'observed_to_trio_expected': a['top3']/a['expected_trio'] if a['expected_trio'] else None,
        'observed_to_trifecta_expected': a['top3']/a['expected_trifecta'] if a['expected_trifecta'] else None,
    }


def add_car(a,r):
    a['cars'] += 1; a['top3'] += int(r['top3'])
    a['expected_trio'] += r['p_trio']; a['expected_trifecta'] += r['p_tri3']


def quintile_by_fixed_rank(records_by_rank, kind='bet'):
    ag = defaultdict(new_bet if kind=='bet' else new_car)
    for rank, rows in records_by_rank.items():
        rows = sorted(rows, key=lambda r:(r['metric'], r.get('race_id',''), r.get('combo',''), r.get('car',0)))
        n=len(rows)
        for i,r in enumerate(rows):
            q=min(4,(i*5)//n)
            if kind=='bet': add_bet(ag[q],r)
            else: add_car(ag[q],r)
    finish = finish_bet if kind=='bet' else finish_car
    return {str(q):finish(ag[q]) for q in range(5)}


def analyze_year(year):
    base=Path(f'data/{year}/s_class_yosen')
    results=load_results(base)
    trio=load_odds(base,'trio_final_odds.csv','=')
    tri3=load_odds(base,'trifecta_final_odds.csv','-')
    d_by_rank=defaultdict(list); i_by_rank=defaultdict(list)
    eligible=0
    for rid in sorted(set(results)&set(trio)&set(tri3)):
        wo=results[rid]; ws=tuple(sorted(wo)); to=trio[rid]; oo=tri3[rid]
        cars=sorted({c for s in to for c in s})
        if len(cars)<5 or len(to)!=(len(cars)*(len(cars)-1)*(len(cars)-2)//6) or len(oo)!=(len(cars)*(len(cars)-1)*(len(cars)-2)):
            continue
        qt=normalize_market(to); qo=normalize_market(oo)
        if len(qt)!=len(to) or len(qo)!=len(oo): continue
        eligible+=1
        qset={s:0.0 for s in qt}
        for o,p in qo.items(): qset[tuple(sorted(o))]+=p
        d={s:safe_log_ratio(qset[s],qt[s]) for s in qt}
        ranked_sets=sorted(qt,key=lambda s:(-qt[s],s))
        for rank,s in enumerate(ranked_sets,1):
            d_by_rank[rank].append({'race_id':rid,'combo':'='.join(map(str,s)),'metric':d[s],
                'win':s==ws,'odds':to[s],'p_trio':qt[s],'p_tri3':qset[s]})
        mt={c:0.0 for c in cars}; mo={c:0.0 for c in cars}
        for s,p in qt.items():
            for c in s: mt[c]+=p
        for o,p in qo.items():
            for c in o: mo[c]+=p
        iv={c:safe_log_ratio(mo[c],mt[c]) for c in cars}
        ranked_cars=sorted(cars,key=lambda c:(-mt[c],c))
        for rank,c in enumerate(ranked_cars,1):
            i_by_rank[rank].append({'race_id':rid,'car':c,'metric':iv[c],'top3':c in ws,'p_trio':mt[c],'p_tri3':mo[c]})
    return {'year':year,'eligible_complete_races':eligible,
        'D_within_same_trio_popularity_rank_quintiles':quintile_by_fixed_rank(d_by_rank,'bet'),
        'I_within_same_trio_rider_rank_quintiles':quintile_by_fixed_rank(i_by_rank,'car')}


def combine(years,key,kind):
    out={}
    for q in range(5):
        a=new_bet() if kind=='bet' else new_car()
        for y in years:
            x=y[key][str(q)]
            if kind=='bet':
                for k in ('bets','wins'): a[k]+=x[k]
                for k in ('expected_trio','expected_trifecta','gross_odds_return'): a[k]+=x[k]
            else:
                for k in ('cars','top3'): a[k]+=x[k]
                for k in ('expected_trio','expected_trifecta'): a[k]+=x[k]
        out[str(q)]=(finish_bet(a) if kind=='bet' else finish_car(a))
    return out


def main():
    years=[analyze_year(y) for y in YEARS]
    full={
        'status':'POPULARITY_CONTROLLED_ODDS_CONTRADICTION_2023_2025',
        'years_read':list(YEARS),'evaluation_year_2026_used':False,'odds_phase':'final',
        'important_limit':'Archived final odds only. Market popularity control uses no result labels to form quintiles.',
        'method':{
            'D_control':'For every exact 3連複 market rank separately (1st favorite, 2nd favorite, ...), sort races by D and split into five equal groups. Then pool equal-D quintiles across ranks. q0=lowest D among tickets with the same trio popularity rank; q4=highest D.',
            'I_control':'For every rider trio-inclusion popularity rank separately (strongest rider, second, ...), sort races by I and split into five equal groups. q4 means trifecta market is unusually stronger than trio market even among riders with the same trio popularity rank.'
        },
        'years':years,
        'combined':{
            'D_within_same_trio_popularity_rank_quintiles':combine(years,'D_within_same_trio_popularity_rank_quintiles','bet'),
            'I_within_same_trio_rider_rank_quintiles':combine(years,'I_within_same_trio_rider_rank_quintiles','car')
        }
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(full,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(full,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

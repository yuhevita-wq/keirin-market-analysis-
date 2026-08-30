from __future__ import annotations

import csv, json, math, re
from collections import defaultdict
from pathlib import Path
from statistics import median

from .simulate_market_scenario_portfolio_2023_v1 import build_race

YEARS=(2023,2024)
OUT=Path('data/audits/fake_favorite_true_middle_2023_2024.json')

FAV_CONS_MAX=0.1884985310418076
SECOND_RATIO_MAX=1.2058823529411764


def read_csv(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def norm3(s):
    xs=tuple(sorted(int(x) for x in re.findall(r'\d+',s or '')))
    return xs if len(xs)==3 and len(set(xs))==3 else None


def entropy(ps):
    s=sum(x for x in ps if x>0)
    if s<=0:return 0.0
    qs=[x/s for x in ps if x>0]
    return -sum(q*math.log(q) for q in qs)


def load_year(year):
    data=Path(f'data/{year}/s_class_yosen')
    tri=defaultdict(list); trio=defaultdict(list)
    for r in read_csv(data/'trifecta_final_odds.csv'): tri[r['race_id']].append(r)
    for r in read_csv(data/'trio_final_odds.csv'): trio[r['race_id']].append(r)
    payouts={}
    for p in read_csv(data/'payouts.csv'):
        if p.get('ticket_type')!='3連複' or p.get('status')!='paid': continue
        c=norm3(p.get('combination',''))
        if not c: continue
        try: py=int(float(p.get('payout_yen') or 0))
        except Exception: continue
        if py>0: payouts[(p['race_id'],c)]=py

    races=[]
    all_candidate_counts=[]
    selected_counts=[]
    for rid, trs in sorted(trio.items()):
        market=build_race(tri.get(rid,[]),trs)
        if not market or len(market)!=35: continue
        ordered=sorted(market,key=lambda c:(market[c]['odds'],c))
        fav, second=ordered[0], ordered[1]
        f=market[fav]
        second_ratio=market[second]['odds']/f['odds']
        fake = f['consensus']<=FAV_CONS_MAX and second_ratio<=SECOND_RATIO_MAX
        if not fake: continue

        favset=set(fav)
        allcars=sorted({x for c in market for x in c})
        outsiders=[x for x in allcars if x not in favset]
        raw=[]; all3=[]
        for outsider in outsiders:
            fam=[]
            for dropped in fav:
                c=tuple(sorted((favset-{dropped})|{outsider}))
                if c in market: fam.append(c)
            pos=[c for c in fam if market[c]['delta']>0]
            if len(pos)>=2:
                rep=max(pos,key=lambda c:(market[c]['delta'],market[c]['p_tri_set'],c))
                raw.append(rep)
            if len(pos)==3:
                rep=max(pos,key=lambda c:(market[c]['delta'],market[c]['p_tri_set'],c))
                all3.append(rep)
        # de-dup representatives across outsider scenarios just in case
        raw=list(dict.fromkeys(raw)); all3=list(dict.fromkeys(all3))
        all_candidate_counts.append(len(raw)); selected_counts.append(len(all3))
        winner_hits=[]
        pay=0
        for c in all3:
            p=payouts.get((rid,c),0)
            if p>0:
                winner_hits.append(c); pay+=p
        races.append({
            'race_id':rid,'favorite':fav,'favorite_odds':f['odds'],'favorite_consensus':f['consensus'],
            'second_to_fav_odds_ratio':second_ratio,'raw_middle_count':len(raw),'all3_middle_count':len(all3),
            'all3_middle': ['-'.join(map(str,c)) for c in all3], 'hit': int(bool(winner_hits)),
            'payout_yen':pay,
        })
    return races


def summarize(races):
    bet=[r for r in races if r['all3_middle_count']>0]
    tickets=sum(r['all3_middle_count'] for r in bet)
    stake=tickets*100
    payout=sum(r['payout_yen'] for r in bet)
    hitr=sum(r['hit'] for r in bet)
    counts=[r['all3_middle_count'] for r in bet]
    raw=[r['raw_middle_count'] for r in races]
    return {
        'fake_favorite_races':len(races),
        'bet_races':len(bet),
        'no_all3_middle_races':len(races)-len(bet),
        'tickets':tickets,
        'avg_tickets_per_bet_race':tickets/len(bet) if bet else 0,
        'median_tickets_per_bet_race':median(counts) if counts else 0,
        'max_tickets_in_race':max(counts) if counts else 0,
        'avg_raw_2of3_middle_scenarios_per_fake_race':sum(raw)/len(raw) if raw else 0,
        'hit_races':hitr,
        'race_hit_rate_pct':100*hitr/len(bet) if bet else 0,
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,
        'roi_pct':100*payout/stake if stake else 0,
        'count_distribution':{str(k):sum(1 for x in counts if x==k) for k in sorted(set(counts))},
    }


def main():
    out={'status':'TRUE_FAKE_FAVORITE_MIDDLE_AUDIT_2023_2024','years_read':[2023,2024],
         'evaluation_year_2025_used':False,'evaluation_year_2026_used':False,
         'definition':'Favorite is shortest final 3連複 odds. Fake favorite gate is frozen F1 AND F2. Middle is rebuilt directly around that exact favorite: for each outsider, inspect all 3 one-car replacements; 3/3 means all three have positive delta=P_trifecta_set-P_trio. Buy strongest-delta representative for every qualifying outsider scenario at flat 100 yen. No old v1 dutching/admission filter is used.',
         'fake_gate':{'fav_consensus_max':FAV_CONS_MAX,'second_to_fav_odds_ratio_max':SECOND_RATIO_MAX},
         'years':{}}
    for y in YEARS:
        races=load_year(y)
        out['years'][str(y)]={'summary':summarize(races),'races':races}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({y:out['years'][str(y)]['summary'] for y in YEARS},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

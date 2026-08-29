from __future__ import annotations

import csv
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

YEAR = 2023
DATA = Path('data/2023/s_class_yosen')
DECISIONS = Path('data/audits/market_scenario_portfolio_2023_v1_decisions.csv')
OUT = Path('data/audits/middle_cross_ticket_simulation_2023.json')
DETAIL = Path('data/audits/middle_cross_ticket_simulation_2023_decisions.csv')
TYPES = ('3連複','ワイド','2車複','2車単','3連単')


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def nums(s: str):
    return tuple(int(x) for x in re.findall(r'\d+', s or ''))


def payout_key(ticket_type: str, combination: str):
    xs = nums(combination)
    if ticket_type in {'ワイド','2車複'} and len(xs) == 2:
        return tuple(sorted(xs))
    if ticket_type == '2車単' and len(xs) == 2:
        return xs
    if ticket_type == '3連複' and len(xs) == 3:
        return tuple(sorted(xs))
    if ticket_type == '3連単' and len(xs) == 3:
        return xs
    return None


def middle_sets(row):
    out = []
    for part in (row.get('tickets') or '').split(' | '):
        if not part.startswith('middle:'):
            continue
        body = part.split(':',1)[1].split('@',1)[0]
        xs = tuple(sorted(int(x) for x in body.split('-')))
        if len(xs) == 3 and len(set(xs)) == 3:
            out.append(xs)
    return sorted(set(out))


def translate(sets3, ticket_type):
    bets = set()
    for s in sets3:
        if ticket_type == '3連複':
            bets.add(tuple(sorted(s)))
        elif ticket_type in {'ワイド','2車複'}:
            for p in itertools.combinations(s,2):
                bets.add(tuple(sorted(p)))
        elif ticket_type == '2車単':
            for a,b in itertools.combinations(s,2):
                bets.add((a,b)); bets.add((b,a))
        elif ticket_type == '3連単':
            bets.update(itertools.permutations(s,3))
    return bets


def main():
    decisions = read_csv(DECISIONS)
    rows = []
    selected = {}
    for r in decisions:
        ms = middle_sets(r)
        if ms:
            selected[r['race_id']] = ms

    paid = defaultdict(lambda: defaultdict(list))
    for p in read_csv(DATA/'payouts.csv'):
        tt = p.get('ticket_type')
        if tt not in TYPES or p.get('status') != 'paid':
            continue
        key = payout_key(tt, p.get('combination',''))
        if key is None:
            continue
        try:
            py = int(float(p.get('payout_yen') or 0))
        except Exception:
            continue
        if py > 0:
            paid[p['race_id']][tt].append((key, py))

    agg = {tt:{'races':0,'points':0,'stake':0,'payout':0,'hit_races':0,'winning_bets':0,'race_points':[]} for tt in TYPES}

    for rid, ms in sorted(selected.items()):
        rec = {'race_id':rid,'middle_sets':'|'.join('-'.join(map(str,s)) for s in ms),'middle_set_count':len(ms)}
        for tt in TYPES:
            bets = translate(ms,tt)
            stake = 100*len(bets)
            payout = 0
            wins = []
            for key,py in paid.get(rid,{}).get(tt,[]):
                if key in bets:
                    payout += py
                    wins.append((key,py))
            a=agg[tt]
            a['races']+=1; a['points']+=len(bets); a['stake']+=stake; a['payout']+=payout; a['race_points'].append(len(bets))
            if wins:
                a['hit_races']+=1
                a['winning_bets']+=len(wins)
            rec[f'{tt}_points']=len(bets)
            rec[f'{tt}_stake_yen']=stake
            rec[f'{tt}_payout_yen']=payout
            rec[f'{tt}_profit_yen']=payout-stake
            rec[f'{tt}_hit']=int(bool(wins))
        rows.append(rec)

    def median(xs):
        ys=sorted(xs); n=len(ys)
        if not ys: return None
        return ys[n//2] if n%2 else (ys[n//2-1]+ys[n//2])/2

    summary={
        'status':'MIDDLE_CROSS_TICKET_SIMULATION_2023',
        'year':2023,
        'years_read':[2023],
        'evaluation_year_2024_used':False,
        'evaluation_year_2025_used':False,
        'evaluation_year_2026_used':False,
        'source_rule':'Use only middle tickets actually selected in frozen market_scenario_portfolio_2023_v1 decisions. No additional rider selection from other ticket types.',
        'staking':'100 yen per unique translated ticket per race; duplicate translated tickets from overlapping middle sets are deduplicated; no dutching.',
        'translation':{
            '3連複':'one 3-car set = 1 trio ticket',
            'ワイド':'all 3 unordered pairs inside each middle 3-car set',
            '2車複':'all 3 unordered pairs inside each middle 3-car set',
            '2車単':'both directions for all 3 pairs = up to 6 ordered tickets per set',
            '3連単':'all 6 permutations of each middle 3-car set',
        },
        'middle_races':len(selected),
        'middle_sets_total':sum(len(v) for v in selected.values()),
        'avg_middle_sets_per_race':sum(len(v) for v in selected.values())/len(selected) if selected else 0,
        'ticket_types':{}
    }
    for tt,a in agg.items():
        summary['ticket_types'][tt]={
            'races':a['races'],
            'total_unique_tickets':a['points'],
            'avg_unique_points_per_race':a['points']/a['races'] if a['races'] else 0,
            'median_unique_points_per_race':median(a['race_points']),
            'hit_races':a['hit_races'],
            'race_hit_rate_pct':100*a['hit_races']/a['races'] if a['races'] else 0,
            'winning_bet_rows':a['winning_bets'],
            'stake_yen':a['stake'],
            'payout_yen':a['payout'],
            'profit_yen':a['payout']-a['stake'],
            'roi_pct':100*a['payout']/a['stake'] if a['stake'] else 0,
        }

    OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    DETAIL.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0].keys()) if rows else ['race_id']
    with DETAIL.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

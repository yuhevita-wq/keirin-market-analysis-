from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

YEARS=(2023,2024)
OUT=Path('data/audits/trio_popularity_takeout_2023_2024.json')
BASELINE=75.0
BUCKETS=[(1,1),(2,3),(4,6),(7,10),(11,15),(16,20),(21,25),(26,30),(31,35)]
TOPNS=[1,2,3,5,10,15,20,25,30,35]


def read_csv(path: Path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def norm3(s: str):
    xs=tuple(sorted(int(x) for x in re.findall(r'\d+',s or '')))
    return xs if len(xs)==3 and len(set(xs))==3 else None


def load_year(year: int):
    data=Path(f'data/{year}/s_class_yosen')
    odds_by=defaultdict(list)
    for r in read_csv(data/'trio_final_odds.csv'):
        if r.get('odds_status')!='available':
            continue
        try: odds=float(r.get('odds') or 0)
        except Exception: continue
        c=norm3(r.get('combination',''))
        if c is None or odds<=1.0: continue
        odds_by[r['race_id']].append((c,odds))

    winners={}
    for p in read_csv(data/'payouts.csv'):
        if p.get('ticket_type')!='3連複' or p.get('status')!='paid':
            continue
        c=norm3(p.get('combination',''))
        if c is None: continue
        try: py=int(float(p.get('payout_yen') or 0))
        except Exception: continue
        if py>0:
            # Rare dead-heat/multiple payout rows are retained as a list per race.
            winners.setdefault(p['race_id'],[]).append((c,py))

    rows=[]
    complete_races=0
    tie_races=0
    for rid, xs in sorted(odds_by.items()):
        # Require a complete 35-combination market and at least one published trio payout.
        uniq={c:o for c,o in xs}
        if len(uniq)!=35 or rid not in winners:
            continue
        complete_races+=1
        ordered=sorted(uniq.items(), key=lambda kv:(kv[1],kv[0]))
        if len({o for _,o in ordered})<35:
            tie_races+=1
        paid=dict(winners[rid])
        for i,(c,o) in enumerate(ordered,1):
            rows.append({
                'race_id':rid,'rank':i,'combination':c,'odds':o,
                'hit':int(c in paid),'payout':paid.get(c,0),
            })
    return rows,complete_races,tie_races


def stats(rows):
    stake=100*len(rows)
    payout=sum(r['payout'] for r in rows)
    wins=sum(r['hit'] for r in rows)
    odds=[r['odds'] for r in rows]
    return {
        'tickets':len(rows),
        'wins':wins,
        'hit_rate_pct':100*wins/len(rows) if rows else 0.0,
        'avg_odds':sum(odds)/len(odds) if odds else None,
        'stake_yen':stake,
        'payout_yen':payout,
        'profit_yen':payout-stake,
        'roi_pct':100*payout/stake if stake else 0.0,
        'abs_deviation_from_75pt':abs((100*payout/stake if stake else 0.0)-BASELINE),
    }


def analyze(rows):
    by_rank=[]
    for k in range(1,36):
        s=stats([r for r in rows if r['rank']==k])
        s['rank']=k
        by_rank.append(s)
    buckets=[]
    for a,b in BUCKETS:
        s=stats([r for r in rows if a<=r['rank']<=b])
        s['rank_from']=a;s['rank_to']=b
        buckets.append(s)
    cumulative=[]
    for n in TOPNS:
        s=stats([r for r in rows if r['rank']<=n])
        s['top_n']=n
        cumulative.append(s)
    return {'by_rank':by_rank,'rank_buckets':buckets,'cumulative_top_n':cumulative}


def main():
    out={
        'status':'TRIO_POPULARITY_TAKEOUT_AUDIT_2023_2024',
        'years_read':[2023,2024],
        'evaluation_year_2025_used':False,
        'evaluation_year_2026_used':False,
        'nominal_reference_roi_pct':BASELINE,
        'question':'Does 3連複 ROI sit closer to the nominal 75% return-rate wall for more popular combinations, with larger deviations for longshots?',
        'method':'Within each race with all 35 final 3連複 odds available, rank combinations from shortest to longest final odds. Flat-bet 100 yen on each exact popularity rank. Report exact ranks, rank buckets, and cumulative top-N. Payout uses published actual 3連複 payout rows. Popularity ties are deterministically ordered by combination only for exact-rank bookkeeping; bucket/cumulative results are more robust to ties.',
        'years':{}
    }
    for y in YEARS:
        rows,n,ties=load_year(y)
        out['years'][str(y)]={
            'complete_races':n,
            'races_with_any_odds_tie':ties,
            **analyze(rows),
        }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

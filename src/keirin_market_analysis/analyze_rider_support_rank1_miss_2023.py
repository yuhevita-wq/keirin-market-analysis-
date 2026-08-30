from __future__ import annotations

import csv, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / '2023' / 's_class_yosen'
OUT = ROOT / 'data' / 'audits' / 'rider_support_rank1_miss_2023.json'


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def parse_combo(s):
    return tuple(sorted(int(x) for x in s.strip().replace('=', '-').replace(',', '-').split('-') if x.strip()))


def mean(xs): return statistics.mean(xs) if xs else None

def median(xs): return statistics.median(xs) if xs else None

def quantile(xs,q):
    a=sorted(xs)
    if not a: return None
    p=(len(a)-1)*q; lo=int(math.floor(p)); hi=int(math.ceil(p))
    if lo==hi:return a[lo]
    return a[lo]+(a[hi]-a[lo])*(p-lo)


def main():
    trio=defaultdict(list)
    for r in read_csv(DATA/'trio_final_odds.csv'):
        if r.get('odds_status')!='available': continue
        try:o=float(r['odds'])
        except: continue
        if o<=0: continue
        trio[r['race_id']].append({'combo':parse_combo(r['combination']),'odds':o,'market_rank':int(r['market_rank']) if r.get('market_rank') else None})

    results=defaultdict(list)
    for r in read_csv(DATA/'results.csv'):
        fp=r.get('finish_position','')
        if fp in {'1','2','3'}:
            results[r['race_id']].append((int(fp),int(r['car_no'])))

    payouts=defaultdict(list)
    for r in read_csv(DATA/'payouts.csv'):
        if r.get('ticket_type')=='3連複' and r.get('status')=='paid':
            payouts[r['race_id']].append({'combo':parse_combo(r['combination']),'payout_yen':int(r['payout_yen']),'popularity':int(r['popularity']) if r.get('popularity') else None})

    rows=[]
    for rid in sorted(set(trio)&set(results)&set(payouts)):
        trs=trio[rid]
        inv={x['combo']:1/x['odds'] for x in trs}
        z=sum(inv.values())
        p={c:w/z for c,w in inv.items()}
        rider_support=defaultdict(float)
        for c,prob in p.items():
            for car in c:rider_support[car]+=prob
        ranked=sorted(rider_support,key=lambda c:(-rider_support[c],c))
        rank={c:i+1 for i,c in enumerate(ranked)}
        s1=rider_support[ranked[0]]; s2=rider_support[ranked[1]]; s3=rider_support[ranked[2]]
        # normalized rider support shares sum to 3 across riders; divide by 3 for intuitive share
        rider_share={c:v/3 for c,v in rider_support.items()}
        top3cars=[c for _,c in sorted(results[rid])[:3]]
        miss=ranked[0] not in top3cars
        entropy=-sum(v*math.log(v) for v in p.values())/math.log(len(p)) if len(p)>1 else 0.0
        trio_sorted=sorted(p.items(),key=lambda kv:(-kv[1],kv[0]))
        top3_conc=sum(v for _,v in trio_sorted[:3])
        win=payouts[rid][0]
        win_ranks=tuple(sorted(rank[c] for c in win['combo']))
        rows.append({
            'race_id':rid,'rank1_car':ranked[0],'rank1_support':s1,'rank1_share':s1/3,
            'rank2_support':s2,'rank3_support':s3,'support_gap_1_2':s1-s2,
            'support_ratio_2_to_1':s2/s1 if s1 else None,'top3_rider_support_sum':s1+s2+s3,
            'trio_entropy':entropy,'trio_top3_concentration':top3_conc,'rank1_miss':miss,
            'winner_rider_support_ranks':win_ranks,'winner_popularity':win['popularity'],'payout_yen':win['payout_yen']
        })

    n=len(rows); misses=sum(r['rank1_miss'] for r in rows); base=misses/n
    feats={
      'rank1_share':'low','support_gap_1_2':'low','support_ratio_2_to_1':'high',
      'top3_rider_support_sum':'low','trio_entropy':'high','trio_top3_concentration':'low'
    }
    comparisons={}
    quintiles={}
    scans={}
    for k,d in feats.items():
        hit=[r[k] for r in rows if not r['rank1_miss']]; miss=[r[k] for r in rows if r['rank1_miss']]
        comparisons[k]={'rank1_top3':{'n':len(hit),'mean':mean(hit),'median':median(hit)},'rank1_miss':{'n':len(miss),'mean':mean(miss),'median':median(miss)}}
        vals=[r[k] for r in rows]; cuts=[quantile(vals,q) for q in (.2,.4,.6,.8)]
        bins=[]
        for i in range(5):
            lo=-float('inf') if i==0 else cuts[i-1]; hi=float('inf') if i==4 else cuts[i]
            sel=[r for r in rows if r[k]>=lo and (r[k]<hi if i<4 else r[k]<=hi)]
            m=sum(r['rank1_miss'] for r in sel)
            bins.append({'bin':i+1,'races':len(sel),'min':min((r[k] for r in sel),default=None),'max':max((r[k] for r in sel),default=None),'miss_rate_pct':100*m/len(sel) if sel else None})
        quintiles[k]=bins
        cand=[]
        for q in [i/10 for i in range(1,10)]:
            t=quantile(vals,q)
            if d=='low': sel=[r for r in rows if r[k]<=t]; expr=f'{k} <= {t}'
            else: sel=[r for r in rows if r[k]>=t]; expr=f'{k} >= {t}'
            if len(sel)<int(.15*n): continue
            m=sum(r['rank1_miss'] for r in sel); rate=m/len(sel)
            cand.append({'expression':expr,'threshold':t,'quantile':q,'races':len(sel),'misses':m,'miss_rate_pct':100*rate,'lift_vs_baseline':rate/base})
        cand.sort(key=lambda x:(x['miss_rate_pct'],x['races']),reverse=True)
        scans[k]=cand

    # pair combinations from each feature's best single threshold, minimum 10% population
    best={k:v[0] for k,v in scans.items() if v}
    pairs=[]
    keys=list(best)
    for i,a in enumerate(keys):
      for b in keys[i+1:]:
        def ok(r,k):
            t=best[k]['threshold']; return r[k]<=t if feats[k]=='low' else r[k]>=t
        sel=[r for r in rows if ok(r,a) and ok(r,b)]
        if len(sel)<int(.10*n): continue
        m=sum(r['rank1_miss'] for r in sel); rate=m/len(sel)
        pairs.append({'features':[a,b],'expressions':[best[a]['expression'],best[b]['expression']], 'races':len(sel),'misses':m,'miss_rate_pct':100*rate,'lift_vs_baseline':rate/base})
    pairs.sort(key=lambda x:(x['miss_rate_pct'],x['races']),reverse=True)

    missrows=[r for r in rows if r['rank1_miss']]
    rank_patterns=Counter('-'.join(map(str,r['winner_rider_support_ranks'])) for r in missrows)
    rank_presence=Counter()
    for r in missrows:
        for rr in r['winner_rider_support_ranks']: rank_presence[str(rr)]+=1
    popb=Counter()
    for r in missrows:
        p=r['winner_popularity']
        if p is None: continue
        if p<=3:b='1-3'
        elif p<=5:b='4-5'
        elif p<=10:b='6-10'
        elif p<=15:b='11-15'
        else:b='16+'
        popb[b]+=1

    out={
      'status':'RIDER_SUPPORT_RANK1_MISS_2023_FULL_POPULATION','year':2023,'years_read':[2023],
      'definition':'Individual rider support is marginal 3連複 market support: sum of normalized 1/odds probability mass over all trio combinations containing the rider. Rank1 miss means the highest-support rider finishes outside top3.',
      'baseline':{'analyzable_races':n,'rank1_top3_races':n-misses,'rank1_miss_races':misses,'rank1_miss_rate_pct':100*base},
      'feature_comparison_top3_vs_miss':comparisons,'feature_quintiles':quintiles,
      'exploratory_single_feature_threshold_scans':scans,'exploratory_top_pair_conditions':pairs[:15],
      'miss_outcomes':{
        'winning_rider_support_rank_patterns_top20':rank_patterns.most_common(20),
        'rider_support_rank_presence':dict(sorted(rank_presence.items(),key=lambda kv:int(kv[0]))),
        'winning_trio_popularity_buckets':dict(popb),
        'payout_yen':{'mean':mean([r['payout_yen'] for r in missrows]),'median':median([r['payout_yen'] for r in missrows]),'min':min(r['payout_yen'] for r in missrows),'max':max(r['payout_yen'] for r in missrows)}
      },
      'warning':'2023 exploratory development only. Thresholds are in-sample and must not be called validated. 2024/2025/2026 not read.'
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

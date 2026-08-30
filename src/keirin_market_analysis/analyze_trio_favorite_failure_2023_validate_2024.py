from __future__ import annotations

import csv, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .simulate_market_scenario_portfolio_2023_v1 import build_race

YEARS=(2023,2024)
OUT=Path('data/audits/trio_favorite_failure_2023_validate_2024.json')


def read_csv(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def norm3(s):
    xs=tuple(sorted(int(x) for x in re.findall(r'\d+',s or '')))
    return xs if len(xs)==3 and len(set(xs))==3 else None


def parse_middle(row):
    out=[]
    for part in (row.get('tickets') or '').split(' | '):
        m=re.match(r'^middle:([0-9-]+)@([0-9.]+)x([0-9]+)u$',part.strip())
        if not m: continue
        cars_s,odds_s,_=m.groups()
        out.append((tuple(sorted(map(int,cars_s.split('-')))),float(odds_s)))
    return out


def entropy(ps):
    s=sum(ps)
    if s<=0:return 0.0
    qs=[p/s for p in ps if p>0]
    return -sum(q*math.log(q) for q in qs)


def load_year(year):
    data=Path(f'data/{year}/s_class_yosen')
    trio=defaultdict(list); tri=defaultdict(list)
    for r in read_csv(data/'trio_final_odds.csv'): trio[r['race_id']].append(r)
    for r in read_csv(data/'trifecta_final_odds.csv'): tri[r['race_id']].append(r)
    payout={}
    for p in read_csv(data/'payouts.csv'):
        if p.get('ticket_type')!='3連複' or p.get('status')!='paid': continue
        c=norm3(p.get('combination',''))
        if c is None: continue
        try: py=int(float(p.get('payout_yen') or 0))
        except Exception: continue
        if py>0: payout.setdefault(p['race_id'],[]).append((c,py))
    decisions={r['race_id']:r for r in read_csv(f'data/audits/market_scenario_portfolio_{year}_v1_decisions.csv')}

    rows=[]
    for rid,trs in sorted(trio.items()):
        available=[]
        for r in trs:
            if r.get('odds_status')!='available': continue
            c=norm3(r.get('combination',''))
            try:o=float(r.get('odds') or 0)
            except Exception:continue
            if c and o>1: available.append((c,o))
        uniq={c:o for c,o in available}
        if len(uniq)!=35 or rid not in payout: continue
        market=build_race(tri.get(rid,[]),trs)
        if not market or len(market)!=35: continue
        ordered=sorted(uniq.items(),key=lambda kv:(kv[1],kv[0]))
        fav,fav_odds=ordered[0]; second,second_odds=ordered[1]
        paid=dict(payout[rid]); fav_hit=int(fav in paid)
        winner=next((c for c,_ in payout[rid] if c in uniq),None)
        if winner is None: continue
        rank_map={c:i for i,(c,_) in enumerate(ordered,1)}
        winner_rank=rank_map[winner]
        vals=[market[c] for c,_ in ordered]
        f=market[fav]; non=[market[c] for c,_ in ordered[1:]]
        p_trios=[x['p_trio'] for x in vals]
        p_tris=[x['p_tri_set'] for x in vals]
        positive=[x['delta'] for x in non if x['delta']>0]
        ratios=[x['ratio'] for x in non if x.get('ratio') is not None]
        # normalized shares from inverse odds, independent of build_race normalization details
        inv=[1/o for _,o in ordered]; invsum=sum(inv)
        fav_share=inv[0]/invsum
        top3_share=sum(inv[:3])/invsum; top5_share=sum(inv[:5])/invsum; top10_share=sum(inv[:10])/invsum
        mids=parse_middle(decisions.get(rid,{}))
        mid_stake=100*len(mids)
        mid_pay=sum(paid.get(c,0) for c,_ in mids)
        rows.append({
            'race_id':rid,'fav_hit':fav_hit,'winner_rank':winner_rank,
            'fav_odds':fav_odds,'second_odds':second_odds,'second_to_fav_odds_ratio':second_odds/fav_odds,
            'fav_market_share':fav_share,'top3_market_share':top3_share,'top5_market_share':top5_share,'top10_market_share':top10_share,
            'trio_entropy':entropy(p_trios),'trifecta_set_entropy':entropy(p_tris),
            'fav_p_trio':f['p_trio'],'fav_p_tri_set':f['p_tri_set'],'fav_delta':f['delta'],'fav_abs_delta':abs(f['delta']),
            'fav_ratio':f['ratio'] or 0.0,'fav_consensus':f['consensus'],
            'nonfav_positive_delta_count':len(positive),'nonfav_positive_delta_sum':sum(positive),
            'nonfav_delta_max':max((x['delta'] for x in non),default=0.0),
            'nonfav_ratio_max':max(ratios,default=0.0),
            'middle_count':len(mids),'middle_stake':mid_stake,'middle_payout':mid_pay,'middle_hit':int(mid_pay>0),
        })
    return rows

FEATURES=['fav_odds','second_to_fav_odds_ratio','fav_market_share','top3_market_share','top5_market_share','top10_market_share',
          'trio_entropy','trifecta_set_entropy','fav_delta','fav_abs_delta','fav_ratio','fav_consensus',
          'nonfav_positive_delta_count','nonfav_positive_delta_sum','nonfav_delta_max','nonfav_ratio_max','middle_count']


def q(vals,p):
    xs=sorted(vals); pos=(len(xs)-1)*p; lo=int(pos); hi=min(lo+1,len(xs)-1); f=pos-lo
    return xs[lo]+(xs[hi]-xs[lo])*f


def summary(rows):
    n=len(rows); hits=sum(r['fav_hit'] for r in rows); miss=n-hits
    ms=[r for r in rows if r['middle_count']>0]
    stake=sum(r['middle_stake'] for r in ms); pay=sum(r['middle_payout'] for r in ms)
    return {'races':n,'fav_hits':hits,'fav_misses':miss,'fav_miss_rate_pct':100*miss/n if n else 0,
            'winner_rank_median':median([r['winner_rank'] for r in rows]) if rows else None,
            'middle_races':len(ms),'middle_hit_races':sum(r['middle_hit'] for r in ms),
            'middle_race_hit_rate_pct':100*sum(r['middle_hit'] for r in ms)/len(ms) if ms else 0,
            'middle_tickets':sum(r['middle_count'] for r in ms),'middle_stake_yen':stake,'middle_payout_yen':pay,
            'middle_profit_yen':pay-stake,'middle_roi_pct':100*pay/stake if stake else 0}


def match(r,c):
    v=float(r[c['feature']]); t=float(c['threshold'])
    return v<=t if c['op']=='<=' else v>=t


def anatomy(rows):
    miss=[r for r in rows if not r['fav_hit']]; hit=[r for r in rows if r['fav_hit']]
    bands=[(2,3),(4,6),(7,10),(11,15),(16,20),(21,35)]
    dist=[]
    for a,b in bands:
        dist.append({'rank_from':a,'rank_to':b,'wins':sum(a<=r['winner_rank']<=b for r in miss),'share_pct':100*sum(a<=r['winner_rank']<=b for r in miss)/len(miss) if miss else 0})
    feat={}
    for f in FEATURES:
        feat[f]={'fav_hit_mean':sum(float(r[f]) for r in hit)/len(hit) if hit else None,
                 'fav_miss_mean':sum(float(r[f]) for r in miss)/len(miss) if miss else None}
    return {'favorite_hit':summary(hit),'favorite_miss':summary(miss),'winner_rank_distribution_when_favorite_misses':dist,'feature_means':feat}


def main():
    r23=load_year(2023); r24=load_year(2024)
    base23=summary(r23); base24=summary(r24)
    candidates=[]
    for f in FEATURES:
        vals=[float(r[f]) for r in r23]
        for t in sorted(set(q(vals,p) for p in (.25,.5,.75))):
            for op in ('<=','>='):
                c={'feature':f,'op':op,'threshold':t}
                a=[r for r in r23 if match(r,c)]; b=[r for r in r24 if match(r,c)]
                if len(a)<150 or len(b)<100: continue
                s23=summary(a); s24=summary(b)
                candidates.append({**c,'train_2023':s23,'validation_2024':s24,
                                   'miss_uplift_2023_pt':s23['fav_miss_rate_pct']-base23['fav_miss_rate_pct'],
                                   'miss_uplift_2024_pt':s24['fav_miss_rate_pct']-base24['fav_miss_rate_pct']})
    by_miss=sorted(candidates,key=lambda x:(-x['miss_uplift_2023_pt'],-x['train_2023']['races']))[:20]
    # Exploratory only: among conditions that genuinely raised 2023 favorite-miss rate, see which monetized middle best.
    eligible=[c for c in candidates if c['miss_uplift_2023_pt']>0]
    by_mid=sorted(eligible,key=lambda x:(-x['train_2023']['middle_roi_pct'],-x['train_2023']['middle_races']))[:20]
    out={'status':'TRIO_FAVORITE_FAILURE_2023_VALIDATE_2024','years_read':[2023,2024],
         'evaluation_year_2025_used':False,'evaluation_year_2026_used':False,
         'definition':'Favorite = shortest final 3連複 odds in a complete 35-combination market. Favorite miss means that exact 3-car set did not win.',
         'discipline':'All race-selection features are pre-result market-only. Single-feature thresholds are discovered from 2023 quartiles and applied unchanged to 2024. Middle performance buys all already-selected frozen v1 middle tickets at flat 100 yen; no ticket pruning.',
         'baseline':{'2023':base23,'2024':base24},'anatomy':{'2023':anatomy(r23),'2024':anatomy(r24)},
         'top20_by_2023_favorite_miss_uplift_with_frozen_2024':by_miss,
         'top20_exploratory_middle_roi_among_2023_miss_uplift_conditions':by_mid}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'baseline':out['baseline'],'top_miss':by_miss[:8],'top_middle':by_mid[:8]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()

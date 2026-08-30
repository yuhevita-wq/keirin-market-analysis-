from __future__ import annotations

import csv, json, math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

YEARS=(2023,2024,2025)
OUT=Path('data/audits/insurance_ticket_236_146_2023_2025.json')
STAKE=100
ENTROPY_MIN=0.7598574338315534
TOP3_CONC_MAX=0.5122247620383481
P123_MAX=0.22900352400362795
RANK1_SHARE_MIN=0.24054261443743438
BASE_PATTERNS=((2,3,6),(1,4,6))
ALL_PATTERNS=tuple(combinations(range(1,8),3))
CANDIDATES=tuple(p for p in ALL_PATTERNS if p not in BASE_PATTERNS)

def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def comb(s):
    s=str(s).strip().replace('=','-').replace(',','-')
    parts=[p for p in s.split('-') if p.strip()]
    try:return tuple(sorted(int(x) for x in parts))
    except:return ()

def max_losing_streak(hits):
    best=cur=0
    for h in hits:
        if h: cur=0
        else:
            cur+=1; best=max(best,cur)
    return best

def load_year(year):
    base=Path(f'data/{year}/s_class_yosen')
    trio_rows=read_csv(base/'trio_final_odds.csv')
    payout_rows=read_csv(base/'payouts.csv')
    trios=defaultdict(list)
    for r in trio_rows:
        try:o=float(r.get('odds') or r.get('final_odds') or r.get('trio_odds'))
        except: continue
        c=comb(r.get('combination') or r.get('bet_code') or '')
        if len(c)==3 and o>0: trios[str(r['race_id'])].append((c,o))
    paid=defaultdict(dict)
    for r in payout_rows:
        if (r.get('ticket_type') or '').strip() not in ('3連複','trio'): continue
        st=str(r.get('status') or '').lower()
        if st not in ('','paid','success','確定'): continue
        c=comb(r.get('combination') or r.get('bet_code') or '')
        if len(c)!=3: continue
        try:p=int(float(r.get('payout_yen') or 0))
        except:p=0
        paid[str(r['race_id'])][c]=p

    races=[]
    for rid,rows in trios.items():
        if rid not in paid or not paid[rid]: continue
        inv=[(c,1/o) for c,o in rows if o>0]
        z=sum(v for _,v in inv)
        if z<=0: continue
        p={c:v/z for c,v in inv}
        riders=sorted({x for c in p for x in c})
        sup={x:sum(v for c,v in p.items() if x in c) for x in riders}
        ranked=sorted(sup,key=lambda x:(-sup[x],x))
        if len(ranked)<6: continue
        vals=sorted(sup.values(),reverse=True)
        n=len(p)
        entropy=-sum(v*math.log(v) for v in p.values())/math.log(n) if n>1 else 0
        top3_conc=sum(sorted(p.values(),reverse=True)[:3])
        rank1_share=vals[0]/3 if vals else 0
        byrank={i+1:c for i,c in enumerate(ranked)}
        c123=tuple(sorted(byrank[k] for k in (1,2,3)))
        p123=p.get(c123,0)
        if not (entropy>=ENTROPY_MIN and top3_conc<=TOP3_CONC_MAX and p123<=P123_MAX and rank1_share>=RANK1_SHARE_MIN):
            continue
        winning=paid[rid]
        races.append((rid,byrank,winning))
    return races

def eval_set(races, patterns):
    stake=payout=0; hit_races=0; hits=[]; bypat={str(p):{'tickets':0,'hits':0,'stake_yen':0,'payout_yen':0} for p in patterns}
    for rid,byrank,winning in races:
        race_hit=False
        for pat in patterns:
            if not all(k in byrank for k in pat): continue
            c=tuple(sorted(byrank[k] for k in pat))
            stake+=STAKE
            d=bypat[str(pat)]; d['tickets']+=1; d['stake_yen']+=STAKE
            if c in winning:
                pay=winning[c]; payout+=pay; d['hits']+=1; d['payout_yen']+=pay; race_hit=True
        if race_hit: hit_races+=1
        hits.append(race_hit)
    return {'selected_races':len(races),'tickets':sum(d['tickets'] for d in bypat.values()),'hit_races':hit_races,'race_hit_rate_pct':100*hit_races/len(races) if races else None,'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi_pct':100*payout/stake if stake else None,'max_losing_streak':max_losing_streak(hits),'by_pattern':bypat}

def main():
    data={y:load_year(y) for y in YEARS}
    base={y:eval_set(data[y],BASE_PATTERNS) for y in YEARS}
    candidate_rows=[]
    for cand in CANDIDATES:
        yearly={y:eval_set(data[y],BASE_PATTERNS+(cand,)) for y in YEARS}
        pooled_stake=sum(v['stake_yen'] for v in yearly.values()); pooled_pay=sum(v['payout_yen'] for v in yearly.values()); pooled_hits=sum(v['hit_races'] for v in yearly.values()); pooled_races=sum(v['selected_races'] for v in yearly.values())
        marginal_new_hits=sum(yearly[y]['hit_races']-base[y]['hit_races'] for y in YEARS)
        row={'candidate':cand,'yearly':yearly,'pooled':{'selected_races':pooled_races,'hit_races':pooled_hits,'race_hit_rate_pct':100*pooled_hits/pooled_races if pooled_races else None,'stake_yen':pooled_stake,'payout_yen':pooled_pay,'profit_yen':pooled_pay-pooled_stake,'roi_pct':100*pooled_pay/pooled_stake if pooled_stake else None,'marginal_new_hit_races':marginal_new_hits,'worst_year_roi_pct':min(yearly[y]['roi_pct'] for y in YEARS),'worst_year_max_losing_streak':max(yearly[y]['max_losing_streak'] for y in YEARS)}}
        candidate_rows.append(row)
    # useful rankings, not a strategy selection
    top_by_streak=sorted(candidate_rows,key=lambda r:(r['pooled']['worst_year_max_losing_streak'],-r['pooled']['roi_pct']))[:12]
    top_roi_over100=sorted([r for r in candidate_rows if r['pooled']['roi_pct']>=100],key=lambda r:(r['pooled']['worst_year_max_losing_streak'],-r['pooled']['roi_pct']))[:20]
    top_by_marginal_hits=sorted(candidate_rows,key=lambda r:(-r['pooled']['marginal_new_hit_races'],-r['pooled']['roi_pct']))[:12]
    out={'status':'INSURANCE_TICKET_SCAN_236_146_2023_2025_EXPLORATORY','years_read':list(YEARS),'gate':'same frozen 2023 gate','base_patterns':BASE_PATTERNS,'base_by_year':base,'candidate_count':len(CANDIDATES),'rankings':{'shortest_worst_year_streak':top_by_streak,'pooled_roi_ge_100_sorted_by_streak':top_roi_over100,'most_marginal_new_hits':top_by_marginal_hits},'all_candidates':candidate_rows,'warning':'Exploratory redesign after 2023-2025 are exposed. Any chosen insurance ticket is a NEW strategy; 2026 remains the only clean forward validation year.'}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out['rankings'],ensure_ascii=False,indent=2))

if __name__=='__main__': main()

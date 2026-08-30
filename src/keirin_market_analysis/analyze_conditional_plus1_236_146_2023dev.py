from __future__ import annotations

import csv, json, math
from collections import defaultdict
from pathlib import Path

YEARS=(2023,2024,2025)
OUT=Path('data/audits/conditional_plus1_236_146_2023dev.json')
STAKE=100
ENTROPY_MIN=0.7598574338315534
TOP3_CONC_MAX=0.5122247620383481
P123_MAX=0.22900352400362795
RANK1_SHARE_MIN=0.24054261443743438
BASE_PATTERNS=((2,3,6),(1,4,6))
CANDIDATES=((1,2,3),(1,3,4))
QUANTILES=(0.50,0.60,0.70,0.75,0.80,0.85,0.90)


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


def quantile(vals,q):
    xs=sorted(vals)
    if not xs:return None
    pos=(len(xs)-1)*q
    lo=int(math.floor(pos)); hi=int(math.ceil(pos))
    if lo==hi:return xs[lo]
    return xs[lo]*(hi-pos)+xs[hi]*(pos-lo)


def pattern_prob(p,byrank,pat):
    if not all(k in byrank for k in pat): return 0.0
    c=tuple(sorted(byrank[k] for k in pat))
    return p.get(c,0.0)


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
        try:pay=int(float(r.get('payout_yen') or 0))
        except:pay=0
        paid[str(r['race_id'])][c]=pay

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
        p123=pattern_prob(p,byrank,(1,2,3))
        if not (entropy>=ENTROPY_MIN and top3_conc<=TOP3_CONC_MAX and p123<=P123_MAX and rank1_share>=RANK1_SHARE_MIN):
            continue
        p236=pattern_prob(p,byrank,(2,3,6)); p146=pattern_prob(p,byrank,(1,4,6))
        ref=max(p236,p146)
        features={}
        for cand in CANDIDATES:
            pc=pattern_prob(p,byrank,cand)
            features[str(cand)]={'p':pc,'ratio_to_base_max':pc/ref if ref>0 else 0.0}
        races.append({'rid':rid,'byrank':byrank,'winning':paid[rid],'features':features})
    return races


def eval_rule(races,cand=None,feature=None,threshold=None):
    stake=payout=0; hit_races=0; hits=[]; added=0; add_hits=0
    pats=list(BASE_PATTERNS)
    for r in races:
        race_hit=False
        for pat in BASE_PATTERNS:
            c=tuple(sorted(r['byrank'][k] for k in pat)); stake+=STAKE
            if c in r['winning']:
                payout+=r['winning'][c]; race_hit=True
        do_add=False
        if cand is not None:
            do_add=r['features'][str(cand)][feature] >= threshold
        if do_add:
            added+=1; stake+=STAKE
            c=tuple(sorted(r['byrank'][k] for k in cand))
            if c in r['winning']:
                payout+=r['winning'][c]; add_hits+=1; race_hit=True
        if race_hit: hit_races+=1
        hits.append(race_hit)
    return {
        'selected_races':len(races),'added_tickets':added,'add_rate_pct':100*added/len(races) if races else None,
        'added_ticket_hits':add_hits,'hit_races':hit_races,'race_hit_rate_pct':100*hit_races/len(races) if races else None,
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi_pct':100*payout/stake if stake else None,
        'max_losing_streak':max_losing_streak(hits)
    }


def main():
    data={y:load_year(y) for y in YEARS}
    base={y:eval_rule(data[y]) for y in YEARS}
    scans=[]
    for cand in CANDIDATES:
        for feature in ('ratio_to_base_max','p'):
            vals=[r['features'][str(cand)][feature] for r in data[2023]]
            for q in QUANTILES:
                thr=quantile(vals,q)
                dev=eval_rule(data[2023],cand,feature,thr)
                scans.append({'candidate':cand,'feature':feature,'quantile_2023':q,'threshold':thr,'dev_2023':dev})

    eligible=[r for r in scans if r['dev_2023']['roi_pct']>=110.0 and r['dev_2023']['add_rate_pct']<=35.0]
    eligible=sorted(eligible,key=lambda r:(r['dev_2023']['max_losing_streak'],-r['dev_2023']['race_hit_rate_pct'],-r['dev_2023']['roi_pct']))
    top=eligible[:10]
    for r in top:
        r['forward_2024']=eval_rule(data[2024],tuple(r['candidate']),r['feature'],r['threshold'])
        r['forward_2025']=eval_rule(data[2025],tuple(r['candidate']),r['feature'],r['threshold'])

    chosen=top[0] if top else None
    out={
        'status':'CONDITIONAL_PLUS1_236_146_2023_DEVELOPMENT_SCAN',
        'years_read':[2023,2024,2025],
        'base_patterns':BASE_PATTERNS,
        'base_by_year':base,
        'development_rule':'Use 2023 only to set candidate/feature/threshold. Constraint: 2023 combined ROI >=110% and add rate <=35%; rank by shortest 2023 max losing streak, then higher hit rate, then ROI.',
        'candidates':CANDIDATES,
        'features':['candidate normalized probability p','candidate p / max(p236,p146)'],
        'top_2023_rules_with_fixed_forward_checks':top,
        'chosen_by_predeclared_objective':chosen,
        'warning':'Exploratory redesign. 2023 is development. 2024/2025 are already exposed project years, so forward checks are replication diagnostics, not clean OOS validation. 2026 remains untouched.'
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'base':base,'top':top[:5],'chosen':chosen},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

from __future__ import annotations

import csv, json, math
from collections import defaultdict
from pathlib import Path

BASE=Path('data/2024/s_class_yosen')
OUT=Path('data/audits/selected_134_136_146_2024.json')
STAKE=100
ENTROPY_MIN=0.7598574338315534
TOP3_CONC_MAX=0.5122247620383481
P123_MAX=0.22900352400362795
RANK1_SHARE_MIN=0.24054261443743438
TARGET_PATTERNS=((1,3,4),(1,3,6),(1,4,6))

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

def main():
    trio_rows=read_csv(BASE/'trio_final_odds.csv')
    payout_rows=read_csv(BASE/'payouts.csv')
    trios=defaultdict(list)
    for r in trio_rows:
        try:o=float(r.get('odds') or r.get('final_odds') or r.get('trio_odds'))
        except: continue
        c=comb(r.get('combination') or r.get('bet_code') or '')
        if len(c)==3 and o>0: trios[str(r['race_id'])].append((c,o))
    paid=defaultdict(dict)
    for r in payout_rows:
        if (r.get('ticket_type') or '').strip() not in ('3連複','trio'): continue
        if r.get('status') not in (None,'','paid','success','確定') and str(r.get('status')).lower() not in ('paid','success'): continue
        c=comb(r.get('combination') or r.get('bet_code') or '')
        if len(c)!=3: continue
        try:p=int(float(r.get('payout_yen') or 0))
        except:p=0
        paid[str(r['race_id'])][c]=p
    selected=[]; race_hits=[]; total_stake=total_payout=0; hit_races=0; analyzable=0
    bypat={str(p):{'tickets':0,'hits':0,'stake_yen':0,'payout_yen':0} for p in TARGET_PATTERNS}
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
        analyzable+=1
        vals=sorted(sup.values(),reverse=True)
        n=len(p)
        entropy=-sum(v*math.log(v) for v in p.values())/math.log(n) if n>1 else 0
        top3_conc=sum(sorted(p.values(),reverse=True)[:3])
        rank1_share=vals[0]/3 if vals else 0
        byrank={i+1:c for i,c in enumerate(ranked)}
        c123=tuple(sorted(byrank[k] for k in (1,2,3)))
        p123=p.get(c123,0)
        if not (entropy>=ENTROPY_MIN and top3_conc<=TOP3_CONC_MAX and p123<=P123_MAX and rank1_share>=RANK1_SHARE_MIN): continue
        race_hit=False; race_pay=0
        for pat in TARGET_PATTERNS:
            c=tuple(sorted(byrank[k] for k in pat))
            total_stake+=STAKE
            d=bypat[str(pat)]; d['tickets']+=1; d['stake_yen']+=STAKE
            if c in paid[rid]:
                pay=paid[rid][c]; total_payout+=pay; race_pay+=pay; d['hits']+=1; d['payout_yen']+=pay; race_hit=True
        if race_hit: hit_races+=1
        race_hits.append(race_hit)
        selected.append({'race_id':rid,'hit':race_hit,'payout_yen':race_pay})
    for d in bypat.values():
        d['profit_yen']=d['payout_yen']-d['stake_yen']
        d['roi_pct']=100*d['payout_yen']/d['stake_yen'] if d['stake_yen'] else None
    out={
      'status':'SELECTED_134_136_146_2024_FROZEN_2023_GATE',
      'year':2024,'years_read':[2024],
      'thresholds':{'entropy_min':ENTROPY_MIN,'top3_conc_max':TOP3_CONC_MAX,'p123_max':P123_MAX,'rank1_share_min':RANK1_SHARE_MIN},
      'strategy':'Buy support-rank trios 1-3-4, 1-3-6, and 1-4-6, 100 yen each, only when frozen 2023 gate passes.',
      'payout_method':'Actual published 3連複 payout_yen from 2024 payouts.csv.',
      'result':{'total_analyzable_races':analyzable,'selected_races':len(selected),'selection_rate_pct':100*len(selected)/analyzable if analyzable else None,
      'tickets':sum(d['tickets'] for d in bypat.values()),'hit_races':hit_races,'race_hit_rate_pct':100*hit_races/len(selected) if selected else None,
      'stake_yen':total_stake,'payout_yen':total_payout,'profit_yen':total_payout-total_stake,'roi_pct':100*total_payout/total_stake if total_stake else None,
      'max_losing_streak':max_losing_streak(race_hits),'by_pattern':bypat},
      'warning':'2024 replication using frozen 2023 gate. Only 1-3-4 added to the existing 1-3-6 and 1-4-6 tickets.'}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out['result'],ensure_ascii=False,indent=2))

if __name__=='__main__': main()

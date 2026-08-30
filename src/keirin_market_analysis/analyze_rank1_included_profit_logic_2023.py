from __future__ import annotations

import csv, json, math, statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'2023'/'s_class_yosen'
OUT=ROOT/'data'/'audits'/'rank1_included_profit_logic_2023.json'
STAKE=100

def read_csv(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def parse_combo(s):
    s=s.strip().replace('=','-').replace(',','-'); return tuple(sorted(int(x) for x in s.split('-') if x.strip()))
def max_ls(xs):
    b=c=0
    for x in xs:
        if x:c=0
        else:c+=1;b=max(b,c)
    return b

def main():
    trio=defaultdict(list)
    for r in read_csv(DATA/'trio_final_odds.csv'):
        if r.get('odds_status')!='available':continue
        try:o=float(r['odds'])
        except:continue
        if o<=0:continue
        trio[r['race_id']].append((parse_combo(r['combination']),o))
    trif=defaultdict(list)
    for r in read_csv(DATA/'trifecta_final_odds.csv'):
        if r.get('odds_status')!='available':continue
        try:o=float(r['odds'])
        except:continue
        if o<=0:continue
        c=tuple(int(x) for x in r['combination'].replace('=','-').replace(',','-').split('-') if x.strip())
        trif[r['race_id']].append((c,o))
    pays=defaultdict(list)
    for r in read_csv(DATA/'payouts.csv'):
        if r.get('ticket_type')=='3連複' and r.get('status')=='paid':
            pays[r['race_id']].append((parse_combo(r['combination']),int(r['payout_yen'])))

    variants=defaultdict(lambda:{k:{'bet_races':0,'tickets':0,'hit_races':0,'stake_yen':0,'payout_yen':0,'hits':[]} for k in range(1,6)})
    rank_combo_diag=defaultdict(lambda:{'tickets':0,'hits':0,'stake_yen':0,'payout_yen':0})

    for rid in sorted(set(trio)&set(pays)):
        rows=trio[rid]; inv={c:1/o for c,o in rows}; z=sum(inv.values()); p={c:w/z for c,w in inv.items()}
        rider=defaultdict(float)
        for c,prob in p.items():
            for x in c:rider[x]+=prob
        ranked=sorted(rider,key=lambda x:(-rider[x],x)); rrank={x:i+1 for i,x in enumerate(ranked)}; r1=ranked[0]
        # pair marginal trio support
        pair_mass=defaultdict(float)
        for c,prob in p.items():
            for a,b in combinations(c,2):pair_mass[tuple(sorted((a,b)))]+=prob
        # trifecta-set normalized mass
        tri_inv=defaultdict(float)
        for order,o in trif.get(rid,[]):
            tri_inv[tuple(sorted(order))]+=1/o
        tz=sum(tri_inv.values()); tri_p={c:w/tz for c,w in tri_inv.items()} if tz else {}
        candidates=[]
        for c,o in rows:
            if r1 not in c:continue
            others=[x for x in c if x!=r1]
            pm=pair_mass[tuple(sorted(others))]
            ind=sum(rider[x] for x in others)
            trio_prob=p[c]
            cross=tri_p.get(c,0)-trio_prob
            cand={'combo':c,'odds':o,'trio_prob':trio_prob,'pair_mass':pm,'other_ind':ind,'cross':cross,'ranks':tuple(sorted(rrank[x] for x in c))}
            candidates.append(cand)
        if not candidates:continue
        # z-score helpers within race
        keys=['pair_mass','other_ind','trio_prob','cross']
        stats={}
        for key in keys:
            vals=[x[key] for x in candidates]; m=statistics.mean(vals); sd=statistics.pstdev(vals) or 1.0; stats[key]=(m,sd)
        def zval(c,k):m,sd=stats[k];return (c[k]-m)/sd
        scored={
            'support_pair':sorted(candidates,key=lambda c:(-(zval(c,'pair_mass')+zval(c,'other_ind')),c['odds'])),
            'value_balance':sorted(candidates,key=lambda c:(-(zval(c,'pair_mass')+zval(c,'other_ind')-zval(c,'trio_prob')),c['odds'])),
            'cross_value':sorted(candidates,key=lambda c:(-(zval(c,'pair_mass')+zval(c,'other_ind')+zval(c,'cross')-zval(c,'trio_prob')),c['odds'])),
            'market_price':sorted(candidates,key=lambda c:(c['odds'],c['combo'])),
        }
        winning=pays[rid]
        for name,arr in scored.items():
            for n in range(1,6):
                sel=arr[:n]; s=variants[name][n]; s['bet_races']+=1;s['tickets']+=len(sel);s['stake_yen']+=STAKE*len(sel)
                pay=sum(py for wc,py in winning if any(x['combo']==wc for x in sel)); hit=pay>0
                s['hit_races']+=int(hit);s['payout_yen']+=pay;s['hits'].append(hit)
        # raw support-rank combination diagnostics for all candidate trios containing rank1
        for c in candidates:
            key='-'.join(map(str,c['ranks'])); d=rank_combo_diag[key];d['tickets']+=1;d['stake_yen']+=100
            py=sum(py for wc,py in winning if wc==c['combo'])
            if py:d['hits']+=1;d['payout_yen']+=py

    out={'status':'RANK1_INCLUDED_PROFIT_LOGIC_2023','year':2023,'years_read':[2023],'definition':'Fix the individual marginal trio-market support rank1 rider in every candidate trio. Rank all rank1-containing trio combinations using market-only features. Evaluate top-N with actual 3連複 payouts.','variants':{},'rank_combo_diagnostic':{}}
    for name,dd in variants.items():
        out['variants'][name]={}
        for n,s in dd.items():
            s['profit_yen']=s['payout_yen']-s['stake_yen'];s['roi_pct']=100*s['payout_yen']/s['stake_yen'] if s['stake_yen'] else None;s['hit_rate_pct']=100*s['hit_races']/s['bet_races'] if s['bet_races'] else None;s['max_losing_streak']=max_ls(s.pop('hits'))
            out['variants'][name][f'top{n}']=s
    for k,d in rank_combo_diag.items():
        d['profit_yen']=d['payout_yen']-d['stake_yen'];d['roi_pct']=100*d['payout_yen']/d['stake_yen'] if d['stake_yen'] else None;d['ticket_hit_rate_pct']=100*d['hits']/d['tickets'] if d['tickets'] else None
        out['rank_combo_diagnostic'][k]=d
    out['warning']='2023 development only. Strategy variants are evaluated in-sample. No 2024/2025/2026 data read.'
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

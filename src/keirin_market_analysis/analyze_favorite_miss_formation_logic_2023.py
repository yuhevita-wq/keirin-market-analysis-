from __future__ import annotations

import csv, json, math, statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'2023'/'s_class_yosen'
OUT=ROOT/'data'/'audits'/'favorite_miss_formation_logic_2023.json'
FAV_SHARE_MAX=0.17724290354711825
ENTROPY_MIN=0.8115383343655119
STAKE=100


def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def parse_combo(s):
    s=s.strip().replace('=','-').replace(',','-')
    return tuple(sorted(int(x) for x in s.split('-') if x.strip()))

def zscore(vals):
    if not vals:return []
    m=statistics.mean(vals); sd=statistics.pstdev(vals)
    if sd==0:return [0.0]*len(vals)
    return [(x-m)/sd for x in vals]

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
        if o>0:trio[r['race_id']].append((parse_combo(r['combination']),o))
    triord=defaultdict(list)
    for r in read_csv(DATA/'trifecta_final_odds.csv'):
        if r.get('odds_status')!='available':continue
        try:o=float(r['odds'])
        except:continue
        if o<=0:continue
        xs=tuple(int(x) for x in r['combination'].replace('=','-').replace(',','-').split('-') if x.strip())
        if len(xs)==3:triord[r['race_id']].append((xs,o))
    payouts=defaultdict(list)
    for r in read_csv(DATA/'payouts.csv'):
        if r.get('ticket_type')=='3連複' and r.get('status')=='paid':
            payouts[r['race_id']].append((parse_combo(r['combination']),int(r['payout_yen'])))

    races=[]
    for rid in sorted(set(trio)&set(payouts)):
        rows=trio[rid]; inv={c:1/o for c,o in rows}; z=sum(inv.values()); p={c:w/z for c,w in inv.items()}
        fav=min(rows,key=lambda x:(x[1],x[0]))[0]; fs=p[fav]
        ent=-sum(v*math.log(v) for v in p.values())/math.log(len(p)) if len(p)>1 else 0
        if not(fs<=FAV_SHARE_MAX and ent>=ENTROPY_MIN):continue
        rider=defaultdict(float)
        for c,pr in p.items():
            for car in c:rider[car]+=pr
        rr=sorted(rider,key=lambda c:(-rider[c],c)); rank={c:i+1 for i,c in enumerate(rr)}
        # 3連単集合確率 proxy normalized within race
        tri_set_raw=defaultdict(float)
        if rid in triord:
            s=sum(1/o for _,o in triord[rid])
            if s>0:
                for order,o in triord[rid]:tri_set_raw[tuple(sorted(order))]+=(1/o)/s
        candidates=[]
        entrants=rr
        for pair in combinations(entrants,2):
            pair=set(pair)
            for x in entrants:
                if x in pair:continue
                c=tuple(sorted((*pair,x)))
                trio_p=p.get(c,0.0)
                tri_p=tri_set_raw.get(c,0.0)
                delta=tri_p-trio_p
                pair_mass=sum(pr for cc,pr in p.items() if pair.issubset(cc))
                outsider_sup=rider[x]
                pair_sup=sum(rider[a] for a in pair)
                candidates.append({'combo':c,'pair':tuple(sorted(pair)),'x':x,'trio_p':trio_p,'tri_p':tri_p,'delta':delta,'pair_mass':pair_mass,'outsider_sup':outsider_sup,'pair_sup':pair_sup,'outsider_rank':rank[x],'contains_fav2':len(set(c)&set(fav))==2,'contains_fav1':len(set(c)&set(fav))==1})
        # Normalize features within race, define a priori coherent score variants
        feats=['trio_p','tri_p','delta','pair_mass','outsider_sup','pair_sup']
        zmap={f:zscore([c[f] for c in candidates]) for f in feats}
        for i,c in enumerate(candidates):
            c['scores']={
                'market_balance': zmap['pair_mass'][i]+zmap['outsider_sup'][i]-zmap['trio_p'][i],
                'cross_market': zmap['delta'][i]+zmap['pair_mass'][i],
                'price_support': zmap['pair_mass'][i]+zmap['outsider_sup'][i]+zmap['delta'][i]-zmap['trio_p'][i],
                'fav2_cross': (zmap['delta'][i]+zmap['outsider_sup'][i]) if c['contains_fav2'] else -999,
            }
        winners={c:pay for c,pay in payouts[rid]}
        races.append({'rid':rid,'fav':fav,'candidates':candidates,'winners':winners})

    out={'status':'FAVORITE_MISS_FORMATION_LOGIC_2023','year':2023,'years_read':[2023],
         'gate':{'favorite_share_max':FAV_SHARE_MAX,'entropy_min':ENTROPY_MIN},
         'logic_note':'Candidate trios are scored only from final trio/trifecta market features. 2023 outcomes are used only to evaluate score variants and top-N formation sizes; this is in-sample development, not validation.',
         'variants':{}}
    for name in ['market_balance','cross_market','price_support','fav2_cross']:
        v={}
        for topn in [1,2,3,4,5]:
            stake=payout=hits=0; seq=[]; chosen=[]
            for r in races:
                cs=sorted(r['candidates'],key=lambda c:(-c['scores'][name],c['combo']))
                sel=[]
                for c in cs:
                    if c['scores'][name]<=-998:continue
                    if c['combo'] not in [x['combo'] for x in sel]:sel.append(c)
                    if len(sel)>=topn:break
                if not sel:continue
                stake+=100*len(sel)
                pay=sum(r['winners'].get(c['combo'],0) for c in sel)
                payout+=pay; hit=pay>0; hits+=hit; seq.append(hit)
                chosen.append({'race_id':r['rid'],'favorite':list(r['fav']),'tickets':[list(c['combo']) for c in sel],'scores':[c['scores'][name] for c in sel],'payout_yen':pay})
            v[f'top{topn}']={'bet_races':len(seq),'tickets':stake//100,'hit_races':hits,'hit_rate_pct':100*hits/len(seq) if seq else None,'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi_pct':100*payout/stake if stake else None,'max_losing_streak':max_ls(seq)}
        out['variants'][name]=v
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out['variants'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()

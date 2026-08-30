import csv, json, math
from collections import defaultdict
from pathlib import Path

BASE=Path('data/2023/s_class_yosen')
AUD=Path('data/audits')
OUT=AUD/'selected_136_146_2023.json'
COMPARE=AUD/'compare_123_vs_146_nearby_2023.json'
RANK1=AUD/'rider_support_rank1_miss_2023.json'
STAKE=100
PATTERNS=[(1,3,6),(1,4,6)]

def comb(s):
    return tuple(sorted(int(x) for x in str(s).replace('-','').replace(' ','') if x.isdigit()))

def median(xs):
    ys=sorted(xs); n=len(ys)
    return ys[n//2] if n%2 else (ys[n//2-1]+ys[n//2])/2

def max_losing_streak(hits):
    best=cur=0
    for h in hits:
        if h: cur=0
        else:
            cur+=1; best=max(best,cur)
    return best

trios=defaultdict(list)
with open(BASE/'trio_final_odds.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        try:o=float(r.get('odds') or r.get('final_odds') or r.get('trio_odds'))
        except:continue
        c=comb(r.get('combination') or r.get('bet_code') or '')
        if len(c)==3 and o>0: trios[r['race_id']].append((c,o))

wins={}; payouts={}
with open(BASE/'payouts.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if (r.get('ticket_type') or '').strip() not in ('3連複','trio'): continue
        c=comb(r.get('combination') or r.get('bet_code') or '')
        if len(c)!=3: continue
        try:p=int(float(r.get('payout_yen') or 0))
        except:p=0
        if p>0:
            wins[r['race_id']]=c; payouts[(r['race_id'],c)]=p

def features(rows):
    inv=[(c,1/o) for c,o in rows]
    z=sum(v for _,v in inv)
    if z<=0:return None
    p={c:v/z for c,v in inv}
    riders=sorted({x for c in p for x in c})
    sup={x:sum(v for c,v in p.items() if x in c) for x in riders}
    ranked=sorted(sup,key=lambda x:(-sup[x],x)); byrank={i+1:x for i,x in enumerate(ranked)}
    vals=sorted(sup.values(),reverse=True)
    n=len(p)
    entropy=-sum(v*math.log(v) for v in p.values())/math.log(n) if n>1 else 0
    top3_conc=sum(sorted(p.values(),reverse=True)[:3])
    rank1_share=vals[0]/3 if vals else 0
    if all(k in byrank for k in (1,2,3)):
        c123=tuple(sorted(byrank[k] for k in (1,2,3))); p123=p.get(c123,0)
    else:p123=0
    return {'entropy':entropy,'top3_conc':top3_conc,'rank1_share':rank1_share,'p123':p123,'byrank':byrank}

cmp=json.loads(COMPARE.read_text(encoding='utf-8'))
c123=cmp['compare']['1-2-3']; c146=cmp['compare']['1-4-6']
# Primary thresholds are fixed BEFORE this simulation as midpoints between the already observed 123 and 146 medians.
thr={
 'entropy_min':(c123['entropy']['median']+c146['entropy']['median'])/2,
 'top3_conc_max':(c123['top3_conc']['median']+c146['top3_conc']['median'])/2,
 'p123_max':(c123['p_123']['median']+c146['p_123']['median'])/2,
 # previously established weakest-quintile boundary from rank1-miss audit; used only as a safety floor, not optimized here.
 'rank1_share_min':0.24054261443743438,
}

def run(use_rank1_floor):
    total_races=selected=tickets=hits=stake=pay=rank1_miss=win123=0
    hit_seq=[]; selected_rows=[]; pattern_detail={str(p):{'tickets':0,'hits':0,'payout_yen':0} for p in PATTERNS}
    for rid,rows in trios.items():
        if rid not in wins: continue
        total_races+=1
        F=features(rows)
        if not F: continue
        shape=(F['entropy']>=thr['entropy_min'] and F['top3_conc']<=thr['top3_conc_max'] and F['p123']<=thr['p123_max'])
        if not shape: continue
        if use_rank1_floor and F['rank1_share']<thr['rank1_share_min']: continue
        selected+=1
        w=wins[rid]
        rank_of={car:r for r,car in F['byrank'].items()}
        wr=tuple(sorted(rank_of.get(x,99) for x in w))
        if 1 not in wr: rank1_miss+=1
        if wr==(1,2,3): win123+=1
        race_hit=False; race_pay=0
        for pat in PATTERNS:
            if not all(k in F['byrank'] for k in pat): continue
            c=tuple(sorted(F['byrank'][k] for k in pat))
            tickets+=1; stake+=STAKE
            d=pattern_detail[str(pat)]; d['tickets']+=1
            if c==w:
                hits+=1; race_hit=True
                py=payouts.get((rid,c),0); pay+=py; race_pay+=py
                d['hits']+=1; d['payout_yen']+=py
        hit_seq.append(race_hit)
        selected_rows.append({'race_id':rid,'winner_support_ranks':wr,'hit':race_hit,'payout_yen':race_pay,'entropy':F['entropy'],'top3_conc':F['top3_conc'],'rank1_share':F['rank1_share'],'p123':F['p123']})
    for d in pattern_detail.values():
        d['stake_yen']=d['tickets']*STAKE
        d['roi_pct']=100*d['payout_yen']/d['stake_yen'] if d['stake_yen'] else 0
    return {'total_analyzable_races':total_races,'selected_races':selected,'selection_rate_pct':100*selected/total_races if total_races else 0,'tickets':tickets,'hit_races':hits,'race_hit_rate_pct':100*hits/selected if selected else 0,'stake_yen':stake,'payout_yen':pay,'profit_yen':pay-stake,'roi_pct':100*pay/stake if stake else 0,'max_losing_streak':max_losing_streak(hit_seq),'rank1_miss_races':rank1_miss,'rank1_miss_rate_pct':100*rank1_miss/selected if selected else 0,'winner_123_races':win123,'winner_123_rate_pct':100*win123/selected if selected else 0,'by_pattern':pattern_detail,'selected_rows':selected_rows}

out={
 'status':'SELECTED_136_146_2023',
 'year':2023,'years_read':[2023],
 'strategy':'Buy support-rank trios 1-3-6 and 1-4-6, 100 yen each, only in preselected market states.',
 'primary_gate':{
   'definition':'entropy >= midpoint(123 median,146 median) AND top3_conc <= midpoint AND p123 <= midpoint AND rank1_share >= previously established weakest-quintile boundary',
   'thresholds':thr,
   'result':run(True)
 },
 'shape_only_sensitivity':{
   'definition':'Same 123-vs-146 midpoint shape gate, without rank1_share safety floor.',
   'result':run(False)
 },
 'warning':'2023 development/in-sample selection. Thresholds are theory-anchored to precomputed 123-vs-146 medians, but this is NOT out-of-sample validation. No 2024/2025/2026 data read.'
}
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'thresholds':thr,'primary':{k:v for k,v in out['primary_gate']['result'].items() if k!='selected_rows'},'shape_only':{k:v for k,v in out['shape_only_sensitivity']['result'].items() if k!='selected_rows'}},ensure_ascii=False,indent=2))

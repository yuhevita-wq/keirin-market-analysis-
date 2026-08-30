import csv, json, math
from collections import defaultdict, Counter
from pathlib import Path

BASE=Path('data/2023/s_class_yosen')
OUT=Path('data/audits/compare_123_vs_146_nearby_2023.json')

def comb(s): return tuple(sorted(int(x) for x in str(s).replace('-','').replace(' ','') if x.isdigit()))

def stats(xs):
    if not xs: return {'n':0}
    ys=sorted(xs); n=len(ys)
    med=ys[n//2] if n%2 else (ys[n//2-1]+ys[n//2])/2
    return {'n':n,'mean':sum(ys)/n,'median':med,'min':ys[0],'max':ys[-1]}

# trio odds by race
trios=defaultdict(list)
with open(BASE/'trio_final_odds.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        try:
            o=float(r.get('odds') or r.get('final_odds') or r.get('trio_odds'))
        except: continue
        c=comb(r.get('combination') or r.get('bet_code') or '')
        if len(c)==3 and o>0: trios[r['race_id']].append((c,o))
# trifecta set mass
tris=defaultdict(lambda:defaultdict(float))
try:
    with open(BASE/'trifecta_final_odds.csv',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:o=float(r.get('odds') or r.get('final_odds') or r.get('trifecta_odds'))
            except:continue
            c=comb(r.get('combination') or r.get('bet_code') or '')
            if len(c)==3 and o>0: tris[r['race_id']][c]+=1/o
except FileNotFoundError: pass
# winners actual trio payout rows
wins=defaultdict(list); payouts={}
with open(BASE/'payouts.csv',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if (r.get('ticket_type') or '').strip() not in ('3連複','trio'): continue
        if str(r.get('status','')).lower() not in ('','paid','success','確定'): pass
        c=comb(r.get('combination') or r.get('bet_code') or '')
        if len(c)!=3: continue
        try:p=int(float(r.get('payout_yen') or 0))
        except:p=0
        wins[r['race_id']].append(c); payouts[(r['race_id'],c)]=p

def race_features(rid, rows):
    inv=[(c,1/o,o) for c,o in rows]
    z=sum(v for _,v,_ in inv)
    if z<=0:return None
    p={c:v/z for c,v,_ in inv}; odds={c:o for c,_,o in inv}
    riders=sorted({x for c in p for x in c})
    sup={x:sum(v for c,v in p.items() if x in c) for x in riders}
    ranks={x:i+1 for i,(x,_) in enumerate(sorted(sup.items(), key=lambda kv:(-kv[1],kv[0])))}
    byrank={r:x for x,r in ranks.items()}
    n=len(p); ent=-sum(v*math.log(v) for v in p.values())/math.log(n) if n>1 else 0
    top3=sum(sorted(p.values(),reverse=True)[:3])
    vals=sorted(sup.values(),reverse=True)
    out={'entropy':ent,'top3_conc':top3,'rank1_share':vals[0]/3 if vals else 0,
         'rider_top3_sum':sum(vals[:3]),'gap12':vals[0]-vals[1] if len(vals)>1 else 0,
         'ratio21':vals[1]/vals[0] if len(vals)>1 and vals[0] else 0,
         'sup':sup,'ranks':ranks,'byrank':byrank,'p':p,'odds':odds}
    # rank combo trio probabilities and cross-market deltas for key combos
    for key in [(1,2,3),(1,4,6),(1,3,6),(1,3,4),(1,4,5),(1,2,6),(1,4,7)]:
        if all(k in byrank for k in key):
            c=tuple(sorted(byrank[k] for k in key))
            out['p_'+''.join(map(str,key))]=p.get(c,0)
            t=tris.get(rid,{}); tz=sum(t.values()) or 1
            out['delta_'+''.join(map(str,key))]=t.get(c,0)/tz-p.get(c,0)
    return out

groups=defaultdict(list); rank_combo_counts=Counter(); all_rows=[]
for rid,rows in trios.items():
    if rid not in wins: continue
    F=race_features(rid,rows)
    if not F: continue
    for w in wins[rid]:
        ranks=tuple(sorted(F['ranks'].get(x,99) for x in w))
        if 99 in ranks: continue
        label='-'.join(map(str,ranks)); rank_combo_counts[label]+=1
        rec={'race_id':rid,'pattern':label,'payout':payouts.get((rid,w),0)}
        for k in ['entropy','top3_conc','rank1_share','rider_top3_sum','gap12','ratio21','p_123','p_146','p_136','p_134','p_145','p_126','delta_123','delta_146','delta_136','delta_134','delta_145','delta_126']:
            rec[k]=F.get(k,0)
        groups[label].append(rec); all_rows.append(rec)

def summarize(label):
    rs=groups.get(label,[])
    keys=['entropy','top3_conc','rank1_share','rider_top3_sum','gap12','ratio21','p_123','p_146','p_136','p_134','p_145','p_126','delta_123','delta_146','delta_136','delta_134','delta_145','delta_126','payout']
    return {k:stats([r[k] for r in rs]) for k in keys}

# nearby = patterns containing rank1, not 123, and rank-distance to 146
base=(1,4,6)
def dist(p): return sum(abs(a-b) for a,b in zip(sorted(p),base))
near=[]
for label,cnt in rank_combo_counts.items():
    p=tuple(map(int,label.split('-')))
    if 1 not in p or p==(1,2,3): continue
    near.append({'pattern':label,'count':cnt,'distance_to_146':dist(p),'summary':summarize(label)})
near=sorted(near,key=lambda x:(x['distance_to_146'],-x['count']))[:15]

# diagnostic profitability for each rank pattern flat one ticket every race where ranks exist
# derive by recreating candidate and seeing if winner; actual payout
patterns=[]
all_candidate_patterns=[]
for a in range(2,8):
  for b in range(a+1,9):
    all_candidate_patterns.append((1,a,b))
for pat in all_candidate_patterns:
    stake=hits=pay=tickets=0
    for rid,rows in trios.items():
        if rid not in wins: continue
        F=race_features(rid,rows)
        if not F or not all(k in F['byrank'] for k in pat): continue
        c=tuple(sorted(F['byrank'][k] for k in pat)); tickets+=1; stake+=100
        if c in wins[rid]:
            hits+=1; pay+=payouts.get((rid,c),0)
    if tickets:
        patterns.append({'pattern':'-'.join(map(str,pat)),'tickets':tickets,'hits':hits,'hit_rate_pct':100*hits/tickets,'payout_yen':pay,'roi_pct':100*pay/stake if stake else 0,'distance_to_146':dist(pat)})
patterns=sorted(patterns,key=lambda x:(x['distance_to_146'],-x['roi_pct']))

out={'status':'COMPARE_123_VS_146_NEARBY_2023','year':2023,'years_read':[2023],
     'definitions':{'support_rank':'individual marginal trio-market support rank','123':'winning top3 rider support ranks exactly 1-2-3','146':'winning ranks exactly 1-4-6','nearby':'rank patterns containing rank1, sorted by L1 distance to 1-4-6'},
     'counts':dict(rank_combo_counts),'compare':{'1-2-3':summarize('1-2-3'),'1-4-6':summarize('1-4-6')},
     'nearby_patterns':near,'flat_pattern_diagnostic':patterns,
     'warning':'2023 development only. Nearby candidates are structural diagnostics, not validated selections. 2024/2025/2026 not read.'}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'counts_123':rank_combo_counts.get('1-2-3',0),'counts_146':rank_combo_counts.get('1-4-6',0),'near':[x['pattern'] for x in near[:8]]},ensure_ascii=False))

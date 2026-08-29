from __future__ import annotations

import json
from pathlib import Path

from .analyze_multifeature_signals_2023_2025 import YEARS, SEGMENTS, load_year, f
from .research_high_payout_structures_v1 import parse_combination
from .research_weakest_line_2023_2025 import weakest_line
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival

OUT=Path('data/audits/dynamic_formations_2023_2025.json')


def role_vec(e):
    win=f(e.get('win_rate')); t2=f(e.get('top2_rate')); t3=f(e.get('top3_rate'))
    return {
        'score':f(e.get('score')),'win':win,'top2':t2,'top3':t3,
        'attack':f(e.get('nige_count'))+f(e.get('makuri_count')),
        'mass1':max(0.0,win),'mass2':max(0.0,t2-win),'mass3':max(0.0,t3-t2),
    }


def build_rows():
    rows=[]
    for y in YEARS:
        races,eb,tri,seg=load_year(y)
        for race in races:
            rid=race['race_id']; s=seg.get(rid)
            if s not in SEGMENTS or not tri.get(rid): continue
            chosen=choose_main_line(eb[rid])
            if not chosen: continue
            mid,main=chosen
            if len(main)<3: continue
            rival=strongest_rival(eb[rid],mid)
            if not rival or len(rival)<2: continue
            weak,reason=weakest_line(eb[rid],mid)
            if not weak: continue
            wid,wmem,_=weak
            if wid==int(rival[0]['line_id']): continue
            A,B,M3=main[0],main[1],main[2]; W1L,W1B=wmem[0],wmem[1]
            ents={'A':A,'B':B,'M3':M3,'W1L':W1L,'W1B':W1B}
            vec={k:role_vec(v) for k,v in ents.items()}
            roles={int(v['car_no']):k for k,v in ents.items()}
            # add rival/other roles only so non-target winners do not accidentally match
            for i,e in enumerate(rival[:2]): roles[int(e['car_no'])]=('R1L','R1B')[i]
            wins=[]
            for combo,pay in tri[rid]:
                cars=parse_combination(combo); wins.append((tuple(roles.get(c,'OTHER') for c in cars),int(pay)))
            rows.append({'year':y,'segment':s,'race_id':rid,'race_date':race['race_date'],'vec':vec,'wins':wins})
    return rows


def pick_w(r,mode):
    v=r['vec']
    if mode=='BOTH': return ['W1L','W1B']
    key={'TOP3':'top3','TOP2':'top2','WIN':'win','ATTACK':'attack','SCORE':'score'}[mode]
    a,b='W1L','W1B'
    if v[a][key]>v[b][key]: return [a]
    if v[b][key]>v[a][key]: return [b]
    return [a]  # deterministic lead preference on tie


def pick_main_pair(r,mode):
    v=r['vec']; ms=['A','B','M3']
    if mode=='AB': return ('A','B')
    key={'SCORE':'score','TOP2':'top2','TOP3':'top3','WIN':'win'}[mode]
    z=sorted(ms,key=lambda x:(-v[x][key],ms.index(x)))
    return tuple(z[:2])


def pick_slots(r,w,mode):
    if mode=='HEAD': return [0]
    if mode=='SECOND': return [1]
    if mode=='THIRD': return [2]
    m=[r['vec'][w]['mass1'],r['vec'][w]['mass2'],r['vec'][w]['mass3']]
    idx=sorted(range(3),key=lambda i:(-m[i],i))
    if mode=='MASS_BEST': return [idx[0]]
    if mode=='MASS_TOP2': return idx[:2]
    raise KeyError(mode)


def orders_for(r,wmode,mmode,smode):
    out=set(); pair=pick_main_pair(r,mmode)
    for w in pick_w(r,wmode):
        for slot in pick_slots(r,w,smode):
            for q in (pair,pair[::-1]):
                a=[None,None,None]; a[slot]=w; rest=[i for i in range(3) if i!=slot]
                a[rest[0]]=q[0]; a[rest[1]]=q[1]; out.add(tuple(a))
    return out


WMODES=['BOTH','TOP3','TOP2','WIN','ATTACK','SCORE']
MMODES=['AB','SCORE','TOP2','TOP3','WIN']
SMODES=['HEAD','SECOND','THIRD','MASS_BEST','MASS_TOP2']


def evaluate(rows,w,m,s):
    stake=pay=hits=0; hp=[]; flags=[]; pts=[]
    for r in sorted(rows,key=lambda z:(z['race_date'],z['race_id'])):
        orders=orders_for(r,w,m,s); p=len(orders); stake+=100*p; pts.append(p)
        ps=[x for o,x in r['wins'] if o in orders]
        if ps:
            hits+=1; v=sum(ps); pay+=v; hp.append(v); flags.append(True)
        else: flags.append(False)
    cur=mx=0
    for h in flags:
        if h: cur=0
        else: cur+=1; mx=max(mx,cur)
    hps=sorted(hp)
    return {'races':len(rows),'hits':hits,'hit_rate':hits/len(rows) if rows else 0,'stake_yen':stake,'payout_yen':pay,'profit_yen':pay-stake,'roi':pay/stake if stake else 0,
            'avg_points':sum(pts)/len(pts) if pts else 0,'min_points':min(pts) if pts else 0,'max_points':max(pts) if pts else 0,
            'median_hit_payout_yen':hps[len(hps)//2] if hps else 0,'high5000_hits':sum(x>=5000 for x in hps),'high10000_hits':sum(x>=10000 for x in hps),'high20000_hits':sum(x>=20000 for x in hps),
            'top1_payout_share':max(hps)/pay if hps and pay else 0,'max_losing_streak':mx}


def main():
    rows=build_rows(); out={'status':'DYNAMIC_FORMATION_SIM_2023_2025_ONLY','years_read':list(YEARS),'evaluation_year_2026_used':False,
        'method':'No outcome-based race filter. Dynamically choose weakest rider, surviving main pair and weakest finishing slot from pre-race score/rates/attack history. A/B order is always kept both ways.',
        'segments':{}}
    for seg in SEGMENTS:
        sr=[r for r in rows if r['segment']==seg]; cand=[]
        for w in WMODES:
          for m in MMODES:
            for s in SMODES:
              per={str(y):evaluate([r for r in sr if r['year']==y],w,m,s) for y in YEARS}; comb=evaluate(sr,w,m,s)
              worst=min(per[str(y)]['roi'] for y in YEARS); min_hits=min(per[str(y)]['hits'] for y in YEARS)
              stable=all(per[str(y)]['roi']>1 for y in YEARS) and min_hits>=2 and all(per[str(y)]['top1_payout_share']<=.65 for y in YEARS)
              cand.append({'w_selector':w,'main_pair_selector':m,'slot_selector':s,'stable':stable,'worst_year_roi':worst,'min_year_hits':min_hits,'by_year':per,'combined':comb})
        cand.sort(key=lambda x:(x['stable'],x['worst_year_roi'],x['min_year_hits'],x['combined']['roi']),reverse=True)
        out['segments'][seg]={'eligible':len(sr),'stable_count':sum(x['stable'] for x in cand),'top':cand[:30]}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({s:{'eligible':out['segments'][s]['eligible'],'stable_count':out['segments'][s]['stable_count'],'top':[{k:x[k] for k in ('w_selector','main_pair_selector','slot_selector','stable','worst_year_roi','min_year_hits')}|{'combined_roi':x['combined']['roi'],'combined_hits':x['combined']['hits'],'avg_points':x['combined']['avg_points']} for x in out['segments'][s]['top'][:12]]} for s in SEGMENTS},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())

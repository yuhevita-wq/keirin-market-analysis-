from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from .analyze_multifeature_signals_2023_2025 import (
    YEARS, SEGMENTS, load_year, role_numeric, pair_features, add_gap_features,
    bool_signal_features, line_structure,
)
from .research_high_payout_structures_v1 import parse_combination
from .research_weakest_line_2023_2025 import weakest_line
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival

OUT = Path('data/audits/weakest_formation_grid_2023_2025.json')


def f(v):
    try: return float(v)
    except Exception: return 0.0


def build_rows():
    rows=[]
    for y in YEARS:
        races, eb, tri, seg = load_year(y)
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
            A,B,M3=main[0],main[1],main[2]
            R1L,R1B=rival[0],rival[1]
            W1L,W1B=wmem[0],wmem[1]
            x={'year':y,'segment':s,'race_id':rid,'race_date':race['race_date'],'weak_len':len(wmem),'main_len':len(main),'rival_len':len(rival)}
            x.update(line_structure(eb[rid]))
            for n,e in [('a',A),('b',B),('r1l',R1L),('r1b',R1B),('w1l',W1L),('w1b',W1B)]: x.update(role_numeric(n,e))
            x.update(pair_features('main',A,B)); x.update(pair_features('rival',R1L,R1B)); x.update(pair_features('weak',W1L,W1B))
            add_gap_features(x,'main','rival','main_rival'); add_gap_features(x,'rival','weak','rival_weak'); add_gap_features(x,'main','weak','main_weak')
            x.update(bool_signal_features(x))
            x.update({
                'weak_hidden_top2_vs_rival': x['weak_score'] < x['rival_score'] and x['weak_top2_rate'] >= x['rival_top2_rate'],
                'weak_hidden_top3_vs_rival': x['weak_score'] < x['rival_score'] and x['weak_top3_rate'] >= x['rival_top3_rate'],
                'weak_hidden_attack_vs_rival': x['weak_score'] < x['rival_score'] and x['weak_attack'] >= x['rival_attack'],
            })
            roles={int(e['car_no']):n for n,e in [('A',A),('B',B),('M3',M3),('R1L',R1L),('R1B',R1B),('W1L',W1L),('W1B',W1B)]}
            wins=[]
            for combo,pay in tri[rid]:
                cars=parse_combination(combo)
                wins.append((tuple(roles.get(c,'OTHER') for c in cars),int(pay)))
            x['wins']=wins
            rows.append(x)
    return rows


PAIRSETS={
    'AB':[('A','B')],
    'AM3':[('A','M3')],
    'BM3':[('B','M3')],
    'MAIN_ANY2':[('A','B'),('A','M3'),('B','M3')],
}
WSETS={'W1L':['W1L'],'W1B':['W1B'],'WBOTH':['W1L','W1B']}
SLOTS={'HEAD':[0],'SECOND':[1],'THIRD':[2],'ANY':[0,1,2],'HEAD_THIRD':[0,2]}


def formation_orders(wset,pairset,slots):
    out=set()
    for w in WSETS[wset]:
        for p in PAIRSETS[pairset]:
            for slot in SLOTS[slots]:
                for q in (p,p[::-1]):
                    a=[None,None,None]; a[slot]=w
                    rest=[i for i in range(3) if i!=slot]
                    a[rest[0]]=q[0]; a[rest[1]]=q[1]
                    out.add(tuple(a))
    return out


FORMS={}
for ws,ps,ss in itertools.product(WSETS,PAIRSETS,SLOTS):
    orders=formation_orders(ws,ps,ss)
    FORMS[f'{ws}_{ps}_{ss}']={'orders':orders,'points':len(orders),'wset':ws,'pairset':ps,'slots':ss}


# Single pre-race conditions only. Coarse cuts are deliberately rounded around the stable 2023-2025 descriptive quartiles.
CONDS={
    'ALL': lambda r: True,
    'WEAK_TOP2_GE50': lambda r: r['weak_top2_rate']>=50,
    'WEAK_TOP3_GE75': lambda r: r['weak_top3_rate']>=75,
    'WEAK_WIN_GE25': lambda r: r['weak_win_rate']>=25,
    'W1L_TOP2_GE28': lambda r: r['w1l_top2_rate']>=28,
    'W1L_TOP3_GE40': lambda r: r['w1l_top3_rate']>=40,
    'W1B_WIN_GE10': lambda r: r['w1b_win_rate']>=10,
    'WEAK_NIGE_GE3': lambda r: r['weak_nige_count']>=3,
    'WEAK_B_GE9': lambda r: r['weak_b_count']>=9,
    'W1L_ATTACK_GE5': lambda r: r['w1l_attack']>=5,
    'RIVAL_WEAK_SCORE_GAP_LE4': lambda r: r['rival_weak_score_gap']<=4,
    'RIVAL_WEAK_SCORE_GAP_LE8': lambda r: r['rival_weak_score_gap']<=8,
    'RIVAL_WEAK_TOP2_GAP_LE0': lambda r: r['rival_weak_top2_rate_gap']<=0,
    'RIVAL_WEAK_TOP2_GAP_LE10': lambda r: r['rival_weak_top2_rate_gap']<=10,
    'MAIN_WEAK_TOP2_GAP_LE20': lambda r: r['main_weak_top2_rate_gap']<=20,
    'HIDDEN_TOP2': lambda r: bool(r['weak_hidden_top2_vs_rival']),
    'HIDDEN_TOP3': lambda r: bool(r['weak_hidden_top3_vs_rival']),
    'HIDDEN_ATTACK': lambda r: bool(r['weak_hidden_attack_vs_rival']),
    'THREE_PLUS_LINES': lambda r: r['line_count_2plus']>=3,
}


def max_streak(seq):
    cur=mx=0
    for hit in seq:
        if hit: cur=0
        else: cur+=1; mx=max(mx,cur)
    return mx


def evaluate(rows,orders):
    rows=sorted(rows,key=lambda r:(r['race_date'],r['race_id']))
    stake=len(rows)*len(orders)*100; pay=0; hits=0; hitp=[]; hitflags=[]
    for r in rows:
        ps=[p for o,p in r['wins'] if o in orders]
        if ps:
            hits+=1; v=sum(ps); pay+=v; hitp.append(v); hitflags.append(True)
        else: hitflags.append(False)
    hp=sorted(hitp)
    return {
        'races':len(rows),'points':len(orders),'hits':hits,'hit_rate':hits/len(rows) if rows else 0,
        'stake_yen':stake,'payout_yen':pay,'profit_yen':pay-stake,'roi':pay/stake if stake else 0,
        'median_hit_payout_yen':hp[len(hp)//2] if hp else 0,
        'mean_hit_payout_yen':sum(hp)/len(hp) if hp else 0,
        'high5000_hits':sum(x>=5000 for x in hp),'high10000_hits':sum(x>=10000 for x in hp),'high20000_hits':sum(x>=20000 for x in hp),
        'top1_payout_share':max(hp)/pay if hp and pay else 0,
        'max_losing_streak':max_streak(hitflags),
    }


def main():
    rows=build_rows(); out={'status':'WEAKEST_FORMATION_GRID_2023_2025_ONLY','years_read':list(YEARS),'evaluation_year_2026_used':False,
      'method':'Formation-first development simulation. W role, W finishing slot, surviving main pair and breadth are enumerated. Conditions are single coarse pre-race atoms only; no two-condition optimization.',
      'formation_count':len(FORMS),'condition_count':len(CONDS),'segments':{}}
    for s in SEGMENTS:
        sr=[r for r in rows if r['segment']==s]; candidates=[]
        for cname,cf in CONDS.items():
            cr=[r for r in sr if cf(r)]
            for fname,fd in FORMS.items():
                per={str(y):evaluate([r for r in cr if r['year']==y],fd['orders']) for y in YEARS}
                comb=evaluate(cr,fd['orders'])
                min_r=min(per[str(y)]['races'] for y in YEARS); min_h=min(per[str(y)]['hits'] for y in YEARS); worst=min(per[str(y)]['roi'] for y in YEARS)
                stable=(min_r>=20 and min_h>=2 and all(per[str(y)]['roi']>1 for y in YEARS) and all(per[str(y)]['top1_payout_share']<=0.65 for y in YEARS) and all(per[str(y)]['high5000_hits']>=1 for y in YEARS))
                candidates.append({'condition':cname,'formation':fname,'points':fd['points'],'wset':fd['wset'],'pairset':fd['pairset'],'slots':fd['slots'],
                    'stable':stable,'worst_year_roi':worst,'min_year_races':min_r,'min_year_hits':min_h,'by_year':per,'combined':comb})
        candidates.sort(key=lambda x:(x['stable'],x['worst_year_roi'],x['min_year_hits'],x['combined']['roi'],-x['points']),reverse=True)
        stable=[x for x in candidates if x['stable']]
        # also retain best unconditional formations to understand formation economics before filtering
        uncond=[x for x in candidates if x['condition']=='ALL']
        uncond.sort(key=lambda x:(x['worst_year_roi'],x['combined']['roi']),reverse=True)
        out['segments'][s]={'eligible_rows':len(sr),'stable_count':len(stable),'top_stable':stable[:30],'top_unconditional':uncond[:20]}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({s:{'eligible':out['segments'][s]['eligible_rows'],'stable_count':out['segments'][s]['stable_count'],'top_stable':[{k:x[k] for k in ('condition','formation','points','worst_year_roi','min_year_races','min_year_hits')}|{'combined_roi':x['combined']['roi'],'combined_hits':x['combined']['hits']} for x in out['segments'][s]['top_stable'][:10]],'top_unconditional':[{k:x[k] for k in ('formation','points','worst_year_roi')}|{'combined_roi':x['combined']['roi'],'combined_hits':x['combined']['hits']} for x in out['segments'][s]['top_unconditional'][:8]]} for s in SEGMENTS},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())

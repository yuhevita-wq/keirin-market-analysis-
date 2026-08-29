from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from .analyze_multifeature_signals_2023_2025 import (
    YEARS, SEGMENTS, f, line_structure, pair_features, role_numeric,
    add_gap_features, load_year, quantile_cut,
)
from .research_high_payout_structures_v1 import parse_combination
from .research_weakest_line_2023_2025 import weakest_line
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival

OUT = Path('data/audits/formation_slot_simulation_2023_2025.json')

ATOMS = {
    'L1': {('W1L','A','B'), ('W1L','B','A')},
    'L2': {('A','W1L','B'), ('B','W1L','A')},
    'L3': {('A','B','W1L'), ('B','A','W1L')},
    'B1': {('W1B','A','B'), ('W1B','B','A')},
    'B2': {('A','W1B','B'), ('B','W1B','A')},
    'B3': {('A','B','W1B'), ('B','A','W1B')},
}


def make_rows():
    rows=[]
    for year in YEARS:
        races, eb, tri, seg = load_year(year)
        for race in races:
            rid=race['race_id']; s=seg.get(rid)
            if s not in SEGMENTS or not tri.get(rid):
                continue
            chosen=choose_main_line(eb[rid])
            if not chosen: continue
            main_id, main=chosen
            if len(main)<3: continue
            rival=strongest_rival(eb[rid],main_id)
            if not rival or len(rival)<2: continue
            weak,reason=weakest_line(eb[rid],main_id)
            if not weak: continue
            weak_id,wmem,_=weak
            if int(rival[0]['line_id'])==weak_id: continue
            A,B,M3=main[0],main[1],main[2]
            R1L,R1B=rival[0],rival[1]
            W1L,W1B=wmem[0],wmem[1]
            x={'year':year,'segment':s,'race_id':rid,'race_date':race['race_date'],'track':race['track'],'race_no':int(race['race_no'])}
            x.update(line_structure(eb[rid]))
            for name,e in (('a',A),('b',B),('r1l',R1L),('r1b',R1B),('w1l',W1L),('w1b',W1B)):
                x.update(role_numeric(name,e))
            x.update(pair_features('main',A,B)); x.update(pair_features('rival',R1L,R1B)); x.update(pair_features('weak',W1L,W1B))
            add_gap_features(x,'main','rival','main_rival'); add_gap_features(x,'rival','weak','rival_weak'); add_gap_features(x,'main','weak','main_weak')
            x['weak_hidden_attack_vs_rival']=x['weak_score']<x['rival_score'] and x['weak_attack']>=x['rival_attack']
            x['weak_hidden_top2_vs_rival']=x['weak_score']<x['rival_score'] and x['weak_top2_rate']>=x['rival_top2_rate']
            x['weak_hidden_top3_vs_rival']=x['weak_score']<x['rival_score'] and x['weak_top3_rate']>=x['rival_top3_rate']
            roles={int(A['car_no']):'A',int(B['car_no']):'B',int(M3['car_no']):'M3',int(R1L['car_no']):'R1L',int(R1B['car_no']):'R1B',int(W1L['car_no']):'W1L',int(W1B['car_no']):'W1B'}
            wins=[]
            for combo,payout in tri[rid]:
                order=tuple(roles.get(c,'OTHER') for c in parse_combination(combo))
                wins.append({'order':order,'payout_yen':payout})
            x['wins']=wins
            rows.append(x)
    return rows


def pooled_thresholds(rows):
    specs={
        'weak_top2_rate':('HIGH',.75),
        'weak_top3_rate':('HIGH',.75),
        'w1l_attack':('HIGH',.75),
        'weak_b_count':('HIGH',.75),
        'rival_weak_attack_gap':('LOW',.25),
        'w1l_top3_rate':('HIGH',.75),
        'w1b_top3_rate':('HIGH',.75),
    }
    out={}
    for s in SEGMENTS:
        sr=[r for r in rows if r['segment']==s]
        out[s]={}
        for feat,(direction,q) in specs.items():
            vals=[float(r[feat]) for r in sr]
            out[s][feat]={'direction':direction,'q':q,'cut':quantile_cut(vals,q)}
    return out


def gate_defs(th):
    gates={}
    for s in SEGMENTS:
        t=th[s]
        gates[s]={
            'ALL': lambda r: True,
            'HIDDEN_ATTACK': lambda r: bool(r['weak_hidden_attack_vs_rival']),
            'HIDDEN_TOP2': lambda r: bool(r['weak_hidden_top2_vs_rival']),
            'HIDDEN_TOP3': lambda r: bool(r['weak_hidden_top3_vs_rival']),
            'WEAK_TOP2_HIGH': lambda r,c=t['weak_top2_rate']['cut']: r['weak_top2_rate']>=c,
            'WEAK_TOP3_HIGH': lambda r,c=t['weak_top3_rate']['cut']: r['weak_top3_rate']>=c,
            'W1L_ATTACK_HIGH': lambda r,c=t['w1l_attack']['cut']: r['w1l_attack']>=c,
            'WEAK_B_HIGH': lambda r,c=t['weak_b_count']['cut']: r['weak_b_count']>=c,
            'RW_ATTACK_GAP_LOW': lambda r,c=t['rival_weak_attack_gap']['cut']: r['rival_weak_attack_gap']<=c,
            'W1L_TOP3_HIGH': lambda r,c=t['w1l_top3_rate']['cut']: r['w1l_top3_rate']>=c,
            'W1B_TOP3_HIGH': lambda r,c=t['w1b_top3_rate']['cut']: r['w1b_top3_rate']>=c,
        }
    return gates


def masks():
    names=list(ATOMS)
    out=[]
    for k in range(1,5):
        for combo in itertools.combinations(names,k):
            orders=set()
            for a in combo: orders |= ATOMS[a]
            out.append({'atoms':combo,'orders':orders,'points':len(orders)})
    return out


def calc(rows,orders,points):
    rows=sorted(rows,key=lambda r:(r['race_date'],r['track'],r['race_no']))
    stake=points*100*len(rows); payout=0; race_hits=0; winning_tickets=0; hit_pays=[]; cur=mx=0
    half=defaultdict(lambda:{'races':0,'hits':0,'stake':0,'payout':0})
    for r in rows:
        key=f"{r['year']}-H{1 if int(r['race_date'][5:7])<=6 else 2}"
        half[key]['races']+=1; half[key]['stake']+=points*100
        rp=0; wt=0
        for w in r['wins']:
            if w['order'] in orders:
                rp += int(w['payout_yen']); wt += 1
        if wt:
            race_hits+=1; winning_tickets+=wt; payout+=rp; hit_pays.append(rp); cur=0
            half[key]['hits']+=1; half[key]['payout']+=rp
        else:
            cur+=1; mx=max(mx,cur)
    hit_pays.sort()
    top1=max(hit_pays) if hit_pays else 0
    top3=sum(sorted(hit_pays,reverse=True)[:3]) if hit_pays else 0
    return {
        'races':len(rows),'points':points,'race_hits':race_hits,'winning_tickets':winning_tickets,
        'hit_rate':race_hits/len(rows) if rows else 0.0,
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi':payout/stake if stake else 0.0,
        'median_hit_payout_yen':hit_pays[len(hit_pays)//2] if hit_pays else 0,
        'mean_hit_payout_yen':sum(hit_pays)/len(hit_pays) if hit_pays else 0.0,
        'ge5000_hits':sum(x>=5000 for x in hit_pays),'ge10000_hits':sum(x>=10000 for x in hit_pays),'ge20000_hits':sum(x>=20000 for x in hit_pays),
        'max_losing_streak':mx,'top1_payout_share':top1/payout if payout else 0.0,'top3_payout_share':top3/payout if payout else 0.0,
        'half_years':{k:{**v,'profit':v['payout']-v['stake'],'roi':v['payout']/v['stake'] if v['stake'] else 0.0} for k,v in sorted(half.items())},
    }


def compact_metric(m):
    return {k:m[k] for k in ('races','points','race_hits','hit_rate','stake_yen','payout_yen','profit_yen','roi','median_hit_payout_yen','ge10000_hits','ge20000_hits','max_losing_streak','top1_payout_share','top3_payout_share')}


def main():
    rows=make_rows(); th=pooled_thresholds(rows); gates=gate_defs(th); ms=masks()
    candidates=[]
    atomic={}
    for s in SEGMENTS:
        sr=[r for r in rows if r['segment']==s]
        atomic[s]={}
        for atom,orders in ATOMS.items():
            atomic[s][atom]={str(y):compact_metric(calc([r for r in sr if r['year']==y],orders,2)) for y in YEARS}
            atomic[s][atom]['combined']=compact_metric(calc(sr,orders,2))
        for gname,gfn in gates[s].items():
            gr=[r for r in sr if gfn(r)]
            for m in ms:
                per={str(y):calc([r for r in gr if r['year']==y],m['orders'],m['points']) for y in YEARS}
                comb=calc(gr,m['orders'],m['points'])
                if min(per[str(y)]['races'] for y in YEARS)<15: continue
                worst=min(per[str(y)]['roi'] for y in YEARS)
                profitable_years=sum(per[str(y)]['roi']>1 for y in YEARS)
                hits_each=min(per[str(y)]['race_hits'] for y in YEARS)
                profitable_halves=sum(v['roi']>1 for y in YEARS for v in per[str(y)]['half_years'].values() if v['races'])
                candidates.append({
                    'segment':s,'gate':gname,'atoms':list(m['atoms']),'points':m['points'],
                    'by_year':{str(y):compact_metric(per[str(y)]) for y in YEARS},'combined':compact_metric(comb),
                    'worst_year_roi':worst,'profitable_years':profitable_years,'min_year_hits':hits_each,'profitable_half_years':profitable_halves,
                })
    candidates.sort(key=lambda c:(c['profitable_years']==3,c['worst_year_roi'],c['min_year_hits'],c['combined']['roi'],c['combined']['race_hits']),reverse=True)
    strict=[c for c in candidates if c['profitable_years']==3 and c['min_year_hits']>=2 and c['combined']['race_hits']>=10 and c['combined']['top1_payout_share']<=0.6]
    out={
        'status':'FORMATION_SLOT_SIMULATION_2023_2025_ONLY','years_read':list(YEARS),'evaluation_year_2026_used':False,
        'method':{
            'atom':'2 tickets: one W role fixed to one finishing position, with A/B order both ways',
            'atoms':{k:['-'.join(x) for x in sorted(v)] for k,v in ATOMS.items()},
            'formation_sizes_tested_points':[2,4,6,8],
            'gates':'ALL plus pre-race descriptive signals from the prior 2023-2025 multi-feature audit; pooled quartile cuts by segment',
            'payout_accounting':'sum every paid 3連単 winning ticket that is included in the formation, including dead-heat multiple winning tickets',
            'not_strategy_freeze':True,
        },
        'thresholds':th,
        'eligible_population':{str(y):{s:sum(r['year']==y and r['segment']==s for r in rows) for s in SEGMENTS} for y in YEARS},
        'atomic_slots':atomic,
        'strict_candidate_count':len(strict),
        'top_strict_candidates':strict[:30],
        'top_overall_candidates':candidates[:30],
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({
        'status':out['status'],'eligible_population':out['eligible_population'],'thresholds':th,'strict_candidate_count':len(strict),
        'top_strict':[{k:c[k] for k in ('segment','gate','atoms','points','worst_year_roi','profitable_years','min_year_hits','profitable_half_years')}|{'combined':c['combined']} for c in strict[:12]],
        'atomic':atomic,
    },ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())

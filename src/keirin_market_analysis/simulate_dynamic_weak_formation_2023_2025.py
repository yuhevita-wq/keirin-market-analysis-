from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from .analyze_multifeature_signals_2023_2025 import build_rows

YEARS=(2023,2024,2025)
SEGMENTS=('前半','中盤','後半')
OUT=Path('data/audits/dynamic_weak_formation_sim_2023_2025.json')

# Each slot is always 2 points because A/B order is retained both ways.
SLOTS={
    'W1L_HEAD':{('W1L','A','B'),('W1L','B','A')},
    'W1L_SECOND':{('A','W1L','B'),('B','W1L','A')},
    'W1L_THIRD':{('A','B','W1L'),('B','A','W1L')},
    'W1B_HEAD':{('W1B','A','B'),('W1B','B','A')},
    'W1B_SECOND':{('A','W1B','B'),('B','W1B','A')},
    'W1B_THIRD':{('A','B','W1B'),('B','A','W1B')},
}

# Predeclared feature states come directly from the descriptive 2023-2025 audit.
# They are deliberately coarse and are not optimized on ROI here.
def signal_states(r):
    out=set(['ALL'])
    s=r['segment']
    if s=='前半':
        if r.get('weak_top2_rate',0)>=52.3: out.add('WEAK_TOP2_HIGH')
        if r.get('weak_top3_rate',0)>=78.6: out.add('WEAK_TOP3_HIGH')
        if r.get('weak_win_rate',0)>=26.6: out.add('WEAK_WIN_HIGH')
        if r.get('rival_weak_score_gap',999)<=3.09: out.add('RIVAL_WEAK_SCORE_CLOSE')
        if r.get('weak_hidden_attack_vs_rival'): out.add('HIDDEN_ATTACK')
        if r.get('weak_hidden_top2_vs_rival'): out.add('HIDDEN_TOP2')
        if r.get('weak_hidden_top3_vs_rival'): out.add('HIDDEN_TOP3')
    elif s=='中盤':
        if r.get('w1l_top3_rate',0)>=40.7: out.add('W1L_TOP3_HIGH')
        if r.get('w1l_top2_rate',0)>=28.5: out.add('W1L_TOP2_HIGH')
        if r.get('weak_top2_rate',0)>=49.925: out.add('WEAK_TOP2_HIGH')
        if r.get('main_weak_top2_rate_gap',999)<=15.2: out.add('MAIN_WEAK_TOP2_CLOSE')
        if r.get('rival_weak_score_gap',999)<=3.3475: out.add('RIVAL_WEAK_SCORE_CLOSE')
        if r.get('weak_nige_count',0)>=3: out.add('WEAK_NIGE_HIGH')
        if r.get('weak_b_count',0)>=9: out.add('WEAK_B_HIGH')
        if r.get('w1l_attack',0)>=5: out.add('W1L_ATTACK_HIGH')
        if r.get('weak_hidden_attack_vs_rival'): out.add('HIDDEN_ATTACK')
        if r.get('weak_hidden_top2_vs_rival'): out.add('HIDDEN_TOP2')
    else:
        if r.get('weak_top3_rate',0)>=73.6: out.add('WEAK_TOP3_HIGH')
        if r.get('rival_weak_score_gap',999)<=2.79: out.add('RIVAL_WEAK_SCORE_CLOSE')
        if r.get('rival_weak_attack_gap',999)<=-1: out.add('WEAK_ATTACK_AT_LEAST_RIVAL')
        if r.get('weak_hidden_attack_vs_rival'): out.add('HIDDEN_ATTACK')
        if r.get('weak_hidden_top2_vs_rival'): out.add('HIDDEN_TOP2')
    return out


def slot_stats(rows, orders):
    n=len(rows); hits=[]
    for r in rows:
        if tuple(r['role_order']) in orders:
            hits.append(r['payout_yen'])
    stake=n*len(orders)*100
    payout=sum(hits)
    return {
        'races':n,'points':len(orders),'hits':len(hits),'hit_rate':len(hits)/n if n else 0.0,
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi':payout/stake if stake else 0.0,
        'median_hit_payout_yen':sorted(hits)[len(hits)//2] if hits else 0,
        'high5000_hits':sum(x>=5000 for x in hits),'high10000_hits':sum(x>=10000 for x in hits),'high20000_hits':sum(x>=20000 for x in hits),
    }


def union_stats(rows, slot_names):
    orders=set().union(*(SLOTS[n] for n in slot_names))
    return slot_stats(rows,orders)


def main():
    _, weakrows, _, _ = build_rows()
    # Attach the actual role order/payout fields already present in build_rows.
    # build_rows returns only pre-race features plus result labels, all from 2023-2025.
    by=defaultdict(list)
    for r in weakrows:
        by[(r['year'],r['segment'])].append(r)

    signals=defaultdict(set)
    for r in weakrows:
        for sig in signal_states(r): signals[r['segment']].add(sig)

    out={
        'status':'DYNAMIC_WEAKEST_FORMATION_SIM_2023_2025_ONLY',
        'years_read':list(YEARS),'evaluation_year_2026_used':False,
        'method':'2-point positional slots; A/B order always retained. Signal thresholds are fixed from prior descriptive quartile audit; no ROI-derived threshold tuning in this script.',
        'slot_definitions':{k:sorted('-'.join(x) for x in v) for k,v in SLOTS.items()},
        'segments':{}
    }

    for s in SEGMENTS:
        seg={}
        # First: raw slot anatomy with no filter.
        raw={}
        allrows=[r for y in YEARS for r in by[(y,s)]]
        for slot,orders in SLOTS.items():
            raw[slot]={
                'combined':slot_stats(allrows,orders),
                'by_year':{str(y):slot_stats(by[(y,s)],orders) for y in YEARS},
            }
        seg['raw_slots']=raw

        # Second: each descriptive signal x each positional slot.
        ss={}
        for sig in sorted(signals[s]):
            selected={y:[r for r in by[(y,s)] if sig in signal_states(r)] for y in YEARS}
            if min(len(selected[y]) for y in YEARS)<20: continue
            ss[sig]={}
            allsel=[r for y in YEARS for r in selected[y]]
            for slot,orders in SLOTS.items():
                ss[sig][slot]={
                    'combined':slot_stats(allsel,orders),
                    'by_year':{str(y):slot_stats(selected[y],orders) for y in YEARS},
                }
        seg['signal_slots']=ss

        # Third: compact candidate formations assembled by opening 1,2,or3 slots.
        # We only consider slot combinations within W1L/W1B + A/B, capped at 6 points.
        cand=[]
        slot_names=list(SLOTS)
        for sig,block in ss.items():
            selected={y:[r for r in by[(y,s)] if sig in signal_states(r)] for y in YEARS}
            if min(len(selected[y]) for y in YEARS)<20: continue
            allsel=[r for y in YEARS for r in selected[y]]
            for k in (1,2,3):
                for combo in itertools.combinations(slot_names,k):
                    per={str(y):union_stats(selected[y],combo) for y in YEARS}
                    comb=union_stats(allsel,combo)
                    cand.append({
                        'signal':sig,'slots':list(combo),'points':2*k,
                        'combined':comb,'by_year':per,
                        'worst_year_roi':min(per[str(y)]['roi'] for y in YEARS),
                        'min_year_hits':min(per[str(y)]['hits'] for y in YEARS),
                        'all_years_profitable':all(per[str(y)]['roi']>1 for y in YEARS),
                    })
        cand.sort(key=lambda x:(x['all_years_profitable'],x['worst_year_roi'],x['min_year_hits'],x['combined']['roi']),reverse=True)
        seg['top_candidates']=cand[:40]
        out['segments'][s]=seg

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({s:out['segments'][s]['top_candidates'][:8] for s in SEGMENTS},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())

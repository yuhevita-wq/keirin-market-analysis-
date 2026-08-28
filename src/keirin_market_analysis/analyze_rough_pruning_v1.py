from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_branching_v1 import classify, mainline_bets, read_csv, strongest_rival
from .simulate_mainline_v1 import choose_main_line, segment_for

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'rough_pruning_v1'


def role_map(entries, main_id, members):
    rival = strongest_rival(entries, main_id)
    if not rival:
        return None
    return {
        'A': int(members[0]['car_no']),
        'B': int(members[1]['car_no']),
        'M3': int(members[2]['car_no']),
        'R1L': int(rival[0]['car_no']),
        'R1B': int(rival[1]['car_no']),
    }


def combos(role_to_car, p1, p2, p3):
    a=[role_to_car[r] for r in p1]
    b=[role_to_car[r] for r in p2]
    c=[role_to_car[r] for r in p3]
    return sorted({f'{x}-{y}-{z}' for x in a for y in b for z in c if len({x,y,z})==3})


CANDIDATES = {
    'P27_base': (['A','B','R1L'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B']),
    'P18_first_A_R1L': (['A','R1L'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B']),
    'P18_first_A_B': (['A','B'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B']),
    'P09_first_A': (['A'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B']),
    'P18_drop_R1B_second': (['A','B','R1L'], ['A','B','R1L'], ['A','B','M3','R1L','R1B']),
    'P21_drop_A_third': (['A','B','R1L'], ['A','B','R1L','R1B'], ['B','M3','R1L','R1B']),
    'P21_drop_B_third': (['A','B','R1L'], ['A','B','R1L','R1B'], ['A','M3','R1L','R1B']),
    'P21_drop_R1L_third': (['A','B','R1L'], ['A','B','R1L','R1B'], ['A','B','M3','R1B']),
    'P21_drop_R1B_third': (['A','B','R1L'], ['A','B','R1L','R1B'], ['A','B','M3','R1L']),
}


def summarize(records, candidate):
    p1,p2,p3=CANDIDATES[candidate]
    stake=payout=hits=points=0
    for r in records:
        cs=combos(r['roles'],p1,p2,p3)
        points += len(cs)
        stake += len(cs)*100
        ret=sum(r['payouts'].get(c,0) for c in cs)
        payout += ret
        hits += int(ret>0)
    n=len(records)
    return {
        'races':n,
        'avg_points':points/n if n else 0,
        'stake_yen':stake,
        'payout_yen':payout,
        'profit_yen':payout-stake,
        'roi':payout/stake if stake else 0,
        'hit_races':hits,
        'hit_rate':hits/n if n else 0,
    }


def main():
    races=read_csv(DATA_DIR/'races.csv')
    entries=read_csv(DATA_DIR/'entries.csv')
    payouts=read_csv(DATA_DIR/'payouts.csv')
    e_by=defaultdict(list); p_by=defaultdict(dict); day=defaultdict(list)
    for e in entries: e_by[e['race_id']].append(e)
    for p in payouts:
        if p.get('ticket_type')=='3連単' and p.get('status')=='paid' and p.get('combination'):
            try: p_by[p['race_id']][p['combination']]=int(p['payout_yen'])
            except ValueError: pass
    for r in races: day[(r['race_date'],r['track'])].append(r)
    seg={}
    for group in day.values():
        group.sort(key=lambda r:int(r['race_no']))
        for pos,r in enumerate(group,1): seg[r['race_id']]=segment_for(pos,len(group))

    rough=[]; mainline=[]
    for race in sorted(races,key=lambda r:(r['race_date'],r['track'],int(r['race_no']))):
        rid=race['race_id']
        if seg.get(rid)!='後半': continue
        es=e_by[rid]; main=choose_main_line(es)
        if not main: continue
        main_id,members=main
        if len(members)<3: continue
        branch,*_=classify(members)
        half='H1' if race['race_date']<='2025-06-30' else 'H2'
        if branch=='B_rough':
            roles=role_map(es,main_id,members)
            if roles: rough.append({'half':half,'roles':roles,'payouts':p_by[rid]})
        elif branch=='A_mainline':
            cs=mainline_bets(es,members)
            ret=sum(p_by[rid].get(c,0) for c in cs)
            mainline.append({'half':half,'stake':len(cs)*100,'payout':ret,'hit':int(ret>0)})

    out={'scope':'branch B pruning; candidate rules predeclared before payout comparison','candidates':{}}
    for name in CANDIDATES:
        allm=summarize(rough,name)
        h1=summarize([r for r in rough if r['half']=='H1'],name)
        h2=summarize([r for r in rough if r['half']=='H2'],name)
        out['candidates'][name]={'all':allm,'H1':h1,'H2':h2,'formation':CANDIDATES[name]}

    # Discovery rule: on H1, require >= 50% hit rate and choose the fewest average points;
    # tie-break higher H1 ROI, then higher H1 hit rate. Freeze before reading H2 in interpretation.
    eligible=[]
    for name,d in out['candidates'].items():
        h1=d['H1']
        if h1['hit_rate']>=0.50:
            eligible.append((h1['avg_points'],-h1['roi'],-h1['hit_rate'],name))
    selected=sorted(eligible)[0][-1] if eligible else 'P27_base'
    out['selection_rule']='Using H1 only: among predeclared candidates with H1 hit rate >=50%, choose fewest points; tie-break higher H1 ROI then hit rate.'
    out['selected_from_H1']=selected

    # Full strategy metrics for selected candidate, with A branch unchanged and C skipped.
    for half in ('H1','H2','ALL'):
        mr=[r for r in mainline if half=='ALL' or r['half']==half]
        rr=[r for r in rough if half=='ALL' or r['half']==half]
        rm=summarize(rr,selected)
        mstake=sum(r['stake'] for r in mr); mpayout=sum(r['payout'] for r in mr); mhits=sum(r['hit'] for r in mr)
        stake=mstake+rm['stake_yen']; payout=mpayout+rm['payout_yen']; bets=len(mr)+len(rr); hits=mhits+rm['hit_races']
        out.setdefault('full_strategy_selected',{})[half]={
            'bet_races':bets,'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,
            'roi':payout/stake if stake else 0,'hit_races':hits,'hit_rate':hits/bets if bets else 0,
            'rough_candidate':selected,
        }

    out['warning']='H1 is discovery. H2 is the cleaner temporal validation. Do not reselect using H2.'
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUT_DIR/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

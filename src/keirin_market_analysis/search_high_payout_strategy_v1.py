from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from .research_high_payout_structures_v1 import load_year, exclusive_roles, parse_combination
from .search_three_year_conditions_v1 import (
    choose_main_line, strongest_rival, feature_row, ATOMS, passes, compatible
)

YEARS=(2023,2024,2025)
OUT=Path('data/audits/high_payout_strategy_search_2023_2025_v1.json')


def perms(items):
    return set(itertools.permutations(items,3))


def make_forms():
    forms={}
    forms['O1_MAINPAIR_6']=perms(['O1','A','B'])
    forms['O1_RIVALPAIR_6']=perms(['O1','R1L','R1B'])

    x=set()
    for m in ('A','B'):
        for r in ('R1L','R1B'):
            x |= perms(['O1',m,r])
    forms['O1_CROSS_24']=x

    x=set()
    core=('A','B','R1L','R1B')
    for pair in itertools.combinations(core,2):
        x |= perms(['O1',*pair])
    forms['O1_CORE_36']=x

    x=set()
    for third in ('A','B','O1'):
        x |= perms(['R1L','R1B',third])
    forms['RIVALPAIR_18']=x

    x=set()
    for third in ('A','B','M3','O1'):
        x |= perms(['R1L','R1B',third])
    forms['RIVALPAIR_24']=x

    x=set()
    for third in ('A','B','R1L'):
        x |= perms(['O1','O2',third])
    forms['O1O2_KEY_18']=x

    x=set()
    key=('A','B','R1L','R1B')
    for first in ('O1','O2'):
        for second in key:
            for third in key:
                if third != second:
                    x.add((first,second,third))
    forms['OUTSIDER_HEAD_24']=x

    x=set()
    tail=('A','B','O1','O2')
    for first in ('R1L','R1B'):
        for second in tail:
            for third in tail:
                if third != second:
                    x.add((first,second,third))
    forms['RIVAL_HEAD_24']=x

    x=set()
    for second in ('O1','R1L','R1B'):
        for third in ('A','O1','R1L','R1B'):
            if third != second and third != 'B':
                x.add(('B',second,third))
    forms['B_HEAD_CHAOS']=x
    return forms

FORMS=make_forms()


def build_data():
    ds={}
    counts={}
    for y in YEARS:
        races,eb,win3,seg=load_year(y)
        ys=defaultdict(list)
        for race in races:
            rid=race['race_id']; s=seg.get(rid)
            if s not in ('前半','中盤','後半') or rid not in win3: continue
            chosen=choose_main_line(eb[rid])
            if not chosen: continue
            mid,main=chosen
            if len(main)<3: continue
            rival=strongest_rival(eb[rid],mid)
            if not rival or len(rival)<2: continue
            roles=exclusive_roles(eb[rid],main,rival)
            combo,pay=win3[rid]
            cars=parse_combination(combo)
            if any(c not in roles for c in cars): continue
            order=tuple(roles[c] for c in cars)
            row=feature_row(race,eb[rid],mid,main,rival)
            row['winner_role_order']=order
            row['winner_payout_yen']=pay
            row['race_date']=race['race_date']
            row['form_eval']={}
            available=set(roles.values())
            for name,orders in FORMS.items():
                valid=[o for o in orders if set(o)<=available]
                points=len(valid)
                hit=order in orders
                row['form_eval'][name]={
                    'points':points,
                    'stake_yen':100*points,
                    'payout_yen':pay if hit else 0,
                    'hit':hit,
                    'high5':hit and pay>=5000,
                    'high10':hit and pay>=10000,
                    'high20':hit and pay>=20000,
                    'recovery_multiple':pay/(100*points) if hit and points else 0.0,
                }
            ys[s].append(row)
        ds[y]=ys
        counts[str(y)]={s:len(ys[s]) for s in ('前半','中盤','後半')}
    return ds,counts


def stats(rows,form):
    n=len(rows)
    stake=sum(r['form_eval'][form]['stake_yen'] for r in rows)
    payout=sum(r['form_eval'][form]['payout_yen'] for r in rows)
    hitrows=[r for r in rows if r['form_eval'][form]['hit']]
    rec=[r['form_eval'][form]['recovery_multiple'] for r in hitrows]
    high5=sum(r['form_eval'][form]['high5'] for r in rows)
    high10=sum(r['form_eval'][form]['high10'] for r in rows)
    high20=sum(r['form_eval'][form]['high20'] for r in rows)
    hitp=[r['form_eval'][form]['payout_yen'] for r in hitrows]
    # max consecutive misses in chronological order
    ordered=sorted(rows,key=lambda r:r['race_date'])
    cur=mx=0
    for r in ordered:
        if r['form_eval'][form]['hit']: cur=0
        else:
            cur+=1; mx=max(mx,cur)
    return {
        'races':n,'hits':len(hitrows),'hit_rate':len(hitrows)/n if n else 0.0,
        'stake_yen':stake,'payout_yen':payout,'profit_yen':payout-stake,'roi':payout/stake if stake else 0.0,
        'high5_hits':high5,'high10_hits':high10,'high20_hits':high20,
        'mean_hit_payout_yen':sum(hitp)/len(hitp) if hitp else 0.0,
        'median_hit_payout_yen':sorted(hitp)[len(hitp)//2] if hitp else 0,
        'mean_recovery_multiple':sum(rec)/len(rec) if rec else 0.0,
        'median_recovery_multiple':sorted(rec)[len(rec)//2] if rec else 0.0,
        'top1_payout_share':max(hitp)/payout if payout and hitp else 0.0,
        'max_losing_streak':mx,
    }


def family(rule_atoms,form):
    feats=sorted(f'{a[1]}:{a[2]}' for a in rule_atoms)
    return form+'|'+'&'.join(feats)


def main():
    ds,counts=build_data()
    out={
      'status':'HIGH_PAYOUT_STRATEGY_SEARCH_2023_2025_ONLY',
      'years_read':list(YEARS),'evaluation_year_2026_used':False,
      'goal':'Trade hit rate for payout/recovery multiple; broad semantic formations allowed.',
      'formations':{k:len(v) for k,v in FORMS.items()},
      'guardrails':{
        'conditions':'0, 1, or 2 coarse predeclared atoms only',
        'min_races_each_year':15,
        'min_hits_each_year':2,
        'roi_each_year':'>1.0',
        'min_high5000_hits_each_year':1,
        'min_total_high10000_hits':3,
        'max_top1_payout_share_each_year':0.80,
        'min_median_hit_recovery_multiple_each_year':2.0,
        'family_support':'at least 2 qualified candidates in same formation+feature/direction family OR unconditional rule',
        'ranking':'worst-year ROI, then worst-year median recovery multiple, then combined ROI, then fewer points',
      },
      'structural_counts':counts,'segments':{}
    }

    rules=[('ALL',[])]
    rules += [(a[0],[a]) for a in ATOMS]
    rules += [(a[0]+'__AND__'+b[0],[a,b]) for i,a in enumerate(ATOMS) for b in ATOMS[i+1:] if compatible(a,b)]

    for seg in ('前半','中盤','後半'):
        candidates=[]
        for rname,atoms in rules:
            selected={y:[r for r in ds[y][seg] if all(passes(r,a) for a in atoms)] for y in YEARS}
            if min(len(selected[y]) for y in YEARS)<15: continue
            for form in FORMS:
                st={y:stats(selected[y],form) for y in YEARS}
                if min(st[y]['hits'] for y in YEARS)<2: continue
                if min(st[y]['roi'] for y in YEARS)<=1.0: continue
                if min(st[y]['high5_hits'] for y in YEARS)<1: continue
                if sum(st[y]['high10_hits'] for y in YEARS)<3: continue
                if max(st[y]['top1_payout_share'] for y in YEARS)>0.80: continue
                if min(st[y]['median_recovery_multiple'] for y in YEARS)<2.0: continue
                total_stake=sum(st[y]['stake_yen'] for y in YEARS)
                total_pay=sum(st[y]['payout_yen'] for y in YEARS)
                c={
                  'rule':rname,'complexity':len(atoms),'formation':form,'points':len(FORMS[form]),
                  'family':family(atoms,form) if atoms else form+'|ALL',
                  'periods':{str(y):st[y] for y in YEARS},
                  'worst_year_roi':min(st[y]['roi'] for y in YEARS),
                  'worst_year_median_recovery_multiple':min(st[y]['median_recovery_multiple'] for y in YEARS),
                  'combined':{
                    'races':sum(st[y]['races'] for y in YEARS),'hits':sum(st[y]['hits'] for y in YEARS),
                    'stake_yen':total_stake,'payout_yen':total_pay,'profit_yen':total_pay-total_stake,
                    'roi':total_pay/total_stake if total_stake else 0.0,
                    'high5_hits':sum(st[y]['high5_hits'] for y in YEARS),
                    'high10_hits':sum(st[y]['high10_hits'] for y in YEARS),
                    'high20_hits':sum(st[y]['high20_hits'] for y in YEARS),
                  }
                }
                candidates.append(c)

        famcount=defaultdict(int)
        for c in candidates: famcount[c['family']]+=1
        robust=[c for c in candidates if c['complexity']==0 or famcount[c['family']]>=2]
        robust.sort(key=lambda c:(c['worst_year_roi'],c['worst_year_median_recovery_multiple'],c['combined']['roi'],-c['points']), reverse=True)
        for c in robust:
            c['family_qualified_count']=famcount[c['family']]
        out['segments'][seg]={
          'qualified_before_family':len(candidates),'qualified_robust':len(robust),
          'recommended':robust[0] if robust else None,
          'shortlist':robust[:20]
        }

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({s:{'qualified':out['segments'][s]['qualified_robust'],'recommended':out['segments'][s]['recommended']} for s in out['segments']},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

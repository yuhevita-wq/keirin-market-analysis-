from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from .research_high_payout_structures_v1 import load_year, parse_combination
from .research_weakest_line_2023_2025 import weakest_line
from .simulate_early_candidate_v0 import pair_strength
from .search_three_year_conditions_v1 import choose_main_line, strongest_rival

YEARS=(2023,2024,2025)
SEGMENTS=('前半','中盤','後半')
OUT=Path('data/audits/weakest_line_formations_2023_2025.json')


def perms(xs): return set(itertools.permutations(xs,3))

def forms():
    f={}
    f['WLEAD_MAINPAIR_6']=perms(['W1L','A','B'])
    f['WSECOND_MAINPAIR_6']=perms(['W1B','A','B'])
    f['WEAK1_MAINPAIR_12']=f['WLEAD_MAINPAIR_6']|f['WSECOND_MAINPAIR_6']
    x=set()
    for third in ('A','B','M3'): x |= perms(['W1L','W1B',third])
    f['WEAKPAIR_MAIN_18']=x
    x=set()
    for third in ('R1L','R1B'): x |= perms(['W1L','W1B',third])
    f['WEAKPAIR_RIVAL_12']=x
    f['WHEAD_AB_4']={('W1L','A','B'),('W1L','B','A'),('W1B','A','B'),('W1B','B','A')}
    f['WPAIR_TOP2_MAIN_6']={(w1,w2,t) for w1,w2 in [('W1L','W1B'),('W1B','W1L')] for t in ('A','B','M3')}
    return f
FORMS=forms()


def stats(rows, form):
    orders=FORMS[form]
    n=len(rows); stake=100*len(orders)*n; pay=0; hits=0; hp=[]; cur=mx=0
    for r in sorted(rows,key=lambda z:z['race_date']):
        if r['role_order'] in orders:
            hits+=1; pay+=r['payout_yen']; hp.append(r['payout_yen']); cur=0
        else:
            cur+=1; mx=max(mx,cur)
    hp.sort()
    return {
      'races':n,'points':len(orders),'hits':hits,'hit_rate':hits/n if n else 0.0,
      'stake_yen':stake,'payout_yen':pay,'profit_yen':pay-stake,'roi':pay/stake if stake else 0.0,
      'median_hit_payout_yen':hp[len(hp)//2] if hp else 0,
      'mean_hit_payout_yen':sum(hp)/len(hp) if hp else 0.0,
      'high5000_hits':sum(x>=5000 for x in hp),'high10000_hits':sum(x>=10000 for x in hp),'high20000_hits':sum(x>=20000 for x in hp),
      'top1_payout_share':max(hp)/pay if hp and pay else 0.0,'max_losing_streak':mx,
    }


def main():
    ds={}
    role_presence={}
    for y in YEARS:
        races,eb,win3,seg=load_year(y); ys=defaultdict(list); pres=defaultdict(lambda:defaultdict(int))
        for race in races:
            rid=race['race_id']; s=seg.get(rid)
            if s not in SEGMENTS or rid not in win3: continue
            chosen=choose_main_line(eb[rid])
            if not chosen: continue
            mid,main=chosen
            if len(main)<3: continue
            rival=strongest_rival(eb[rid],mid)
            if not rival or len(rival)<2: continue
            weak,reason=weakest_line(eb[rid],mid)
            if not weak: continue
            wid,wmem,wscore=weak
            rid2=int(rival[0]['line_id'])
            if wid==rid2: continue
            roles={int(e['car_no']):name for name,e in [('A',main[0]),('B',main[1]),('M3',main[2]),('R1L',rival[0]),('R1B',rival[1]),('W1L',wmem[0]),('W1B',wmem[1])]}
            combo,payout=win3[rid]; cars=parse_combination(combo); order=tuple(roles.get(c,'OTHER') for c in cars)
            row={'year':y,'segment':s,'race_id':rid,'race_date':race['race_date'],'role_order':order,'payout_yen':payout}
            ys[s].append(row)
            for role in ('W1L','W1B'):
                if role in order:
                    pres[s][f'{role}_top3']+=1
                if order[0]==role:
                    pres[s][f'{role}_head']+=1
                if len(order)>1 and order[1]==role:
                    pres[s][f'{role}_second']+=1
                if len(order)>2 and order[2]==role:
                    pres[s][f'{role}_third']+=1
        ds[y]=ys; role_presence[str(y)]={s:dict(pres[s])|{'eligible':len(ys[s])} for s in SEGMENTS}

    out={'status':'WEAKEST_LINE_FORMATION_AUDIT_2023_2025_ONLY','years_read':list(YEARS),'evaluation_year_2026_used':False,'formations':{k:len(v) for k,v in FORMS.items()},'role_presence':role_presence,'segments':{}}
    for s in SEGMENTS:
        out['segments'][s]={}
        for form in FORMS:
            per={str(y):stats(ds[y][s],form) for y in YEARS}
            allrows=[r for y in YEARS for r in ds[y][s]]
            out['segments'][s][form]={'by_year':per,'combined':stats(allrows,form),'worst_year_roi':min(per[str(y)]['roi'] for y in YEARS)}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())

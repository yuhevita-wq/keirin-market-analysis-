from __future__ import annotations

import csv, json
from collections import Counter, defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, segment_for
from .simulate_early_candidate_v0 import strongest_rival

OUT=Path('data/audits/early_v2_third_role.json')


def read_csv(p):
    with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def audit(year:int):
    d=Path(f'data/{year}/s_class_yosen')
    races=read_csv(d/'races.csv'); entries=read_csv(d/'entries.csv'); results=read_csv(d/'results.csv')
    eb=defaultdict(list); rb=defaultdict(list); groups=defaultdict(list)
    for e in entries: eb[e['race_id']].append(e)
    for r in results: rb[r['race_id']].append(r)
    for r in races: groups[(r['race_date'],r['track'])].append(r)
    seg={}
    for g in groups.values():
        g.sort(key=lambda r:int(r['race_no']))
        for i,r in enumerate(g,1): seg[r['race_id']]=segment_for(i,len(g))
    thirds=Counter(); orders=Counter(); n=0
    for race in races:
        rid=race['race_id']
        if seg.get(rid)!='前半': continue
        main=choose_main_line(eb[rid])
        if not main: continue
        mid,m=main
        if len(m)<3: continue
        rival=strongest_rival(eb[rid],mid)
        if not rival or len(rival)<2: continue
        roles={int(m[0]['car_no']):'A',int(m[1]['car_no']):'B',int(m[2]['car_no']):'M3',int(rival[0]['car_no']):'R1L',int(rival[1]['car_no']):'R1B'}
        for j,e in enumerate(m[3:],4): roles[int(e['car_no'])]=f'M{j}'
        other=1
        for e in eb[rid]:
            c=int(e['car_no'])
            if c not in roles:
                roles[c]=f'O{other}'; other+=1
        finish={r.get('finish_position'):int(r['car_no']) for r in rb[rid] if r.get('finish_position') in {'1','2','3'}}
        if len(finish)<3: continue
        top2={finish['1'],finish['2']}
        if top2 != {int(rival[0]['car_no']),int(rival[1]['car_no'])}: continue
        n+=1
        thirds[roles.get(finish['3'],'OTHER')]+=1
        orders[f"{roles.get(finish['1'],'OTHER')}-{roles.get(finish['2'],'OTHER')}-{roles.get(finish['3'],'OTHER')}"]+=1
    return {
      'rival_pair_top2_races':n,
      'third_roles':[{'role':k,'count':v,'rate':v/n if n else 0.0} for k,v in thirds.most_common()],
      'top_sequences':[{'seq':k,'count':v,'rate':v/n if n else 0.0} for k,v in orders.most_common(12)],
      'A_or_B_third_rate':(thirds['A']+thirds['B'])/n if n else 0.0,
    }

def main():
    out={'scope':'2024+2025 development only; structure labels only; no payout data','2024':audit(2024),'2025':audit(2025)}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

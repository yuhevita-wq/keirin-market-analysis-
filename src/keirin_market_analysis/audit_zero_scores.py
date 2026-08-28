from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .audit_input_roles import expected_roles
from .simulate_mainline_v1 import segment_for


def read(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def audit(year: int):
    base=Path(f'data/{year}/s_class_yosen')
    races=read(base/'races.csv'); entries=read(base/'entries.csv')
    eb=defaultdict(list); groups=defaultdict(list)
    for e in entries: eb[e['race_id']].append(e)
    for r in races: groups[(r['race_date'],r['track'])].append(r)
    seg={}
    for g in groups.values():
        g.sort(key=lambda r:int(r['race_no'])); n=len(g)
        for pos,r in enumerate(g,1): seg[r['race_id']]=segment_for(pos,n)
    zeros=[e for e in entries if float(e['score'])==0.0]
    zero_ids={(e['race_id'],e['car_no']) for e in zeros}
    role_hits=[]
    for r in races:
        mode={'前半':'early','中盤':'middle','後半':'late'}[seg[r['race_id']]]
        roles=expected_roles(eb[r['race_id']], mode)
        if not roles: continue
        for role,car in roles.items():
            if car and (r['race_id'],str(car)) in zero_ids:
                role_hits.append({'race_id':r['race_id'],'date':r['race_date'],'track':r['track'],'race_no':r['race_no'],'segment':seg[r['race_id']],'role':role,'car_no':car})
    return {
        'zero_score_entries':len(zeros),
        'zero_score_races':len({e['race_id'] for e in zeros}),
        'zero_score_selected_role_occurrences':len(role_hits),
        'selected_role_examples':role_hits[:20],
    }


def main():
    out={'2024':audit(2024),'2025':audit(2025)}
    p=Path('data/audits/zero_score_audit_2024_2025.json'); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

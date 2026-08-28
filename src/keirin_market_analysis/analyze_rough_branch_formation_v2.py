from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .simulate_mainline_v1 import choose_main_line, segment_for
from .analyze_rough_branch_candidates_v1 import branch_b, role_map, normal_top3, formation_combos

DATA_DIR = Path('data/2025/s_class_yosen')
OUT_DIR = DATA_DIR / 'simulations' / 'rough_branch_formation_v2'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main() -> None:
    races=read_csv(DATA_DIR/'races.csv'); entries=read_csv(DATA_DIR/'entries.csv'); results=read_csv(DATA_DIR/'results.csv')
    ebr=defaultdict(list); rbr=defaultdict(list); groups=defaultdict(list)
    for e in entries: ebr[e['race_id']].append(e)
    for r in results: rbr[r['race_id']].append(r)
    for r in races: groups[(r['race_date'],r['track'])].append(r)
    seg={}
    for g in groups.values():
        g.sort(key=lambda x:int(x['race_no'])); n=len(g)
        for i,r in enumerate(g,1): seg[r['race_id']]=segment_for(i,n)

    defs={
      'F27': (['A','B','R1L'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B']),
      'F36': (['A','B','R1L'], ['A','B','M3','R1L','R1B'], ['A','B','M3','R1L','R1B']),
      'F24_no_R1B_second': (['A','B','R1L'], ['A','B','R1L'], ['A','B','M3','R1L','R1B']),
      'F18_old': (['A','R1L'], ['A','B','R1L','R1B'], ['A','B','M3','R1L','R1B']),
    }
    rows=[]
    for race in races:
        rid=race['race_id']
        if seg.get(rid)!='後半': continue
        main=choose_main_line(ebr[rid])
        if not main: continue
        mid,mem=main
        if len(mem)<3 or not branch_b(mem): continue
        top3,normal=normal_top3(rbr[rid])
        roles=role_map(ebr[rid],mid,mem)
        rows.append({'half':'H1' if race['race_date']<='2025-06-30' else 'H2','normal':normal,'top3':top3,'roles':roles})

    def summ(sub):
        normal=[r for r in sub if r['normal']]
        out={'races':len(sub),'normal_races':len(normal),'formations':{}}
        for name,(p1,p2,p3) in defs.items():
            pts=hit=0
            for r in normal:
                combos=formation_combos(r['roles'],p1,p2,p3)
                pts += len(combos)
                hit += int(tuple(r['top3']) in combos)
            out['formations'][name]={'avg_points':pts/len(normal) if normal else 0,'hit_races':hit,'hit_rate':hit/len(normal) if normal else 0}
        return out
    summary={'scope':'branch B rough-lean, structural exact-order capture only','all':summ(rows),'H1':summ([r for r in rows if r['half']=='H1']),'H2':summ([r for r in rows if r['half']=='H2'])}
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUT_DIR/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

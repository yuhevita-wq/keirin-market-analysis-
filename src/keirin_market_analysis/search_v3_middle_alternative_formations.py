from __future__ import annotations

import json
from pathlib import Path

from .audit_v3_failure_decomposition import PERIODS, load_period
from .simulate_mainline_v1 import choose_main_line
from .simulate_early_candidate_v0 import strongest_rival
from .search_three_year_conditions_v1 import ATOMS, compatible, feature_row, formations, passes as atom_passes
from .search_v3_four_period_conditions import MIN_RACES, MIN_HITS, fin

OUT=Path('data/audits/v3_middle_alternative_formation_search.json')
FORMS=['MAIN4_RIVAL','MAIN6','RIVAL4']


def main():
    protocol=json.loads(Path('data/audits/v3_research_protocol.json').read_text(encoding='utf-8'))
    assert protocol['phase2_predeclared_formation_families']['middle']==['MIX2','MAIN4_RIVAL','MAIN6','RIVAL4']
    data={p:[] for p in PERIODS}
    for period,path in PERIODS.items():
        races,eb,tri,_rb,seg=load_period(path)
        for race in races:
            rid=race['race_id']
            if seg.get(rid)!='中盤': continue
            es=eb[rid]; chosen=choose_main_line(es)
            if not chosen: continue
            mid,main=chosen
            if len(main)<3: continue
            rival=strongest_rival(es,mid)
            if not rival or len(rival)<2: continue
            row=feature_row(race,es,mid,main,rival); fs=formations(row,es)
            row['form_outcomes']={form:(100*len(fs[form]),sum(tri[rid].get(b,0) for b in fs[form])) for form in FORMS}
            data[period].append(row)
    rules=[(a[0],[a]) for a in ATOMS]+[(a[0]+'__AND__'+b[0],[a,b]) for i,a in enumerate(ATOMS) for b in ATOMS[i+1:] if compatible(a,b)]
    out={'status':'V3_MIDDLE_ALTERNATIVE_FORMATION_SEARCH_COMPLETE','forms':FORMS,'results':{}}
    for form in FORMS:
        q=[]; evaluated=0
        for name,atoms in rules:
            sel={p:[r for r in data[p] if all(atom_passes(r,a) for a in atoms)] for p in PERIODS}
            if any(len(sel[p])<MIN_RACES[p] for p in PERIODS): continue
            evaluated+=1
            stats={p:fin([{'stake':r['form_outcomes'][form][0],'payout':r['form_outcomes'][form][1]} for r in sel[p]]) for p in PERIODS}
            if any(stats[p]['hits']<MIN_HITS[p] for p in PERIODS): continue
            if any(stats[p]['roi']<=1 for p in PERIODS): continue
            if any(stats[p]['top1_payout_share']>0.70 for p in PERIODS): continue
            rois=sorted(stats[p]['roi'] for p in PERIODS); st=sum(stats[p]['stake_yen'] for p in PERIODS); py=sum(stats[p]['payout_yen'] for p in PERIODS)
            q.append({'rule':name,'complexity':len(atoms),'periods':stats,'worst_period_roi':rois[0],'median_period_roi':(rois[1]+rois[2])/2,'combined_roi':py/st,'combined_profit_yen':py-st})
        q.sort(key=lambda x:(x['worst_period_roi'],x['median_period_roi'],x['combined_roi'],-x['complexity']),reverse=True)
        out['results'][form]={'evaluated':evaluated,'qualified_count':len(q),'top_candidates':q[:25]}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({f:{'qualified':out['results'][f]['qualified_count'],'top':out['results'][f]['top_candidates'][:4]} for f in FORMS},ensure_ascii=False,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())

from __future__ import annotations
import json
from pathlib import Path

SRC=Path('data/audits/multifeature_signals_2023_2025.json')
OUT=Path('data/audits/multifeature_signals_compact_2023_2025.json')
OUTCOME={'high10000','high20000','weak_included','weak_main2','weak_head'}

def fam(feat):
    if 'discord' in feat or 'hidden' in feat: return 'DISCORDANCE'
    if any(s in feat for s in ('nige','makuri','sashi','mark','attack','finish','b_count','s_count')): return 'TACTICAL_COUNTS'
    if any(s in feat for s in ('win_rate','top2_rate','top3_rate')): return 'HISTORICAL_RATES'
    if 'score' in feat: return 'SCORE'
    if any(s in feat for s in ('line_count','line_len','solo_count','main_len','rival_len','weak_len','max_line_len')): return 'LINE_STRUCTURE'
    return 'OTHER'

def slim(x):
    return {
        'feature':x['feature'],'family':fam(x['feature']),'direction':x['direction'],
        'q25':x.get('q25'),'q75':x.get('q75'),'min_effect_pp':round(x['min_abs_effect_pp'],3),
        'mean_effect_pp':round(x['mean_abs_effect_pp'],3),
        'year_effect_pp':{y:round(v.get('high_minus_low_pp',v.get('true_minus_false_pp',0.0)),3) for y,v in x['by_year'].items()}
    }

def main():
    d=json.load(SRC.open(encoding='utf-8'))
    out={'status':'MULTIFEATURE_COMPACT_2023_2025_ONLY','years_read':d['years_read'],'evaluation_year_2026_used':False,'targets':{}}
    for target,segs in d['targets'].items():
        out['targets'][target]={}
        for seg,z in segs.items():
            nums=[x for x in z['top_stable_numeric'] if x['feature'] not in OUTCOME]
            bools=[x for x in z['top_stable_boolean'] if x['feature'] not in OUTCOME]
            combined=nums+bools
            champs={}
            for x in combined:
                fx=fam(x['feature'])
                if fx not in champs or x['min_abs_effect_pp']>champs[fx]['min_abs_effect_pp']:
                    champs[fx]=x
            out['targets'][target][seg]={
                'baseline_pct':{y:round(v['rate']*100,2) for y,v in z['baseline'].items()},
                'top_numeric':[slim(x) for x in nums[:8]],
                'top_boolean':[slim(x) for x in bools[:6]],
                'family_champions':{k:slim(v) for k,v in champs.items()},
            }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

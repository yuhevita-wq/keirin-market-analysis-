from __future__ import annotations
import json
from pathlib import Path

src=json.loads(Path('data/audits/frozen_structure_2024_vs_2025.json').read_text(encoding='utf-8'))
out={'population':{},'years':{}}
for y in ('2024','2025'):
    p=src['population'][y]
    out['population'][y]={
        'dataset_races':p['dataset_races'],
        'segments':{seg:{
            'all':v['all'],'main3plus':v['main3plus'],
            'main3plus_rate':v['main3plus']/v['all'] if v['all'] else 0
        } for seg,v in p['stages'].items()}
    }
    yy=src['years'][y]['strategies']
    out['years'][y]={}
    for s in ('early_v0','middle_v0','late_v1'):
        d=yy[s]['decision_structure']; f=yy[s]['financial']; o=yy[s]['outcome_structure']
        out['years'][y][s]={
            'eligible':d['decision_rows'],'purchased':d['purchased_races'],'purchase_rate':d['purchase_rate'],
            'decision_counts':d['decision_counts'],
            'pre_medians':{
                'pair_win_all':(d.get('pair_win_all') or {}).get('median'),
                'pair_win_bought':(d.get('pair_win_bought') or {}).get('median'),
                'b_score_all':(d.get('b_score_all') or {}).get('median'),
                'b_score_bought':(d.get('b_score_bought') or {}).get('median'),
                'pair_top3_all':(d.get('pair_top3_all') or {}).get('median'),
                'b_top3_all':(d.get('b_top3_all') or {}).get('median'),
            },
            'financial':{k:f[k] for k in ('hits','hit_rate','stake_yen','payout_yen','profit_yen','roi','avg_payout_per_hit','median_payout_per_hit','max_payout_yen','top1_payout_share','roi_excluding_largest_hit','profitable_hit_races','losing_or_flat_hit_races')},
            'outcome_counts':o['counts'],
        }
    lb=yy['late_v1']['branch_financial']
    out['years'][y]['late_branches']={}
    for b in ('mainline_4pt','rough_18pt'):
        f=lb[b]
        out['years'][y]['late_branches'][b]={k:f[k] for k in ('purchased_races','hits','hit_rate','stake_yen','payout_yen','profit_yen','roi','avg_payout_per_hit','median_payout_per_hit','max_payout_yen','profitable_hit_races','losing_or_flat_hit_races')}
Path('data/audits/frozen_structure_compact.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))

from __future__ import annotations
import json
from pathlib import Path
SRC=Path('data/audits/high_payout_strategy_simple_2023_2025_v1.json')
OUT=Path('data/audits/high_payout_strategy_simple_2023_2025_v1_compact.json')
def main():
    d=json.load(open(SRC,encoding='utf-8'))
    out={'status':'COMPACT_SIMPLE_HIGH_PAYOUT_2023_2025_ONLY','years_read':d['years_read'],'evaluation_year_2026_used':d['evaluation_year_2026_used'],'segments':{}}
    for s,v in d['segments'].items():
        r=v['recommended']
        out['segments'][s]={'qualified':v['qualified'],'recommended':None if r is None else {'rule':r['rule'],'formation':r['formation'],'points':r['points'],'worst_year_roi':r['worst_year_roi'],'worst_year_median_recovery_multiple':r['worst_year_median_recovery_multiple'],'combined':r['combined'],'periods':{y:{k:r['periods'][y][k] for k in ('races','hits','stake_yen','payout_yen','profit_yen','roi','high5_hits','high10_hits','high20_hits','median_hit_payout_yen','median_recovery_multiple','top1_payout_share','max_losing_streak')} for y in ('2023','2024','2025')}}}
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

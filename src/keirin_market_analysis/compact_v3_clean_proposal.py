from __future__ import annotations
import json
from pathlib import Path
SRC=Path('data/audits/v3_clean_2023_2025_proposal.json')
OUT=Path('data/audits/v3_clean_2023_2025_proposal_compact.json')

def main():
    d=json.loads(SRC.read_text(encoding='utf-8'))
    assert d['years_read']==[2023,2024,2025]
    out={'status':'CLEAN_2023_2025_ONLY','years_read':d['years_read'],'guardrails':d['guardrails'],'segments':{}}
    for seg in ('前半','中盤','後半'):
        s=d['segments'][seg]
        r=s['recommended']
        if r is None:
            out['segments'][seg]={'robust_count':s['robust_family_supported_count'],'recommended':None}
            continue
        out['segments'][seg]={
          'robust_count':s['robust_family_supported_count'],
          'recommended':{
            'rule':r['rule'],'formation':r['formation'],'complexity':r['complexity'],
            'family':r['family'],'family_qualifying_variants':r['family_qualifying_variants'],
            '2023':{k:r['periods']['2023'][k] for k in ('races','hits','stake_yen','payout_yen','profit_yen','roi','top1_payout_share')},
            '2024':{k:r['periods']['2024'][k] for k in ('races','hits','stake_yen','payout_yen','profit_yen','roi','top1_payout_share')},
            '2025':{k:r['periods']['2025'][k] for k in ('races','hits','stake_yen','payout_yen','profit_yen','roi','top1_payout_share')},
            'combined_roi':r['combined_roi'],'combined_profit_yen':r['combined_profit_yen'],
            'worst_year_roi':r['worst_year_roi'],'min_races_per_year':r['min_races_per_year'],
            'min_hits_per_year':r['min_hits_per_year'],'max_top1_share':r['max_top1_share'],
            'profitable_halves':r['profitable_halves'],'worst_half_roi':r['worst_half_roi'],
          }
        }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

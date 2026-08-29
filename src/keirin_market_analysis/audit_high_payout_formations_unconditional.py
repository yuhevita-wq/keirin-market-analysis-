from __future__ import annotations
import json
from pathlib import Path
from .search_high_payout_strategy_v1 import YEARS, FORMS, build_data, stats

OUT=Path('data/audits/high_payout_formations_unconditional_2023_2025.json')

def main():
    ds,counts=build_data()
    out={'status':'UNCONDITIONAL_HIGH_PAYOUT_FORMATIONS_2023_2025_ONLY','years_read':list(YEARS),'evaluation_year_2026_used':False,'formations':{k:len(v) for k,v in FORMS.items()},'segments':{}}
    for seg in ('前半','中盤','後半'):
        rows=[]
        for form in FORMS:
            st={str(y):stats(ds[y][seg],form) for y in YEARS}
            total_stake=sum(st[str(y)]['stake_yen'] for y in YEARS)
            total_pay=sum(st[str(y)]['payout_yen'] for y in YEARS)
            rows.append({'formation':form,'points':len(FORMS[form]),'periods':st,'worst_year_roi':min(st[str(y)]['roi'] for y in YEARS),'combined_roi':total_pay/total_stake if total_stake else 0.0,'combined_profit_yen':total_pay-total_stake})
        rows.sort(key=lambda x:(x['worst_year_roi'],x['combined_roi']),reverse=True)
        out['segments'][seg]=rows
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({s:out['segments'][s][:3] for s in out['segments']},ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())

from __future__ import annotations
import json
from pathlib import Path
SRC=Path('data/audits/weakest_line_formations_2023_2025.json')
OUT=Path('data/audits/weakest_line_formations_2023_2025_compact.json')
def main():
 d=json.load(SRC.open(encoding='utf-8'))
 out={'status':'COMPACT_WEAKEST_LINE_FORMATIONS_2023_2025_ONLY','years_read':d['years_read'],'evaluation_year_2026_used':d['evaluation_year_2026_used'],'role_presence':d['role_presence'],'segments':{}}
 for seg,fs in d['segments'].items():
  rows=[]
  for name,z in fs.items():
   c=z['combined']; per=z['by_year']
   rows.append({'formation':name,'points':c['points'],'combined_races':c['races'],'combined_hits':c['hits'],'combined_roi':c['roi'],'combined_profit_yen':c['profit_yen'],'median_hit_payout_yen':c['median_hit_payout_yen'],'high10000_hits':c['high10000_hits'],'high20000_hits':c['high20000_hits'],'max_losing_streak':c['max_losing_streak'],'year_roi':{y:per[y]['roi'] for y in ('2023','2024','2025')},'year_hits':{y:per[y]['hits'] for y in ('2023','2024','2025')},'worst_year_roi':z['worst_year_roi']})
  rows.sort(key=lambda x:(x['worst_year_roi'],x['combined_roi']),reverse=True)
  out['segments'][seg]=rows
 OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

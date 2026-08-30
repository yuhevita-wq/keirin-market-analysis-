from __future__ import annotations
import json
from pathlib import Path

SRC=Path('data/audits/trio_popularity_takeout_2023_2024.json')
OUT=Path('data/audits/trio_popularity_takeout_2023_2024_compact.json')

def keep(x, keys):
    return {k:x.get(k) for k in keys}

def main():
    d=json.loads(SRC.read_text(encoding='utf-8'))
    out={
        'status':'TRIO_POPULARITY_TAKEOUT_AUDIT_2023_2024_COMPACT',
        'years_read':d['years_read'],
        'nominal_reference_roi_pct':d['nominal_reference_roi_pct'],
        'years':{}
    }
    for y,yd in d['years'].items():
        out['years'][y]={
            'complete_races':yd['complete_races'],
            'rank1':keep(yd['by_rank'][0],['wins','hit_rate_pct','avg_odds','roi_pct','abs_deviation_from_75pt']),
            'rank_buckets':[keep(x,['rank_from','rank_to','tickets','wins','hit_rate_pct','avg_odds','roi_pct','abs_deviation_from_75pt']) for x in yd['rank_buckets']],
            'cumulative_top_n':[keep(x,['top_n','tickets','wins','hit_rate_pct','avg_odds','roi_pct','abs_deviation_from_75pt']) for x in yd['cumulative_top_n']],
        }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

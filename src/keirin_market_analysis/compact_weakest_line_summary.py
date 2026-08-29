from __future__ import annotations

import json
from pathlib import Path

SRC = Path('data/audits/weakest_line_structure_2023_2025.json')
OUT = Path('data/audits/weakest_line_structure_2023_2025_compact.json')


def slim(s):
    keys = ('races','share','median_payout_yen','mean_payout_yen','ge_5000_rate','ge_10000_rate','ge_20000_rate','ge_5000_enrichment_vs_all','ge_10000_enrichment_vs_all','ge_20000_enrichment_vs_all')
    return {k:s[k] for k in keys if k in s}


def main():
    d=json.load(SRC.open(encoding='utf-8'))
    out={
      'status':'COMPACT_WEAKEST_LINE_2023_2025_ONLY',
      'years_read':d['years_read'],
      'evaluation_year_2026_used':d['evaluation_year_2026_used'],
      'definition':d['definition'],
      'structural_counts':d['structural_counts'],
      'exclusions':d['exclusions'],
      'segments':{}
    }
    for seg,blob in d['segments'].items():
        c=blob['combined_2023_2025']
        out['segments'][seg]={
          'eligible_races':c['eligible_races'],
          'overall':slim(c['overall']),
          'weak0':slim(c['weakest_top3_count']['0']),
          'weak1':slim(c['weakest_top3_count']['1']),
          'weak2':slim(c['weakest_top3_count']['2']),
          'weakest_included':slim(c['weakest_included']),
          'weakest_head':slim(c['weakest_head']),
          'both_weakest_in_top3':slim(c['both_weakest_in_top3']),
          'weakest_pair_top2':slim(c['weakest_pair_top2']),
          'weak_mix_counts':c['weak_mix_counts'],
          'top_high20000_orders':c['top_high_payout_weak_orders']['20000'][:12],
          'by_year':{}
        }
        for y,ys in blob['by_year'].items():
            out['segments'][seg]['by_year'][y]={
              'eligible_races':ys['eligible_races'],
              'weakest_included':slim(ys['weakest_included']),
              'weakest_head':slim(ys['weakest_head']),
              'both_weakest_in_top3':slim(ys['both_weakest_in_top3']),
            }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())

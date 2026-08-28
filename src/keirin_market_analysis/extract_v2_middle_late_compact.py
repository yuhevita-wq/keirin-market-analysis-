from __future__ import annotations

import json
from pathlib import Path

SRC = Path('data/audits/v2_middle_late_structure.json')
OUT = Path('data/audits/v2_middle_late_structure_compact.json')


def pick_candidate(rows, name):
    for r in rows:
        if r.get('name') == name:
            return {k:r[k] for k in ('name','races','hits','hit_rate','avg_points')}
    return None


def main():
    d=json.loads(SRC.read_text(encoding='utf-8'))
    mid=d['middle']; late=d['late']
    rough_names=['P27_base','P18_first_A_R1L','P18_first_A_B','P09_first_A','P18_drop_R1B_second']
    out={
      'scope':d['scope'],
      'validation_plan':d['validation_plan'],
      'middle':{
        'basic':mid['basic'],
        'top_AB_screens':mid['stable_AB_top2_screens'][:8],
        'top_R1_screens':mid['stable_R1_top2_screens'][:8],
        'top_sequences':{
          y: mid['role_sequences'][y]['top_sequences'][:12] for y in ('2024','2025')
        },
      },
      'late':{
        'all_basic':late['all_basic'],
        'mainline':{
          'basic':late['mainline_branch']['basic'],
          'third_after_AB_top2':late['mainline_branch']['third_after_AB_top2'],
          'current_4pt_structural_hit':late['mainline_branch']['current_4pt_structural_hit'],
        },
        'rough':{
          'basic':late['rough_branch']['basic'],
          'top_sequences':{
            y: late['rough_branch']['role_sequences'][y]['top_sequences'][:12] for y in ('2024','2025')
          },
          'selected_candidates':{
            y:[pick_candidate(late['rough_branch']['predeclared_pruning_candidates'][y],n) for n in rough_names]
            for y in ('2024','2025')
          }
        }
      }
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

from __future__ import annotations
import json
from pathlib import Path

SRC=Path('data/audits/threshold_overfit_2025.json')
OUT=Path('data/audits/threshold_overfit_2025_compact.json')

def main():
    d=json.loads(SRC.read_text(encoding='utf-8'))
    e=d['early_v0']; m=d['middle_v0']; l=d['late_routing']; r=d['late_rough_ticket_path']
    out={
      'scope':d['scope'],
      'early':{
        'H1_rows':e['H1_rows'],'H2_rows':e['H2_rows'],
        'midpoint_candidates':e['one_feature_midpoint_candidates'],
        'valid_min_leaf20':e['valid_candidates_min_leaf20'],
        'frozen_rank':e['frozen_rank_by_H1_gini_gain'],
        'frozen_split':e['frozen_22_35_split'],'best_split':e['best_H1_split'],
        'nearby':{k:{'H1_structure':v['H1_structure'],'H2_structure':v['H2_structure'],'ALL_roi':v['ALL_betting']['roi']} for k,v in e['nearby_threshold_sensitivity'].items()},
        'provenance_gap':e['selection_provenance_gap'],
      },
      'middle':{
        'structure_rows':m['structure_rows_all_2025'],'actionable_rows':m['actionable_rows_with_rival'],
        'features_scanned':m['features_scanned'],'valid_candidates':m['valid_H1_feature_threshold_candidates_min_leaf20'],
        'frozen_split':m['frozen_105_structural_split'],'global_rank':m['frozen_105_global_gini_rank'],
        'feature_best_rank':m['frozen_105_rank_among_feature_best'],'best_structural_split':m['best_H1_structural_split'],
        'roi_scan':m['counterfactual_B_score_threshold_scan'],'provenance_gap':m['final_rule_provenance_gap'],
      },
      'late_routing':l,
      'rough':{
        'search_breadth':r['explicit_search_breadth_minimum'],
        'pruning_selected':r['pruning_candidates_H1_selection']['selected'],
        'pruning_selected_metrics':r['pruning_candidates_H1_selection']['all_candidates'][r['pruning_candidates_H1_selection']['selected']],
        'score_ge_10':r['confidence_zone_search']['score_ge_10'],
        'final_pruning':r['same_2025_payout_driven_final_pruning'],
      },
      'readout':d['methodological_readout'],
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

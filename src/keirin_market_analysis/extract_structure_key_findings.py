from __future__ import annotations
import json
from pathlib import Path

SRC = Path('data/audits/frozen_structure_2024_vs_2025.json')
OUT = Path('data/audits/frozen_structure_key_findings.json')

def slim_strategy(s: dict) -> dict:
    d = s['decision_structure']; f = s['financial']; o = s['outcome_structure']
    return {
        'decision_rows': d['decision_rows'],
        'purchased_races': d['purchased_races'],
        'purchase_rate': d['purchase_rate'],
        'decision_counts': d['decision_counts'],
        'pair_win_all': d.get('pair_win_all'),
        'pair_win_bought': d.get('pair_win_bought'),
        'b_score_all': d.get('b_score_all'),
        'b_score_bought': d.get('b_score_bought'),
        'pair_top3_all': d.get('pair_top3_all'),
        'b_top3_all': d.get('b_top3_all'),
        'financial': f,
        'outcomes': o,
        'branch_financial': s.get('branch_financial', {}),
    }

def main():
    src = json.loads(SRC.read_text(encoding='utf-8'))
    out = {'population': src['population'], 'years': {}}
    for year in ('2024','2025'):
        y = src['years'][year]
        out['years'][year] = {
            'combined_financial': y['combined_financial'],
            'early_v0': slim_strategy(y['strategies']['early_v0']),
            'middle_v0': slim_strategy(y['strategies']['middle_v0']),
            'late_v1': slim_strategy(y['strategies']['late_v1']),
        }
    a = out['years']['2024']; b = out['years']['2025']
    out['derived'] = {
        'early_hit_rate_ratio_2024_over_2025': a['early_v0']['financial']['hit_rate'] / b['early_v0']['financial']['hit_rate'] if b['early_v0']['financial']['hit_rate'] else None,
        'middle_hit_rate_ratio_2024_over_2025': a['middle_v0']['financial']['hit_rate'] / b['middle_v0']['financial']['hit_rate'] if b['middle_v0']['financial']['hit_rate'] else None,
        'late_hit_rate_ratio_2024_over_2025': a['late_v1']['financial']['hit_rate'] / b['late_v1']['financial']['hit_rate'] if b['late_v1']['financial']['hit_rate'] else None,
        'late_avg_payout_per_hit_ratio_2024_over_2025': a['late_v1']['financial']['avg_payout_per_hit'] / b['late_v1']['financial']['avg_payout_per_hit'] if b['late_v1']['financial']['avg_payout_per_hit'] else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

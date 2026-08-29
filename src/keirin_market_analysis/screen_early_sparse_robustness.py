from __future__ import annotations

import json
from pathlib import Path

SRC = Path('data/audits/early_sparse_high_return_search.json')
OUT = Path('data/audits/early_sparse_robustness_screen.json')


def main():
    data = json.loads(SRC.read_text(encoding='utf-8'))
    cands = data['top_candidates']
    # Development-only robustness diagnostic. No 2023 data is read.
    screened = []
    for c in cands:
        if min(c['2024']['hits'], c['2025']['hits']) < 3:
            continue
        if max(c['2024']['top1_payout_share'], c['2025']['top1_payout_share']) > 0.70:
            continue
        screened.append(c)
    screened.sort(key=lambda c: (c['worst_year_roi'], c['combined']['roi'], -c['complexity']), reverse=True)
    out = {
        'scope': '2024+2025 development only; reads only the previously generated candidate search output; 2023 prohibited.',
        'screen': {
            'min_hits_each_year': 3,
            'max_top1_payout_share_each_year': 0.70,
            'note': 'This is a payout-concentration diagnostic, not a new race-selection feature.'
        },
        'survivor_count': len(screened),
        'survivors': screened[:20],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

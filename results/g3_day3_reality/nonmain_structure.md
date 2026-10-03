# G3三日目 本命以外ゼロベース再分析

本命A-Bだけ固定。2番手ラインを特別扱いせず、外側をライン順位・単騎・同一ユニット/異ユニットで分解。train=2022-2024 / test=2025-2026H1

## train: 1467R
### 本命A/Bの3着内残存人数
- 2人: 570 (38.9%)
- 1人: 573 (39.1%)
- 0人: 324 (22.1%)
### A/Bが1人だけ残った時、外2人の構造
- 1M+two_different_out_units: 255 (44.5%)
- 1M+same_out_line_pair: 247 (43.1%)
- 1M+singleton+other: 62 (10.8%)
- 1M+2singletons: 9 (1.6%)
### A/Bが2人とも消えた時、外3人の構造
- 0M+same_out_line_pair+other: 198 (61.1%)
- 0M+same_out_line_triple: 84 (25.9%)
- 0M+three_different_out_units: 17 (5.2%)
- 0M+singleton+two_other_units: 16 (4.9%)
- 0M+2plus_singletons: 9 (2.8%)
### 構造別 実現率 vs 市場期待（全レース基準）
- 0M+2plus_singletons: actual 0.61% / market 0.47% / ratio 1.295
- 0M+same_out_line_pair+other: actual 13.50% / market 14.19% / ratio 0.951
- 0M+same_out_line_triple: actual 5.73% / market 3.76% / ratio 1.523
- 0M+singleton+two_other_units: actual 1.09% / market 1.27% / ratio 0.859
- 0M+three_different_out_units: actual 1.16% / market 2.12% / ratio 0.547
- 1M+2singletons: actual 0.61% / market 0.48% / ratio 1.291
- 1M+same_out_line_pair: actual 16.84% / market 17.05% / ratio 0.987
- 1M+singleton+other: actual 4.23% / market 4.59% / ratio 0.920
- 1M+two_different_out_units: actual 17.38% / market 20.16% / ratio 0.862
- 2M+one_out: actual 38.85% / market 35.91% / ratio 1.082

## test: 793R
### 本命A/Bの3着内残存人数
- 2人: 283 (35.7%)
- 1人: 325 (41.0%)
- 0人: 185 (23.3%)
### A/Bが1人だけ残った時、外2人の構造
- 1M+two_different_out_units: 138 (42.5%)
- 1M+same_out_line_pair: 137 (42.2%)
- 1M+singleton+other: 46 (14.2%)
- 1M+2singletons: 4 (1.2%)
### A/Bが2人とも消えた時、外3人の構造
- 0M+same_out_line_pair+other: 108 (58.4%)
- 0M+same_out_line_triple: 46 (24.9%)
- 0M+singleton+two_other_units: 12 (6.5%)
- 0M+three_different_out_units: 11 (5.9%)
- 0M+2plus_singletons: 8 (4.3%)
### 構造別 実現率 vs 市場期待（全レース基準）
- 0M+2plus_singletons: actual 1.01% / market 0.79% / ratio 1.275
- 0M+same_out_line_pair+other: actual 13.62% / market 14.44% / ratio 0.943
- 0M+same_out_line_triple: actual 5.80% / market 3.87% / ratio 1.500
- 0M+singleton+two_other_units: actual 1.51% / market 1.59% / ratio 0.951
- 0M+three_different_out_units: actual 1.39% / market 2.05% / ratio 0.676
- 1M+2singletons: actual 0.50% / market 0.63% / ratio 0.797
- 1M+same_out_line_pair: actual 17.28% / market 16.55% / ratio 1.044
- 1M+singleton+other: actual 5.80% / market 5.74% / ratio 1.010
- 1M+two_different_out_units: actual 17.40% / market 19.91% / ratio 0.874
- 2M+one_out: actual 35.69% / market 34.42% / ratio 1.037


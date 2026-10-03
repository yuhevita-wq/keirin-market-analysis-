# G3三日目 本命以外ゼロベース再分析

本命A-Bだけ固定。2番手ラインを特別扱いせず、外側をライン順位・単騎・同一ユニット/異ユニットで分解。

## train: 139R
### 本命A/Bの3着内残存人数
- 2人: 54 (38.8%)
- 1人: 52 (37.4%)
- 0人: 33 (23.7%)
### A/Bが1人だけ残った時、外2人の構造
- 1M+same_out_line_pair: 23 (44.2%)
- 1M+two_different_out_units: 22 (42.3%)
- 1M+singleton+other: 7 (13.5%)
### A/Bが2人とも消えた時、外3人の構造
- 0M+same_out_line_pair+other: 17 (51.5%)
- 0M+same_out_line_triple: 7 (21.2%)
- 0M+three_different_out_units: 6 (18.2%)
- 0M+2plus_singletons: 2 (6.1%)
- 0M+singleton+two_other_units: 1 (3.0%)
### 構造別 実現率 vs 市場期待（全レース基準）
- 0M+2plus_singletons: actual 1.44% / market 0.73% / ratio 1.982
- 0M+same_out_line_pair+other: actual 12.23% / market 14.08% / ratio 0.869
- 0M+same_out_line_triple: actual 5.04% / market 3.76% / ratio 1.341
- 0M+singleton+two_other_units: actual 0.72% / market 1.32% / ratio 0.547
- 0M+three_different_out_units: actual 4.32% / market 2.55% / ratio 1.692
- 1M+2singletons: actual 0.00% / market 0.70% / ratio 0.000
- 1M+same_out_line_pair: actual 16.55% / market 16.01% / ratio 1.034
- 1M+singleton+other: actual 5.04% / market 4.85% / ratio 1.039
- 1M+two_different_out_units: actual 15.83% / market 20.13% / ratio 0.786
- 2M+one_out: actual 38.85% / market 35.89% / ratio 1.082

## test: 2121R
### 本命A/Bの3着内残存人数
- 2人: 799 (37.7%)
- 1人: 846 (39.9%)
- 0人: 476 (22.4%)
### A/Bが1人だけ残った時、外2人の構造
- 1M+two_different_out_units: 371 (43.9%)
- 1M+same_out_line_pair: 361 (42.7%)
- 1M+singleton+other: 101 (11.9%)
- 1M+2singletons: 13 (1.5%)
### A/Bが2人とも消えた時、外3人の構造
- 0M+same_out_line_pair+other: 289 (60.7%)
- 0M+same_out_line_triple: 123 (25.8%)
- 0M+singleton+two_other_units: 27 (5.7%)
- 0M+three_different_out_units: 22 (4.6%)
- 0M+2plus_singletons: 15 (3.2%)
### 構造別 実現率 vs 市場期待（全レース基準）
- 0M+2plus_singletons: actual 0.71% / market 0.58% / ratio 1.228
- 0M+same_out_line_pair+other: actual 13.63% / market 14.29% / ratio 0.953
- 0M+same_out_line_triple: actual 5.80% / market 3.80% / ratio 1.526
- 0M+singleton+two_other_units: actual 1.27% / market 1.39% / ratio 0.917
- 0M+three_different_out_units: actual 1.04% / market 2.07% / ratio 0.502
- 1M+2singletons: actual 0.61% / market 0.52% / ratio 1.180
- 1M+same_out_line_pair: actual 17.02% / market 16.93% / ratio 1.005
- 1M+singleton+other: actual 4.76% / market 5.01% / ratio 0.951
- 1M+two_different_out_units: actual 17.49% / market 20.07% / ratio 0.872
- 2M+one_out: actual 37.67% / market 35.35% / ratio 1.066


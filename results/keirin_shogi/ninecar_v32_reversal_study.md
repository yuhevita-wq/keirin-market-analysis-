# 9車 v3.2 序列崩壊・逆転条件研究

現行の強さベース順位を基準とし、その順位が崩れる条件を別問題として研究する。
結果・オッズ・人気・払戻は推論特徴に使わない。

## 主要率
- 2024: 強さ1位勝率 0.373, 4位以下評価の勝利 0.270, 2着モデル4位以下の実2着 0.487
- 2025 forward: 強さ1位勝率 0.380, 4位以下評価の勝利 0.264, 2着モデル4位以下の実2着 0.448
- 2026 H1 forward: 強さ1位勝率 0.370, 4位以下評価の勝利 0.265, 2着モデル4位以下の実2着 0.409

## 2025/2026で安定して差が大きい条件
- second_rank_ge4 / num_lines: spread 2025=0.524, 2026=0.636
- top_strength_loses / num_lines: spread 2025=0.389, 2026=0.404
- winner_rank_ge4 / num_lines: spread 2025=0.429, 2026=0.364
- winner_rank_ge4 / top_line_size: spread 2025=0.313, 2026=0.444
- top_strength_loses / top_line_size: spread 2025=0.307, 2026=0.556
- top_strength_loses / top_line_position: spread 2025=0.266, 2026=0.383
- top_strength_loses / hidden_challenger_count: spread 2025=0.419, 2026=0.250
- second_rank_ge4 / top_line_size: spread 2025=0.214, 2026=0.667
- top_strength_loses / score_gap_1_2: spread 2025=0.214, 2026=0.305
- second_rank_ge4 / top3_same_line_count: spread 2025=0.213, 2026=0.232
- top_strength_loses / p1_top: spread 2025=0.244, 2026=0.210
- top_strength_loses / p1_gap_1_2: spread 2025=0.219, 2026=0.200

# v35 Top2-conditioned third

v31の21個のTop2組確率を変更せず、各組に対する残り5人の3着条件付き確率を周辺化する。
v21/v31は候補除外に使わない。2026 H1は再開封診断であり、完全未来試験ではない。

| test block | third capture | avg third candidates |
|---|---:|---:|
| 2024-10-01〜2024-12-31 | 53.74% | 2.815 |
| 2025-04-01〜2025-06-30 | 53.18% | 2.779 |
| 2025-10-27〜2025-12-28 | 52.64% | 2.842 |
| reopened 2025-12-29〜2026-06-28 | 54.40% | 2.806 |

判定: **開発不合格**

```bash
python scripts/keirin_shogi_v35_top2_conditioned_third.py
```

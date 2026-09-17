# v37 Shrunk board third

v36の強い再重み付け過適合を受け、全21組を残したままgamma=0.25へ固定。
候補ポリシーだけを2025-10-27〜12-28で選び、2026 H1は開発診断として扱う。

| evaluation | third | complete board | third given first+second | avg third |
|---|---:|---:|---:|---:|
| calibration | 51.46% | 32.52% | 72.83% | 2.782 |
| reopened 2026 H1 | 53.42% | 32.09% | 74.89% | 2.840 |

判定: **完全未来試験へ凍結**

```bash
python scripts/keirin_shogi_v37_shrunk_board_third.py
```

# v36 Board-consistent third

v35の全21組を残し、v21の1着確率とv31 membershipで組確率をソフト再重み付けする。
候補除外は行わず、重み係数と候補数は2025-10-27〜12-28のみで校正する。

| evaluation | third | complete board | third given first+second | avg third |
|---|---:|---:|---:|---:|
| calibration | 53.40% | 34.47% | 77.17% | 2.845 |
| reopened 2026 H1 | 52.25% | 31.90% | 74.43% | 2.883 |

判定: **開発不合格**

```bash
python scripts/keirin_shogi_v36_board_consistent_third.py
```

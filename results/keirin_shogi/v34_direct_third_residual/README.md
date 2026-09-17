# v34 Direct Third Residual

Top3集合だけでは実3着の順序を分離できなかったため、7人を直接比較する独立3着モデルへ変更。
v31は候補除外に使わず、校正された重複ペナルティとしてだけ使う。

| block | third capture | avg candidates |
|---|---:|---:|
| 2024-10-01〜2024-12-31 | 54.50% | 2.785 |
| 2025-04-01〜2025-06-30 | 51.26% | 2.780 |
| 2025-10-27〜2025-12-28 | 52.53% | 2.799 |

判定: **開発不合格**

```bash
python scripts/keirin_shogi_v34_direct_third_residual.py
```

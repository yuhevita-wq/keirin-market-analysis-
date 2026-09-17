# v32 Top3 Membership

## 結論

- 判定: **不採用**
- 2026年前半ホールドアウト: 511R
- 3着捕捉率: 46.77%
- 平均3着候補数: 2.830人
- 完全盤面捕捉率: 24.85%
- 選択方式: B_top3_minus_top2 (beta=1.0, gap3=0.6, gap4=None)

## 時系列

モデル候補は2024年後半、2025年前半、2025年7月〜10月26日の順で検証した。
最終モデルは2024年1月1日〜2025年10月26日だけで再学習した。
A/B/Cの方式と2〜4人の可変候補ルールは2025年10月27日〜12月28日だけで校正した。
2026年1月以降はすべて固定後に一度だけ評価した。

## A/B/C

- A: `Top3Membership`
- B: `Top3Membership - Top2Membership`
- C: `Top3Membership - beta * Top2Membership`（betaは校正期間だけで選択）

3着候補は常に2人を置き、2位と3位の標準化スコア差が小さい時だけ3人目を追加する。
4人目は3位と4位も接近した場合だけ許可し、校正目的関数で強く減点する。

## 再現

```bash
python -m pip install scikit-learn
python scripts/keirin_shogi_v32_top3_membership.py
```

オッズ、人気、支持率は読み込まない。v21とv31のルールは変更していない。

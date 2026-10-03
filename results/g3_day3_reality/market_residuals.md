# G3三日目 市場残差分析

頻度ではなく、実現率と正規化市場期待確率の差を見る。train=2022-2024 / test=2025-2026H1

## ライン長・最弱ライン・単騎の実現率
### train: 1467R
- 本命ライン長: 2車=503R, 3車=910R, 4車=49R, 5車=5R
- 最弱ライン1人以上TOP3: 40.7% / 最弱ライン先頭TOP3 21.2% / 番手TOP3 25.6% / 最弱ライン勝者 15.7% / 最弱ペア同時TOP3 8.4%
- 単騎あり 569R: 単騎1人以上TOP3 29.7% / 単騎勝者 9.1%

### test: 793R
- 本命ライン長: 2車=302R, 3車=469R, 4車=20R, 5車=2R
- 最弱ライン1人以上TOP3: 40.1% / 最弱ライン先頭TOP3 19.5% / 番手TOP3 25.2% / 最弱ライン勝者 15.0% / 最弱ペア同時TOP3 7.2%
- 単騎あり 341R: 単騎1人以上TOP3 32.3% / 単騎勝者 7.0%

## 構造カテゴリ 市場較正
### AB_pair
- train: actual 38.9% / market 35.9% / actual-market +2.95pt / ratio 1.082
- test: actual 35.7% / market 34.4% / actual-market +1.27pt / ratio 1.037

### AB_plus_outside
- train: actual 27.3% / market 27.6% / actual-market -0.33pt / ratio 0.988
- test: actual 24.8% / market 26.3% / actual-market -1.50pt / ratio 0.943

### main_tail_any
- train: actual 19.6% / market 17.5% / actual-market +2.16pt / ratio 1.124
- test: actual 20.7% / market 17.6% / actual-market +3.05pt / ratio 1.173

### CD_pair
- train: actual 18.3% / market 16.9% / actual-market +1.41pt / ratio 1.083
- test: actual 17.9% / market 16.1% / actual-market +1.81pt / ratio 1.112

### second_tail_any
- train: actual 9.9% / market 10.1% / actual-market -0.19pt / ratio 0.982
- test: actual 10.8% / market 10.6% / actual-market +0.24pt / ratio 1.023

### weak_pair
- train: actual 8.4% / market 8.6% / actual-market -0.21pt / ratio 0.975
- test: actual 7.2% / market 8.5% / actual-market -1.30pt / ratio 0.847

### weak_any
- train: actual 40.7% / market 44.2% / actual-market -3.49pt / ratio 0.921
- test: actual 40.1% / market 44.1% / actual-market -3.97pt / ratio 0.910

### weak_leader_any
- train: actual 21.2% / market 23.4% / actual-market -2.19pt / ratio 0.907
- test: actual 19.5% / market 22.8% / actual-market -3.23pt / ratio 0.858

### weak_second_any
- train: actual 25.6% / market 26.9% / actual-market -1.39pt / ratio 0.949
- test: actual 25.2% / market 27.3% / actual-market -2.05pt / ratio 0.925

### weak_tail_any
- train: actual 7.4% / market 7.1% / actual-market +0.25pt / ratio 1.034
- test: actual 8.4% / market 7.7% / actual-market +0.79pt / ratio 1.103

### singleton_any
- train: actual 11.5% / market 11.2% / actual-market +0.33pt / ratio 1.029
- test: actual 13.9% / market 14.2% / actual-market -0.37pt / ratio 0.974

### singleton_2plus
- train: actual 1.2% / market 0.9% / actual-market +0.28pt / ratio 1.293
- test: actual 1.5% / market 1.4% / actual-market +0.09pt / ratio 1.062

### three_units
- train: actual 18.7% / market 22.5% / actual-market -3.74pt / ratio 0.834
- test: actual 19.9% / market 23.7% / actual-market -3.82pt / ratio 0.839

### any_non_top2_line
- train: actual 46.6% / market 49.8% / actual-market -3.27pt / ratio 0.934
- test: actual 45.8% / market 49.4% / actual-market -3.66pt / ratio 0.926

## 単騎 rider-level 市場較正
- train: 単騎延べ767人 top3 24.6% / win 6.8% / 市場top3期待 23.4% / ratio 1.054
- test: 単騎延べ480人 top3 26.0% / win 5.0% / 市場top3期待 26.0% / ratio 1.001

## 代表的な1点構造 市場期待との差と実ROI
### AB_best_singleton
- train: n=569 hit 6.85% / market 5.07% / ratio 1.351 / ROI 94.2% / median odds 20.8
- test: n=341 hit 5.28% / market 5.52% / ratio 0.956 / ROI 56.8% / median odds 19.4

### AB_main3
- train: n=964 hit 17.32% / market 12.39% / ratio 1.398 / ROI 104.8% / median odds 7.1
- test: n=491 hit 17.52% / market 12.94% / ratio 1.353 / ROI 97.6% / median odds 6.8

### AB_rank2_leader
- train: n=1463 hit 6.70% / market 6.83% / ratio 0.980 / ROI 63.4% / median odds 13.3
- test: n=793 hit 5.42% / market 5.85% / ratio 0.926 / ROI 83.8% / median odds 15.9

### AB_rank2_second
- train: n=1463 hit 6.97% / market 7.36% / ratio 0.948 / ROI 71.1% / median odds 12.4
- test: n=793 hit 6.43% / market 7.17% / ratio 0.896 / ROI 62.7% / median odds 12.0

### AB_weak_leader
- train: n=1337 hit 3.07% / market 3.32% / ratio 0.922 / ROI 72.6% / median odds 31.3
- test: n=717 hit 4.04% / market 3.15% / ratio 1.284 / ROI 97.5% / median odds 32.4

### AB_weak_second
- train: n=1337 hit 3.52% / market 3.89% / ratio 0.904 / ROI 54.9% / median odds 26.0
- test: n=717 hit 3.21% / market 3.74% / ratio 0.858 / ROI 54.3% / median odds 28.8

### CD_A
- train: n=1463 hit 4.44% / market 4.28% / ratio 1.037 / ROI 74.9% / median odds 21.9
- test: n=793 hit 3.91% / market 3.80% / ratio 1.028 / ROI 76.1% / median odds 26.1

### CD_B
- train: n=1463 hit 3.69% / market 4.18% / ratio 0.882 / ROI 51.5% / median odds 22.9
- test: n=793 hit 4.16% / market 3.94% / ratio 1.055 / ROI 103.9% / median odds 23.8

### CD_best_singleton
- train: n=565 hit 2.48% / market 2.21% / ratio 1.122 / ROI 63.1% / median odds 48.4
- test: n=341 hit 1.76% / market 2.16% / ratio 0.814 / ROI 69.0% / median odds 44.1

### CD_rank2_3rd
- train: n=731 hit 7.11% / market 5.08% / ratio 1.401 / ROI 119.0% / median odds 19.5
- test: n=395 hit 8.35% / market 5.19% / ratio 1.608 / ROI 116.2% / median odds 17.9

### CD_weak_leader
- train: n=1337 hit 1.94% / market 1.44% / ratio 1.348 / ROI 122.8% / median odds 73.3
- test: n=717 hit 1.12% / market 1.32% / ratio 0.846 / ROI 35.5% / median odds 75.1

### CD_weak_second
- train: n=1337 hit 1.72% / market 1.71% / ratio 1.006 / ROI 78.8% / median odds 61.2
- test: n=717 hit 1.53% / market 1.60% / ratio 0.956 / ROI 47.7% / median odds 67.7

### WEAKPAIR_A
- train: n=1463 hit 1.91% / market 1.88% / ratio 1.016 / ROI 67.1% / median odds 65.8
- test: n=793 hit 1.13% / market 1.72% / ratio 0.661 / ROI 55.8% / median odds 70.1

### WEAKPAIR_B
- train: n=1463 hit 1.71% / market 1.80% / ratio 0.948 / ROI 56.7% / median odds 70.0
- test: n=793 hit 1.64% / market 1.76% / ratio 0.929 / ROI 106.9% / median odds 66.6

### WEAKPAIR_best_singleton
- train: n=569 hit 1.58% / market 1.57% / ratio 1.008 / ROI 54.8% / median odds 74.6
- test: n=341 hit 1.17% / market 1.58% / ratio 0.743 / ROI 33.5% / median odds 67.1


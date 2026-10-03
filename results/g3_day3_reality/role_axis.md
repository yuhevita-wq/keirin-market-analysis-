# G3三日目 ライン非依存・役割軸分析

役割は事前個体成績のみ。ライン情報・予想印は候補選定に不使用。
- W（勝ち切り役） = win_rate
- Q2（2着残存役） = top2_rate - win_rate
- Q3（3着滑り込み役） = top3_rate - top2_rate
- 各レース内でW/Q2/Q3を別々に順位付け。
- formation Wn × Q2m × Q3k。重複選手は除外し3人が異なる組だけ購入。

train=2022-2024 / test=2025-2026H1

## 勝ち組3人に各役割上位が含まれる率
### train: 1404R
- W: top1 50.1% / top2以内 73.6% / top3以内 89.1%
- Q2: top1 41.9% / top2以内 68.3% / top3以内 84.9%
- Q3: top1 33.8% / top2以内 58.3% / top3以内 77.5%
- 役割1位の脚質 W1={'追': 221, '逃': 871, '両': 312} / Q2-1={'逃': 561, '追': 577, '両': 266} / Q3-1={'追': 869, '両': 257, '逃': 278}
### test: 798R
- W: top1 53.1% / top2以内 79.8% / top3以内 90.1%
- Q2: top1 44.4% / top2以内 69.4% / top3以内 84.1%
- Q3: top1 38.3% / top2以内 65.2% / top3以内 79.7%
- 役割1位の脚質 W1={'逃': 407, '両': 259, '追': 132} / Q2-1={'逃': 320, '追': 272, '両': 206} / Q3-1={'追': 491, '逃': 150, '両': 157}

## trainだけで選んだ2〜8点帯 上位フォーメーション
### W1 × Q23 × Q33
- train: 6.49点 hit 16.45% / market 15.68% / ratio 1.049 / diff +0.77% / ROI 60.6%
- test: 6.20点 hit 17.54% / market 17.03% / ratio 1.030 / diff +0.51% / ROI 58.5%
### W1 × Q22 × Q33
- train: 4.46点 hit 12.18% / market 11.42% / ratio 1.067 / diff +0.76% / ROI 61.6%
- test: 4.19点 hit 13.16% / market 12.32% / ratio 1.068 / diff +0.84% / ROI 66.3%
### W3 × Q21 × Q32
- train: 4.35点 hit 10.54% / market 10.07% / ratio 1.047 / diff +0.47% / ROI 74.9%
- test: 4.19点 hit 10.40% / market 10.88% / ratio 0.956 / diff -0.48% / ROI 46.5%
### W1 × Q22 × Q34
- train: 5.84点 hit 15.17% / market 14.72% / ratio 1.031 / diff +0.45% / ROI 58.3%
- test: 5.49点 hit 16.29% / market 15.40% / ratio 1.058 / diff +0.89% / ROI 62.7%
### W2 × Q21 × Q34
- train: 5.79点 hit 14.53% / market 14.12% / ratio 1.029 / diff +0.41% / ROI 56.5%
- test: 5.44点 hit 14.04% / market 14.72% / ratio 0.954 / diff -0.68% / ROI 56.4%
### W2 × Q22 × Q32
- train: 5.81点 hit 13.60% / market 13.24% / ratio 1.028 / diff +0.36% / ROI 58.7%
- test: 5.47点 hit 14.16% / market 14.42% / ratio 0.982 / diff -0.26% / ROI 50.0%
### W1 × Q21 × Q34
- train: 3.02点 hit 8.55% / market 8.26% / ratio 1.035 / diff +0.29% / ROI 58.9%
- test: 2.80点 hit 8.27% / market 8.21% / ratio 1.007 / diff +0.06% / ROI 51.4%
### W1 × Q22 × Q32
- train: 3.05点 hit 8.19% / market 7.98% / ratio 1.026 / diff +0.21% / ROI 60.1%
- test: 2.85点 hit 8.15% / market 8.38% / ratio 0.972 / diff -0.23% / ROI 56.9%
### W2 × Q21 × Q32
- train: 2.99点 hit 7.62% / market 7.45% / ratio 1.023 / diff +0.17% / ROI 53.6%
- test: 2.82点 hit 7.77% / market 7.99% / ratio 0.973 / diff -0.22% / ROI 49.0%
### W3 × Q22 × Q31
- train: 4.34点 hit 9.54% / market 9.37% / ratio 1.018 / diff +0.17% / ROI 52.5%
- test: 4.20点 hit 10.90% / market 11.02% / ratio 0.990 / diff -0.11% / ROI 51.8%
### W2 × Q21 × Q33
- train: 4.42点 hit 11.04% / market 10.88% / ratio 1.014 / diff +0.16% / ROI 54.5%
- test: 4.15点 hit 10.90% / market 11.61% / ratio 0.939 / diff -0.71% / ROI 55.1%
### W3 × Q21 × Q33
- train: 6.39点 hit 14.60% / market 14.45% / ratio 1.011 / diff +0.15% / ROI 66.2%
- test: 6.14点 hit 14.41% / market 15.66% / ratio 0.920 / diff -1.25% / ROI 60.6%

## 得点上位BOX benchmark
### score top3 BOX
- train: 1.00点 hit 9.97% / market 9.65% / ratio 1.033 / ROI 65.0%
- test: 1.00点 hit 9.15% / market 9.93% / ratio 0.921 / ROI 59.4%
### score top4 BOX
- train: 4.00点 hit 23.72% / market 24.47% / ratio 0.969 / ROI 66.5%
- test: 4.00点 hit 23.43% / market 24.89% / ratio 0.942 / ROI 62.6%
### score top5 BOX
- train: 10.00点 hit 39.53% / market 41.47% / ratio 0.953 / ROI 62.2%
- test: 10.00点 hit 39.10% / market 41.71% / ratio 0.937 / ROI 65.5%
### score top6 BOX
- train: 20.00点 hit 57.83% / market 59.87% / ratio 0.966 / ROI 63.1%
- test: 20.00点 hit 56.02% / market 60.14% / ratio 0.931 / ROI 55.6%
### score top7 BOX
- train: 35.00点 hit 73.72% / market 75.69% / ratio 0.974 / ROI 61.9%
- test: 35.00点 hit 75.44% / market 76.73% / ratio 0.983 / ROI 68.0%
### score top8 BOX
- train: 54.61点 hit 88.75% / market 89.51% / ratio 0.991 / ROI 60.8%
- test: 53.53点 hit 88.72% / market 89.91% / ratio 0.987 / ROI 62.3%

## 勝ち組の役割順位シグネチャ上位
### train
- best W/Q2/Q3 ranks=(1, 1, 1): 78 (5.56%)
- best W/Q2/Q3 ranks=(1, 1, 2): 75 (5.34%)
- best W/Q2/Q3 ranks=(1, 1, 3): 65 (4.63%)
- best W/Q2/Q3 ranks=(1, 2, 1): 59 (4.20%)
- best W/Q2/Q3 ranks=(2, 1, 1): 52 (3.70%)
- best W/Q2/Q3 ranks=(1, 2, 2): 43 (3.06%)
- best W/Q2/Q3 ranks=(1, 2, 3): 41 (2.92%)
- best W/Q2/Q3 ranks=(1, 3, 1): 38 (2.71%)
- best W/Q2/Q3 ranks=(1, 1, 4): 35 (2.49%)
- best W/Q2/Q3 ranks=(2, 1, 2): 30 (2.14%)
- best W/Q2/Q3 ranks=(3, 1, 1): 30 (2.14%)
- best W/Q2/Q3 ranks=(1, 3, 3): 27 (1.92%)
- best W/Q2/Q3 ranks=(1, 4, 1): 27 (1.92%)
- best W/Q2/Q3 ranks=(3, 1, 2): 27 (1.92%)
- best W/Q2/Q3 ranks=(2, 2, 1): 26 (1.85%)
### test
- best W/Q2/Q3 ranks=(1, 1, 1): 65 (8.15%)
- best W/Q2/Q3 ranks=(1, 1, 2): 53 (6.64%)
- best W/Q2/Q3 ranks=(1, 2, 1): 40 (5.01%)
- best W/Q2/Q3 ranks=(2, 1, 1): 34 (4.26%)
- best W/Q2/Q3 ranks=(2, 1, 2): 29 (3.63%)
- best W/Q2/Q3 ranks=(2, 2, 1): 26 (3.26%)
- best W/Q2/Q3 ranks=(1, 1, 4): 26 (3.26%)
- best W/Q2/Q3 ranks=(1, 2, 2): 26 (3.26%)
- best W/Q2/Q3 ranks=(1, 3, 1): 25 (3.13%)
- best W/Q2/Q3 ranks=(1, 1, 3): 22 (2.76%)
- best W/Q2/Q3 ranks=(3, 1, 1): 19 (2.38%)
- best W/Q2/Q3 ranks=(1, 3, 2): 16 (2.01%)
- best W/Q2/Q3 ranks=(1, 2, 3): 15 (1.88%)
- best W/Q2/Q3 ranks=(2, 2, 2): 15 (1.88%)
- best W/Q2/Q3 ranks=(1, 3, 3): 14 (1.75%)

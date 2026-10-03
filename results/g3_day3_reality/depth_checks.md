# G3三日目 深掘り再検証

最弱ラインは3本以上の複数ラインがあるレースだけ。単騎は得点順位別。第2強度3車ラインは年別・競走種別で再検証。

## 本当の最弱ライン 市場較正
### weak_any
- train: n=1337 actual 38.4% / market 41.6% / diff -3.27pt / ratio 0.922
- test: n=717 actual 37.4% / market 41.5% / diff -4.17pt / ratio 0.900

### weak_pair
- train: n=1337 actual 7.0% / market 7.5% / diff -0.50pt / ratio 0.933
- test: n=717 actual 5.9% / market 7.5% / diff -1.66pt / ratio 0.779

### weak_leader
- train: n=1337 actual 19.6% / market 21.7% / diff -2.15pt / ratio 0.901
- test: n=717 actual 18.0% / market 21.5% / diff -3.48pt / ratio 0.838

### weak_second
- train: n=1337 actual 23.5% / market 25.1% / diff -1.59pt / ratio 0.937
- test: n=717 actual 23.0% / market 25.4% / diff -2.39pt / ratio 0.906

### weak_tail
- train: n=1337 actual 6.4% / market 6.1% / diff +0.35pt / ratio 1.058
- test: n=717 actual 7.3% / market 6.1% / diff +1.11pt / ratio 1.181

### weak_size_2
- train: n=809 actual 31.5% / market 34.6% / diff -3.04pt / ratio 0.912
- test: n=458 actual 28.6% / market 34.7% / diff -6.13pt / ratio 0.823

### weak_size_3
- train: n=514 actual 47.9% / market 51.8% / diff -3.96pt / ratio 0.924
- test: n=255 actual 52.2% / market 53.2% / diff -1.06pt / ratio 0.980

### weak_size_4
- train: n=13 actual 84.6% / market 75.8% / diff +8.83pt / ratio 1.117
- test: n=4 actual 100.0% / market 77.3% / diff +22.69pt / ratio 1.294

## 単騎 得点順位別 top3市場較正
### score_rank 1-3
- train: n=139 top3 43.2% / win 18.7% / market 43.2% / ratio 0.999
- test: n=148 top3 56.1% / win 21.6% / market 56.3% / ratio 0.996

### score_rank 4-6
- train: n=270 top3 27.8% / win 8.1% / market 25.7% / ratio 1.082
- test: n=218 top3 26.1% / win 4.1% / market 27.5% / ratio 0.951

### score_rank 7-9
- train: n=358 top3 15.1% / win 1.1% / market 13.9% / ratio 1.081
- test: n=247 top3 17.0% / win 0.8% / market 15.6% / ratio 1.088

## 第2強度ライン3車丸ごと 3連複
- train: n=731 hits=52 hit 7.11% ROI 119.0% median_odds 19.5
- test: n=395 hits=33 hit 8.35% ROI 116.2% median_odds 17.9
### 年別
- 2022: n=228 hits=14 hit 6.14% ROI 103.7% median_odds 18.5
- 2023: n=249 hits=18 hit 7.23% ROI 101.6% median_odds 19.5
- 2024: n=254 hits=20 hit 7.87% ROI 149.7% median_odds 21.0
- 2025: n=264 hits=19 hit 7.20% ROI 116.3% median_odds 17.2
- 2026: n=131 hits=14 hit 10.69% ROI 116.0% median_odds 18.7
### 競走種別 n>=40
- Ｓ級特選: n=317 hits=28 hit 8.83% ROI 141.9% median_odds 19.2
- Ｓ級選抜: n=294 hits=24 hit 8.16% ROI 99.8% median_odds 18.5
- Ｓ級準決勝: n=268 hits=16 hit 5.97% ROI 143.4% median_odds 33.6
- Ｓ級一般: n=234 hits=16 hit 6.84% ROI 84.3% median_odds 13.1

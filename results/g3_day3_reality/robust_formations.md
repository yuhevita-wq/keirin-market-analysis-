# G3三日目 ロバスト3連複フォーメーション

目的: 想定外展開を拾いつつ、市場期待確率より実現率が上回る形を探す。train=2022-2024 / test=2025-2026H1

## CORE4
- train: 1462R 平均4.00点 hit 21.8% / market 22.7% / ratio 0.963 / diff -0.84% / ROI 65.3%
  - AB崩れ時 hit 13.3% (894R) / AB・CD両ペア不成立時 hit 0.0% (625R) / 主力外勝者時 hit 17.0% / 的中のうち深いズレ 0.0%
- test: 792R 平均4.00点 hit 19.9% / market 20.8% / ratio 0.960 / diff -0.83% / ROI 81.8%
  - AB崩れ時 hit 12.6% (509R) / AB・CD両ペア不成立時 hit 0.0% (367R) / 主力外勝者時 hit 15.8% / 的中のうち深いズレ 0.0%

## CORE4_LINE
- train: 1462R 平均5.16点 hit 36.7% / market 33.3% / ratio 1.102 / diff +3.41% / ROI 75.5%
  - AB崩れ時 hit 19.1% (894R) / AB・CD両ペア不成立時 hit 0.0% (625R) / 主力外勝者時 hit 24.3% / 的中のうち深いズレ 0.0%
- test: 792R 平均5.12点 hit 35.0% / market 31.4% / ratio 1.114 / diff +3.58% / ROI 87.0%
  - AB崩れ時 hit 19.1% (509R) / AB・CD両ペア不成立時 hit 0.0% (367R) / 主力外勝者時 hit 24.0% / 的中のうち深いズレ 0.0%

## CROSS_TAILS
- train: 1462R 平均8.62点 hit 27.3% / market 28.8% / ratio 0.947 / diff -1.53% / ROI 62.2%
  - AB崩れ時 hit 22.3% (894R) / AB・CD両ペア不成立時 hit 12.8% (625R) / 主力外勝者時 hit 22.3% / 的中のうち深いズレ 20.1%
- test: 792R 平均8.47点 hit 27.4% / market 27.1% / ratio 1.012 / diff +0.32% / ROI 75.1%
  - AB崩れ時 hit 24.2% (509R) / AB・CD両ペア不成立時 hit 16.1% (367R) / 主力外勝者時 hit 22.3% / 的中のうち深いズレ 27.2%

## CROSS_VALUE
- train: 1462R 平均10.07点 hit 27.8% / market 29.5% / ratio 0.945 / diff -1.62% / ROI 60.0%
  - AB崩れ時 hit 23.2% (894R) / AB・CD両ペア不成立時 hit 14.1% (625R) / 主力外勝者時 hit 23.3% / 的中のうち深いズレ 21.6%
- test: 792R 平均9.78点 hit 28.2% / market 27.7% / ratio 1.017 / diff +0.47% / ROI 78.4%
  - AB崩れ時 hit 25.3% (509R) / AB・CD両ペア不成立時 hit 17.7% (367R) / 主力外勝者時 hit 23.0% / 的中のうち深いズレ 29.1%

## CROSS_SINGLE
- train: 1462R 平均10.17点 hit 29.3% / market 30.9% / ratio 0.949 / diff -1.59% / ROI 65.8%
  - AB崩れ時 hit 25.6% (894R) / AB・CD両ペア不成立時 hit 17.6% (625R) / 主力外勝者時 hit 24.4% / 的中のうち深いズレ 25.6%
- test: 792R 平均10.20点 hit 29.8% / market 29.6% / ratio 1.008 / diff +0.23% / ROI 73.6%
  - AB崩れ時 hit 27.9% (509R) / AB・CD両ペア不成立時 hit 21.3% (367R) / 主力外勝者時 hit 25.5% / 的中のうち深いズレ 33.1%

## INSURANCE_TAIL
- train: 1462R 平均6.60点 hit 37.3% / market 34.0% / ratio 1.098 / diff +3.32% / ROI 69.2%
  - AB崩れ時 hit 20.0% (894R) / AB・CD両ペア不成立時 hit 1.3% (625R) / 主力外勝者時 hit 25.3% / 的中のうち深いズレ 1.5%
- test: 792R 平均6.43点 hit 35.7% / market 32.0% / ratio 1.117 / diff +3.73% / ROI 89.5%
  - AB崩れ時 hit 20.2% (509R) / AB・CD両ペア不成立時 hit 1.6% (367R) / 主力外勝者時 hit 24.8% / 的中のうち深いズレ 2.1%

## INSURANCE_TAIL_SINGLE
- train: 1462R 平均8.14点 hit 39.3% / market 36.1% / ratio 1.091 / diff +3.27% / ROI 72.3%
  - AB崩れ時 hit 23.4% (894R) / AB・CD両ペア不成立時 hit 6.1% (625R) / 主力外勝者時 hit 27.4% / 的中のうち深いズレ 6.6%
- test: 792R 平均8.15点 hit 38.1% / market 34.5% / ratio 1.106 / diff +3.65% / ROI 84.6%
  - AB崩れ時 hit 24.0% (509R) / AB・CD両ペア不成立時 hit 6.8% (367R) / 主力外勝者時 hit 28.0% / 的中のうち深いズレ 8.3%

## FOLLOWER_HUB
- train: 879R 平均9.88点 hit 42.5% / market 40.8% / ratio 1.042 / diff +1.73% / ROI 70.4%
  - AB崩れ時 hit 26.2% (543R) / AB・CD両ペア不成立時 hit 10.3% (388R) / 主力外勝者時 hit 28.2% / 的中のうち深いズレ 10.7%
- test: 491R 平均9.94点 hit 42.8% / market 40.1% / ratio 1.068 / diff +2.71% / ROI 77.3%
  - AB崩れ時 hit 27.4% (303R) / AB・CD両ペア不成立時 hit 14.0% (221R) / 主力外勝者時 hit 30.2% / 的中のうち深いズレ 14.8%

## LEADER_HUB
- train: 879R 平均9.88点 hit 41.9% / market 39.9% / ratio 1.048 / diff +1.92% / ROI 67.4%
  - AB崩れ時 hit 25.0% (543R) / AB・CD両ペア不成立時 hit 8.8% (388R) / 主力外勝者時 hit 27.5% / 的中のうち深いズレ 9.2%
- test: 491R 平均9.94点 hit 40.9% / market 38.9% / ratio 1.051 / diff +1.99% / ROI 73.1%
  - AB崩れ時 hit 24.4% (303R) / AB・CD両ペア不成立時 hit 10.0% (221R) / 主力外勝者時 hit 29.0% / 的中のうち深いズレ 10.9%


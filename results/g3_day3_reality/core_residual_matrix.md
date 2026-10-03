# G3三日目 上位2ライン×単騎・下位ライン 市場乖離

A-B=最上位ライン、C-D=2番手ライン。下位ラインは強度3位以下。市場期待は3連複1/oddsのレース内正規化。

## 市場乖離が両期間で残る上位候補
- **CD+SINGLE_ALL**: train n=565 pts=1.33 hit=3.01% market=2.64% ratio=1.141 ROI=68.8% medhit=27.9 / test n=341 pts=1.41 hit=2.93% market=2.71% ratio=1.082 ROI=78.0% medhit=39.3
- **CD+RESID_BACK_ONLY**: train n=1032 pts=1.33 hit=1.94% market=1.83% ratio=1.056 ROI=52.2% medhit=30.1 / test n=563 pts=1.41 hit=2.31% market=1.99% ratio=1.159 ROI=109.6% medhit=48.1
- **CD+RESID_ROLE_ALL**: train n=1032 pts=1.33 hit=1.94% market=1.83% ratio=1.056 ROI=52.2% medhit=30.1 / test n=563 pts=1.41 hit=2.31% market=1.99% ratio=1.159 ROI=109.6% medhit=48.1
- **SAME+WEAK_LEADER**: train n=1337 pts=2.00 hit=5.01% market=4.77% ratio=1.051 ROI=97.7% medhit=23.0 / test n=717 pts=2.00 hit=5.16% market=4.47% ratio=1.155 ROI=66.5% medhit=18.4
- **SAME+LOWER_LEADERS**: train n=1337 pts=2.75 hit=7.26% market=7.11% ratio=1.020 ROI=93.0% medhit=20.6 / test n=717 pts=2.74 hit=7.53% market=6.59% ratio=1.142 ROI=82.8% medhit=19.5
- **CD+LOWER_ALL**: train n=1337 pts=3.22 hit=5.61% market=4.84% ratio=1.160 ROI=92.3% medhit=46.1 / test n=717 pts=3.19 hit=4.60% market=4.53% ratio=1.016 ROI=76.6% medhit=34.2
- **CD+LOWER_LEADERS**: train n=1337 pts=1.38 hit=2.69% market=2.12% ratio=1.272 ROI=108.0% medhit=40.5 / test n=717 pts=1.37 hit=1.95% market=1.94% ratio=1.009 ROI=65.2% medhit=29.6
- **SAME+LOWER_ALL**: train n=1337 pts=6.43 hit=16.16% market=16.06% ratio=1.006 ROI=80.2% medhit=19.4 / test n=717 pts=6.37 hit=15.62% market=15.16% ratio=1.031 ROI=76.5% medhit=18.7
- **SAME+RESID_R3_SINGLE**: train n=1438 pts=5.54 hit=15.30% market=15.42% ratio=0.992 ROI=71.7% medhit=17.1 / test n=785 pts=5.61 hit=15.67% market=15.28% ratio=1.025 ROI=82.4% medhit=15.8
- **CD+RESID_R3_SINGLE**: train n=1438 pts=2.77 hit=4.73% market=4.65% ratio=1.016 ROI=71.9% medhit=30.6 / test n=785 pts=2.81 hit=4.46% market=4.49% ratio=0.992 ROI=83.1% medhit=35.3
- **AB+RESID_R3_SINGLE**: train n=1438 pts=2.77 hit=10.57% market=10.76% ratio=0.982 ROI=71.5% medhit=12.9 / test n=785 pts=2.81 hit=11.21% market=10.79% ratio=1.039 ROI=81.7% medhit=12.7
- **SAME+RESID_BACK_ONLY**: train n=1032 pts=2.66 hit=6.88% market=5.98% ratio=1.150 ROI=60.2% medhit=17.3 / test n=563 pts=2.83 hit=6.75% market=6.91% ratio=0.976 ROI=87.8% medhit=14.0
- **SAME+RESID_ROLE_ALL**: train n=1032 pts=2.66 hit=6.88% market=5.98% ratio=1.150 ROI=60.2% medhit=17.3 / test n=563 pts=2.83 hit=6.75% market=6.91% ratio=0.976 ROI=87.8% medhit=14.0
- **CROSS+SINGLE_BEST**: train n=565 pts=4.00 hit=5.31% market=5.45% ratio=0.974 ROI=85.6% medhit=31.5 / test n=341 pts=4.00 hit=5.57% market=5.78% ratio=0.964 ROI=66.2% medhit=23.1
- **CD+LOWER_SECONDS**: train n=1337 pts=1.38 hit=2.69% market=2.42% ratio=1.113 ROI=96.9% medhit=43.4 / test n=717 pts=1.37 hit=2.23% market=2.32% ratio=0.963 ROI=62.0% medhit=29.4
- **SAME+R3_SECOND**: train n=1337 pts=2.00 hit=6.21% market=6.45% ratio=0.962 ROI=75.8% medhit=14.6 / test n=717 pts=2.00 hit=6.14% market=6.22% ratio=0.986 ROI=68.5% medhit=14.4
- **SAME+SINGLE_ALL**: train n=565 pts=2.67 hit=10.62% market=8.68% ratio=1.223 ROI=77.5% medhit=15.7 / test n=341 pts=2.82 hit=9.09% market=9.46% ratio=0.961 ROI=77.3% medhit=11.5
- **CD+R3_SECOND**: train n=1337 pts=1.00 hit=1.94% market=1.95% ratio=0.997 ROI=84.1% medhit=27.0 / test n=717 pts=1.00 hit=1.81% market=1.89% ratio=0.959 ROI=67.7% medhit=34.2
- **AB+SINGLE_BEST**: train n=565 pts=1.00 hit=6.90% market=5.04% ratio=1.371 ROI=94.8% medhit=9.7 / test n=341 pts=1.00 hit=5.28% market=5.52% ratio=0.956 ROI=56.8% medhit=6.8
- **CD+WEAK_SECOND**: train n=1337 pts=1.00 hit=1.72% market=1.71% ratio=1.006 ROI=78.8% medhit=40.7 / test n=717 pts=1.00 hit=1.53% market=1.60% ratio=0.956 ROI=47.7% medhit=19.5
- **ALL2+WEAK_LEADER**: train n=1337 pts=6.00 hit=8.53% market=8.80% ratio=0.969 ROI=66.4% medhit=31.3 / test n=717 pts=6.00 hit=7.81% market=8.23% ratio=0.949 ROI=60.1% medhit=25.2
- **AB+R3_SECOND**: train n=1337 pts=1.00 hit=4.26% market=4.50% ratio=0.947 ROI=67.5% medhit=12.5 / test n=717 pts=1.00 hit=4.32% market=4.33% ratio=0.998 ROI=69.3% medhit=12.4
- **ALL2+RESID_BACK_ONLY**: train n=1032 pts=7.97 hit=11.05% market=10.54% ratio=1.048 ROI=62.2% medhit=23.2 / test n=563 pts=8.48 hit=11.55% market=12.27% ratio=0.941 ROI=71.1% medhit=21.1
- **ALL2+RESID_ROLE_ALL**: train n=1032 pts=7.97 hit=11.05% market=10.54% ratio=1.048 ROI=62.2% medhit=23.2 / test n=563 pts=8.48 hit=11.55% market=12.27% ratio=0.941 ROI=71.1% medhit=21.1
- **AB+LOWER_ALL**: train n=1337 pts=3.22 hit=10.55% market=11.22% ratio=0.940 ROI=68.0% medhit=13.4 / test n=717 pts=3.19 hit=11.02% market=10.63% ratio=1.037 ROI=76.4% medhit=14.4
- **ALL2+SINGLE_BEST**: train n=565 pts=6.00 hit=14.69% market=12.70% ratio=1.157 ROI=83.4% medhit=17.4 / test n=341 pts=6.00 hit=12.61% market=13.46% ratio=0.937 ROI=65.1% medhit=13.2
- **SAME+LOWER_SECONDS**: train n=1337 pts=2.75 hit=8.08% market=8.00% ratio=1.010 ROI=81.0% medhit=17.3 / test n=717 pts=2.74 hit=7.11% market=7.63% ratio=0.932 ROI=61.5% medhit=17.2
- **ALL2+WEAK_TAIL**: train n=528 pts=6.17 hit=3.60% market=3.87% ratio=0.931 ROI=45.1% medhit=57.0 / test n=259 pts=6.09 hit=4.25% market=3.98% ratio=1.066 ROI=99.3% medhit=61.8
- **BD+WEAK_LEADER**: train n=1337 pts=1.00 hit=0.90% market=0.97% ratio=0.924 ROI=51.6% medhit=37.0 / test n=717 pts=1.00 hit=0.98% market=1.04% ratio=0.939 ROI=85.1% medhit=41.7
- **AB+WEAK_LEADER**: train n=1337 pts=1.00 hit=3.07% market=3.32% ratio=0.922 ROI=72.6% medhit=16.1 / test n=717 pts=1.00 hit=4.04% market=3.15% ratio=1.284 ROI=97.5% medhit=17.4
- **SAME+WEAK_TAIL**: train n=528 pts=2.06 hit=2.08% market=2.08% ratio=1.000 ROI=44.7% medhit=37.6 / test n=259 pts=2.03 hit=1.93% market=2.10% ratio=0.920 ROI=102.4% medhit=59.3
- **AB+LOWER_SECONDS**: train n=1337 pts=1.38 hit=5.39% market=5.58% ratio=0.965 ROI=65.1% medhit=12.9 / test n=717 pts=1.37 hit=4.88% market=5.31% ratio=0.919 ROI=60.9% medhit=13.0
- **SAME+SINGLE_BEST**: train n=565 pts=2.00 hit=9.38% market=7.24% ratio=1.295 ROI=79.0% medhit=13.6 / test n=341 pts=2.00 hit=7.04% market=7.68% ratio=0.916 ROI=62.9% medhit=9.0
- **AB+LOWER_LEADERS**: train n=1337 pts=1.38 hit=4.56% market=5.00% ratio=0.913 ROI=78.1% medhit=14.7 / test n=717 pts=1.37 hit=5.58% market=4.66% ratio=1.197 ROI=100.4% medhit=15.7
- **AB+SINGLE_ALL**: train n=565 pts=1.33 hit=7.61% market=6.04% ratio=1.260 ROI=86.1% medhit=9.8 / test n=341 pts=1.41 hit=6.16% market=6.75% ratio=0.913 ROI=76.6% medhit=8.5

## 全結果はJSON参照

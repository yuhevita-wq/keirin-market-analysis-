# G3三日目 3連複 固定ペア再設計

有効2ライン以上: 2273R

## 固定ペアが実際にTOP3へ2人とも残る率
- AB: 857/2273 = 37.7%
- CD: 417/2273 = 18.3%
- ABまたはCD: 1274/2273 = 56.0%

## ABが残った時、3人目は誰だったか
- main3: 254/857 = 29.6%
- D_rival_second: 154/857 = 18.0%
- C_rival_leader: 140/857 = 16.3%
- other_line_second: 108/857 = 12.6%
- other_line_leader: 103/857 = 12.0%
- singleton: 65/857 = 7.6%
- rival3plus: 18/857 = 2.1%
- other: 12/857 = 1.4%
- main4plus: 3/857 = 0.4%

## CDが残った時、3人目は誰だったか
- A_main_leader: 97/417 = 23.3%
- rival3: 88/417 = 21.1%
- B_main_second: 86/417 = 20.6%
- other_line_leader: 77/417 = 18.5%
- other_line_second: 53/417 = 12.7%
- main3: 9/417 = 2.2%
- other: 7/417 = 1.7%

## AB固定 2〜3点
- AB2_M3D: train 2.0点 hit 20.9% ROI 82.4% / test 2.0点 hit 19.8% ROI 85.6% | test外勝者捕捉 2.7% 主力勝者捕捉 37.8%
- AB2_market: train 2.0点 hit 24.1% ROI 78.3% / test 2.0点 hit 23.9% ROI 82.2% | test外勝者捕捉 5.4% 主力勝者捕捉 43.3%
- AB2_M3C: train 2.0点 hit 20.0% ROI 77.9% / test 2.0点 hit 19.7% ROI 87.3% | test外勝者捕捉 4.0% 主力勝者捕捉 36.3%
- AB3_M3CD: train 3.0点 hit 27.5% ROI 78.1% / test 3.0点 hit 25.7% ROI 75.8% | test外勝者捕捉 5.4% 主力勝者捕捉 46.9%
- AB3_market: train 3.0点 hit 30.7% ROI 82.1% / test 3.0点 hit 28.1% ROI 75.3% | test外勝者捕捉 6.2% 主力勝者捕捉 51.0%
- AB3_top3: train 3.0点 hit 20.7% ROI 68.2% / test 3.0点 hit 21.4% ROI 86.6% | test外勝者捕捉 5.7% 主力勝者捕捉 37.8%
- AB2_top3: train 2.0点 hit 14.3% ROI 68.0% / test 2.0点 hit 16.2% ROI 83.3% | test外勝者捕捉 4.2% 主力勝者捕捉 28.8%
- AB2_score: train 2.0点 hit 19.0% ROI 80.2% / test 2.0点 hit 17.3% ROI 67.5% | test外勝者捕捉 4.9% 主力勝者捕捉 30.3%
- AB3_CDscore: train 3.0点 hit 22.9% ROI 74.3% / test 3.0点 hit 20.2% ROI 67.2% | test外勝者捕捉 6.4% 主力勝者捕捉 34.7%
- AB2_CD: train 2.0点 hit 13.6% ROI 66.5% / test 2.0点 hit 11.8% ROI 71.9% | test外勝者捕捉 4.7% 主力勝者捕捉 19.2%
- AB3_score: train 3.0点 hit 25.0% ROI 77.8% / test 3.0点 hit 22.0% ROI 64.9% | test外勝者捕捉 6.4% 主力勝者捕捉 38.3%

## CD固定 相手2〜6人
- CD3_top3: train 3.0点 hit 10.9% ROI 79.3% / test 3.0点 hit 10.5% ROI 83.3% | test外勝者捕捉 16.8% 主力勝者捕捉 3.9%
- CD2_market: train 2.0点 hit 10.7% ROI 78.6% / test 2.0点 hit 11.0% ROI 82.0% | test外勝者捕捉 17.3% 主力勝者捕捉 4.4%
- CD5_market: train 4.9点 hit 17.0% ROI 76.4% / test 5.0点 hit 16.8% ROI 76.9% | test外勝者捕捉 28.1% 主力勝者捕捉 4.9%
- CD3_market: train 3.0点 hit 13.6% ROI 76.1% / test 3.0点 hit 14.4% ROI 90.5% | test外勝者捕捉 23.7% 主力勝者捕捉 4.7%
- CD5_top3: train 4.9点 hit 15.0% ROI 74.8% / test 5.0点 hit 15.4% ROI 79.5% | test外勝者捕捉 25.4% 主力勝者捕捉 4.9%
- CD2_score: train 2.0点 hit 9.4% ROI 74.4% / test 2.0点 hit 8.7% ROI 95.4% | test外勝者捕捉 12.6% 主力勝者捕捉 4.7%
- CD4_top3: train 4.0点 hit 13.1% ROI 73.9% / test 4.0点 hit 13.5% ROI 79.8% | test外勝者捕捉 21.7% 主力勝者捕捉 4.9%
- CD6_top3: train 5.8点 hit 16.7% ROI 73.4% / test 5.9点 hit 16.9% ROI 76.7% | test外勝者捕捉 28.4% 主力勝者捕捉 4.9%
- CD6_market: train 5.8点 hit 17.7% ROI 73.2% / test 5.9点 hit 17.6% ROI 76.1% | test外勝者捕捉 29.6% 主力勝者捕捉 4.9%
- CD4_score: train 4.0点 hit 13.8% ROI 73.0% / test 4.0点 hit 13.9% ROI 87.8% | test外勝者捕捉 22.5% 主力勝者捕捉 4.9%
- CD4_market: train 4.0点 hit 15.5% ROI 72.8% / test 4.0点 hit 15.7% ROI 80.3% | test外勝者捕捉 25.9% 主力勝者捕捉 4.9%
- CD3_score: train 3.0点 hit 11.7% ROI 71.3% / test 3.0点 hit 11.4% ROI 91.3% | test外勝者捕捉 17.8% 主力勝者捕捉 4.7%
- CD6_score: train 5.8点 hit 17.1% ROI 70.3% / test 5.9点 hit 16.4% ROI 74.6% | test外勝者捕捉 27.4% 主力勝者捕捉 4.9%
- CD6_ABscore: train 5.8点 hit 16.9% ROI 69.8% / test 5.9点 hit 16.2% ROI 73.1% | test外勝者捕捉 26.9% 主力勝者捕捉 4.9%
- CD5_score: train 4.9点 hit 15.6% ROI 69.7% / test 5.0点 hit 15.2% ROI 75.3% | test外勝者捕捉 24.9% 主力勝者捕捉 4.9%
- CD5_ABscore: train 4.9点 hit 15.0% ROI 68.8% / test 5.0点 hit 14.9% ROI 74.9% | test外勝者捕捉 24.4% 主力勝者捕捉 4.9%
- CD2_top3: train 2.0点 hit 7.5% ROI 71.8% / test 2.0点 hit 6.4% ROI 64.5% | test外勝者捕捉 9.4% 主力勝者捕捉 3.4%
- CD2_ABscore: train 2.0点 hit 8.1% ROI 63.0% / test 2.0点 hit 8.0% ROI 87.5% | test外勝者捕捉 11.1% 主力勝者捕捉 4.7%
- CD4_ABscore: train 4.0点 hit 12.4% ROI 61.9% / test 4.0点 hit 12.8% ROI 82.5% | test外勝者捕捉 20.2% 主力勝者捕捉 4.9%
- CD3_ABscore: train 3.0点 hit 10.3% ROI 59.7% / test 3.0点 hit 9.5% ROI 76.1% | test外勝者捕捉 13.8% 主力勝者捕捉 4.9%

注: 100円均等。候補選定はすべてレース前情報のみ。市場順は確定3連複オッズを使用。結果は候補選定に未使用。

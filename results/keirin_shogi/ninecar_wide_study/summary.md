# Nine-car v3.2 -> Wide forward study

No prediction-time odds/popularity. 100 yen per selected wide pair.

## Coverage
- 2024: model 2211, payout-covered 2165, participants 1303
- 2025: model 2201, payout-covered 2201, participants 1643
- 2026: model 941, payout-covered 941, participants 770

## Stable shortlist

| strategy | 2024 ROI | 2025 ROI | 2026H1 ROI | pooled ROI | profit | hit rate | avg tickets | +all |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| cross_joint_rank4to8 | 78.7% | 78.6% | 79.9% | 78.9% | -389,470 | 60.2% | 4.97 | NO |
| cross_joint_rank3to6 | 78.1% | 77.9% | 78.0% | 78.0% | -326,610 | 60.9% | 4.00 | NO |
| cross_joint_rank3to8 | 78.9% | 77.5% | 78.5% | 78.2% | -483,730 | 72.0% | 5.97 | NO |
| joint_rank3to5 | 79.1% | 77.9% | 77.4% | 78.2% | -242,870 | 52.9% | 3.00 | NO |
| cross_joint_rank3to5 | 79.5% | 76.8% | 76.7% | 77.8% | -248,040 | 51.8% | 3.00 | NO |
| joint_rank3to6 | 77.4% | 79.9% | 76.6% | 78.4% | -321,570 | 61.7% | 4.00 | NO |
| joint_rank2to8 | 76.6% | 78.9% | 76.8% | 77.7% | -580,900 | 80.4% | 7.00 | NO |
| joint_q120 | 77.3% | 77.9% | 76.5% | 77.4% | -684,580 | 89.8% | 8.15 | NO |
| cross_joint_top8 | 78.1% | 76.6% | 76.3% | 77.0% | -679,660 | 88.5% | 7.97 | NO |
| joint_rank3to8 | 76.2% | 80.6% | 79.1% | 78.7% | -473,850 | 75.2% | 6.00 | NO |
| cross_joint_rank2to8 | 78.9% | 76.4% | 76.1% | 77.2% | -590,400 | 78.2% | 6.97 | NO |
| joint_top8 | 76.1% | 78.8% | 76.8% | 77.5% | -670,350 | 89.8% | 8.00 | NO |
| cross_joint_q100 | 76.1% | 77.0% | 78.1% | 76.9% | -749,100 | 89.4% | 8.72 | NO |
| joint_q80 | 76.1% | 77.1% | 77.2% | 76.8% | -1,302,500 | 97.3% | 15.08 | NO |
| cross_joint_q120 | 77.5% | 76.6% | 75.9% | 76.8% | -610,300 | 86.2% | 7.07 | NO |
| joint_q70 | 75.9% | 75.9% | 75.9% | 75.9% | -1,569,530 | 98.4% | 17.54 | NO |
| cross_joint_q80 | 77.7% | 75.6% | 77.5% | 76.7% | -886,200 | 91.1% | 10.25 | NO |
| cross_joint_q70 | 77.8% | 75.6% | 76.3% | 76.5% | -950,560 | 91.7% | 10.89 | NO |
| cross_joint_top6 | 77.3% | 76.6% | 75.3% | 76.6% | -522,540 | 83.7% | 6.00 | NO |
| joint_rank4to8 | 75.0% | 82.1% | 80.2% | 79.2% | -386,720 | 63.6% | 5.00 | NO |

## Interpretation guard

A strategy is not adopted merely because it leads this table. 2024/2025/2026H1 are shown separately to expose regime dependence and payout concentration. The development_2024_selection_then_validation section selects only on 2024 and treats 2025/2026H1 as untouched validation. Pair-probability rank is also checked as a no-odds proxy for cheap/expensive-looking wide combinations.

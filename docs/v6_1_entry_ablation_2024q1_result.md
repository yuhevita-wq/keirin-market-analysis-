# v6.1 Entry Gate Ablation — 2024Q1

Status: development analysis. Formation held fixed to v8.4-F05 for every eligible race; only entry conditions vary.

Population: 1,191 complete races.

Actual v6.1 pass/fail conditions after LS defines A/B:
- PS_AB: top two adjacent-pair PS lines are exactly A and B.
- H_AB: top two H1 riders are exactly one from A and one from B.
- H_POS: top two H1 riders are leader or second wheel of their lines.
- H_RATIO: top H1 < 2 × second H1.

LS top-two lines are the A/B definition, not a separate pass/fail condition.

## Individual condition pass counts

| Condition | Pass races | Pass rate |
|---|---:|---:|
| PS_AB | 604 | 50.71% |
| H_AB | 535 | 44.92% |
| H_POS | 1,175 | 98.66% |
| H_RATIO | 688 | 57.77% |

## Main variants

| Variant | Races | Hit rate | Avg tickets | ROI | Profit | Profitable-hit share |
|---|---:|---:|---:|---:|---:|---:|
| No gate | 1,191 | 40.05% | 16.27 | 70.59% | -569,820 | 54.30% |
| PS only | 604 | 37.42% | 16.63 | 75.39% | -247,130 | 67.70% |
| H_AB only | 535 | 38.88% | 18.52 | 65.75% | -339,300 | 63.46% |
| H_POS only | 1,175 | 40.34% | 16.31 | 71.16% | -552,600 | 54.01% |
| H_RATIO only | 688 | 40.26% | 19.04 | 66.13% | -443,770 | 53.07% |
| PS + H_AB | 377 | 39.52% | 18.71 | 69.34% | -216,310 | 71.14% |
| Full v6.1 | 260 | 39.62% | 21.21 | 65.74% | -188,910 | 65.05% |

## Drop-one-condition analysis versus full v6.1

### Drop PS_AB
Newly admitted: 68 races.
- Hit rate: 39.71%
- Avg tickets: 21.85
- ROI: 50.95%
- Profit: -72,890
- Profitable-hit share: 40.74%
- Avg hit payout: ¥2,804

Interpretation: PS_AB is a strong protective filter in Q1.

### Drop H_AB
Newly admitted: 137 races.
- Hit rate: 33.58%
- Avg tickets: 15.17
- ROI: 87.35%
- Profit: -26,280
- Profitable-hit share: 63.04%
- Avg hit payout: ¥3,946
- Hits >= ¥10,000: 5

Interpretation: H_AB appears too restrictive as a hard gate in Q1. The excluded group has lower hit rate but materially lower ticket count and much better ROI than the retained full-v6.1 population.

### Drop H_POS
Newly admitted: 0 races.

Interpretation: H_POS is completely redundant conditional on the other full-v6.1 conditions in 2024Q1.

### Drop H_RATIO
Newly admitted: 117 races.
- Hit rate: 39.32%
- Avg tickets: 13.16
- ROI: 82.21%
- Profit: -27,400
- Profitable-hit share: 84.78%
- Avg hit payout: ¥2,752

Interpretation: H_RATIO is strongly suspect as a hard exclusion in Q1. It removes a group with similar hit rate, far fewer tickets, a much higher profitable-hit share, and better ROI.

## Q1 development conclusion

On 2024Q1 only:
1. Keep PS_AB as the strongest hard-gate candidate.
2. Remove H_POS from the gate candidate set because it adds no filtering after the other conditions.
3. Reconsider H_AB as a hard condition; use as a diagnostic/score candidate rather than mandatory exclusion.
4. Reconsider H_RATIO as a hard condition; its rejected group is especially attractive for the current objective of lower ticket count with retained hit rate and improved ROI.
5. Do not promote these Q1 findings to a final scheme without fixed-rule validation on later quarters.

Descriptive payout bands (¥5,000/¥10,000) are analysis outputs only, not entry thresholds.

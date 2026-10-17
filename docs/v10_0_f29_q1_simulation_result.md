# v10.0-F29 — 2024Q1 Simulation Result

## Execution
- Scheme: `v10.0-F29`
- Dataset: `2024Q1`
- Scheme changed for run: **No**
- GitHub Actions run: `33601024883`
- Job: `100154517048`
- Conclusion: **success**
- Only 2024Q1 was intentionally evaluated by this v10 workflow.

## Overall
| Metric | Result |
|---|---:|
| Population | 1,191 |
| Bet races | 533 |
| Hits | 446 |
| Hit rate | 83.68% |
| Tickets | 51,410 |
| Avg tickets / bet race | 96.45 |
| Stake | ¥5,141,000 |
| Payout | ¥3,787,710 |
| Profit | **-¥1,353,290** |
| ROI | **73.68%** |

## By group
| Group | Bets | Hits | Avg tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| QUALIFYING | 173 | 141 | 88.20 | ¥1,525,800 | ¥893,770 | **-¥632,030** | **58.58%** |
| GENERAL | 101 | 89 | 101.78 | ¥1,028,000 | ¥798,210 | **-¥229,790** | **77.65%** |
| SEMIFINAL | 109 | 99 | 95.52 | ¥1,041,200 | ¥1,083,020 | **+¥41,820** | **104.02%** |
| SPECIAL | 130 | 100 | 102.31 | ¥1,330,000 | ¥862,600 | **-¥467,400** | **64.86%** |
| FINAL | 20 | 17 | 108.00 | ¥216,000 | ¥150,110 | **-¥65,890** | **69.50%** |

## Decision counts
- Estimated ROI > 1.0 => BUY: 533 races
- Estimated ROI <= 1.0 => NO BET: 635 races
- No valid rectangular tickets: 23 races

## Estimated ROI diagnostic
Bought races had model-estimated ROI:
- min: 1.00025
- mean: 1.25689
- max: 2.29654

Actual aggregate ROI of those bought races was **0.73677**.

This demonstrates that the v10.0-F29 equal geometric market/fundamental pool is not calibrated as a true probability model for economic edge. Its `EstimatedROI > 1` gate substantially overstates value on 2024Q1.

## Formation breadth
Most bought formations were extremely broad. Ticket-count distribution included:
- 60 tickets: 97 races
- 80 tickets: 32 races
- 100 tickets: 22 races
- 120 tickets: 326 races

The average was **96.45 tickets per bought race**.

## Interpretation
This frozen Q1 run rejects v10.0-F29 as the desired practical universal betting scheme. It behaves primarily as a broad high-hit-rate coverage model rather than the intended market-psychology/value-selection system. The result should not be rescued by fitting Q1 thresholds after the fact.

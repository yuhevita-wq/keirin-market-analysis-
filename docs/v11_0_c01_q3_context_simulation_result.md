# v11.0-C01 — 2024Q3 Race-card Context Simulation

## Status
- Base betting engine: `v8.25-F26`
- Race-card layer: `v11.0-C01`
- Dataset: `2024Q3`
- User-requested simulation
- Context definition changed from Q1/Q2: **No**
- Bet logic changed vs v8.25: **No**
- GitHub Actions run: `33611852083`
- Job: `100188612298`
- Conclusion: **success**

Scientific caution: Q3 outcomes had already been observed earlier in the project during the v8.25 validation, so this is not a pristine project-level out-of-sample test. However, the v11.0-C01 context definitions were applied unchanged from Q1/Q2.

## Overall
| Metric | Result |
|---|---:|
| Population | 1,186 |
| Bet races | 162 |
| Hits | 28 |
| Hit rate | 17.28% |
| Tickets | 1,004 |
| Avg tickets | 6.20 |
| Stake | ¥100,400 |
| Payout | ¥87,440 |
| Profit | **-¥12,960** |
| ROI | **87.09%** |

## One-dimensional context summaries

### H state
- H_BALANCED: 115 bets / 13 hits / **-¥20,250 / ROI 68.94%**
- H_CONCENTRATED: 47 bets / 15 hits / **+¥7,290 / ROI 120.71%**

### Head context
- HEAD_CHALLENGED: 22 bets / 4 hits / **+¥14,840 / ROI 270.57%**
- HEAD_SUPPORTED: 140 bets / 24 hits / **-¥27,800 / ROI 69.68%**

### Line context
- LINE_CHALLENGED: 44 bets / 8 hits / **-¥12,520 / ROI 57.41%**
- LINE_SUPPORTED: 118 bets / 20 hits / **-¥440 / ROI 99.38%**

## Full context
| Context | Bets | Hits | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|
| H_BALANCED · HEAD_CHALLENGED · LINE_CHALLENGED | 7 | 0 | ¥1,800 | ¥0 | **-¥1,800** | **0.00%** |
| H_BALANCED · HEAD_CHALLENGED · LINE_SUPPORTED | 12 | 2 | ¥4,300 | ¥16,820 | **+¥12,520** | **391.16%** |
| H_BALANCED · HEAD_SUPPORTED · LINE_CHALLENGED | 23 | 4 | ¥17,700 | ¥13,090 | **-¥4,610** | **73.95%** |
| H_BALANCED · HEAD_SUPPORTED · LINE_SUPPORTED | 73 | 7 | ¥41,400 | ¥15,040 | **-¥26,360** | **36.33%** |
| H_CONCENTRATED · HEAD_CHALLENGED · LINE_CHALLENGED | 1 | 1 | ¥600 | ¥1,090 | **+¥490** | **181.67%** |
| H_CONCENTRATED · HEAD_CHALLENGED · LINE_SUPPORTED | 2 | 1 | ¥2,000 | ¥5,630 | **+¥3,630** | **281.50%** |
| H_CONCENTRATED · HEAD_SUPPORTED · LINE_CHALLENGED | 13 | 3 | ¥9,300 | ¥2,700 | **-¥6,600** | **29.03%** |
| H_CONCENTRATED · HEAD_SUPPORTED · LINE_SUPPORTED | 31 | 10 | ¥23,300 | ¥33,070 | **+¥9,770** | **141.93%** |

## Cross-quarter fixed-context comparison

### HEAD_SUPPORTED
- Q1: ROI **140.35%**
- Q2: ROI **137.03%**
- Q3: ROI **69.68%**

This one-dimensional context does **not** reproduce in Q3.

### H_CONCENTRATED · HEAD_SUPPORTED · LINE_SUPPORTED
- Q1: 35 bets / 11 hits / ROI **173.54%**
- Q2: 47 bets / 22 hits / ROI **115.43%**
- Q3: 31 bets / 10 hits / ROI **141.93%**

This full context remained profitable in all three quarters under the unchanged v11 context definitions. It is a stronger stability candidate than HEAD_SUPPORTED alone, but it must not yet be promoted to a live buy/no-buy rule solely from these observed periods.

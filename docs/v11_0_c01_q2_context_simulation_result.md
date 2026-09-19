# v11.0-C01 — 2024Q2 Race-card Context Simulation

## Status
- Base betting engine: `v8.25-F26`
- Race-card layer: `v11.0-C01`
- Dataset: `2024Q2`
- Context definition changed from Q1: **No**
- Bet logic changed vs v8.25: **No**
- GitHub Actions run: `33611257471`
- Job: `100186689868`
- Conclusion: **success**

## Overall

| Metric | Result |
|---|---:|
| Population | 1,172 |
| Bet races | 186 |
| Hits | 45 |
| Hit rate | 24.19% |
| Tickets | 1,164 |
| Avg tickets | 6.26 |
| Stake | ¥116,400 |
| Payout | ¥145,640 |
| Profit | **+¥29,240** |
| ROI | **125.12%** |

## Full context

| Context | Bets | Hits | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|
| H_BALANCED · HEAD_CHALLENGED · LINE_CHALLENGED | 5 | 0 | ¥1,200 | ¥0 | **-¥1,200** | **0.00%** |
| H_BALANCED · HEAD_CHALLENGED · LINE_SUPPORTED | 9 | 1 | ¥9,200 | ¥3,130 | **-¥6,070** | **34.02%** |
| H_BALANCED · HEAD_SUPPORTED · LINE_CHALLENGED | 24 | 5 | ¥14,700 | ¥33,690 | **+¥18,990** | **229.18%** |
| H_BALANCED · HEAD_SUPPORTED · LINE_SUPPORTED | 87 | 14 | ¥49,500 | ¥63,830 | **+¥14,330** | **128.95%** |
| H_CONCENTRATED · HEAD_CHALLENGED · LINE_CHALLENGED | 2 | 0 | ¥2,000 | ¥0 | **-¥2,000** | **0.00%** |
| H_CONCENTRATED · HEAD_SUPPORTED · LINE_CHALLENGED | 12 | 3 | ¥8,500 | ¥8,860 | **+¥360** | **104.24%** |
| H_CONCENTRATED · HEAD_SUPPORTED · LINE_SUPPORTED | 47 | 22 | ¥31,300 | ¥36,130 | **+¥4,830** | **115.43%** |

## One-dimensional context

### H state
- H_BALANCED: 125 bets / 20 hits / **+¥26,050 / ROI 134.92%**
- H_CONCENTRATED: 61 bets / 25 hits / **+¥3,190 / ROI 107.63%**

### Head context
- HEAD_CHALLENGED: 16 bets / 1 hit / **-¥9,270 / ROI 25.24%**
- HEAD_SUPPORTED: 170 bets / 44 hits / **+¥38,510 / ROI 137.03%**

### Line context
- LINE_CHALLENGED: 43 bets / 8 hits / **+¥16,150 / ROI 161.17%**
- LINE_SUPPORTED: 143 bets / 37 hits / **+¥13,090 / ROI 114.54%**

## Cross-quarter observation with fixed definitions

Q1 → Q2:
- HEAD_SUPPORTED: 140.35% → 137.03% — strong stability.
- HEAD_CHALLENGED: 120.54% → 25.24% — not stable.
- LINE_SUPPORTED: 143.95% → 114.54% — positive in both.
- LINE_CHALLENGED: 125.33% → 161.17% — positive in both, but magnitude varies.
- H_BALANCED + HEAD_SUPPORTED + LINE_SUPPORTED: 137.99% → 128.95% — positive in both and relatively stable.
- H_CONCENTRATED + HEAD_SUPPORTED + LINE_SUPPORTED: 173.54% → 115.43% — positive in both, but weaker in Q2.
- H_CONCENTRATED + HEAD_SUPPORTED + LINE_CHALLENGED: 259.25% → 104.24% — remains above break-even but the Q1 magnitude did not reproduce.
- H_BALANCED + HEAD_SUPPORTED + LINE_CHALLENGED: 31.20% → 229.18% — sign reversal, therefore not a stable directional rule.

No context split is promoted to a buy/no-buy rule from Q1+Q2 alone in this file.

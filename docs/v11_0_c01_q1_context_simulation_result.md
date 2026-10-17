# v11.0-C01 — 2024Q1 Race-card Context Simulation

## Status
- Base betting engine: `v8.25-F26`
- Race-card layer: `v11.0-C01`
- Dataset: `2024Q1`
- User-requested simulation
- Bet logic changed vs v8.25: **No**
- GitHub Actions run: `33608607030`
- Job: `100178239254`
- Conclusion: **success**

The race-card layer is observational here. It tags the existing v8.25 market-psychology bets by whether the market's leading head and leading line are supported or challenged by deterministic race-card fundamentals. It does not add, remove, or prune tickets.

## Overall

| Metric | Result |
|---|---:|
| Population | 1,191 |
| Bet races | 171 |
| Hits | 34 |
| Hit rate | 19.88% |
| Tickets | 1,011 |
| Avg tickets | 5.91 |
| Stake | ¥101,100 |
| Payout | ¥140,430 |
| Profit | **+¥39,330** |
| ROI | **138.90%** |

## Context definitions

`H_CONCENTRATED`: market H1 >= 2 * H2.

`H_BALANCED`: market H1 < 2 * H2.

`HEAD_SUPPORTED`: market H1 is inside the natural top fundamental head block derived from the largest adjacent F-score gap.

`HEAD_CHALLENGED`: market H1 is outside that block.

`LINE_SUPPORTED`: market's strongest trio-supported line is also the strongest line by summed race-card F.

`LINE_CHALLENGED`: the market's strongest line is not strongest by summed race-card F.

## Bought-race performance by full context

| Context | Bets | Hits | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|
| H_BALANCED · HEAD_CHALLENGED · LINE_CHALLENGED | 4 | 2 | ¥3,200 | ¥8,920 | **+¥5,720** | **278.75%** |
| H_BALANCED · HEAD_CHALLENGED · LINE_SUPPORTED | 8 | 0 | ¥3,000 | ¥0 | **-¥3,000** | **0.00%** |
| H_BALANCED · HEAD_SUPPORTED · LINE_CHALLENGED | 21 | 2 | ¥15,000 | ¥4,680 | **-¥10,320** | **31.20%** |
| H_BALANCED · HEAD_SUPPORTED · LINE_SUPPORTED | 90 | 12 | ¥46,700 | ¥64,440 | **+¥17,740** | **137.99%** |
| H_CONCENTRATED · HEAD_CHALLENGED · LINE_CHALLENGED | 2 | 0 | ¥1,200 | ¥0 | **-¥1,200** | **0.00%** |
| H_CONCENTRATED · HEAD_SUPPORTED · LINE_CHALLENGED | 11 | 7 | ¥8,000 | ¥20,740 | **+¥12,740** | **259.25%** |
| H_CONCENTRATED · HEAD_SUPPORTED · LINE_SUPPORTED | 35 | 11 | ¥24,000 | ¥41,650 | **+¥17,650** | **173.54%** |

## One-dimensional context summaries

### H state
- H_BALANCED: 123 bets / 16 hits / **+¥10,140 / ROI 114.93%**
- H_CONCENTRATED: 48 bets / 18 hits / **+¥29,190 / ROI 187.92%**

### Head context
- HEAD_CHALLENGED: 14 bets / 2 hits / **+¥1,520 / ROI 120.54%**
- HEAD_SUPPORTED: 157 bets / 32 hits / **+¥37,810 / ROI 140.35%**

### Line context
- LINE_CHALLENGED: 38 bets / 11 hits / **+¥6,940 / ROI 125.33%**
- LINE_SUPPORTED: 133 bets / 23 hits / **+¥32,390 / ROI 143.95%**

## Interpretation guardrail

These Q1 context splits are diagnostic only. They must not be promoted directly into a new buy/no-buy rule from Q1 results alone. The safe next scientific question is whether the same psychological meanings persist in later periods when the definitions are held fixed.

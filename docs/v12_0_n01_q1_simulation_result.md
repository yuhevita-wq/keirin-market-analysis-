# v12.0-N01 — 2024Q1 Simulation Result

## Status
- Scheme: `v12.0-N01 MARKET CONNECTION TOPOLOGY`
- Dataset: `2024Q1`
- User-requested simulation
- Scheme changed for run: **No**
- GitHub Actions run: `33627030673`
- Job: `100237087735`
- Scientific status: development test; Q1 is not pristine project-level OOS because earlier Q1-Q3 outcomes were already known when v12 was designed.
- Conclusion: **REJECTED AS CURRENT BETTING SCHEME**

## Overall

| Metric | Result |
|---|---:|
| Population | 1,191 |
| Bet races | **1,191** |
| Buy rate | **100.00%** |
| Hits | 685 |
| Hit rate | 57.51% |
| Tickets | 44,842 |
| Avg tickets / race | **37.65** |
| Stake | ¥4,484,200 |
| Payout | ¥3,428,300 |
| Profit | **-¥1,055,900** |
| ROI | **76.45%** |

## By group

| Group | Bets | Hits | Avg tickets | Profit | ROI |
|---|---:|---:|---:|---:|---:|
| QUALIFYING | 360 | 208 | 35.77 | -¥223,420 | 82.65% |
| GENERAL | 214 | 130 | 39.34 | -¥151,860 | 81.96% |
| SEMIFINAL | 204 | 119 | 37.45 | -¥288,440 | 62.25% |
| SPECIAL | 346 | 191 | 37.77 | -¥330,510 | 74.71% |
| FINAL | 67 | 37 | 42.36 | -¥61,670 | 78.27% |

Every race group was negative.

## Structural diagnosis

The fresh-root topology concept did not produce a meaningful participation gate. Every one of the 1,191 eligible Q1 races produced at least one accepted two-view connection, so the scheme bought 100% of the population.

The average formation expanded to 37.65 exact trifecta tickets per race. This is structurally incompatible with the intended objective: identify a market-psychology thesis and express it selectively, rather than broadly cover the outcome space.

This failure resembles the earlier v10 failure in economic shape: high coverage and materially sub-break-even ROI. The mechanism is different, but the practical result is the same broad-coverage problem.

## Guardrail

Do not rescue v12.0-N01 by mining Q1 ticket-count buckets, head-count buckets, race groups, or adding a fitted maximum-ticket cutoff after seeing these results. Some individual buckets are positive in Q1, but selecting them now would be outcome-informed fitting.

The correct conclusion is that the current connection-acceptance architecture is too permissive and must not remain as an inherited active betting layer.

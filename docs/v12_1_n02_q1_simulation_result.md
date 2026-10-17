# v12.1-N02 — 2024Q1 Simulation Result

## Status
- Scheme: `v12.1-N02 — ANOMALY FIRST`
- Dataset: `2024Q1`
- User-requested simulation
- Scheme changed for run: **No**
- GitHub Actions run: `33632411788`
- Job: `100254863609`
- Scientific status: development test; Q1 is not pristine project-level OOS because earlier project outcomes were already known before v12.1 was designed.

## Overall

| Metric | Result |
|---|---:|
| Population | 1,191 |
| Bet races | **84** |
| Buy rate | **7.05%** |
| Hits | **0** |
| Hit rate | **0.00%** |
| Tickets | 212 |
| Avg tickets / race | **2.52** |
| Stake | ¥21,200 |
| Payout | **¥0** |
| Profit | **-¥21,200** |
| ROI | **0.00%** |

## Race-selection behavior

The race gate did materially restrict participation:

- `NO_ROBUST_UNDERATTACHED_CONNECTION_ANOMALY`: 1,107 races
- `N02_ROBUST_UNDERATTACHED_CONNECTION`: 84 races

This is structurally different from rejected v12.0-N01, which bought all 1,191 Q1 races.

Selected anomaly counts per purchased race:

- 1 selected connection: 68 races
- 2 selected connections: 11 races
- 3 selected connections: 4 races
- 4 selected connections: 1 race

Thus the participation gate and ticket-count control functioned mechanically, but the selected betting thesis failed completely in Q1.

## By race group

Every group had zero hits:

- FINAL: 6 races / 14 tickets / 0 hits
- GENERAL: 24 races / 56 tickets / 0 hits
- QUALIFYING: 23 races / 60 tickets / 0 hits
- SEMIFINAL: 10 races / 32 tickets / 0 hits
- SPECIAL: 21 races / 50 tickets / 0 hits

## Interpretation

The current N02 thesis selects connections where the trio market strongly packages `{head,a,b}` but the head-conditioned trifecta market attaches that same pair to the head abnormally less than the race's normal connection pattern, while both market and race-card head hierarchies support the head.

Q1 shows that converting this **under-attachment anomaly** directly into a trifecta bet on `head-a-b / head-b-a` is not supported. The gate is selective, but the direction/meaning assigned to the anomaly is wrong or incomplete.

Do not rescue N02 by fitting a different modified-z cutoff, race-type branch, ticket-count bucket, or other Q1-derived filter. Any next redesign should preserve useful feature extraction and the race-selection-first principle while reconsidering what an extreme cross-market residual actually means before turning it into a ticket.

# v8.21-F22 FINAL Heat Quarantine — 2024Q1+Q2 Development Result

Status: DEVELOPMENT. Q1 and Q2 have both been observed during development. The next untouched validation set for the current v8 lineage is Q3.

## Decision

FINAL (`Ｓ級決勝`) is changed to **NO BET / diagnostic only**.

The decision is not a fitted cutoff. It follows repeated failure of semantic market-only FINAL hypotheses to reproduce between Q1 and Q2:

1. total 3連複 / 3連単 vote heat,
2. location of head and formation heat,
3. formation price viability / overheat,
4. line-count and line-size structure,
5. 3連複 vs collapsed-3連単 set-ranking agreement,
6. whole-rider structural price compression,
7. head certainty × conditional tail diffusion.

The current FINAL candidates are universally hotter than same-day same-track S-class peers, but the extra heat does not produce a reproducible value edge. In Q2 the three current-rule hits were each race-level losing at 100 yen per ticket, indicating that consensus was often too efficiently priced.

## v8.21-F22 rule change

All v8.20-F21 branches remain unchanged except FINAL.

- FINAL population remains in diagnostics.
- Prior v8.20 FINAL candidate formation is retained as metadata when available.
- Betting action is always zero.
- No individual ticket pruning.
- No outcome-fitted FINAL threshold.

## Whole scheme results

| Dataset | Population | Bets | Hits | Tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 1191 | 135 | 43 | 1354 | 135,400 | 152,490 | **+17,090** | **112.62%** |
| 2024Q2 | 1172 | 142 | 48 | 1334 | 133,400 | 134,880 | **+1,480** | **101.11%** |

## Q1 by group

| Group | Bets | Hits | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|
| QUALIFYING | 63 | 16 | 57,800 | 58,500 | +700 | 101.21% |
| GENERAL | 7 | 5 | 5,200 | 11,970 | +6,770 | 230.19% |
| SEMIFINAL | 56 | 19 | 62,800 | 70,300 | +7,500 | 111.94% |
| SPECIAL (unified INITIAL_SPECIAL only) | 9 | 3 | 9,600 | 11,720 | +2,120 | 122.08% |
| FINAL | 0 | 0 | 0 | 0 | 0 | — |

## Q2 by group

| Group | Bets | Hits | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|
| QUALIFYING | 51 | 23 | 41,600 | 27,010 | **-14,590** | 64.93% |
| GENERAL | 16 | 8 | 11,100 | 17,210 | +6,110 | 155.05% |
| SEMIFINAL | 59 | 14 | 63,100 | 71,620 | +8,520 | 113.50% |
| SPECIAL (unified INITIAL_SPECIAL only) | 16 | 3 | 17,600 | 19,040 | +1,440 | 108.18% |
| FINAL | 0 | 0 | 0 | 0 | 0 | — |

## Effect of FINAL quarantine

Under v8.20-F21, Q2 overall was -11,910 yen. The prior FINAL candidate branch contributed -13,390 yen in Q2. With FINAL quarantined and all other branches unchanged, v8.21-F22 becomes **+1,480 yen / ROI 101.11%** in Q2.

In Q1 the prior FINAL branch had contributed +2,860 yen; quarantining it lowers Q1 profit but Q1 remains positive at **+17,090 yen / ROI 112.62%**.

## Current interpretation

FINAL market heat is real, but in the observed development data it behaves more like an efficiency / overpricing warning for this market-center-following rectangular scheme than a robust entry edge. Until a structurally different FINAL model is discovered without outcome-fitted rescue thresholds, FINAL remains no-bet.

Current remaining weakness is QUALIFYING: Q2 is -14,590 yen. Adding H_AB to QUALIFYING was already rejected because it worsened both the structural interpretation and Q1/Q2 results. Future QUALIFYING narrowing must use a different market concept.

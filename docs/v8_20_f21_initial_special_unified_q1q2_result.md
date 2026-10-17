# v8.20-F21 INITIAL_SPECIAL Unified: 2024Q1 + Q2 Development Result

Status: DEVELOPMENT. Q1 and Q2 have both been observed and used in development; neither is untouched OOS for v8.20-F21.

Workflow run: 33572671822
Job: 100069789761
Runner head commit: 8678f6cda46b624c59f099ed760c47ff530577f8
Engine: `src/keirin_market_analysis/v8_20_f21_initial_special_unified.py`
Evaluator: `src/keirin_market_analysis/evaluate_v8_20_q1q2.py`

## Correction

`Ｓ級初特選` and `Ｓ級初日特選` are not treated as separate market regimes. They are unified as the same `INITIAL_SPECIAL` race type for scheme design and evaluation.

The prior v8.19-F20 conclusion that `Ｓ級初特選` could be active while `Ｓ級初日特選` was quarantined is withdrawn because it depended on an invalid label split.

## Unified INITIAL_SPECIAL rule

Both labels use the exact same rule:

1. F09 `PS_AB` must pass.
2. Require `H_AB == H_CONCENTRATED`.
3. Translate H state into first-place pool:
   - `H1 >= 2*H2`: first = H1 only.
   - `H1 < 2*H2`: first = top two H riders.
4. Keep F09 second/third pools.
5. Whole-rider price contraction may affect only second/third pools; the H-derived first pool is protected.
6. No individual-ticket pruning, rider ability, result, payout, or fitted numeric cutoff is used.

## Unified INITIAL_SPECIAL result

| Dataset | Population | Bets | Hits | Hit rate | Tickets | Avg tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 63 | 9 | 3 | 33.3333% | 96 | 10.6667 | 9,600 | 11,720 | +2,120 | 122.0833% |
| 2024Q2 | 67 | 16 | 3 | 18.7500% | 176 | 11.0000 | 17,600 | 19,040 | +1,440 | 108.1818% |
| Q1+Q2 | 130 | 25 | 6 | 24.0000% | 272 | 10.8800 | 27,200 | 30,760 | +3,560 | 113.0882% |

The unified branch is positive in both observed quarters.

## Label-level display only, not separate strategy branches

For audit only, the source labels still appear separately in the raw dataset:

### 2024Q1
- `Ｓ級初特選`: 46 population, 7 bets, 2 hits, +1,160 yen, ROI 116.8116%.
- `Ｓ級初日特選`: 17 population, 2 bets, 1 hit, +960 yen, ROI 135.5556%.

### 2024Q2
- `Ｓ級初特選`: 47 population, 12 bets, 3 hits, +6,640 yen, ROI 153.5484%.
- `Ｓ級初日特選`: 20 population, 4 bets, 0 hits, -5,200 yen, ROI 0%.

These label-level outcomes must not be used to create different betting rules because the two labels represent the same race regime for this project.

## Full v8.20-F21 result

### 2024Q1
- Population: 1191
- Bet races: 151
- Hits: 50
- Hit rate: 33.1126%
- Tickets: 1572
- Average tickets/race: 10.4106
- Stake: 157,200 yen
- Payout: 177,150 yen
- Profit: +19,950 yen
- ROI: 112.6908%

By group:
- QUALIFYING: +700 yen, ROI 101.2111%
- GENERAL: +6,770 yen, ROI 230.1923%
- SEMIFINAL: +7,500 yen, ROI 111.9427%
- SPECIAL (unified INITIAL_SPECIAL only): +2,120 yen, ROI 122.0833%
- FINAL: +2,860 yen, ROI 113.1193%

### 2024Q2
- Population: 1172
- Bet races: 152
- Hits: 51
- Hit rate: 33.5526%
- Tickets: 1496
- Average tickets/race: 9.8421
- Stake: 149,600 yen
- Payout: 137,690 yen
- Profit: -11,910 yen
- ROI: 92.0388%

By group:
- QUALIFYING: -14,590 yen, ROI 64.9279%
- GENERAL: +6,110 yen, ROI 155.0450%
- SEMIFINAL: +8,520 yen, ROI 113.5024%
- SPECIAL (unified INITIAL_SPECIAL only): +1,440 yen, ROI 108.1818%
- FINAL: -13,390 yen, ROI 17.3457%

## Interpretation

The classification correction is accepted. `Ｓ級初特選` and `Ｓ級初日特選` are one branch from now on. The unified INITIAL_SPECIAL branch remains positive in both Q1 and Q2, although the overall Q2 scheme is still negative because QUALIFYING and FINAL remain large loss sources.

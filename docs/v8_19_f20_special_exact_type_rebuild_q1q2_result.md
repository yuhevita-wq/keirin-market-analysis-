# v8.19-F20 SPECIAL Exact-Type Rebuild: 2024Q1 + Q2 Development Result

Status: DEVELOPMENT. Both Q1 and Q2 have been used in branch development and are not untouched OOS validation for this scheme.

Workflow run: 33570643199
Job: 100063625143
Runner head commit: 0b0f4d1c67719415a5128a749f028f0782589e16
Engine: `src/keirin_market_analysis/v8_19_f20_special_exact_type_rebuild.py`
Evaluator: `src/keirin_market_analysis/evaluate_v8_19_q1q2.py`

## SPECIAL redesign

The former SPECIAL group is no longer treated as one betting branch.

- `Ｓ級初特選`: active rebuilt branch.
- `Ｓ級特選`: no bet / quarantined because candidate H interpretations changed sign between Q1 and Q2.
- `Ｓ級初日特選`: no bet / quarantined because the branch was unstable and the adaptive candidate had zero Q2 hits.
- `Ｓ級選抜`: remains disagreement-model diagnostic only / no bet.

### Active `Ｓ級初特選` rule

1. F09 `PS_AB` must pass.
2. Require pre-existing H-state consistency: `H_AB == H_CONCENTRATED`.
3. Translate the H state directly into the first-place pool:
   - `H1 >= 2*H2`: first-place pool = H1 only.
   - `H1 < 2*H2`: first-place pool = top two H riders.
4. Second- and third-place pools remain the F09 market-cliff pools.
5. After the clean formation exists, price compression may remove whole riders only from second/third pools. The H-derived first pool is protected.
6. No individual-ticket pruning, rider ability, result, payout, or fitted Q1/Q2 numeric threshold is used to generate the bet.

## SPECIAL result

| Dataset | Bets | Hits | Hit rate | Tickets | Avg tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 7 | 2 | 28.5714% | 69 | 9.8571 | 6,900 | 8,060 | +1,160 | 116.8116% |
| 2024Q2 | 12 | 3 | 25.0000% | 124 | 10.3333 | 12,400 | 19,040 | +6,640 | 153.5484% |
| Q1+Q2 | 19 | 5 | 26.3158% | 193 | 10.1579 | 19,300 | 27,100 | +7,800 | 140.4145% |

The active SPECIAL branch is therefore positive in both observed development quarters.

## Diagnostic reason for exact-type split

### Ordinary `Ｓ級特選`

Current branch (PS_AB + H_CONCENTRATED, unchanged F09 formation):
- Q1: +5,680 yen, ROI 117.0060%.
- Q2: -6,940 yen, ROI 75.5634%.
- Combined: -1,260 yen, ROI 97.9612%.

H1 anchor:
- Q1: -7,550 yen, ROI 41.0156%.
- Q2: +5,400 yen, ROI 140.9091%.

H1 anchor + H_AB:
- Q1: -2,680 yen, ROI 54.5763%.
- Q2: +2,170 yen, ROI 129.7260%.

Semifinal-style H-adaptive + H_AB:
- Q1: -31,860 yen, ROI 43.6106%.
- Q2: -9,060 yen, ROI 88.3697%.

No tested semantic H translation was stable across both quarters, so `Ｓ級特選` is quarantined rather than forced into a fitted rule.

### `Ｓ級初日特選`

The H-state-consistent adaptive first-pool candidate:
- Q1: 2 bets, 1 hit, +960 yen, ROI 135.5556%.
- Q2: 4 bets, 0 hits, -5,200 yen, ROI 0%.

This exact type is therefore quarantined.

## Full v8.19-F20 result

Only SPECIAL changed from v8.18-F19. The rebuilt semifinal and all other branches are inherited unchanged.

### 2024Q1

- Population: 1191
- Bet races: 149
- Hits: 49
- Hit rate: 32.8859%
- Tickets: 1545
- Average tickets/race: 10.3691
- Stake: 154,500 yen
- Payout: 173,490 yen
- Profit: +18,990 yen
- ROI: 112.2913%

By group:
- QUALIFYING: +700 yen, ROI 101.2111%
- GENERAL: +6,770 yen, ROI 230.1923%
- SEMIFINAL: +7,500 yen, ROI 111.9427%
- SPECIAL: +1,160 yen, ROI 116.8116%
- FINAL: +2,860 yen, ROI 113.1193%

Every active group is positive in Q1.

### 2024Q2

- Population: 1172
- Bet races: 148
- Hits: 51
- Hit rate: 34.4595%
- Tickets: 1444
- Average tickets/race: 9.7568
- Stake: 144,400 yen
- Payout: 137,690 yen
- Profit: -6,710 yen
- ROI: 95.3532%

By group:
- QUALIFYING: -14,590 yen, ROI 64.9279%
- GENERAL: +6,110 yen, ROI 155.0450%
- SEMIFINAL: +8,520 yen, ROI 113.5024%
- SPECIAL: +6,640 yen, ROI 153.5484%
- FINAL: -13,390 yen, ROI 17.3457%

Compared with v8.18-F19 Q2 (-31,940 yen, ROI 82.3243%), v8.19-F20 improves Q2 profit by +25,230 yen and raises ROI by about 13.029 percentage points while reducing bets from 180 to 148 and tickets from 1807 to 1444.

## Interpretation

The SPECIAL rebuild succeeds as a development branch: SPECIAL itself is positive in both Q1 and Q2. The improvement came from respecting exact race type instead of treating all special-stage races as one market regime.

The full scheme is still -6,710 yen in Q2. The remaining negative groups are QUALIFYING (-14,590 yen) and FINAL (-13,390 yen). No untouched validation result for v8.19-F20 is recorded here, and the next validation set must not be tuned after observation.

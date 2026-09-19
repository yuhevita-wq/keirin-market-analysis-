# v9.2-F28 — Fundamental Line-Set Mispricing Gate (2024Q1–Q3 Development)

## Status

**DEVELOPMENT — NOT OUT-OF-SAMPLE VALIDATION**

v9.2-F28 was designed after the v8.25-F26 2024Q3 out-of-sample result had already been observed. Therefore 2024Q1, Q2, and Q3 are all development data for v9.2-F28. 2024Q4 remains untouched and is reserved for a later fixed validation after v9.2-F28 is frozen.

## Why race-card data are eligible

The stored `entries.csv` values were collected later from historical Rakuten K-Dreams race pages, but the values themselves behave as historical race/meeting snapshots rather than current-profile values copied backward. For the same rider, score, B count, rates, and age change across historical dates while remaining stable within a meeting. The raw historical row cells are also retained in `raw_row_json`.

The fundamental layer excludes `prediction_mark` and `evaluation` because they are external prediction/evaluation fields rather than independent race-card fundamentals.

## Universal fundamental definition

The same transformation is used for every race type. No race-type-specific fundamental weights are used.

Inputs:
- `score`
- `top3_rate`
- `b_count`
- `nige_count`
- `makuri_count`
- `sashi_count`
- `mark_count`
- `line_position`
- `line_size`

Within each seven-rider race, each numeric performance field is converted to an ordinal rank scaled from worst=0 to best=1, with ties receiving the same value.

- `attack = mean(rank(B), rank(nige), rank(makuri))`
- `follow = mean(rank(sashi), rank(mark))`
- `role_fit = attack` for a line leader
- `role_fit = follow` for a line follower
- `role_fit = mean(attack, follow)` for a singleton
- `F = mean(rank(score), rank(top3_rate), role_fit)`

Fundamental line support:

`FundLS(L) = sum(F_i for riders i in line L)`

Market line support remains the existing trio-market definition:

`MarketLS(L) = sum(S_i for riders i in line L)`

where `S_i` is the rider support derived from the normalized 3-way-combination market.

## v9.2-F28 gate

1. Build the complete existing v8.25-F26 market candidate first.
2. If v8.25-F26 says no bet, v9.2-F28 also says no bet.
3. Rank all predicted lines by `MarketLS`.
4. Independently rank the same lines by `FundLS`.
5. Compare the **set of the top two lines**, ignoring their internal order.
6. If the market top-two line set **equals** the fundamental top-two line set: **NO BET**.
7. If the two top-two line sets **differ**: **BUY the exact original v8.25-F26 tickets unchanged**.

The race-card layer does not create a formation, change a positional set, or prune individual tickets. It is only an independent race-entry / mispricing gate.

## Guardrails

- No fitted numeric cutoff.
- No race-type-specific fundamental weights.
- No `prediction_mark` or `evaluation` in the fundamental score.
- No result or payout used to build the candidate.
- No individual ticket pruning at the v9 layer.
- Existing v8.25-F26 ticket formation remains unchanged when the gate passes.

## Rejected predecessor: v9.0-F27

An earlier attempt used the fundamental score to directly veto first-position rider branches. It was rejected because it worsened the previously observed Q3 result from v8.25-F26 ROI 87.09% to approximately ROI 67.46%. This indicated that race-card fundamentals should not directly replace the market's head judgement; doing so can remove valuable market/fundamental disagreement.

## Development results

| Quarter | Bets | Hits | Hit rate | Tickets | Avg tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 36 | 6 | 16.67% | 139 | 3.86 | 13,900 | 46,850 | **+32,950** | **337.05%** |
| 2024Q2 | 28 | 6 | 21.43% | 143 | 5.11 | 14,300 | 15,590 | **+1,290** | **109.02%** |
| 2024Q3 | 31 | 5 | 16.13% | 186 | 6.00 | 18,600 | 27,510 | **+8,910** | **147.90%** |
| **Q1–Q3 combined** | **95** | **17** | **17.89%** | **468** | **4.93** | **46,800** | **89,950** | **+43,150** | **192.20%** |

## Interpretation

The same semantic market-vs-fundamental disagreement gate is profitable in all three development quarters without changing the fundamental formula, adding a quarter-specific cutoff, or altering the underlying market ticket formation.

This is evidence that the independent race-card layer may be useful as a **mispricing detector** rather than as a direct finishing-order predictor. The signal is specifically that the market and the historical race-card fundamentals disagree over which two predicted lines are structurally strongest.

This is promising development evidence, not validation. Because Q3 was already observed before v9.2-F28 was designed, Q3 cannot be used as an out-of-sample claim for this scheme. The next valid untouched test is 2024Q4, after the scheme is frozen without further Q1–Q3 tuning.

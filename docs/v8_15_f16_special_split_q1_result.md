# v8.15-F16 SPECIAL split rebuild — 2024Q1 development result

Status: DEVELOPMENT / Q1 in-sample
Actions run: 33514648086
Job: 99878510129
Population: 1191 complete F1 S-class seven-car races

## Fixed branch rules

- QUALIFYING: unchanged from v8.14-F15.
- GENERAL: unchanged from v8.14-F15.
- SEMIFINAL: unchanged from v8.14-F15.
- SELECTION (`Ｓ級選抜`): no bet / branch quarantined. Population 98.
- SPECIAL (`Ｓ級特選`): PS_AB + H_CONCENTRATED, where H_CONCENTRATED means `H1_top >= 2 * H1_second`; then keep the unchanged F09 market-cliff formation. H_AB is diagnostic only. No forced H1 anchor preservation.
- INITIAL_SPECIAL (`Ｓ級初特選` + `Ｓ級初日特選`): PS_AB + state consistency `H_AB == H_CONCENTRATED`; then keep the unchanged F09 market-cliff formation.
- FINAL: still quarantined / no bet.
- No individual-ticket pruning, no per-ticket odds cutoff, no fixed place counts, no nesting requirement.

## Overall Q1 result

- Bet races: 228
- Hits: 80
- Hit rate: 35.087719%
- Tickets: 2325
- Average tickets per race: 10.197368
- Stake: 232,500 yen
- Payout: 258,140 yen
- Profit: **+25,640 yen**
- ROI: **111.027957%**
- Acceptance: **PASS_Q1_PROFIT**

## SPECIAL split result

### SELECTION / 選抜

- Population: 98
- Bet races: 0
- Rule: quarantine / no bet pending a distinct market model.
- Previous diagnostic: PS_AB baseline and semantic formation-price viability both remained severely negative, so no Q1-fitted rescue rule was added.

### SPECIAL / 特選

- Population: 185
- Bet races: 30
- Hits: 10
- Hit rate: 33.333333%
- Tickets: 334
- Average tickets: 11.133333
- Stake: 33,400 yen
- Payout: 39,080 yen
- Profit: **+5,680 yen**
- ROI: **117.005988%**

Important negative diagnostic: forcing GENERAL-style H1 anchor preservation reduced this branch to ROI about 40%, so ordinary 特選 must not inherit the GENERAL formation rule.

### INITIAL_SPECIAL / 初特選系

Includes both `Ｓ級初特選` and `Ｓ級初日特選`.

- Population: 63
- Bet races: 17
- Hits: 7
- Hit rate: 41.176471%
- Tickets: 135
- Average tickets: 7.941176
- Stake: 13,500 yen
- Payout: 24,560 yen
- Profit: **+11,060 yen**
- ROI: **181.925926%**

The decisive signal is the pre-race H-state consistency itself. Extra anchor preservation or structural price-knee compression did not change these 17 selected races.

## All active branches in v8.15-F16 Q1

- QUALIFYING: 63 races, 16 hits, +700 yen, ROI 101.211073%
- GENERAL: 7 races, 5 hits, +6,770 yen, ROI 230.192308%
- SEMIFINAL: 111 races, 42 hits, +1,430 yen, ROI 101.166395%
- SPECIAL combined: 47 races, 17 hits, +16,740 yen, ROI 135.692964%
- FINAL: 0 bets, still pending rebuild.

## Development interpretation

The former broad SPECIAL bucket was invalid. Selection, ordinary special, and initial-special exhibit materially different pre-race market structures and should not share one gate.

The Q1 result passes the project requirement that a candidate must be profitable in Q1. It is not OOS validation. These SPECIAL rules should now be frozen before later validation; further Q1 tuning would increase overfit risk.

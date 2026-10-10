# v8.18-F19 Semifinal Rebuild: 2024Q1 + Q2 Development Result

Status: DEVELOPMENT. 2024Q2 had already been observed under v8.17-F18, so Q2 is no longer treated as untouched out-of-sample data for this rebuild.

Actions run: 33570015251
Job: 100061711098
Runner commit: 559e0424e707eb6d787454476f2beec4fb096647
Engine: `src/keirin_market_analysis/v8_18_f19_semifinal_h_adaptive.py`
Evaluator: `src/keirin_market_analysis/evaluate_v8_18_q1q2.py`

## Rebuilt semifinal rule

1. Require the existing F09 `PS_AB` market structure.
2. Require `H_AB`: the top two first-place H riders must map one each to the two strongest lines A/B.
3. Translate H concentration directly into the first-place formation pool:
   - if `H1 >= 2*H2`, first-place pool = H1 only;
   - if `H1 < 2*H2`, first-place pool = top two H riders.
4. Second- and third-place pools remain the F09 market-cliff pools.
5. The completed formation is priced only afterward. Structural price contraction may remove whole riders from second/third pools only. The H-derived first-place pool is protected.
6. No individual-ticket pruning, fixed ticket count, rider ability, score, style, result, or payout is used to generate the bet.
7. No Q1/Q2-optimized numeric threshold was introduced.

## Semifinal: v8.17-F18 vs v8.18-F19

| Dataset | Version | Bets | Hits | Hit rate | Tickets | Avg tickets | Stake | Payout | Profit | ROI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | v8.17-F18 | 111 | 42 | 37.8378% | 1226 | 11.0450 | 122600 | 124030 | +1430 | 101.1664% |
| 2024Q1 | v8.18-F19 | 56 | 19 | 33.9286% | 628 | 11.2143 | 62800 | 70300 | +7500 | 111.9427% |
| 2024Q2 | v8.17-F18 | 110 | 27 | 24.5455% | 1220 | 11.0909 | 122000 | 72630 | -49370 | 59.5328% |
| 2024Q2 | v8.18-F19 | 59 | 14 | 23.7288% | 631 | 10.6949 | 63100 | 71620 | +8520 | 113.5024% |

### Semifinal deltas

2024Q1:
- Bet races: -55
- Tickets: -598
- Profit: +6070 yen versus v8.17-F18
- ROI: +10.7763 percentage points

2024Q2:
- Bet races: -51
- Tickets: -589
- Profit: +57890 yen versus v8.17-F18
- ROI: +53.9696 percentage points

New semifinal branch across Q1+Q2 combined:
- Bet races: 115
- Hits: 33
- Hit rate: 28.6956521739%
- Tickets: 1259
- Average tickets/race: 10.9478260870
- Stake: 125900 yen
- Payout: 141920 yen
- Profit: +16020 yen
- ROI: 112.7243844321%

## Full v8.18-F19 scheme

Only the semifinal branch changed. All other branches were inherited unchanged from v8.17-F18.

### 2024Q1
- Population: 1191
- Bet races: 189
- Hits: 64
- Hit rate: 33.8624338624%
- Tickets: 1945
- Average tickets/race: 10.2910052910
- Stake: 194500 yen
- Payout: 229070 yen
- Profit: +34570 yen
- ROI: 117.7737789203%

### 2024Q2
- Population: 1172
- Bet races: 180
- Hits: 61
- Hit rate: 33.8888888889%
- Tickets: 1807
- Average tickets/race: 10.0388888889
- Stake: 180700 yen
- Payout: 148760 yen
- Profit: -31940 yen
- ROI: 82.3242944106%

Q2 by group after the semifinal rebuild:

| Group | Bets | Hits | Profit | ROI |
|---|---:|---:|---:|---:|
| QUALIFYING | 51 | 23 | -14590 | 64.9279% |
| GENERAL | 16 | 8 | +6110 | 155.0450% |
| SEMIFINAL | 59 | 14 | +8520 | 113.5024% |
| SPECIAL | 44 | 13 | -18590 | 61.8275% |
| FINAL | 10 | 3 | -13390 | 17.3457% |

## Interpretation

The semifinal branch is materially improved: it is profitable in both observed development quarters while roughly halving the number of semifinal bets and tickets versus v8.17-F18. The key change is not a new fitted cutoff; it is a change in market semantics. The trifecta first-place H hierarchy now determines the shape of the first-place pool instead of being used only as a diagnostic or a simple concentrated/not-concentrated gate.

The full scheme is still not profitable in 2024Q2 because QUALIFYING, SPECIAL, and FINAL remain negative. Therefore v8.18-F19 is a successful semifinal branch rebuild candidate, not a completed overall scheme.

No v8.18-F19 validation has been run on the next untouched validation set in this result record. The next validation set must not be tuned after observing its result.

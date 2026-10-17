# v8.17-F18 2024Q2 OOS Simulation Result

Status: OUT-OF-SAMPLE / FIXED AFTER 2024Q1
Actions run: 33568022342
Job: 100055551472
Trigger commit: abd998aa0980ce212660f277ae34152bc5eaba77
No Q2 tuning: true

The v8.17-F18 rules were carried from 2024Q1 into 2024Q2 without changing the market gates, race-type branch logic, thresholds, or formation-generation rules.

Population: 1172 complete F1 S-class seven-car races.
Selection: 92 races, diagnostic-only / no bet.
Final: PS_AB + H_CONCENTRATED, unchanged F09 formation.

## Overall
- Bet races: 231
- Hits: 74
- Hit rate: 32.03463203463203%
- Tickets: 2396
- Average tickets/race: 10.372294372294371
- Stake: 239600 yen
- Payout: 149770 yen
- Profit: -89830 yen
- ROI: 62.50834724540901%
- Max losing streak: 21
- Acceptance: FAIL_Q2_OOS_NOT_PROFITABLE

## By group

| Group | Population | Bets | Hits | Hit rate | Tickets | Avg tickets | Stake | Payout | Profit | ROI | Max losing streak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QUALIFYING | 348 | 51 | 23 | 45.0980% | 416 | 8.1569 | 41600 | 27010 | -14590 | 64.9279% | 9 |
| GENERAL | 232 | 16 | 8 | 50.0000% | 111 | 6.9375 | 11100 | 17210 | +6110 | 155.0450% | 3 |
| SEMIFINAL | 195 | 110 | 27 | 24.5455% | 1220 | 11.0909 | 122000 | 72630 | -49370 | 59.5328% | 17 |
| SPECIAL | 333 | 44 | 13 | 29.5455% | 487 | 11.0682 | 48700 | 30110 | -18590 | 61.8275% | 5 |
| FINAL | 64 | 10 | 3 | 30.0000% | 162 | 16.2000 | 16200 | 2810 | -13390 | 17.3457% | 2 |

## Exact race types
- S級一般: population 232, bets 16, hits 8, hit rate 50.0000%, profit +6110, ROI 155.0450%.
- S級予選: population 348, bets 51, hits 23, hit rate 45.0980%, profit -14590, ROI 64.9279%.
- S級初日特選: population 20, bets 6, hits 0, profit -7000, ROI 0.0000%.
- S級初特選: population 47, bets 13, hits 2, hit rate 15.3846%, profit -4650, ROI 65.0376%.
- S級決勝: population 64, bets 10, hits 3, hit rate 30.0000%, profit -13390, ROI 17.3457%.
- S級準決勝: population 195, bets 110, hits 27, hit rate 24.5455%, profit -49370, ROI 59.5328%.
- S級特選: population 174, bets 25, hits 11, hit rate 44.0000%, profit -6940, ROI 75.5634%.
- S級選抜: population 92, bets 0; all 92 received market-disagreement diagnostics.

## Q1 vs Q2

| Group | Q1 Profit | Q1 ROI | Q1 Hit rate | Q2 Profit | Q2 ROI | Q2 Hit rate |
|---|---:|---:|---:|---:|---:|---:|
| OVERALL | +28500 | 111.2072% | 35.6557% | -89830 | 62.5083% | 32.0346% |
| QUALIFYING | +700 | 101.2111% | 25.3968% | -14590 | 64.9279% | 45.0980% |
| GENERAL | +6770 | 230.1923% | 71.4286% | +6110 | 155.0450% | 50.0000% |
| SEMIFINAL | +1430 | 101.1664% | 37.8378% | -49370 | 59.5328% | 24.5455% |
| SPECIAL | +16740 | 135.6930% | 36.1702% | -18590 | 61.8275% | 29.5455% |
| FINAL | +2860 | 113.1193% | 43.7500% | -13390 | 17.3457% | 30.0000% |

## Observations (not new tuned rules)
- GENERAL is the only group with positive profit in both Q1 and Q2.
- SEMIFINAL is the largest Q2 loss source (-49370 yen) and its Q1 result had only been marginally positive.
- QUALIFYING hit rate rose sharply in Q2, but ROI fell to 64.93%, indicating that hit frequency alone did not preserve payout economics.
- FINAL failed severely in Q2 after being positive in Q1.
- SPECIAL also reversed from strong Q1 profit to Q2 loss; both regular special and initial-special exact types were negative in Q2.
- SELECTION remains no-bet, so it contributed neither profit nor loss in this validation.

No Q2-derived rule changes are made in this result record.

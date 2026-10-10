# v8.13-F14 GENERAL Rebuild 2024Q1 Result

Status: DEVELOPMENT / REJECTED

## Diagnostic finding before F14
For S-class GENERAL races in 2024Q1:
- Population: 214
- PS_AB pass: 113
- All PS_AB with structural price stage: 113 bets, 26 hits, hit rate 23.01%, avg 12.42 tickets, ROI 46.15%.
- H-state HAB1_HRATIO0 was the strongest pre-existing GENERAL entry state: 12 bets, 6 hits, hit rate 50.00%, avg 14.33 tickets, ROI 73.02%.
- The broad 19+ ticket band hit 47.83% but returned only 44.82% ROI and only 18.18% of hit races were individually profitable.
- The generic structural price compression activated in only 2 of 113 PS_AB GENERAL races, and in 0 of the 12 HAB1_HRATIO0 races.

Interpretation: the GENERAL entry signal is not the main failure. The larger failure is that GENERAL clean formations can remain too broad/cheap and the generic price-knee is too passive for this race type.

## F14 rule tested
GENERAL only:
- Keep PS_AB + H_CONCENTRATED + H_AB entry.
- Build the full clean market-cliff formation first.
- Only after formation completion inspect prices.
- Walk whole-rider structural compression path only.
- Select the earliest clean rectangle satisfying:
  - gm_return_multiple >= 1.0
  - profitable_q_share >= 0.5
- If none exists, skip.
- No individual ticket pruning.

The 1.0 and 0.5 boundaries are semantic break-even / majority boundaries, not Q1-optimized cutoffs.

## F14 2024Q1 result
Overall with QUALIFYING and SEMIFINAL unchanged and SPECIAL/FINAL quarantined:
- Bet races: 186
- Hits: 63
- Hit rate: 33.87%
- Tickets: 1,960
- Avg tickets: 10.54
- Stake: 196,000 yen
- Payout: 194,500 yen
- Profit: -1,500 yen
- ROI: 99.23%
- Verdict: REJECT_Q1_NOT_PROFITABLE

By active group:
- QUALIFYING: 63 bets, 16 hits, +700 yen, ROI 101.21% (unchanged).
- SEMIFINAL: 111 bets, 42 hits, +1,430 yen, ROI 101.17% (unchanged).
- GENERAL: 12 bets, 5 hits, hit rate 41.67%, 156 tickets, avg 13.0 tickets, stake 15,600 yen, payout 11,970 yen, profit -3,630 yen, ROI 76.73%, compressed races 3.

Comparison of GENERAL old vs F14:
- Old: 12 bets, 6 hits, avg 14.33 tickets, profit -4,640 yen, ROI 73.02%.
- F14: 12 bets, 5 hits, avg 13.00 tickets, profit -3,630 yen, ROI 76.73%.
- F14 saved 1,600 yen of stake but lost one hit. Net GENERAL improvement was +1,010 yen, insufficient to make the full scheme profitable.

## Development conclusion
- Preserve the GENERAL entry structure for now; it remains the clearest high-hit H-state.
- Do not accept F14 as final GENERAL logic.
- The price stage needs a stronger GENERAL-specific concept than simple break-even/majority viability.
- Avoid blindly reducing ticket count: <=6 ticket GENERAL formations had only 10.64% hit rate in the diagnostic, while 19+ ticket formations had 47.83% hit rate but poor price economics.
- Next GENERAL research should target how to retain the broad formation's hit coverage while identifying and structurally removing the cheap branch that causes losing hits, without individual-ticket pruning or irregular formations.

# v8.25-F26 — 2024Q3 Out-of-Sample Result

Status: OUT-OF-SAMPLE VALIDATION RESULT. The v8.25-F26 rules were frozen after 2024Q1+Q2 development and applied to 2024Q3 with no Q3 tuning or rule changes.

## Overall 2024Q3

| Metric | Result |
|---|---:|
| Population | 1,186 |
| Bet races | 162 |
| Hits | 28 |
| Hit rate | 17.28% |
| Tickets | 1,004 |
| Avg tickets/race | 6.20 |
| Stake | 100,400 yen |
| Payout | 87,440 yen |
| Profit | **-12,960 yen** |
| ROI | **87.09%** |

Verdict: **FAIL — overall profit negative and ROI below 100%.**

## By group

| Group | Bets | Hits | Tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| QUALIFYING | 13 | 4 | 74 | 7,400 | 23,610 | **+16,210** | **319.05%** |
| GENERAL | 15 | 4 | 126 | 12,600 | 7,320 | **-5,280** | **58.10%** |
| SEMIFINAL | 59 | 15 | 609 | 60,900 | 31,570 | **-29,330** | **51.84%** |
| SPECIAL | 48 | 3 | 141 | 14,100 | 20,550 | **+6,450** | **145.74%** |
| FINAL | 27 | 2 | 54 | 5,400 | 4,390 | **-1,010** | **81.30%** |

SPECIAL is a combined reporting group. Within it:
- INITIAL_SPECIAL_UNIFIED: 12 bets, 3 hits, 105 tickets, stake 10,500, payout 20,550, **+10,050 / ROI 195.71%**.
- SELECTION_ULTRA_ROI: 36 bets, 0 hits, 36 tickets, stake 3,600, payout 0, **-3,600 / ROI 0%**.
- Ordinary `Ｓ級特選`: 0 bets (quarantined).

## FINAL branch

SET LOCK × ORDER SPLIT remained exactly two tickets per qualifying FINAL race:
- 27 bets
- 2 hits
- 54 tickets
- stake 5,400
- payout 4,390
- **-1,010 / ROI 81.30%**

## Scientific interpretation

The frozen v8.25-F26 scheme did not validate as a whole on 2024Q3. The largest negative contribution came from SEMIFINAL (-29,330 / ROI 51.84%). QUALIFYING remained strongly positive, and INITIAL_SPECIAL_UNIFIED also remained positive. GENERAL, SELECTION_ULTRA_ROI, SEMIFINAL, and FINAL were negative in this quarter.

No Q3-derived threshold, branch deletion, or rule adjustment is applied in this result. Any subsequent redesign using these outcomes makes Q3 development data for the redesigned scheme and requires a later untouched dataset for validation.

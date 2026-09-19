# v8.23-F24 — QUALIFYING Single-Pole + SELECTION Ultra-ROI — 2024Q1+Q2 Development Result

Status: DEVELOPMENT. 2024Q1 and 2024Q2 have both been observed during development. The current v8.23-F24 rules are now candidates to freeze before untouched Q3 validation.

## QUALIFYING inherited from v8.22-F23

QUALIFYING uses the hard/narrow single-pole branch:

1. F09 PS_AB must pass.
2. Require H_CONCENTRATED: H1 >= 2*H2.
3. Require H_AB == False, interpreted as one-sided head concentration rather than neat two-line head agreement.
4. First place = H1 only.
5. Second/third begin from F09 market-cliff pools.
6. Structural price contraction may remove whole riders only from second/third pools; H1 first is protected.
7. No individual-ticket pruning and no fitted odds cutoff.

QUALIFYING results:

| Dataset | Bets | Hits | HR | Tickets | Avg tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 21 | 3 | 14.29% | 129 | 6.14 | 12,900 | 25,850 | **+12,950** | **200.39%** |
| 2024Q2 | 26 | 15 | 57.69% | 147 | 5.65 | 14,700 | 15,590 | **+890** | **106.05%** |

## SELECTION ultra-ROI branch

SELECTION has the opposite objective from QUALIFYING. Low hit rate is acceptable; the branch seeks sparse, high-return trifecta exposure from cross-market disagreement.

Rule for exact `Ｓ級選抜`:

1. H must be BALANCED: H1 < 2*H2.
2. Rank all 35 trio sets by normalized trio support P_trio.
3. Among the trio top3 sets with D_log < 0, choose the set with the most negative D_log, where D_log = log(Q_tf_set / P_trio) and Q_tf_set is the normalized 210-way trifecta market collapsed back to the same unordered 35 sets.
4. The target set must include global H1.
5. The target set must span exactly two predicted lines.
6. Within that set, first place is the rider with the highest H support.
7. The remaining two riders create two possible tail orders. Buy only the **higher-odds / less-supported** tail order.
8. Exactly one trifecta ticket per qualifying SELECTION race.
9. No fitted numeric odds threshold and no result/payout input.
10. This is the formation-definition rule itself, not post-hoc deletion of one ticket from a broader purchased formation.

SELECTION results:

| Dataset | Population | Bets | Hits | HR | Tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 98 | 50 | 2 | 4.00% | 50 | 5,000 | 9,970 | **+4,970** | **199.40%** |
| 2024Q2 | 92 | 39 | 1 | 2.56% | 39 | 3,900 | 11,490 | **+7,590** | **294.62%** |
| Q1+Q2 | 190 | 89 | 3 | 3.37% | 89 | 8,900 | 21,460 | **+12,560** | **241.12%** |

The opposite tail-order control was poor: on the same H-balanced / H1-in-target / two-line target state, buying the lower-odds tail order produced Q1 ROI 62.4% and Q2 ROI 39.23%. The higher-odds order is therefore the coherent contrarian order translation in the observed development data.

## Whole v8.23-F24 results

| Dataset | Population | Bets | Hits | HR | Tickets | Avg tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 1191 | 143 | 32 | 22.38% | 955 | 6.68 | 95,500 | 129,810 | **+34,310** | **135.93%** |
| 2024Q2 | 1172 | 156 | 41 | 26.28% | 1104 | 7.08 | 110,400 | 134,950 | **+24,550** | **122.24%** |

Q1 group profits:
- QUALIFYING +12,950
- GENERAL +6,770
- SEMIFINAL +7,500
- SPECIAL +7,090, consisting of INITIAL_SPECIAL +2,120 and SELECTION +4,970
- FINAL 0 / diagnostic only

Q2 group profits:
- QUALIFYING +890
- GENERAL +6,110
- SEMIFINAL +8,520
- SPECIAL +9,030, consisting of INITIAL_SPECIAL +1,440 and SELECTION +7,590
- FINAL 0 / diagnostic only

## Scientific caution / freeze point

The SELECTION branch is intentionally extreme and has only **3 hits across Q1+Q2**. Its ROI is therefore highly sensitive to a tiny number of outcomes. The rule has a coherent pre-race market interpretation and no fitted numeric cutoff, but it is not validated.

Do not continue mining Q1/Q2 for a prettier SELECTION number. Freeze v8.23-F24 here if it is to be tested. The next meaningful test for the current v8 lineage is untouched 2024Q3, with no v8.23 rule changes after viewing Q3.

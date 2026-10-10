# v8.25-F26 — FINAL SET LOCK × ORDER SPLIT — 2024Q1+Q2 Development Result

Status: DEVELOPMENT. 2024Q1 and 2024Q2 have both been observed during development. 2024Q3 remains untouched by the current v8.25-F26 lineage and is the next meaningful validation dataset after freeze.

## FINAL active rule: SET LOCK × ORDER SPLIT

Exact `Ｓ級決勝` only.

1. H must be BALANCED: H1 < 2*H2.
2. Rank the 35 trio sets by normalized trio support P_trio.
3. Collapse the normalized 210-way trifecta market q back to the same 35 unordered sets as Q_tf_set.
4. The top P_trio set must exactly equal the top Q_tf_set set. This is SET LOCK.
5. The locked three-rider set must contain global H1 and H2.
6. Under H1 as first place, compare the two possible orders of the remaining two riders and retain the higher-odds / less-supported tail order.
7. Repeat the same under H2 as first place.
8. Buy only when those two selected tickets themselves form one clean rectangular two-ticket formation. Otherwise skip.
9. Exactly two trifecta tickets per qualifying FINAL race.
10. No fitted numeric odds cutoff, no result or payout input, and no individual-ticket pruning.

## HEAD LOCK × TAIL SPLIT disposition

The predefined v8.24 comparison also tested HEAD LOCK × TAIL SPLIT for H1 >= 2*H2. It produced Q1 -1,460 yen / ROI 94.69% and Q2 +9,320 yen / ROI 155.15%. Because the project requires a development candidate to be positive in Q1, this branch is not active in v8.25-F26. It remains diagnostic-only; no further Q1/Q2 threshold mining was used to rescue it.

## FINAL results — active SET LOCK × ORDER SPLIT only

| Dataset | FINAL population | Bets | Hits | HR | Tickets | Avg tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 67 | 28 | 2 | 7.14% | 56 | 2.00 | 5,600 | 10,620 | **+5,020** | **189.64%** |
| 2024Q2 | 64 | 30 | 4 | 13.33% | 60 | 2.00 | 6,000 | 10,690 | **+4,690** | **178.17%** |

## Whole v8.25-F26 results

All non-FINAL branches are unchanged from v8.23-F24.

| Dataset | Population | Bets | Hits | HR | Tickets | Avg tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | 1191 | 171 | 34 | 19.88% | 1011 | 5.91 | 101,100 | 140,430 | **+39,330** | **138.90%** |
| 2024Q2 | 1172 | 186 | 45 | 24.19% | 1164 | 6.26 | 116,400 | 145,640 | **+29,240** | **125.12%** |

Q1 group profits:
- QUALIFYING +12,950 / ROI 200.39%
- GENERAL +6,770 / ROI 230.19%
- SEMIFINAL +7,500 / ROI 111.94%
- SPECIAL +7,090 / ROI 148.56%
- FINAL +5,020 / ROI 189.64%

Q2 group profits:
- QUALIFYING +890 / ROI 106.05%
- GENERAL +6,110 / ROI 155.05%
- SEMIFINAL +8,520 / ROI 113.50%
- SPECIAL +9,030 / ROI 142.00%
- FINAL +4,690 / ROI 178.17%

## Freeze point / scientific caution

v8.25-F26 is a Q1+Q2 development scheme, not an out-of-sample validated scheme. The active FINAL branch has only 6 total hits across Q1+Q2, so the apparent ROI is still sensitive to a small number of outcomes. Do not tune v8.25-F26 against Q3 after viewing Q3. If the current scheme is frozen, the next meaningful test is the untouched 2024Q3 dataset with no rule changes beforehand.

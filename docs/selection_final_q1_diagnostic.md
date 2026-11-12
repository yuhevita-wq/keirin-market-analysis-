# Selection / Final 2024Q1 diagnostic

Status: DEVELOPMENT / Q1 in-sample diagnostic
Actions run: 33525952558
Job: 99916722482
Population: 1191 complete F1 S-class seven-car races.
Diagnostic formation: unchanged v8.8-F09 Market Cliff after PS_AB. No result/payout is used to create a candidate subgroup; result/payout is used only to evaluate the pre-race structural groups listed below.

## S級選抜

Population 98. PS_AB pass 40, fail 58.

- ALL PS_AB: 40 races, 10 hits, 433 tickets, stake 43,300, payout 15,030, profit -28,270, ROI 34.71%.
- H_CONCENTRATED: 7 races, 1 hit, stake 5,200, payout 700, profit -4,500, ROI 13.46%.
- H_NOT_CONCENTRATED: 33 races, 9 hits, profit -23,770, ROI 37.61%.
- H_AB: 24 races, 7 hits, profit -19,060, ROI 37.10%.
- H_NOT_AB: 16 races, 3 hits, profit -9,210, ROI 29.15%.
- H_STATE_CONSISTENT: 21 races, 4 hits, profit -11,910, ROI 27.38%.
- H_STATE_MISMATCH: 19 races, 6 hits, profit -16,360, ROI 39.18%.
- First pool singleton: 28 races, 7 hits, avg 5.64 tickets, profit -9,100, ROI 42.41%.
- First pool >=2: 12 races, 3 hits, avg 22.92 tickets, profit -19,170, ROI 30.29%.
- All 40 PS_AB-pass races already satisfy weighted-GM break-even; 39/40 have profitable-q majority, so these price metrics do not discriminate Selection.
- Ticket buckets: <=6 points ROI 50.51%; 7-12 ROI 29.46%; >=13 ROI 31.79%.

Interpretation: no broad semantic H-state, first-pool shape, price-break-even state, or point bucket produces a viable Selection branch under the current PS_AB + F09 framework. Continue quarantine rather than fit a Q1 rescue threshold.

## S級決勝

Population 67. PS_AB pass 37, fail 30.

- ALL PS_AB: 37 races, 10 hits, 464 tickets, stake 46,400, payout 31,250, profit -15,150, ROI 67.35%. All 10 hits are race-profitable, so the main problem is misses, not cheap hits.
- H_CONCENTRATED (`H1_top >= 2*H1_second`): 16 races, 7 hits, HR 43.75%, 218 tickets, avg 13.63, stake 21,800, payout 24,660, profit +2,860, ROI 113.12%. All 7 hits are race-profitable.
- H_NOT_CONCENTRATED: 21 races, 3 hits, HR 14.29%, profit -18,010, ROI 26.79%.
- H_AB: 19 races, 7 hits, ROI 77.27%; H_NOT_AB: 18 races, 3 hits, ROI 53.91%.
- H_STATE_CONSISTENT: ROI 61.27%; mismatch ROI 78.36%. Consistency is not the Final rule.
- First pool singleton: 23 races, 4 hits, ROI 65.58%. First pool >=2: 14 races, 6 hits, ROI 68.33%. Do not force a singleton first-place anchor in Finals.
- Ticket <=6: HR 8.33%, ROI 30.00%; 7-12: HR 30.77%, ROI 87.27%; >=13: HR 41.67%, ROI 64.89%. Tiny formations miss too many Finals.

H_CONCENTRATED is robust across both H_AB states:
- HAB0 + concentrated: 7 races, 2 hits, stake 5,800, payout 8,480, profit +2,680, ROI 146.21%.
- HAB1 + concentrated: 9 races, 5 hits, stake 16,000, payout 16,180, profit +180, ROI 101.13%.

Interpretation: the clean Final candidate is `PS_AB + H_CONCENTRATED`, with H_AB diagnostic only and unchanged F09 formation. This uses the pre-existing semantic 2x H threshold and works on both H_AB branches, so it is structurally preferable to selecting one winning H-state cell. It remains Q1 development and requires later untouched validation.

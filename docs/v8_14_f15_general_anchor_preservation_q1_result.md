# v8.14-F15 GENERAL Anchor Preservation — 2024Q1 Result

Status: DEVELOPMENT / PASS_Q1_PROFIT

## Why GENERAL was reconsidered
The existing GENERAL entry (`PS_AB + H_CONCENTRATED + H_AB`) was strong on hit rate but weak economically:
- 12 bets
- 6 hits
- hit rate 50.00%
- 172 tickets, avg 14.33
- stake 17,200 yen
- payout 12,560 yen
- profit -4,640 yen
- ROI 73.02%

A separate AnchorSpread diagnostic did not discriminate: all 12 GENERAL-entry races already had second/third-position markets more dispersed than first-position market.

The decisive structural split was the completed formation's first-place pool:
- first-place pool size 1: 7 races, 5 hits, 52 tickets, payout 11,970 yen, profit +6,770 yen, ROI 230.19%
- first-place pool size >=2: 5 races, 1 hit, 120 tickets, payout 590 yen, profit -11,410 yen, ROI about 4.92%

This exposed a scheme inconsistency. GENERAL entry explicitly requires a strong H1 first-place concentration (`H1_top >= 2 * H1_second`), yet the generic market-cliff growth was sometimes allowed to expand the first-place pool across that same H1 cliff.

## v8.14-F15 GENERAL rule
This is not a global fixed-one-rider rule.

For GENERAL only:
1. Keep `PS_AB + H_CONCENTRATED + H_AB` entry.
2. Build the normal clean market-cliff formation first without price pruning.
3. Preserve the H1 dominant block identified by the GENERAL entry. The formation is not allowed to cross the H1 top/second concentration cliff that justified entry.
4. If the H1 anchor cannot coexist with the completed second/third place sets as one valid clean rectangle, skip the race rather than deform the formation.
5. Only after the clean anchored formation exists, apply the existing structural price-compression knee.
6. Individual-ticket pruning remains prohibited.

Under the current GENERAL entry (`H1_top >= 2 * H1_second`), the H1 dominant block ends at the top rider, but the rule is conceptually "preserve the market cliff", not "first place must always contain exactly one rider".

## 2024Q1 result
### Overall scheme
- Population: 1,191
- Bet races: 181
- Hits: 63
- Hit rate: 34.8066%
- Tickets: 1,856
- Average tickets: 10.2541
- Stake: 185,600 yen
- Payout: 194,500 yen
- Profit: **+8,900 yen**
- ROI: **104.7953%**
- Verdict: **PASS_Q1_PROFIT**

### QUALIFYING (unchanged)
- 63 bets
- 16 hits
- Hit rate 25.3968%
- 578 tickets, avg 9.1746
- Stake 57,800 yen
- Payout 58,500 yen
- Profit +700 yen
- ROI 101.2111%

### GENERAL (rebuilt)
- 7 bets
- 5 hits
- Hit rate **71.4286%**
- 52 tickets
- Average tickets **7.4286**
- Stake 5,200 yen
- Payout 11,970 yen
- Profit **+6,770 yen**
- ROI **230.1923%**

The seven GENERAL bets were:
- `5-23-2347` — 6 tickets — miss
- `1-57-23567` — 8 tickets — hit, payout 1,190 yen
- `7-13-13456` — 8 tickets — hit, payout 4,760 yen
- `3-45-124567` — 10 tickets — miss
- `3-57-12457` — 8 tickets — hit, payout 790 yen
- `7-124-1245` — 9 tickets — hit, payout 4,030 yen
- `4-3-125` — 3 tickets — hit, payout 1,200 yen

Five of the prior 12 GENERAL-entry races were skipped because restricting the first-place set to the H1 anchor did not leave a valid clean rectangle with the completed second/third-place sets (`GENERAL_INVALID_ANCHORED_RECTANGLE`). The scheme skips those races rather than creating an irregular ticket set.

### SEMIFINAL (unchanged)
- 111 bets
- 42 hits
- Hit rate 37.8378%
- 1,226 tickets, avg 11.0450
- Stake 122,600 yen
- Payout 124,030 yen
- Profit +1,430 yen
- ROI 101.1664%

SPECIAL and FINAL remain quarantined pending dedicated rebuilds.

## Development caution
This is a strong 2024Q1 development result, but GENERAL has only seven accepted races after the correction. The rule was developed after inspecting Q1 GENERAL failures, so this is not independent validation. Further Q1 tuning of GENERAL should be avoided; the next meaningful test is untouched later-period validation after the remaining race-type branches are developed/frozen.

# QUALIFYING H_AB and FINAL Market Heat: 2024Q1+Q2 Diagnostic

Status: DEVELOPMENT DIAGNOSTIC. Q1 and Q2 are observed development data for the current v8.20 lineage.

## QUALIFYING: proposed H_AB tightening rejected

Current qualifying branch versus adding H_AB as an additional entry requirement:

| Dataset | Rule | Bets | Hits | Tickets | Stake | Payout | Profit | ROI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | Current | 63 | 16 | 578 | 57,800 | 58,500 | +700 | 101.21% |
| Q1 | Current + H_AB | 39 | 12 | 406 | 40,600 | 26,400 | -14,200 | 65.02% |
| Q1 | Removed H_AB=0 side | 24 | 4 | 172 | 17,200 | 32,100 | +14,900 | 186.63% |
| Q2 | Current | 51 | 23 | 416 | 41,600 | 27,010 | -14,590 | 64.93% |
| Q2 | Current + H_AB | 24 | 7 | 245 | 24,500 | 10,440 | -14,060 | 42.61% |
| Q2 | Removed H_AB=0 side | 27 | 16 | 171 | 17,100 | 16,570 | -530 | 96.90% |

Conclusion: do **not** add H_AB to qualifying. In Q1 the H_AB=0 side contains most of the profitable payout, and in Q2 the H_AB filter worsens ROI. The semifinal H_AB logic does not transfer to qualifying.

## FINAL: race-level total-vote heat exists, but does not discriminate

Heat was measured relative to other eligible S-class races at the same track/date using `total_votes` from the complete 3連複 and 3連単 markets.

Definitions tested:

- `trio_heat = final_trio_votes / median(other_same_day_S_trio_votes)`
- `tf_heat = final_tf_votes / median(other_same_day_S_tf_votes)`
- vote level versus the maximum other S race on the same day
- final share of all same-day eligible S-class market votes
- relative amplification of 3連単 heat versus 3連複 heat

All current FINAL candidates in both Q1 and Q2 had both 3連複 and 3連単 total votes above every other eligible S-class race at the same track/date. Therefore simple `hot/not-hot` race-level vote volume cannot select profitable finals.

### Current FINAL baseline

| Dataset | Bets | Hits | Tickets | Stake | Payout | Profit | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 16 | 7 | 218 | 21,800 | 24,660 | +2,860 | 113.12% |
| Q2 | 10 | 3 | 162 | 16,200 | 2,810 | -13,390 | 17.35% |

### Same-day vote-majority states

3連単 final market taking >=50% of the eligible same-day S-class 3連単 votes:
- Q1: 3 bets, 2 hits, +190 yen, ROI 105.28%.
- Q2: 2 bets, 1 hit, -2,310 yen, ROI 35.83%.

Both 3連複 and 3連単 final markets taking >=50% of their same-day S-class votes:
- Q1: 2 bets, 1 hit, -890 yen, ROI 50.56%.
- Q2: 1 bet, 0 hits, -1,800 yen, ROI 0%.

These do not reproduce across quarters.

### 3連単 heat amplification versus 3連複

`(final_tf / peer_max_tf) / (final_trio / peer_max_trio) >= 1`:
- Q1: 14 bets, 6 hits, -180 yen, ROI 99.01%.
- Q2: 9 bets, 3 hits, -12,190 yen, ROI 18.73%.

The opposite side is too small and does not reproduce:
- Q1: 2 bets, 1 hit, +3,040 yen, ROI 184.44%.
- Q2: 1 bet, 0 hits, -1,200 yen, ROI 0%.

Conclusion: FINAL heat should not be interpreted as total race volume. Finals are universally hotter. The next useful concept is **heat location**: where the extra final money lands inside the 210-order market and 35-set market.

## Next FINAL heat variables to study

Use the already available total market votes together with normalized market probabilities, without rider ability or result inputs:

1. **Head heat**: `TF_total_votes * H(i)` for each rider, especially H1 and H2.
2. **Head heat cliff**: absolute/relative separation between H1 and H2 after scaling by final total votes.
3. **Formation heat**: `TF_total_votes * q_mass(formation)`.
4. **Heat per ticket**: formation heat divided by formation ticket count.
5. **A/B line heat**: total absolute head/set support landing on lines A and B.
6. Normalize these against same-track/date S-class peer races so venue/day market size does not create a false signal.

The structural question becomes: not 'is the final hot?' but **'where is the final-specific extra money concentrated, and does the formation preserve that concentration?'**

# v8.8-F09 — Numbered Scheme Design

Status: DEVELOPMENT DESIGN / NOT YET VALIDATED
Dataset for next simulation: 2024Q1

## 0. Target population
- F1 meetings
- S class
- 7 riders
- complete 3連複 35 odds
- complete 3連単 210 odds
- complete Rakuten K-Dreams predicted line formation
- market data only for decision logic

## 1. Normalize the markets
### 1.1 3連複
For each 3-rider combination c:
P(c) = (1 / odds_c) / sum(1 / odds)

For rider i:
S(i) = (1/3) * sum_{c contains i} P(c)

For each predicted line L:
LS(L) = sum_{i in L} S(i)

Define A and B as the top two lines by LS.
Tie break: higher LS, then earlier predicted-line index.

### 1.2 Adjacent-pair support
For adjacent pair (i,j):
PS(i,j) = sum_{c contains i,j} P(c)

Rank all adjacent pairs across all predicted lines.

### 1.3 3連単
For exact-order ticket t:
q(t) = (1 / odds_t) / sum(1 / odds)

For rider i first-place support:
H1(i) = sum_{t:first=i} q(t)

## 2. Hard entry gate
Only one v6.1-derived hard gate remains:

PS_AB = PASS iff the top two adjacent-pair PS rows come from exactly line A and line B, one each.

If PS_AB fails: SKIP.
If PS_AB passes: continue.

Rationale from 2024Q1 ablation: PS_AB was the strongest protective hard filter. Dropping it admitted 68 races with ROI 50.95% under the held-fixed v8.4 formation.

## 3. H-state classification, not exclusion
The following are recorded as race-type features and do NOT reject the race:

H_AB:
- true iff the top two H1 riders are exactly one from A and one from B.

H_RATIO:
- true iff top H1 < 2 * second H1.

Classify every PS_AB-passing race into one of four states:
1. H_AB=1, H_RATIO=1
2. H_AB=1, H_RATIO=0
3. H_AB=0, H_RATIO=1
4. H_AB=0, H_RATIO=0

H_POS is removed from the active gate because it admitted zero additional races when dropped from the full gate in 2024Q1.

## 4. Start formation from the exact-order market center
Take the exact trifecta ticket with the largest q(t) as the seed.

If seed = a-b-c:
F1={a}, F2={b}, F3={c}.

The final output must remain one clean branch-free rectangular formation F1-F2-F3.
F1, F2 and F3 are independent sets. Nested closure is NOT required.

## 5. Build one deterministic growth path
At each step, consider adding exactly one not-yet-present rider to exactly one position set F1, F2 or F3.

For each candidate addition:
- generate only the newly created valid trifecta tickets,
- calculate their q values,
- calculate marginal market mass gain Δq,
- calculate added ticket count ΔN,
- calculate marginal density D = Δq / ΔN.

Choose the candidate addition with the largest D.
Tie break deterministically by position order 1st -> 2nd -> 3rd, then smaller car number.

Repeat until no further rider can be added within the structural rider universe.

No fixed number of riders by place.
No fixed maximum ticket count.
No price cut.

## 6. Detect the market cliff and stop before it
For every accepted growth step k, calculate the geometric-mean q support of the newly created ticket block:

E_k = exp(mean(log(q(t)))) over newly created tickets at step k.

For adjacent growth steps calculate the support-drop ratio:

Drop_k = E_k / E_{k+1}.

Find the largest Drop_k.
Select the formation state immediately BEFORE that largest drop.

Interpretation: retain the dense market center and stop before the strongest transition into the thin outer layer.

Tie break for equal largest drops: choose the earlier cliff, preserving the smaller formation.

## 7. Final formation and ticket set
Generate every valid exact-order ticket represented by the selected rectangular F1-F2-F3:
(a,b,c) where a in F1, b in F2, c in F3, and all three riders are distinct.

Stake = 100 yen per ticket.

No individual ticket is removed because of its odds.

## 8. Payout-potential diagnostics
After the compact formation is fixed, calculate race-level price structure from pre-race odds.

Let N = final ticket count.

Break-even support:
ProfitMass_1x = sum q(t) for selected tickets with odds(t) >= N.

2x-return support:
ProfitMass_2x = sum q(t) for selected tickets with odds(t) >= 2N.

Also record:
- q mass of the whole formation
- median selected-ticket odds
- market-weighted geometric mean selected-ticket odds

These are diagnostics/classification variables in v8.8-F09. They are NOT yet hard thresholds, avoiding Q1 outcome-fitted price gates.

## 9. Simulation outputs required
For 2024Q1 report:
- population
- PS_AB pass races
- four H-state race counts
- bet races
- hit races / hit rate
- total tickets
- average / min / max tickets per race
- stake / payout / profit / ROI
- maximum losing streak
- profitable-hit races
- losing-hit races
- profitable-hit share
- payout bands
- size-pattern distribution
- performance by four H states
- performance by ProfitMass_1x / ProfitMass_2x quantiles for analysis only

## 10. Development discipline
- 2024Q1 is development data.
- Do not alter this rule set during the Q1 simulation.
- Any threshold discovered from Q1 must be explicitly labeled Q1-developed and later validated out of sample.
- Primary design objective: reduce tickets while retaining meaningful hit rate and improving ROI.
- A Q1 result with negative profit is not accepted as a completed winning scheme.

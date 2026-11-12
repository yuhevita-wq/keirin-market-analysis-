# v8.9-F10 — Compact Entry / Market-Cliff Scheme

Status: Q1-DEVELOPED DESIGN / REQUIRES OUT-OF-SAMPLE VALIDATION
Base formation engine: v8.8-F09

## 0. Target population
- F1 meetings
- S class
- 7 riders
- complete 3連複 35 odds
- complete 3連単 210 odds
- complete Rakuten K-Dreams predicted line formation
- market data only for decision logic

## 1. Normalize markets
Use the same normalized 3連複 and 3連単 market measures as v8.8-F09.

From 3連複:
- P(c)
- S(i)
- LS(L)
- adjacent-pair PS

Define A and B as the top two lines by LS.

From 3連単:
- q(t)
- H1(i) = first-place support for rider i

## 2. Hard gate 1 — PS_AB
PASS only when the top two adjacent-pair PS rows come from exactly line A and line B, one each.

If PS_AB fails: SKIP.

## 3. Hard gate 2 — H_CONCENTRATED
Rank riders by H1.

Let:
- H1_top = largest H1
- H1_second = second-largest H1

PASS only when:

H1_top >= 2 * H1_second

If not: SKIP.

This is intentionally the complement of the old v6.1 H_RATIO pass condition.
The rule is Q1-developed from the v8.8-F09 H-state analysis and must be validated out of sample before being treated as final.

## 4. H_AB is classification only
H_AB records whether the top two H1 riders come one each from A and B.

H_AB does NOT reject a race in v8.9-F10.

Reason: both H_AB=0 and H_AB=1 subgroups inside H_CONCENTRATED performed materially better than their H_RATIO=1 counterparts in Q1. Selecting only one H_AB subgroup would be a more aggressive in-sample restriction.

H_POS remains removed.

## 5. Formation seed
After both hard gates pass, use the same v8.8-F09 formation engine.

Start from the strongest valid exact-order 3連単 ticket by normalized q within the structural rider pools.

## 6. Grow one rider at a time
At each step, add one rider to one position set F1, F2 or F3.

For each candidate addition:
- calculate added valid tickets ΔN
- calculate added 3連単 market mass Δq
- marginal density = Δq / ΔN

Take the candidate with the greatest marginal density.

No fixed rider counts by place.
No fixed maximum ticket count.

## 7. Stop at the market cliff
For each accepted growth step, calculate geometric-mean q support E_k for the newly created ticket block.

For adjacent blocks:
Drop_k = E_k / E_{k+1}

Find the largest drop and select the formation immediately before it.

## 8. Final ticket set
Use one clean branch-free rectangular F1-F2-F3.

Generate all valid distinct-rider exact-order tickets represented by that formation.

Stake: ¥100 per ticket.

No individual price cut.
No odds > point-count pruning.
No nested-set requirement.

## 9. Payout-potential diagnostics
After formation is fixed, continue recording:
- ProfitMass_1x
- ProfitMass_2x
- total q mass
- median selected-ticket odds
- market-weighted geometric-mean selected-ticket odds

These remain diagnostics only in v8.9-F10. They do not form an additional hard gate yet.

## 10. Expected race-count effect from existing Q1 segmentation
This is not a new simulation result; it is arithmetic from the already completed v8.8-F09 Q1 H-state counts.

v8.8-F09 PS_AB pass: 604 races.
H_RATIO=0 states:
- H_AB=0 / H_RATIO=0: 88 races
- H_AB=1 / H_RATIO=0: 117 races

Therefore the new hard gate corresponds to 205 of the previously classified Q1 races, about one-third of the 604 PS_AB-pass races.

Previously measured combined Q1 totals for these two groups, before a dedicated v8.9-F10 rerun:
- races: 205
- hits: 73
- tickets: 2,221
- average tickets: about 10.83
- stake: ¥222,100
- payout: ¥196,720
- profit: -¥25,380
- ROI: about 88.57%

These values are descriptive development evidence, not validation.

## 11. Development discipline
- 2024Q1 is development data.
- The new H_CONCENTRATED gate was motivated by Q1 segmentation and must be labeled Q1-developed.
- Do not tune an additional H_AB or ProfitMass threshold from the same Q1 result and call it validation.
- The formation logic remains fixed from v8.8-F09.
- Primary objective: materially reduce purchased races while preserving useful hit rate, compact ticket counts and improving ROI.
- Negative Q1 profit is not accepted as a completed winning scheme.

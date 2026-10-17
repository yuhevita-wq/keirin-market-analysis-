# v8.11-F12 Race-Type Adaptive

Status: DEVELOPMENT DESIGN / NOT YET VALIDATED

## Core idea
The same market structure must not be interpreted identically across every S-class race type.

The scheme keeps the proven construction order:
1. Read the trio market and define A/B.
2. Require PS_AB market structure.
3. Build one clean rectangular trifecta formation by the v8.8 market-cliff method.
4. Interpret H-market structure according to race type.
5. Only after a clean formation exists, inspect trifecta prices.
6. If price compression is useful, remove whole riders from a place-set only.
7. Never remove individual cheap tickets and never leave an irregular ticket set.

## Five race-type groups

| Group | Included race types | Provisional market rule |
|---|---|---|
| QUALIFYING | S-class qualifying / preliminary races | PS_AB + H_CONCENTRATED |
| GENERAL | S-class general races | PS_AB + H_CONCENTRATED + H_AB |
| SEMIFINAL | S-class semifinals | PS_AB only; H state diagnostic |
| SPECIAL | S-class selection / special-selection / initial-special-selection | PS_AB only; H state diagnostic |
| FINAL | S-class finals | PS_AB + H_BALANCED |

Definitions:
- H_CONCENTRATED: H1_top >= 2 * H1_second, equivalent to H_RATIO=False.
- H_BALANCED: H1_top < 2 * H1_second, equivalent to H_RATIO=True.
- H_AB: the top two H riders are split one each across A/B.

## Why the rules differ

### QUALIFYING
The working hypothesis is that a clear first-place anchor is meaningful in qualifying races. Therefore concentration remains useful.

### GENERAL
General races are treated more strictly. A concentrated first-place market alone is not enough; the second major H support must still connect to the other A/B market axis via H_AB.

### SEMIFINAL
Semifinals can contain multiple viable advancement scenarios. Requiring a single dominant first-place market may reject useful two-pole structures, so PS_AB is the hard structural requirement and H is classification only.

### SPECIAL
Selection and special-selection races are treated similarly to semifinals in the first development version. PS_AB defines the market backbone; H concentration is not forced.

### FINAL
A final is treated as structurally different. A single overwhelming first-place anchor is not assumed to be desirable. The provisional hypothesis prefers a balanced top H market while still requiring PS_AB.

## Formation and price discipline
- No fixed riders per place.
- No fixed ticket-count cap.
- No nested-set requirement.
- No individual ticket pruning.
- The clean F1-F2-F3 formation must exist before price is inspected.
- Price correction may only shrink a place-set by removing a whole rider.
- Every price-stage candidate must remain one clean rectangular formation.
- The v8.10 support-vs-price knee remains the price-stage selector in F12.

## Development discipline
These race-type rules are conceptual hypotheses and are not claimed to be validated.
The 2x H concentration boundary is inherited from the existing H_RATIO definition, not fitted separately by race type.
No Q1 result, hit, payout, or ROI is used inside the engine.
The next step is a fixed-rule 2024Q1 simulation with performance reported separately for all five groups.
Any later modification discovered from Q1 remains development and requires untouched out-of-sample validation.

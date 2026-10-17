# CROSS LAB: missing race scopes (v5.3)

The page previously failed on abbreviated race types such as `予選`, `特一般`, and `準決勝`, and retained the preceding race's selected class. Resolve class from all published entrant classes and reset it on every load. Preserve G1/G2/G3 meeting metadata.

The seven-car A1/A2 and S coefficients are unchanged. Two independent ordered-pair / conditional-third models add S-class nine-car and A3 seven-car support. They use the same interaction features, with the means divided by the actual car count. This is not a conversion of the existing nine-car pair-only diagnostic to a four-ticket probability.

Protocol fixed before evaluation: train through 2024-12-31; SHA256 race-ID sampling, train cap 3,000 per scope, holdout cap 1,500 per period; ridge 0.01. Data: `data/grade_races/**/*.zip` for S9, `data/a3_challenge/*/{races,entries,results}.csv` for A3. Require published line assignments, full positive score/age and all other numeric fields, completed top-three outcomes, and no reported incidents. Data ends 2026-06-30; no October race outcome enters fitting or selection. Run `OPENBLAS_NUM_THREADS=2 python research/cross_extended_v1/train.py`.

S9 trains on 2,712 races. Holdouts: 1,500 from 2025 and 822 from 2026H1. A3 trains on 3,000 sampled races; both holdouts contain 1,500 races. `validation.json` records out-of-sample ordered-top-three log loss. Comparing with uniform is a sanity check, not proof of calibrated probabilities, improved cross-ticket selection, or profitability. No payout/odds claims are made.

The new scopes are experimental. The page compares the sum of the four joint outcome probabilities; parenthesized supplementary thirds remain separate. Without complete odds, ranking is by four-ticket probability and not expected return. Existing supported scopes continue to use their original coexistence model.

Validation: page-module regression loads all 23 frozen race cards (Keiokaku 1–12 and Yokkaichi 1–11, October 17) through the real load/analyze functions. It checks routing, a unique main answer, full-distribution normalization, model provenance, and absence of odds-value claims. `node --test docs/cross-lab/*.test.mjs` passes 70 tests.

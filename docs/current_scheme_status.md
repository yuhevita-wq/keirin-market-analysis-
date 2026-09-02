# Current Scheme Status

## Current scheme candidate

**v12.1-N02 — ANOMALY FIRST**

Status: **Q1 SIMULATED — BETTING THESIS FAILED; FEATURE/RACE-SELECTION ARCHITECTURE RETAINED FOR REDESIGN**

The v12.1-N02 code was not changed during the Q1 simulation.

## 2024Q1 result

- population: 1,191
- bet races: 84
- buy rate: 7.05%
- hits: 0
- tickets: 212
- average tickets: 2.52
- stake: ¥21,200
- payout: ¥0
- ROI: 0.00%

The entrance mechanism did materially restrict participation compared with v12.0-N01, but the specific N02 interpretation of a robust negative cross-market residual as a bet on `head-a-b / head-b-a` failed completely.

## Priority

**Race selection remains the first and most important decision.**

The active architecture must decide whether a race contains an unusual market structure before it is allowed to generate tickets.

## What remains useful

`market_connection_features_v1.py` remains a pure feature utility and retains:

- complete trio 35-way market;
- complete trifecta 210-way market;
- market head support H;
- head-conditioned companion-pair attachment;
- trio-set companion attachment;
- cross-market connection residual;
- tail-order asymmetry;
- deterministic historical race-card F.

Line information may enter only inside F role fit. Line membership is not a direct betting rule.

## What Q1 rejected

Do not treat an extreme **negative** cross-market connection residual, even when market-head, F-head, and trio-package support align, as sufficient reason to buy the corresponding head-fixed tail-swap trifectas.

Do not rescue N02 with a Q1-fitted modified-z cutoff, race-type split, ticket-count bucket, or other post-result filter.

## Rejected accumulation rule

When a version fails, remove or replace the failed betting logic, not every useful feature or representation that version introduced.

The Q1 result remains as an audit record. A future redesign should preserve the race-selection-first architecture while changing the interpretation or direction of the anomaly before ticket generation.

## Execution rule

Creating or updating a scheme does not authorize simulation. Simulation runs only when explicitly requested by the user.

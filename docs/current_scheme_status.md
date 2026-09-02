# Current Scheme Status

## Active scheme

**None.**

The most recent candidate, `v12.0-N01 — MARKET CONNECTION TOPOLOGY`, was rejected after its fixed 2024Q1 simulation.

## v12.0-N01 rejection

2024Q1 fixed simulation:

- population: 1,191
- bet races: 1,191
- buy rate: 100.00%
- hits: 685
- hit rate: 57.51%
- total tickets: 44,842
- average tickets per race: 37.65
- stake: ¥4,484,200
- payout: ¥3,428,300
- profit: -¥1,055,900
- ROI: 76.45%

The scheme failed structurally because its connection-acceptance rule produced at least one buy connection in every eligible race and expanded into broad exact-trifecta coverage.

## Removal rule applied

Because the user requires rejected betting logic not to accumulate into later versions, the following v12 active implementation files were removed after the simulation:

- v12 betting-scheme implementation
- v12 Q1 evaluator
- v12 Q1 workflow
- v12 scheme specification

The Q1 result document remains only as an audit record of why v12 was rejected. Future schemes must not import or inherit v12 betting logic.

`racecard_fundamentals_v1.py` remains available because it is a deterministic feature utility rather than a betting scheme.

## Execution rule

Creating or updating a scheme does not authorize simulation. Simulation runs only when explicitly requested by the user.

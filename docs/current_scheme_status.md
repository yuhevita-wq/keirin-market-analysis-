# Current Scheme Status

## Current scheme

**v11.0-C01 — MARKET PSYCHOLOGY + RACECARD CONTEXT**

Status: **RESTORED AS DEVELOPMENT BASELINE**

The project has been rolled back from the rejected v12/v12.1 betting logic to v11.0-C01.

## Why v11.0-C01 is the restoration point

v11.0-C01 kept the useful market-psychology engine intact and added deterministic historical race-card context without immediately changing tickets.

It observes:

- complete 35-way trio market;
- complete 210-way trifecta market;
- market first-place support and concentration;
- market head support/challenge from race-card fundamentals;
- market line support/challenge from race-card fundamentals;
- deterministic race-card F components.

The race-card context does not directly add, delete, or prune trifecta tickets in v11.0-C01.

## What was good and must remain

- market psychology remains primary;
- race-card information explains or challenges market psychology rather than replacing it;
- no direct fundamental hard veto;
- no blended pseudo-probability model;
- no post-result ticket pruning;
- no automatic conversion of disagreement into an opposite-line bet;
- the existing v8.25 formation is kept fixed while race-selection work is developed.

## The specific weakness to change next

The betting entrance is still inherited from v8.25-F26.

That means v11.0-C01 successfully *observes* race-card context but does not yet use that context to improve the most important decision: **whether to enter the race at all**.

This is the only active redesign target.

### Development constraint

**Change race selection only. Keep the formation/ticket-generation logic fixed while the entrance is being developed.**

Do not simultaneously change:

- race-selection rule;
- first/second/third formation;
- ticket pruning;
- stake sizing.

Otherwise the cause of improvement or failure becomes unidentifiable.

## v12 / v12.1 cleanup

Rejected betting logic has been removed from the active branch:

- v12.1-N02 betting implementation;
- v12.1 Q1 evaluator;
- v12.1 Q1 workflow;
- v12.1 scheme specification.

The v12/v12.1 simulation-result documents remain only as audit evidence of failure.

`market_connection_features_v1.py` is retained only as a pure descriptive feature utility. It has no betting authority and is not part of the active v11 betting decision.

## Execution rule

Creating or updating a scheme does not authorize simulation. Simulation runs only when explicitly requested by the user.

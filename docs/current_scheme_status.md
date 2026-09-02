# Current Scheme Status

## Current scheme

**v12.1-N02 — ANOMALY FIRST**

Status: **DESIGN_FROZEN_PRE_SIMULATION**

No simulation has been run for v12.1-N02.

## Priority

**Race selection is the first and most important decision.**

The active architecture must decide whether a race contains an unusual market structure before it is allowed to generate tickets.

## What was wrong with v12.0-N01

The useful market-connection concept was not the part rejected by the Q1 test. The rejected parts were the betting/participation rules:

- broad natural top blocks;
- two-of-three overlap treated as sufficient evidence;
- one accepted ordinary connection was enough to enter the race;
- every accepted connection became tickets.

That architecture bought all 1,191 Q1 races and averaged 37.65 tickets per race. Those rules are removed and must not be inherited.

## What is retained

`market_connection_features_v1.py` restores the useful connection-analysis layer as a pure feature utility. It cannot buy a race or create a ticket.

It retains:

- complete trio 35-way market;
- complete trifecta 210-way market;
- market head support H;
- head-conditioned companion-pair attachment;
- trio-set companion attachment;
- cross-market connection residual;
- tail-order asymmetry;
- deterministic historical race-card F.

Line information may enter only inside F role fit. Line membership is not a direct betting rule.

## v12.1 race gate

v12.1 examines all 105 possible head/pair connections inside one race and searches for a robust within-race cross-market anomaly.

The race may participate only when an under-attached connection is an extreme modified-z residual and is also structurally supported by the market head hierarchy, the independent F head hierarchy, and the trio package structure.

If there is no such anomaly, the race is **NO BET**.

The robust-outlier cutoff is the conventional modified-z value 3.5, fixed before any v12.1 simulation and not selected from keirin outcomes.

## Rejected accumulation rule

When a version fails, remove the failed betting logic, not every useful feature or representation that version introduced.

The audit result for v12.0 remains available to explain why its participation rule was rejected, but v12.1 does not import the v12.0 betting implementation.

## Execution rule

Creating or updating a scheme does not authorize simulation. Simulation runs only when explicitly requested by the user.

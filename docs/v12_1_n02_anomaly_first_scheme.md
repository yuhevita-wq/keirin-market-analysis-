# v12.1-N02 — ANOMALY FIRST

## Status

**DESIGN_FROZEN_PRE_SIMULATION**

No simulation has been run for v12.1-N02.

## Core correction

Race selection is the first decision.

The rejected v12.0-N01 mistake was not the market-connection concept itself. The mistake was its participation rule:

- broad natural top blocks;
- two-of-three overlap counted as enough evidence;
- one accepted connection was enough to authorize a race;
- every surviving connection became tickets.

That made at least one connection appear in all 1,191 Q1 races and produced 100% participation with 37.65 average exact-trifecta tickets.

Those betting rules remain rejected and are not inherited.

## Good v12 ideas retained

A pure feature module, `market_connection_features_v1.py`, preserves only the useful non-betting structure:

- complete 35-way trio market;
- complete 210-way trifecta market;
- first-place market support H;
- for every possible head and unordered companion pair, head-conditioned trifecta attachment;
- corresponding trio-set attachment;
- cross-market connection log residual;
- exact tail-order asymmetry;
- deterministic historical race-card fundamental F;
- line information only inside F role fit, never as a direct ticket rule.

The feature module cannot buy a race or create a ticket.

## Race selection first

For each race there are 7 possible heads × 15 companion pairs = 105 directed-head / unordered-tail connections.

For each connection:

`R(h,{a,b}) = log(TF_head_pair / TRIO_set_pair)`

This compares how strongly the exact-order market attaches `{a,b}` to head `h` against how strongly the unordered trio market packages the same three-rider set.

The 105 residuals create that race's own internal reference distribution.

### Robust anomaly detector

Residuals are standardized by the modified z-score:

`Mz = 0.6745 * (R - median(R)) / MAD(R)`

A connection is an extreme residual only when `|Mz| >= 3.5`.

The value 3.5 is the conventional robust-outlier criterion. It was fixed before any v12.1 simulation and was not selected from Q1/Q2/Q3 betting outcomes.

This changes the question from:

> Does a connection exist?

to:

> Is this connection abnormally different from the other 104 connections in this same race?

## Current v12.1 thesis

v12.1-N02 only expresses a negative extreme residual: the trio market strongly packages a three-rider set, while the trifecta market attaches that companion pair to a particular head abnormally less than the race's normal connection pattern.

That connection is eligible only when all of the following are true before the result:

1. the residual is a robust negative outlier (`Mz <= -3.5`);
2. the head belongs to the market's natural head block;
3. the same head belongs to the independent race-card F head block;
4. the companion pair belongs to that head's natural upper trio package.

If no connection satisfies all four conditions, the race is **NO BET**.

This is the key architecture change: a race cannot enter merely because ordinary market structure exists.

## Ticket construction

Ticket construction occurs only after the race-selection gate passes.

For each selected anomalous connection `h + {a,b}`:

- `h-a-b`
- `h-b-a`

are retained because the thesis identifies the head and unordered companion set, not a reliable second/third ordering.

No fixed ticket-count cap is used. If the anomaly gate produces no connection, there is no bet.

## Explicitly removed logic

v12.1 does not contain:

- v12.0 two-of-three overlap buy rule;
- one-connection-is-enough participation rule based on ordinary top blocks;
- race-type-specific buying branches;
- line-supported / line-challenged ticket rules;
- result or payout input;
- Q1-derived ticket-count filters;
- estimated ROI gates;
- post-result pruning.

## Research discipline

The market-connection feature idea is retained because it was not itself disproven by the v12.0 Q1 result. The broad participation and ticket-generation rules were disproven and removed.

No simulation is authorized by creation of this design. Simulation is run only on explicit user request.

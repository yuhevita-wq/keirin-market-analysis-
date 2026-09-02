# Current Scheme Status

## Current scheme

**v12.0-N01 — MARKET CONNECTION TOPOLOGY**

Status: **DESIGN_FROZEN_PRE_SIMULATION**

No simulation has been run for v12.0-N01.

## Architecture

v12 is a fresh betting-scheme root. It does not inherit the betting decisions, race-type branches, formations, or buy/no-buy gates of v8, v9, v10, or v11.

It uses:

- complete 35-way trio market;
- complete 210-way trifecta market;
- market first-place support;
- head-conditioned companion-pair attachment;
- trio companion-pair attachment;
- deterministic historical race-card fundamental F;
- natural top blocks derived from within-race structure;
- two-of-three informative-view support at the individual connection level.

Tickets are directed connection paths, not line formations.

Line information may affect only race-card role fit inside F. Line membership does not directly create, replace, or delete a v12 ticket.

`race_type` is metadata only and cannot change the decision rule.

## Rejected accumulation rule

The rejected `v11.1-C02 HEAD SURVIVES × LINE FADE` code, evaluator, Q1 workflow, specification, and Q1 result were removed before v12 was created.

Going forward, a betting logic that is explicitly rejected should not become another inherited layer of the active scheme. A replacement scheme should either remove the rejected active implementation or start from a clean root that does not import it.

## Reused utility

`racecard_fundamentals_v1.py` remains because it is a deterministic feature utility, not a betting scheme. It contains no outcome-dependent betting gate.

## Execution rule

Creating or updating a scheme does **not** authorize simulation. Simulation runs only when explicitly requested by the user.

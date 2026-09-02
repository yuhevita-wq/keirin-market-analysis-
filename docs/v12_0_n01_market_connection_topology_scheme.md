# v12.0-N01 — MARKET CONNECTION TOPOLOGY

## Status

**DESIGN_FROZEN_PRE_SIMULATION**

No simulation was run while creating this scheme.

This is a fresh betting-scheme root. It does **not** inherit the betting decisions, race-type branches, formations, or buy/no-buy gates of v8, v9, v10, or v11.

The rejected `v11.1-C02 HEAD SURVIVES × LINE FADE` implementation, evaluator, workflow, spec, and result were removed before v12 was created.

## Why v12 exists

The previous failure was not merely that C02 used too few tickets. The deeper mistake was translating a market contradiction back into conventional keirin language:

- strong head -> fix first;
- weak line -> delete that line;
- stronger rival line -> use that line.

That interpretation was tidy but it reintroduced ordinary race-prediction assumptions.

v12 instead treats the betting market itself as a **connection network**.

The question becomes:

> If rider `i` is the head, which two-rider companion set does each information source connect to `i`?

No line label is allowed to answer that question.

## Eligible input

- F1
- S-class
- 7 riders
- complete 35-way trio market
- complete 210-way trifecta market
- complete historical race-card rows for all 7 riders

`race_type` is metadata only. It cannot change the rule.

## Independent race-card view

The existing deterministic `racecard_fundamentals_v1` utility is reused because it is not a betting rule.

For each rider it uses only historical race-card snapshot fields:

- score
- top3_rate
- B
- nige
- makuri
- sashi
- mark
- line_position / line_size for role fit

It produces `F(i)` with equal, fixed components and no fitted coefficient.

Line information stops here. It can describe a rider's role fit, but **line membership never directly creates or deletes a ticket in v12**.

## Market head block

Normalize the complete 210-way trifecta market:

`q(t) = (1 / odds_t) / Σ(1 / odds)`

First-place support:

`H(i) = Σ q(i, *, *)`

The head block is the natural upper market tier determined by the largest adjacent support-ratio drop.

There is no fixed number of heads.

If the head view is completely flat, the race is NO BET because the market has not expressed a usable head structure.

## The 15 companion pairs for each head

For one head `h`, the other six riders create exactly 15 unordered companion pairs `{a,b}`.

Each pair is evaluated from three different views.

### View A — trifecta head-conditioned attachment

`TF_h(a,b) = [q(h,a,b) + q(h,b,a)] / H(h)`

This asks:

> Conditional on the market believing `h` wins, which two riders does it package with that head?

This is not a line calculation.

### View B — trio set attachment

Normalize the 35-way trio market to `P(set)`.

For head `h`:

`TRIO_h(a,b) = P({h,a,b}) / Σ P({h,x,y})`

This asks:

> In the unordered three-rider market, which companion pair is attached to `h`?

### View C — independent fundamental pair strength

`FPAIR_h(a,b) = F(a) + F(b)`

This asks:

> Ignoring the market's packaging, how much independent race-card strength do these two companions carry?

## No fixed cutoffs: each view creates its own natural top block

For the two probability views, the upper block is determined by the largest adjacent support-ratio drop.

For the F-pair view, the upper block is determined by the largest adjacent absolute F gap.

If a view is completely flat and returns all 15 pairs, that view is considered **uninformative** and does not vote.

This is important: v12 does not force every data source to have an opinion.

## Connection rule

For a head to produce tickets, at least **two views must be informative**.

A companion pair survives only when it belongs to the natural top block of at least **two informative views**.

Examples:

- trifecta + trio support it, fundamentals dissent -> survives;
- trifecta + fundamentals support it, trio dissent -> survives;
- trio + fundamentals support it, trifecta dissent -> survives;
- only one view supports it -> rejected.

This is not an agreement/disagreement race filter. The disagreement is preserved at the **individual connection level**.

## Ticket generation

For every surviving connection `{a,b}` under head `h`, buy both exact orders:

- `h-a-b`
- `h-b-a`

The scheme deliberately refuses to pretend that the current F model can reliably separate second from third place.

Tickets are therefore generated as a **directed connection path set**, not a rectangular line formation and not a post-hoc pruning of a larger formation.

## What v12 deliberately does not do

- no race-type-specific branch;
- no result or payout input;
- no Q1/Q2/Q3 fitted threshold;
- no fixed ticket-count cutoff;
- no estimated-ROI gate;
- no `HEAD_SUPPORTED` buy gate;
- no `LINE_SUPPORTED` / `LINE_CHALLENGED` buy gate;
- no rule saying a strong head must be fixed alone;
- no rule saying a challenged line must be deleted;
- no replacement of one line with another line;
- no prediction mark / evaluation mark;
- no individual ticket deletion after seeing a rectangular formation.

## Scientific interpretation

v12 is testing a different object from the previous schemes.

It is not primarily asking:

> Who is strong and which line wins?

It is asking:

> Which exact rider connections are simultaneously visible in at least two of three independent structural views?

The market is treated as something that has a topology: heads, attachments, packages, and conditional paths.

The race card is used to challenge or support those connections without being allowed to replace the market with a conventional race prediction.

## Evaluation discipline

This design is frozen before its first simulation.

A request to create/update the scheme does not authorize simulation. Only an explicit simulation request may run a historical period.

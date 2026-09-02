# v11.1-C02 — HEAD SURVIVES × LINE FADE

## Status
PRE-SIMULATION DESIGN.

This branch has been designed after observing Q1-Q3 v11 context splits, so Q1-Q3 are development data for this branch. Do not treat any later Q1-Q3 simulation as out-of-sample validation.

## Target context

Only the existing v11.0 context:

`H_CONCENTRATED | HEAD_SUPPORTED | LINE_CHALLENGED`

Interpretation:

- the trifecta market is strongly concentrated on one first-place rider H1;
- deterministic race-card fundamentals independently keep H1 inside the natural top head block;
- the market's strongest trio-supported line is not the strongest line by race-card fundamentals.

## Sharpening principle

Do not throw away the head merely because the line is challenged.

Treat the contradiction literally:

**HEAD SURVIVES. MARKET LINE FADES.**

The branch is only active when the market's strongest line is H1's own line. If the market's strongest line is some other line, the conflict has a different psychological meaning and C02 does not force a bet.

## Ticket construction

When C02 applies:

1. Fix market H1 in first place.
2. Identify H1's line. It must equal the market top line.
3. Identify the strongest line by summed race-card F. It must be a different line.
4. Rank riders inside that rival fundamental line by F.
5. Keep only the top two rival-line riders.
6. Buy the two tail orders only:
   - H1 → A → B
   - H1 → B → A

Exactly **2 tickets**.

No H1 line-mate is allowed in the C02 tail.

## Why this is deliberately sharp

The hypothesis is not "fundamentals beat the market". The market still determines the head. Fundamentals are used only to challenge the packaged line accompaniment around that head.

This preserves the strongest part of the market-psychology framework while expressing the disagreement in the smallest coherent trifecta rectangle.

## No fitted thresholds

C02 adds no learned odds cutoff, no learned F-gap cutoff, and no payout/result condition.

The only new structural requirement is that H1 belongs to the market top line, because without that geometry the phrase "head survives, market line fades" is not logically defined.

## Evaluation discipline

- No simulation was run when this design was created.
- Q1-Q3 are development data because their v11 outcomes/context performance were already observed.
- Any tuning after seeing Q1-Q3 C02 results must remain development-only.
- A future untouched period is required for real validation after the branch is frozen.

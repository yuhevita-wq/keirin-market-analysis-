# v8.17-F18 Race-Type Market Semantics

Status: DEVELOPMENT / 2024Q1 architecture

## Core principle

The same market signal does not have the same meaning in every S-class race type.
The scheme therefore interprets market hierarchy by race type before deciding whether a race is buyable.

Common constraints remain unchanged:
- F1, S-class, seven-car races only.
- Complete trio 35 and trifecta 210 markets.
- Complete predicted line formation.
- Market data + line structure only for decisions.
- No rider ability, score, style or result information in the decision engine.
- Build a clean rectangular formation first.
- No individual cheap-ticket pruning.
- Price compression, where used, removes whole riders from a positional set only.

## Race-type branches

### S-class Qualifying
- Entry: PS_AB + H_CONCENTRATED.
- H_CONCENTRATED means H1_top >= 2 * H1_second.
- Formation: F09 Market Cliff.
- Price phase: existing structural whole-rider compression.
- Development posture: micro-adjust only.

### S-class General
- Entry: PS_AB + H_CONCENTRATED + H_AB.
- Preserve the dominant H1 rider as the first-place anchor because the entry itself was justified by that first-place concentration cliff.
- Formation: F09 Market Cliff corrected so formation growth cannot contradict the H1 cliff used at entry.
- Price phase: structural whole-rider compression only after the anchored rectangle exists.
- No global one-rider first-place rule outside this branch.

### S-class Semifinal
- Entry: PS_AB only.
- H state is diagnostic, not a hard gate.
- Formation: F09 Market Cliff.
- Price phase: existing structural whole-rider compression.
- Development posture: micro-adjust only.

### S-class Selection
- Formal market meaning: MARKET DISAGREEMENT branch.
- Do NOT interpret PS_AB, H_AB or H concentration as a direct winner-strength signal.
- Compare the same 35 unordered three-rider sets across the two markets:
  - P_trio(c): normalized trio implied probability.
  - Q_tf_set(c): normalized trifecta implied probability aggregated across all six orders of set c.
  - D_log(c) = log(Q_tf_set(c) / P_trio(c)).
- Diagnostics include total-variation distance between P_trio and Q_tf_set, top-set disagreement, top-3/top-5 overlap, rank shifts and the sets with the strongest positive/negative D.
- Current action: NO BET / diagnostic-only.
- No Q1-fitted divergence threshold is permitted. A later live Selection rule must be semantic, pre-race, and fixed before untouched validation.

### S-class Special
- Entry: PS_AB + H_CONCENTRATED.
- H_AB is diagnostic only.
- Formation: unchanged F09 Market Cliff.
- Do not force a one-rider first-place anchor.

### S-class Initial Special / First-Day Special
- Treat `Ｓ級初特選` and `Ｓ級初日特選` as the same subtype.
- Entry: PS_AB + H-state consistency, where H_AB == H_CONCENTRATED.
- Formation: unchanged F09 Market Cliff.

### S-class Final
- Entry: PS_AB + H_CONCENTRATED.
- H_AB is diagnostic only.
- Formation: unchanged F09 Market Cliff.
- Do not force a singleton first-place anchor.
- Do not add an extra price gate.
- This replaces the rejected H_BALANCED final hypothesis.

## Interpretation summary

- Qualifying: buy a concentrated market.
- General: buy concentration and preserve the first-place cliff that justified entry.
- Semifinal: buy the A/B market structure without requiring first-place concentration.
- Selection: read disagreement between markets, not raw support strength.
- Special: buy concentrated first-place support after PS_AB.
- Initial Special: buy consistency between line-side and first-place H states.
- Final: trust unusually strong first-place concentration after PS_AB.

## Anti-overfitting rule

Selection remains quarantined until a market-disagreement rule is defined without payout/result-tuned thresholds. The diagnostic layer is part of the scheme; a betting gate is not yet part of the scheme.

# QUALIFYING two-path diagnostic — 2024Q1 + Q2

Status: DEVELOPMENT DIAGNOSTIC. Q1 and Q2 are observed development data. No fitted numeric cutoff was used in this comparison.

## Question

For S-class qualifying races, should the rebuild pursue:

1. **HARD / narrow-consensus**: enter races with a strongly concentrated first-place market and reduce the clean rectangular formation aggressively; or
2. **HOLE / dispersed-disagreement**: enter races with balanced first-place support or trio-vs-trifecta top-set disagreement and seek higher-priced outcomes?

The neutral comparison uses the unchanged v8.8-F09 formation after PS_AB and splits races only by pre-existing market semantics:
- H_CONCENTRATED: H1 >= 2*H2
- H_BALANCED: H1 < 2*H2
- H_AB: top two H riders split one each across top lines A/B
- TOP_SET_AGREE / DISAGREE: exact equality of the #1 unordered trio set between the 3連複 market and the 3連単 market collapsed to 35 unordered sets.

## Current qualifying branch

Current v8.21 lineage inherits the qualifying rule `PS_AB + H_CONCENTRATED`, then structural whole-rider price compression.

| Dataset | Races | Hits | HR | Avg tickets | Stake | Payout | Profit | ROI | Avg payout / hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 63 | 16 | 25.40% | 9.17 | 57,800 | 58,500 | +700 | 101.21% | 3,656 |
| Q2 | 51 | 23 | 45.10% | 8.16 | 41,600 | 27,010 | -14,590 | 64.93% | 1,174 |

Interpretation: Q2 is not failing because the branch cannot identify likely outcomes. Hit rate rises sharply, but payout per hit collapses. The main problem is price / ticket burden.

## Neutral F09 semantic comparison

### Broad H state

| State | Q1 ROI | Q2 ROI | Q1 HR | Q2 HR | Q1 avg tickets | Q2 avg tickets |
|---|---:|---:|---:|---:|---:|---:|
| H_CONCENTRATED | 104.25% | 62.96% | 26.98% | 45.10% | 9.63 | 8.41 |
| H_BALANCED | 61.83% | 53.38% | 25.00% | 32.50% | 8.48 | 12.11 |

The balanced / ostensibly more open market is materially worse in both quarters under the same neutral formation.

### Concentrated H split by H_AB

| State | Q1 races | Q1 ROI | Q2 races | Q2 ROI | Q1 avg payout/hit | Q2 avg payout/hit |
|---|---:|---:|---:|---:|---:|---:|
| H_CONC + H_AB=0 | 24 | **183.48%** | 27 | **91.04%** | 7,376 | 1,036 |
| H_CONC + H_AB=1 | 39 | 65.02% | 24 | 42.27% | 2,200 | 1,491 |

The strongest stable direction is not “more agreement.” It is **first-place concentration without the top-two H riders being neatly split one each across A/B**. This suggests a single dominant head/line block rather than two-line consensus.

For Q2 H_CONC + H_AB=0, the neutral F09 formation uses 182 tickets across 27 races (6.74/race) and returns 16,570 yen. Break-even at 100 yen/ticket would require no more than 165.7 tickets, or about **6.14 tickets/race** if payout were unchanged. That is only about a 9% ticket reduction from the neutral formation. By contrast, H_BALANCED Q2 uses 12.11 tickets/race and would need roughly 6.46 tickets/race at unchanged payout, requiring almost half the ticket burden to disappear.

### Trio-vs-trifecta top-set disagreement

Top-set disagreement does not rescue the hole direction:

| State | Q1 ROI | Q2 ROI |
|---|---:|---:|
| TOP_SET_AGREE | 82.36% | 54.70% |
| TOP_SET_DISAGREE | 56.45% | 63.44% |

Disagreement slightly improves Q2 versus agreement but remains deeply negative and is worse in Q1. It is not a sufficient hole-entry model.

## Development conclusion

**Prefer HARD / narrow-consensus rebuild. Do not pursue a broad hole-seeking qualifying model first.**

But the new HARD model must differ fundamentally from the current qualifying branch:

1. Entry should represent a **single dominant head block**, not generic strong consensus.
2. `H_CONCENTRATED` remains structurally relevant.
3. `H_AB=0` is a promising development diagnostic because it distinguishes one-sided head concentration from clean two-line agreement. It is not yet validated and must not be advertised as a finished rule.
4. Formation should translate that concentration into a smaller rectangle, probably by protecting H1 in first place and removing whole rider branches from second/third place.
5. No individual cheap-ticket deletion.
6. No fitted odds threshold.
7. Freeze the resulting semantic rule on Q1+Q2 development data, then use untouched Q3 for validation.

The hole route is not impossible in principle, but the broad semantic versions tested here (H_BALANCED and top-set disagreement) show no evidence strong enough to justify making it the primary qualifying rebuild.

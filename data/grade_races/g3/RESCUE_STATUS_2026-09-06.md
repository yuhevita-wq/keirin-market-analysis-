# G3 rescue status — 2026-09-06

This file records the emergency recovery state of the G3 historical datasets.

## Confirmed-live artifacts rescued locally and awaiting binary import to main

| Dataset | Artifact ID | SHA256 of downloaded artifact ZIP |
|---|---:|---|
| G3 2024 Q1 | 9785544974 | `1c7e4ca4b6eea44cec3eb378d673a5a61fa03dd7a6b56653d1f94580097edaac` |
| G3 2024 Q2 | 9790740336 | `e89b9768dd331618ab21b4bc3512a48519cd4a795d2b71531b685775ae9554c1` |
| G3 2024 Q3 | 9796166761 | `ceee1476c03b7dfa9719ed2800c90604180d6211b80cc2abe954615cffe46b1e` |
| G3 2024 Q4 | 9802470186 | `07799921346f2070f5330425983127e884ce95ba8d13b7dbc02b725a21cb7eef` |
| G3 2025 Q1 | 9810481117 | `1edae314aadc767266b6cde860d0eeea67e9f21d16822a80d6b762248c18ed23` |
| G3 2025 Q2 | 9810185856 | `b83c5cebb7cb5d5c5e589c8bbaa2aaeba8af61ee3c2b0ecd2464cd1a0b7fb679` |
| G3 2025 Q3 | 9810152428 | `d9ea4b2541d6daf7cb7ff2d9e48c322746a877bf8ba8fcf2b68d42b8317ce1ef` |
| G3 2025 Q4 | 9810151108 | `893269283f2e1b94de3e5c090fbb8e45c01929ead1b526410ca38cecd7992e64` |
| G3 2026 Q1 | 9825216498 | `b8a1a7cc1244123d627e2f597f1372f5b9103c4a3ea51c4fcfa7baab2791ebde` |
| G3 2026 Q2 | 9824781860 | `ebdf5ff21f13e56518921c1dd56b181411715dfde108b6757ebf091dc14f97fc` |

All 10 downloaded ZIPs passed `unzip -tq` integrity validation on 2026-09-06.

## Intended durable main paths

- `data/grade_races/g3/2024/g3-2024-q1.zip` through `g3-2024-q4.zip`
- `data/grade_races/g3/2025/g3-2025-q1.zip` through `g3-2025-q4.zip`
- `data/grade_races/g3/2026_h1/g3-2026-q1.zip`
- `data/grade_races/g3/2026_h1/g3-2026-q2.zip`

## Recovery automation

`.github/workflows/rescue-g3-artifacts-2026-09-06.yml` was committed to main to copy only the ten confirmed-live artifacts into durable Git storage and verify all ten paths after push.

The first run (`34032891025`) failed before any workflow step started (job had no steps / no assigned runner). Therefore **this status file must not be interpreted as proof that the ten binary ZIP files are already on main**. The completion criterion remains: all ten binary ZIPs must be readable from `origin/main`, with hashes matching this table.

## Retention warning

The source acquisition workflows used `retention-days: 7`. Do not treat Actions artifacts as durable storage. The Git main branch is the authoritative durable location once the binary import completes.

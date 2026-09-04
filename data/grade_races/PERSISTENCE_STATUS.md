# Grade-race persistence status

Status: `MAIN_BINARY_COPY_BLOCKED`

This file is intentionally explicit: the G1/G2 and G3 binary dataset archives are **not yet present on `main`**. Do not report them as main-saved until the ZIP files themselves are verified under `data/grade_races/` on `main`.

## Durable source already secured

The acquired datasets are preserved in the GitHub Release tagged `grade-race-databank-2026-09-02`, so they are not dependent on the 7-day Actions artifact retention window.

Scope currently preserved there:

- G1/G2 non-girls: 2021 corrected edition, 2022, 2023, 2024, 2025.
- G3: 2022 Q1-Q4, 2023 Q1-Q4, 2024 Q1-Q4, 2025 Q1-Q4, 2026 Q1-Q2 non-girls.

`RELEASE_INDEX.csv` records the canonical release assets, byte sizes and SHA256 digests.

## Main-copy workflow

A workflow on branch `archive-grade-races-2026-09-02` is prepared to download, ZIP-test, SHA256-check, commit and re-verify the archives on `main`.

Attempts on Ubuntu 24.04, Ubuntu 22.04 and a minimal macOS runner all failed before any workflow step was started. This indicates a GitHub-hosted runner/account-side availability or usage restriction rather than a collector or persistence-script failure. The exact account-side reason is not exposed by the available API response and must not be guessed.

Completion criterion remains:

1. ZIP archives physically present on `main`.
2. Manifest and SHA256 verification complete.
3. Re-fetch `main` and verify the files exist there.

Until all three are true, answer `mainにはまだない`.

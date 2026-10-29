# Grade-race persistence status

Status: `MAIN_RESCUE_COMPLETE`

Verified on `main` as of 2026-09-06.

## G1/G2

The five non-girls archives are physically present under `data/grade_races/g1g2/`:

- 2021 corrected edition
- 2022
- 2023
- 2024
- 2025

All five byte sizes match `RELEASE_INDEX.csv` exactly. The files were sourced from GitHub Release `grade-race-databank-2026-09-02`; the release SHA256 digests remain the canonical integrity references.

## G3

The historical G3 archives are physically present under `data/grade_races/g3/` in year folders:

- 2022 Q1-Q4
- 2023 Q1-Q4
- 2024 Q1-Q4
- 2025 Q1-Q4
- 2026 H1 Q1-Q2

The 2022 and 2023 archives were copied from the durable GitHub Release and their byte sizes match `RELEASE_INDEX.csv` exactly. The 2024-2026H1 rescued archives were previously ZIP-tested and SHA256-verified before import, and their byte sizes on `main` match the rescued source files exactly.

## Durable backup

GitHub Release `grade-race-databank-2026-09-02` remains the canonical secondary backup. `RELEASE_INDEX.csv` records canonical asset names, byte sizes and SHA256 digests.

## Authoritative layout

- G1/G2: `data/grade_races/g1g2/`
- G3 2022: `data/grade_races/g3/2022/`
- G3 2023: `data/grade_races/g3/2023/`
- G3 2024: `data/grade_races/g3/2024/`
- G3 2025: `data/grade_races/g3/2025/`
- G3 2026 H1: `data/grade_races/g3/2026_h1/`

Completion criterion satisfied: archives exist on `main`, expected byte sizes match, and the final layout was re-fetched directly from `main` after the rescue commits.

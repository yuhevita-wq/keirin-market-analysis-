# Grade-race persistence status

Status: `MAIN_PARTIAL_RESCUE_COMPLETE`

## Verified on main as of 2026-09-06

### G3
The following 10 rescued ZIP archives are physically present on `main` under `data/grade_races/g3/` and were re-fetched from GitHub with byte sizes matching the rescued source files exactly:

- 2024 Q1-Q4
- 2025 Q1-Q4
- 2026 H1 Q1-Q2

### G1/G2
The following 5 non-girls ZIP archives are physically present on `main` and their byte sizes match `RELEASE_INDEX.csv` exactly:

- 2021 corrected edition
- 2022
- 2023
- 2024
- 2025

They were uploaded on 2026-09-06 and currently live at `data/grade_races/g3/g1-g2-*.zip` because they were committed one directory higher than the intended `data/grade_races/g1g2/` target. This is an organization issue only; the binary files are durably stored on `main`.

## Durable release source

The GitHub Release tagged `grade-race-databank-2026-09-02` remains the canonical release backup. `RELEASE_INDEX.csv` records canonical asset names, byte sizes and SHA256 digests.

Release scope includes:

- G1/G2 non-girls: 2021 corrected edition, 2022, 2023, 2024, 2025.
- G3: 2022 Q1-Q4, 2023 Q1-Q4, 2024 Q1-Q4, 2025 Q1-Q4, 2026 Q1-Q2 non-girls.

## Still not copied to main

The release-backed G3 2022 Q1-Q4 and 2023 Q1-Q4 archives are not yet physically copied into the rescued `main` G3 archive set. Do not report those eight archives as main-copied until their ZIP files are directly verified on `main`.

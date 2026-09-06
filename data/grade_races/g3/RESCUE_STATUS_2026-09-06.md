# G3 rescue status — 2026-09-06

Status: `COMPLETE`

The historical G3 archive is now durably present on `main`.

## Verified main coverage

- 2022 Q1-Q4: `data/grade_races/g3/2022/`
- 2023 Q1-Q4: `data/grade_races/g3/2023/`
- 2024 Q1-Q4: `data/grade_races/g3/2024/`
- 2025 Q1-Q4: `data/grade_races/g3/2025/`
- 2026 H1 Q1-Q2: `data/grade_races/g3/2026_h1/`

The 2022 and 2023 archives were copied from GitHub Release `grade-race-databank-2026-09-02`; every file byte size matches `RELEASE_INDEX.csv` exactly.

The 2024-2026H1 archives were rescued from the confirmed-live acquisition artifacts. Before import, all ten ZIPs passed `unzip -tq` validation and SHA256 verification. Their byte sizes on `main` match the rescued source ZIPs exactly.

The 2023 archives were initially uploaded into the 2022 folder by mistake, then relocated on `main` using the exact same Git blob SHAs. No binary rewrite occurred during that correction.

GitHub Release `grade-race-databank-2026-09-02` remains the canonical secondary backup and SHA256 reference.

Completion criterion satisfied: all expected G3 ZIP archives are physically present on `main` in the final year-based layout and were re-fetched from `main` after the rescue commits.

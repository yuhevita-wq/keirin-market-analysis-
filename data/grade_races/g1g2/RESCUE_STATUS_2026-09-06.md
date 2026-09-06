# G1/G2 Rescue Status — 2026-09-06

Status: `COMPLETE`

Durable source release: `grade-race-databank-2026-09-02`

The five non-girls archives are physically present on `main` under `data/grade_races/g1g2/`:

| Period | File | Bytes | SHA256 |
|---|---|---:|---|
| 2021 | `g1-g2-2021-non-girls-corrected.zip` | 2993916 | `d3d409920cbbb465347044ba086b7eab03428df045d34c33b41ce7067e8eff3b` |
| 2022 | `g1-g2-2022-non-girls.zip` | 2927894 | `f477f21d50e1d047bb6056d4a96e60e772ef6343d8426db8fde7de7dfc00cf44` |
| 2023 | `g1-g2-2023-non-girls.zip` | 3127130 | `34ea23b896c04e618cd08ff66400ad89b6dd9d450970aba3623d5407b1f3d08d` |
| 2024 | `g1-g2-2024-non-girls.zip` | 3162558 | `814e106c4b77c914b2832b159099591ee61226b4b8353101314405d7e07c9e48` |
| 2025 | `g1-g2-2025-non-girls.zip` | 3262247 | `22a7e95ffb086b30c34e255a525fed28ad7f726d094e971790baaf81cfbcdd0a` |

All five byte sizes on `main` match the release index exactly. The files were initially uploaded under the G3 directory by mistake, then relocated to `data/grade_races/g1g2/` using the same Git blob SHAs, so the binary content was not rewritten during the move.

Completion criterion satisfied: all five ZIPs exist in the final main path and were re-fetched from `main` after relocation.

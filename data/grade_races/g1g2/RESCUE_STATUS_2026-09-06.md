# G1/G2 Rescue Status — 2026-09-06

Target: permanently copy the already-secured G1/G2 non-girls release archives onto `main`.

Durable source release: `grade-race-databank-2026-09-02`

Expected files:

| Period | File | Bytes | SHA256 |
|---|---|---:|---|
| 2021 | `g1-g2-2021-non-girls-corrected.zip` | 2993916 | `d3d409920cbbb465347044ba086b7eab03428df045d34c33b41ce7067e8eff3b` |
| 2022 | `g1-g2-2022-non-girls.zip` | 2927894 | `f477f21d50e1d047bb6056d4a96e60e772ef6343d8426db8fde7de7dfc00cf44` |
| 2023 | `g1-g2-2023-non-girls.zip` | 3127130 | `34ea23b896c04e618cd08ff66400ad89b6dd9d450970aba3623d5407b1f3d08d` |
| 2024 | `g1-g2-2024-non-girls.zip` | 3162558 | `814e106c4b77c914b2832b159099591ee61226b4b8353101314405d7e07c9e48` |
| 2025 | `g1-g2-2025-non-girls.zip` | 3262247 | `22a7e95ffb086b30c34e255a525fed28ad7f726d094e971790baaf81cfbcdd0a` |

Completion criterion:

1. All five ZIPs physically exist under `data/grade_races/g1g2/` on `main`.
2. Each byte size matches the table above.
3. Each Git blob matches the uploaded source bytes; SHA256 is verified where binary readback is available.
4. Re-fetch `main` and confirm all five files are visible.

Current status: `WAITING_FOR_MAIN_UPLOAD`.

# 型ラボ：全アーカイブ復号・実データ接続 実行記録（2026-10-22）

## 実行結果
- GitHub Actions: https://github.com/yuhevita-wq/keirin-market-analysis-/actions/runs/39789535577 (attempt after true owner RSA key installation)
- Workflow result: `private-key-preflight` and all six `full-year-join` jobs **success**.
- Code source revision reported within runner: `7452ef8bc7aaee9917b41323733244ea90e171f8`.
- Historical scope: 2021-01-01 through 2026-06-30, all cataloged 7-market ciphertext files (not sampled).
- Plaintext data / owner private key were **not** committed.

| Year | SHA-verified and decrypted CMS files | Full-market verified distinct races | Exact race ID ↔ actual result matched | Of matched: 7/9-car | No verified result | Processed days |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 490 | 23,797 | 23,669 | 17,968 | 128 | 365 |
| 2022 | 473 | 24,445 | 24,331 | 20,668 | 114 | 363 |
| 2023 | 484 | 25,085 | 24,966 | 23,138 | 119 | 365 |
| 2024 | 445 | 24,700 | 24,598 | 22,834 | 102 | 366 |
| 2025 | 482 | 24,372 | 24,234 | 22,752 | 138 | 365 |
| 2026 H1 | 205 | 11,937 | 11,875 | 11,167 | 62 | 181 |
| **Total** | **2,579** | **134,336** | **133,673** | **118,527** | **663** | **2,005** |

## What "connected" means
1. The original encrypted 7-market archives were hashed, opened and verified, with rerun duplication resolved per race ID. No selected 42,975-race plaintext subset was substituted for the full archive.
2. Actual podium results were matched by exact source race ID; 663 races without verified results remain unlabelled.
3. The workflow output is split into **CMS-encrypted** all-market odds and verified-result-label artifacts, plus a plaintext aggregate count report, per year. GitHub Actions artifacts use 30-day retention; the original ciphertext archives remain at the source repository.
4. 118,527 counts 7/9-car races with market/result labels; it is **not** a claim that the separate original A–F Type Lab engine has already received historical rider-level features or pre-race 1st/3rd probabilities.
5. Historical prior independent settlement audit isolated 188 ambiguous dead-heat settlement outcomes. **This full-join run does not apply that independent payout/settlement quarantine**, so 133,673 should NOT be mislabelled as 133,485 clean training samples. Payout-level tie handling and independent label audit must precede model fit.
6. Historical odds here are **final odds**, not genuine 15-minute-preclose snapshots. Betting/ROI testing with contemporary prices is not validated.

## Next integration gate
- Decode the CMS artifacts within an owner-authorized private environment and join all 7/9-car racecard/entrant/line fields, with as-of-time provenance.
- Enforce payout/dead-heat quarantine and no hindsight leakage.
- Create pre-race p1/p3 probabilities and valid Type Lab A–F classifications; only then begin chronological backtests and model fitting.
- Never export plaintext original markets or RSA private key to a public repo.

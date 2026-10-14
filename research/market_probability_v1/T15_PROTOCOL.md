# T-15 execution discipline — market probability research

## Decision cutoff (not the closing price)

**Default: 15 minutes before scheduled race start** (`race.start_jst - 15min`). An explicit `race.t15_reference = "sales_deadline"` switches to 15 minutes before sales deadline. Both variants require `deadline_jst` and reject a decision if it is not BEFORE the closing deadline.

Each decision needs:
1. Pre-decision race card and timestamp (`card_as_of_jst`), rider stats/lineups.
2. Full **live** 3連単 and 3連複 odds for all available runner combinations (210+35 for 7 riders).
3. **Separate** on-page update timestamps for the two pools (`market_as_of_jst`), captured UTC/JST wall-clock time, source URL and race ID.
4. Both market timestamps and the capture time at or before T-15, with maximum 5-minute staleness at the cutoff.
5. A frozen pre-result decision file with input SHA-256. No overwrite.

If any item fails, **no model decision**. Do NOT take archived final odds and falsely label them T-15. The collection tool explicitly refuses a published final-odds page.

## How to produce one snapshot

Create an *at-the-time* race-card JSON file **before** the race. Example structural record:

```json
{
  "race": {
    "race_id": "verified-source-race-id",
    "track": "verified-track",
    "start_jst": "YYYY-MM-DDTHH:MM:SS+09:00",
    "deadline_jst": "YYYY-MM-DDTHH:MM:SS+09:00",
    "card_as_of_jst": "YYYY-MM-DDTHH:MM:SS+09:00",
    "source_url": "https://keirin.kdreams.jp/...",
    "predicted_line_formation": "1-2-3/4-5/6-7",
    "t15_reference": "race_start"
  },
  "entries": [
    {"car_no": 1, "score": 100.5, "win_rate": 20.2, "top3_rate": 45,
     "age": 31, "style": "逃", "b_count": 4, "nige_count": 5,
     "makuri_count": 2, "sashi_count": 1, "mark_count": 0}
  ]
}
```

`entries` must include *every* rider. Other features can be supplied using matching entries.csv field names; verify race-card freshness and completeness. Do not rely on invented timestamps.

Run the collector **around** the T-15 timestamp using a scheduler with sufficient precision; aim to finish within the 5-minute lookback and before T-15. A cron/action job starting every five minutes does not GUARANTEE this: startup delay and data-provider freshness can cause disqualification. If the site cannot provide a full time-stamped live book, no snapshot is accepted.

```bash
python -m pip install numpy scipy beautifulsoup4 lxml requests
PYTHONPATH=src python research/market_probability_v1/t15_capture.py \
  --race-card path/to/frozen_pre_race_card.json \
  --out path/to/immutable_live_t15_snapshot.json

python research/market_probability_v1/t15_decision.py \
  --snapshot path/to/immutable_live_t15_snapshot.json \
  --out path/to/frozen_t15_paper_decision.json
```

The two scripts have **no betting account integration** and do not execute any financial transaction. `t15_decision.py` uses currently stored final-odds-trained coefficients **provisionally** on the live pool, labels output `PAPER_ONLY_UNVALIDATED_T15_MODEL`, and stores zero real stakes. Candidate sensitivity is shown for a user-configurable odds haircut, not a statistical confidence interval. Data-driven re-estimation and calibrated uncertainty cannot be skipped.

## Current numerical position

- Archived final-odds learning: 16,833 joined races, split 2023–2026H1. This ONLY tests probability correction using **final** odds.
- Archived 2025: log loss market 4.15265 / corrected 4.14722. Final-odds-only max EV >=1.10 heuristic: 1,434 races, 15 winners, 132.95% ROI.
- Archived 2026H1: log loss market 4.19031 / corrected 4.17894. Same heuristic: 373 races, 3 winners, 93.65% ROI.
- **Verified strict T-15 training/test races: 0 currently certified in this project.** Existing one-day timestamped odds records cannot be presumed T-15 without matched race deadline/start and source-update checks.
- Market-corrected inference at T-15 is NOT yet shown to beat the T-15 market, nor to produce positive return.

The real hurdle is to collect MANY prospective independent T-15 decision records, freeze the model **before** them, and compare at the same market snapshot:
- log loss vs T-15 market and T-15 market-only temperature calibrator;
- Brier score, decile calibration, race-clustered confidence intervals, period-level stability;
- gross return based on **actual official settled payouts**, total stake per **frozen** candidate, including void/refund treatment;
- realistic price impact and T-15-to-closing drift, bet-capacity evidence, zero-bet races and drawdown.

**Pass conditions**: improvement in probability scoring *and* sufficiently well-supported positive net return in future records without after-the-fact threshold selection. A single year above 100% gross return is insufficient.

**Honest limitation:** even this new capture adapter's compatibility with the *live* KDreams odds markup still requires a live trial; offline structural tests do not establish website compatibility. 15-minute automated buying requires separate authorized transaction integration and explicit risk controls.

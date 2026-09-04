from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import keirin_market_analysis.prediction_engine as prediction_engine_module
from keirin_market_analysis.prediction_cli import calibrate_policy_2024
from keirin_market_analysis.prediction_engine import EngineContract
from keirin_market_analysis.prediction_features import (
    RelationshipTracker,
    _numeric_pair_columns,
    _pair_row,
    add_race_relative_features,
)


def build_pairwise_rows_day_safe(
    entries: pd.DataFrame,
    results: pd.DataFrame,
    *,
    relationship_tracker: RelationshipTracker | None = None,
    update_relationships: bool = True,
) -> tuple[pd.DataFrame, RelationshipTracker]:
    """Build pair rows without allowing results from one race to leak into another race on the same day."""
    e = add_race_relative_features(entries)
    if "race_date" not in e.columns:
        raise KeyError("race_date is required for leakage-safe relationship history")

    e = e.copy()
    e["race_date"] = pd.to_datetime(e["race_date"], errors="coerce").dt.normalize()
    if e["race_date"].isna().any():
        raise ValueError("race_date contains invalid values")

    results = results.copy()
    results["race_id"] = results["race_id"].astype(str)
    tracker = relationship_tracker or RelationshipTracker()
    result_groups = {str(rid): g for rid, g in results.groupby("race_id", sort=False)}
    numeric_base = _numeric_pair_columns(e)
    rows: list[dict] = []

    race_order = e[["race_id", "race_date"]].drop_duplicates().sort_values(["race_date", "race_id"])

    for _, day_races in race_order.groupby("race_date", sort=True):
        pending_updates: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        for raw_race_id in day_races["race_id"]:
            race_id = str(raw_race_id)
            if race_id not in result_groups:
                continue

            rg = result_groups[race_id]
            order = {
                int(r.car_no): float(r.order_numeric)
                for r in rg[["car_no", "order_numeric"]].itertuples(index=False)
                if pd.notna(r.car_no) and pd.notna(r.order_numeric)
            }
            g = e[e["race_id"].astype(str) == race_id].copy().sort_values("car_no")
            recs = list(g.to_dict("records"))

            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    a, b = recs[i], recs[j]
                    ca, cb = int(a["car_no"]), int(b["car_no"])
                    if ca not in order or cb not in order or order[ca] == order[cb]:
                        continue
                    row = _pair_row(a, b, tracker, numeric_base)
                    row["label_a_beats_b"] = int(order[ca] < order[cb])
                    rows.append(row)

            if update_relationships:
                pending_updates.append((g, rg))

        # Only after every prediction row for this date is frozen may that date enter history.
        if update_relationships:
            for g, rg in pending_updates:
                tracker.update_race(g, rg)

    return pd.DataFrame(rows), tracker


def main() -> int:
    # KeirinPredictionEngine.fit resolves this module-level symbol at runtime.
    prediction_engine_module.build_pairwise_rows = build_pairwise_rows_day_safe

    root = Path(".")
    out_dir = Path("artifacts/prediction_engine_v1_q4_safe")
    _, report = calibrate_policy_2024(root, out_dir, EngineContract())

    q4 = report["q4_validation"]
    gate = report["q4_gate"]
    payload = {
        "selected_policy": report["selected_policy"],
        "q3": report["selected_q3"],
        "q4": q4,
        "gate": gate,
        "leakage_rule": "All races on date D use relationship history only through D-1; date-D results enter history after all date-D pair rows are frozen.",
        "2025_status": "NOT_OPENED",
        "2026_status": "NOT_OPENED",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "q4_safe_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

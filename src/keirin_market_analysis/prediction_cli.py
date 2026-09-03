from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .prediction_engine import EngineContract, KeirinPredictionEngine
from .prediction_models import blend_trifecta
from .prediction_schema import DatasetCatalog, Segment, concat_labels, concat_pre_race, load_payouts
from .prediction_strategy import TicketPolicy, evaluate_decisions, policy_grid, select_tickets, trifecta_payout_map


def _segments(catalog: DatasetCatalog, names: Iterable[str]) -> tuple[Segment, ...]:
    wanted = set(names)
    found = tuple(s for s in catalog.segments if s.name in wanted)
    if {s.name for s in found} != wanted:
        missing = wanted - {s.name for s in found}
        raise KeyError(f"dataset segments not found: {sorted(missing)}")
    return found


def _concat_payouts(segments: Iterable[Segment], *, unlock_sealed: bool = False) -> pd.DataFrame:
    frames = [load_payouts(s, unlock_sealed=unlock_sealed) for s in segments]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _filter_labels(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    ids = set(races["race_id"].astype(str))
    out = results.copy()
    out["race_id"] = out["race_id"].astype(str)
    return out[out["race_id"].isin(ids)].copy()


def _fit_on_segments(root: Path, segments: tuple[Segment, ...], contract: EngineContract) -> KeirinPredictionEngine:
    races, entries = concat_pre_race(segments, seven_rider_only=contract.seven_rider_only)
    results = _filter_labels(concat_labels(segments), races)
    return KeirinPredictionEngine(contract).fit(races, entries, results)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return str(obj)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _decision_from_pack(pack: dict, policy: TicketPolicy) -> dict:
    d = select_tickets(
        pack["trifecta"],
        pack["statistical_trifecta"],
        pack["similarity_trifecta"],
        policy,
    )
    meta = pack["meta"]
    return {
        "race_id": str(pack["race_id"]),
        "race_date": meta.get("race_date"),
        "track": meta.get("track"),
        "race_no": meta.get("race_no"),
        "race_type": meta.get("race_type"),
        "buy": d["buy"],
        "tickets": d["tickets"],
        "stake_yen": d["stake_yen"],
        "skip_reasons": d["skip_reasons"],
        "metrics": d["metrics"],
    }


def _prediction_packs(engine: KeirinPredictionEngine, races: pd.DataFrame, entries: pd.DataFrame):
    by_race = {str(rid): g for rid, g in entries.groupby("race_id", sort=False)}
    ordered = engine._sort_races(races)
    for r in ordered.itertuples(index=False):
        rid = str(getattr(r, "race_id"))
        eg = by_race.get(rid)
        if eg is None or len(eg) != 7:
            continue
        race_row = races[races["race_id"].astype(str) == rid].head(1)
        yield engine.predict_race(race_row, eg)


def _score_policy_on_packs(packs: list[dict], payouts: pd.DataFrame, policy: TicketPolicy) -> dict:
    decisions = [_decision_from_pack(pack, policy) for pack in packs]
    _, summary = evaluate_decisions(decisions, payouts)
    return summary


def calibrate_policy_2024(root: Path, out_dir: Path, contract: EngineContract) -> tuple[TicketPolicy, dict]:
    """Use 2024H1 for model fit, then Q3/Q4 separately for a robust price-blind policy choice."""
    catalog = DatasetCatalog(root)
    h1 = _segments(catalog, ("2024_q1", "2024_q2"))
    q3 = _segments(catalog, ("2024_q3",))
    q4 = _segments(catalog, ("2024_q4",))
    engine = _fit_on_segments(root, h1, contract)

    q3_races, q3_entries = concat_pre_race(q3, seven_rider_only=True)
    q4_races, q4_entries = concat_pre_race(q4, seven_rider_only=True)
    q3_payouts = _concat_payouts(q3)
    q4_payouts = _concat_payouts(q4)

    # The model/history snapshot is frozen at 2024-06-30 for both Q3 and Q4.
    # Q3/Q4 outcomes are never fed back into the provisional predictor.
    q3_packs = list(_prediction_packs(engine, q3_races, q3_entries))
    q4_packs = list(_prediction_packs(engine, q4_races, q4_entries))

    records: list[dict] = []
    best: tuple[float, int, int, TicketPolicy, dict, dict] | None = None
    for policy_id, policy in enumerate(policy_grid()):
        s3 = _score_policy_on_packs(q3_packs, q3_payouts, policy)
        s4 = _score_policy_on_packs(q4_packs, q4_payouts, policy)
        min_sample = min(int(s3["bought_races"]), int(s4["bought_races"]))
        worst_roi = min(float(s3["roi"]), float(s4["roi"]))
        worst_dd = max(int(s3["max_drawdown_yen"]), int(s4["max_drawdown_yen"]))
        eligible = min_sample >= 60
        record = {
            "policy_id": policy_id,
            "policy": asdict(policy),
            "eligible": eligible,
            "worst_quarter_roi": worst_roi,
            "minimum_quarter_bought_races": min_sample,
            "worst_drawdown_yen": worst_dd,
            "q3": s3,
            "q4": s4,
        }
        records.append(record)
        if eligible:
            key = (worst_roi, min_sample, -worst_dd)
            if best is None or key > best[:3]:
                best = (worst_roi, min_sample, -worst_dd, policy, s3, s4)

    if best is None:
        raise RuntimeError("No 2024 policy candidate met the pre-declared minimum sample requirement")
    selected = best[3]
    report = {
        "method": "fit model on 2024H1; select policy by worst ROI across 2024Q3 and Q4; minimum 60 bought races each",
        "contract": asdict(contract),
        "selected_policy": asdict(selected),
        "selected_q3": best[4],
        "selected_q4": best[5],
        "candidate_count": len(records),
        "important": "2025 was not used for policy selection; final odds were not used as prediction or ticket features.",
    }
    _write_json(out_dir / "2024_policy_candidates.json", records)
    _write_json(out_dir / "2024_selected_policy.json", report)
    return selected, report


def destruction_test_2025(root: Path, out_dir: Path, policy: TicketPolicy, contract: EngineContract) -> dict:
    catalog = DatasetCatalog(root)
    development = _segments(catalog, ("2024_q1", "2024_q2", "2024_q3", "2024_q4"))
    h1 = _segments(catalog, ("2025_q1", "2025_q2"))
    h2 = _segments(catalog, ("2025_q3", "2025_q4"))
    engine = _fit_on_segments(root, development, contract)

    h1_races, h1_entries = concat_pre_race(h1, seven_rider_only=True)
    h1_packs = list(_prediction_packs(engine, h1_races, h1_entries))
    h1_decisions = [_decision_from_pack(pack, policy) for pack in h1_packs]
    h1_payouts = _concat_payouts(h1)
    h1_eval, h1_summary = evaluate_decisions(h1_decisions, h1_payouts)

    # Historical memory is refreshed only at the half-year boundary. Coefficients and policy stay frozen.
    h1_results = _filter_labels(concat_labels(h1), h1_races)
    engine.refresh_history(h1_races, h1_entries, h1_results)

    h2_races, h2_entries = concat_pre_race(h2, seven_rider_only=True)
    h2_packs = list(_prediction_packs(engine, h2_races, h2_entries))
    h2_decisions = [_decision_from_pack(pack, policy) for pack in h2_packs]
    h2_payouts = _concat_payouts(h2)
    h2_eval, h2_summary = evaluate_decisions(h2_decisions, h2_payouts)

    all_decisions = h1_decisions + h2_decisions
    all_payouts = pd.concat([h1_payouts, h2_payouts], ignore_index=True)
    all_eval, all_summary = evaluate_decisions(all_decisions, all_payouts)

    out_dir.mkdir(parents=True, exist_ok=True)
    h1_eval.to_csv(out_dir / "2025_h1_evaluation.csv", index=False)
    h2_eval.to_csv(out_dir / "2025_h2_evaluation.csv", index=False)
    all_eval.to_csv(out_dir / "2025_full_evaluation.csv", index=False)
    _write_json(out_dir / "2025_h1_decisions.json", h1_decisions)
    _write_json(out_dir / "2025_h2_decisions.json", h2_decisions)

    report = {
        "contract": asdict(contract),
        "policy": asdict(policy),
        "2025_h1": h1_summary,
        "2025_h2": h2_summary,
        "2025_full": all_summary,
        "history_refresh": "2025H1 outcomes enter relationship/similarity memory only after every H1 prediction is already frozen; model coefficients are not refit.",
        "sealed_2026_status": "NOT_OPENED",
        "market_rule": "historical final odds were not loaded by this command",
    }
    _write_json(out_dir / "2025_destruction_test_summary.json", report)
    _write_json(out_dir / "engine_manifest_after_2025_h1_refresh.json", engine.manifest(policy))
    return report


def run_develop_and_test(root: Path, out_dir: Path) -> dict:
    contract = EngineContract()
    policy, development_report = calibrate_policy_2024(root, out_dir, contract)
    test_report = destruction_test_2025(root, out_dir, policy, contract)
    combined = {
        "development": development_report,
        "destruction_test": test_report,
        "sealed_2026_status": "NOT_OPENED",
    }
    _write_json(out_dir / "prediction_engine_v1_report.json", combined)
    return combined


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-safe keirin Prediction Engine v1")
    p.add_argument("command", choices=("develop-and-test", "calibrate-2024", "destruction-test-2025", "manifest"))
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="artifacts/prediction_engine_v1")
    p.add_argument("--policy-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out)
    contract = EngineContract()

    if args.command == "manifest":
        payload = {
            "contract": asdict(contract),
            "sealed_2026_status": "LOCKED",
            "available_commands": ["calibrate-2024", "destruction-test-2025", "develop-and-test"],
        }
    elif args.command == "calibrate-2024":
        policy, report = calibrate_policy_2024(root, out, contract)
        payload = report
    elif args.command == "destruction-test-2025":
        if not args.policy_json:
            raise SystemExit("--policy-json is required for destruction-test-2025")
        spec = json.loads(Path(args.policy_json).read_text(encoding="utf-8"))
        raw = spec.get("selected_policy", spec)
        policy = TicketPolicy(**raw)
        payload = destruction_test_2025(root, out, policy, contract)
    else:
        payload = run_develop_and_test(root, out)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

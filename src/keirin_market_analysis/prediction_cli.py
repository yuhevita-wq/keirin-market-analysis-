from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .prediction_engine import EngineContract, KeirinPredictionEngine
from .prediction_models import distribution_metrics
from .prediction_schema import DatasetCatalog, Segment, concat_labels, concat_pre_race, load_payouts
from .prediction_strategy import TicketPolicy, evaluate_decisions, model_tv_distance, policy_grid, select_tickets


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _decision_from_pack(pack: dict, policy: TicketPolicy) -> dict:
    d = select_tickets(pack["trifecta"], pack["statistical_trifecta"], pack["similarity_trifecta"], policy)
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


def _compact_pack(pack: dict) -> dict:
    tri = pack["trifecta"].sort_values("probability", ascending=False)
    meta = pack["meta"]
    metrics = distribution_metrics(tri)
    metrics["model_tv_distance"] = model_tv_distance(pack["statistical_trifecta"], pack["similarity_trifecta"])
    return {
        "race_id": str(pack["race_id"]),
        "race_date": meta.get("race_date"),
        "track": meta.get("track"),
        "race_no": meta.get("race_no"),
        "race_type": meta.get("race_type"),
        "combos": tri["combo"].astype(str).tolist(),
        "probabilities": tri["probability"].astype(float).tolist(),
        "metrics": metrics,
    }


def _decision_from_compact(pack: dict, policy: TicketPolicy) -> dict:
    m = pack["metrics"]
    reasons: list[str] = []
    if m["normalized_entropy"] > policy.max_normalized_entropy:
        reasons.append("entropy")
    if m["top10_mass"] < policy.min_top10_mass:
        reasons.append("diffuse_top10")
    if m["model_tv_distance"] > policy.max_model_tv_distance:
        reasons.append("model_disagreement")

    tickets: list[str] = []
    mass = 0.0
    if not reasons:
        for combo, prob in zip(pack["combos"], pack["probabilities"]):
            if len(tickets) >= policy.max_tickets:
                break
            tickets.append(combo)
            mass += float(prob)
            if mass >= policy.cumulative_mass:
                break
        if not tickets:
            reasons.append("no_ticket")

    buy = not reasons
    return {
        "race_id": pack["race_id"],
        "race_date": pack.get("race_date"),
        "track": pack.get("track"),
        "race_no": pack.get("race_no"),
        "race_type": pack.get("race_type"),
        "buy": buy,
        "tickets": tickets if buy else [],
        "stake_yen": len(tickets) * policy.unit_yen if buy else 0,
        "covered_probability_mass": mass if buy else 0.0,
        "skip_reasons": reasons,
        "metrics": m,
    }


def _prediction_packs(engine: KeirinPredictionEngine, races: pd.DataFrame, entries: pd.DataFrame):
    by_race = {str(rid): g for rid, g in entries.groupby("race_id", sort=False)}
    for r in engine._sort_races(races).itertuples(index=False):
        rid = str(getattr(r, "race_id"))
        eg = by_race.get(rid)
        if eg is None or len(eg) != 7:
            continue
        race_row = races[races["race_id"].astype(str) == rid].head(1)
        yield engine.predict_race(race_row, eg)


def _monthly_summaries(evaluated: pd.DataFrame) -> list[dict]:
    if evaluated.empty:
        return []
    df = evaluated.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["month"] = df["race_date"].dt.month
    out: list[dict] = []
    for month, g in df.groupby("month", dropna=True):
        bought = g[g["buy"]]
        stake = int(bought["stake_yen"].sum()) if not bought.empty else 0
        ret = int(bought["return_yen"].sum()) if not bought.empty else 0
        out.append({
            "month": int(month),
            "evaluated_races": int(len(g)),
            "bought_races": int(len(bought)),
            "stake_yen": stake,
            "return_yen": ret,
            "roi": float(ret / stake) if stake else 0.0,
        })
    return sorted(out, key=lambda x: x["month"])


def _score_policy_on_compact(packs: list[dict], payouts: pd.DataFrame, policy: TicketPolicy) -> dict:
    decisions = [_decision_from_compact(pack, policy) for pack in packs]
    evaluated, summary = evaluate_decisions(decisions, payouts)
    summary["monthly"] = _monthly_summaries(evaluated)
    monthly_with_sample = [m for m in summary["monthly"] if m["bought_races"] >= 15]
    summary["minimum_monthly_roi_15plus"] = (
        min(m["roi"] for m in monthly_with_sample) if monthly_with_sample else 0.0
    )
    summary["months_with_15plus_buys"] = len(monthly_with_sample)
    return summary


def calibrate_policy_2024(root: Path, out_dir: Path, contract: EngineContract) -> tuple[TicketPolicy, dict]:
    """Q3 selects exactly one policy. Q4 is pass/fail only and may not select a replacement."""
    catalog = DatasetCatalog(root)
    h1 = _segments(catalog, ("2024_q1", "2024_q2"))
    q3 = _segments(catalog, ("2024_q3",))
    q4 = _segments(catalog, ("2024_q4",))
    engine = _fit_on_segments(root, h1, contract)

    q3_races, q3_entries = concat_pre_race(q3, seven_rider_only=True)
    q3_payouts = _concat_payouts(q3)
    q3_packs = [_compact_pack(pack) for pack in _prediction_packs(engine, q3_races, q3_entries)]

    candidates: list[dict] = []
    best: tuple[float, float, int, int, TicketPolicy, dict] | None = None
    for policy_id, policy in enumerate(policy_grid()):
        score = _score_policy_on_compact(q3_packs, q3_payouts, policy)
        eligible = int(score["bought_races"]) >= 60 and int(score["months_with_15plus_buys"]) == 3
        record = {"policy_id": policy_id, "policy": asdict(policy), "eligible": eligible, "q3": score}
        candidates.append(record)
        if eligible:
            key = (
                float(score["minimum_monthly_roi_15plus"]),
                float(score["roi"]),
                int(score["bought_races"]),
                -int(score["max_drawdown_yen"]),
            )
            if best is None or key > best[:4]:
                best = (*key, policy, score)
    if best is None:
        raise RuntimeError("No Q3 policy candidate met the pre-declared sample/stability requirement")

    selected = best[4]
    selected_q3 = best[5]

    # Q4 is opened only after the Q3 winner is frozen. No alternative policy is scored on Q4.
    q4_races, q4_entries = concat_pre_race(q4, seven_rider_only=True)
    q4_payouts = _concat_payouts(q4)
    q4_packs = [_compact_pack(pack) for pack in _prediction_packs(engine, q4_races, q4_entries)]
    q4_score = _score_policy_on_compact(q4_packs, q4_payouts, selected)
    q4_months_ok = sum(1 for m in q4_score["monthly"] if m["bought_races"] >= 15 and m["roi"] >= 0.85)
    validation_passed = (
        int(q4_score["bought_races"]) >= 60
        and float(q4_score["roi"]) >= 1.00
        and q4_months_ok >= 2
    )

    report = {
        "method": "fit model on 2024H1; choose one policy on Q3 only; Q4 is an irreversible validation gate",
        "contract": asdict(contract),
        "selected_policy": asdict(selected),
        "selected_q3": selected_q3,
        "q4_validation": q4_score,
        "q4_gate": {
            "minimum_bought_races": 60,
            "minimum_full_quarter_roi": 1.00,
            "minimum_months_with_15plus_buys_and_roi_0_85": 2,
            "observed_qualifying_months": q4_months_ok,
            "passed": validation_passed,
        },
        "candidate_count_q3_only": len(candidates),
        "important": "No Q4 result can select a replacement policy. 2025 must remain unopened if this gate fails.",
    }
    _write_json(out_dir / "2024_q3_policy_candidates.json", candidates)
    _write_json(out_dir / "2024_selected_policy_and_q4_gate.json", report)
    return selected, report


def destruction_test_2025(root: Path, out_dir: Path, policy: TicketPolicy, contract: EngineContract) -> dict:
    catalog = DatasetCatalog(root)
    development = _segments(catalog, ("2024_q1", "2024_q2", "2024_q3", "2024_q4"))
    h1 = _segments(catalog, ("2025_q1", "2025_q2"))
    h2 = _segments(catalog, ("2025_q3", "2025_q4"))
    engine = _fit_on_segments(root, development, contract)

    h1_races, h1_entries = concat_pre_race(h1, seven_rider_only=True)
    h1_decisions = [_decision_from_pack(pack, policy) for pack in _prediction_packs(engine, h1_races, h1_entries)]
    h1_payouts = _concat_payouts(h1)
    h1_eval, h1_summary = evaluate_decisions(h1_decisions, h1_payouts)
    h1_summary["monthly"] = _monthly_summaries(h1_eval)

    h1_results = _filter_labels(concat_labels(h1), h1_races)
    engine.refresh_history(h1_races, h1_entries, h1_results)

    h2_races, h2_entries = concat_pre_race(h2, seven_rider_only=True)
    h2_decisions = [_decision_from_pack(pack, policy) for pack in _prediction_packs(engine, h2_races, h2_entries)]
    h2_payouts = _concat_payouts(h2)
    h2_eval, h2_summary = evaluate_decisions(h2_decisions, h2_payouts)
    h2_summary["monthly"] = _monthly_summaries(h2_eval)

    all_decisions = h1_decisions + h2_decisions
    all_payouts = pd.concat([h1_payouts, h2_payouts], ignore_index=True)
    all_eval, all_summary = evaluate_decisions(all_decisions, all_payouts)
    all_summary["monthly"] = _monthly_summaries(all_eval)

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
        "history_refresh": "2025H1 enters relationship/similarity memory only after all H1 predictions are frozen; coefficients and policy are unchanged.",
        "sealed_2026_status": "NOT_OPENED",
        "market_rule": "historical final odds were not loaded by this command",
    }
    _write_json(out_dir / "2025_destruction_test_summary.json", report)
    _write_json(out_dir / "engine_manifest_after_2025_h1_refresh.json", engine.manifest(policy))
    return report


def run_develop_and_test(root: Path, out_dir: Path) -> dict:
    contract = EngineContract()
    policy, development_report = calibrate_policy_2024(root, out_dir, contract)
    if not bool(development_report["q4_gate"]["passed"]):
        combined = {
            "development": development_report,
            "destruction_test": "NOT_RUN_BECAUSE_Q4_GATE_FAILED",
            "sealed_2026_status": "NOT_OPENED",
        }
        _write_json(out_dir / "prediction_engine_v1_report.json", combined)
        return combined
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
        _, payload = calibrate_policy_2024(root, out, contract)
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

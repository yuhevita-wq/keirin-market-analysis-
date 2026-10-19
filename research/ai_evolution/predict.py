"""GEN-001: strict live T-15 paper-only inference from an immutable checkpoint."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "market_probability_v1"))
from t15_decision import calculate, FEATURES

MODEL = Path(__file__).parent / "generations" / "g001" / "model.json"

def predict(snapshot: dict, minimum_gross_ev: float = 1.0) -> dict:
    """Return every qualifying trifecta outcome (not an arbitrary four-ticket limit)."""
    if not 1 <= minimum_gross_ev <= 5:
        raise ValueError("minimum_gross_ev must be between 1 and 5")
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    if model["features"] != list(FEATURES):
        raise ValueError("model schema mismatch")
    result = calculate(snapshot, model, min_edge=minimum_gross_ev - 1,
                       odds_haircut=0, max_candidates=210)
    result["generation_id"] = "GEN-001"
    result["model_train_through"] = "2025-12-31"
    result["screened_trifecta_outcomes"] = result["market_sample_size"]
    result["all_qualifying_trifecta_candidates"] = len(result["paper_decisions"])
    result["selection_minimum_estimated_ev"] = minimum_gross_ev
    result["trading_status"] = "PAPER_ONLY_UNVALIDATED_T15_PROFITABILITY"
    result["research_warning"] = "Final-odds trained model; EV may not survive T-15 prices or market movement"
    return result

def main():
    parser = argparse.ArgumentParser(description="GEN-001 simulated 3連単 screen; no purchases")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-gross-ev", type=float, default=1.0)
    args = parser.parse_args()
    snap = json.loads(args.snapshot.read_text(encoding="utf-8"))
    out = predict(snap, args.min_gross_ev)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"generation":"GEN-001", "race_id":out["race_id"],
        "screened":out["screened_trifecta_outcomes"],
        "candidates":out["all_qualifying_trifecta_candidates"],
        "real_stakes":0,"output":str(args.out)},ensure_ascii=False))

if __name__ == "__main__":
    main()

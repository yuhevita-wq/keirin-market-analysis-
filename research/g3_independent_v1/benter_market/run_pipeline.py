#!/usr/bin/env python3
"""Single command G3 independent model -> market odds audit -> Benter bridge.

Requires REAL pre-race card and REAL observed market price snapshot as JSON.
No scraping or post-result backfilling here. All frozen G3 models read-only.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from predict_g3_first import independent_first,DEFAULT_MODEL
from evaluate import evaluate_race

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path,
                   help="JSON with raw racer card and separately timestamped 7-market snapshot")
    p.add_argument("--first",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--calibration",type=Path,default=None)
    p.add_argument("--mode",choices=("research","live"),default="research")
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    request=json.loads(args.input.read_text(encoding="utf-8"))
    if set(request)!={"racecard","market_snapshot"}:
        raise ValueError("Only raw racecard and observed market_snapshot are permitted")
    firstmodel=json.loads(args.first.read_text(encoding="utf-8"))
    qcard=independent_first(request["racecard"],firstmodel)
    calibration=json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    report=evaluate_race(qcard,request["market_snapshot"],calibration,mode=args.mode)
    report["source_card_input_sha256"]=hashlib.sha256(args.input.read_bytes()).hexdigest()
    report["G3_original_first_win"] = qcard["first_win"]
    report["G3_independent_first_feature_provenance"]={
        "source_url":qcard["source_url"],
        "captured_at_utc":qcard["snapshot_captured_at_utc"],
        "source_model":qcard["source_model"]
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_BENTER_MARKET_PIPELINE",report["status"],report["race_id"],flush=True)
    return report

if __name__=="__main__":
    main()

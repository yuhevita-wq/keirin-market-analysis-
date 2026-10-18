#!/usr/bin/env python3
"""New-chat entrypoint for the G3-only Benter market bridge.

This does not retrieve odds or run today's races automatically.
Provide model q + predicted lines, complete observed odds books, and phase.
Until separately fitted market weights exist, output only diagnostics.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from engine import SUPPORTED, analyze,load_json

HERE=Path(__file__).resolve().parent
G3ROOT=HERE.parent

def main():
    p=argparse.ArgumentParser(description="G3 frozen first/second/third + Benter-stage odds evaluation")
    p.add_argument("--card",required=True,type=Path,help="G3 independent prior-race first q and published lines; NO results")
    p.add_argument("--odds",required=True,type=Path,help="timestamped odds JSON with snapshot and markets")
    p.add_argument("--second",type=Path,default=G3ROOT/"models/g3_second_conditional_v1.json")
    p.add_argument("--third",type=Path,default=G3ROOT/"models/g3_third_conditional_v1.json")
    p.add_argument("--calibrations",type=Path,help="optional JSON mapping wager type to train_fusion.py report")
    p.add_argument("--historical-final",action="store_true",
                   help="research only: permit old finalized historical quotes, not a 15min prediction")
    p.add_argument("--require-all-seven",action="store_true",
                   help="refuse to call a full-market audit complete until all 7 quote pools and calibrations are validated")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    odds=load_json(args.odds)
    if set(odds)!={"snapshot","markets"}:
        raise ValueError("Odds input must contain exactly snapshot and markets")
    cal=load_json(args.calibrations) if args.calibrations else None
    required=set(SUPPORTED)
    if args.require_all_seven:
        if set(odds["markets"])!=required:
            raise ValueError("All-seven request incomplete: missing odds types "+str(sorted(required-set(odds["markets"]))))
        if not cal or required-set(cal):
            raise ValueError("All-seven calibrated market data unavailable. ワイド needs nonexclusive calibration.")
    report=analyze(load_json(args.card),load_json(args.second),load_json(args.third),
                   odds["markets"],odds["snapshot"],cal,args.historical_final)
    if args.require_all_seven and report["calibration_missing_markets"]:
        raise ValueError("Full-market calibration not established: "+str(report["calibration_missing_markets"]))
    data=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(data,encoding="utf-8")
    print(data)

if __name__=="__main__":
    main()

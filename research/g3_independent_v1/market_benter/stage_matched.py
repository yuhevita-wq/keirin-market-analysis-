#!/usr/bin/env python3
"""Strictly join G3 prioryear independent probabilities, complete odds, actual result.

Input is a canonical JSONL of MATCHED G3 races. All original archive
identifiers and quote phases must be audited by the caller. This tool rejects
incomplete market odds and duplicate race IDs: it does not invent outcomes,
market prices, snapshots, or other wager types.

Output is one separate set of complete Benter second-stage fitting records per
EXCLUSIVE market, plus a WIN proxy derived from the exhaustive 3連単 book.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path
from engine import (EXCLUSIVE,aggregate_probability,frame_mapping,
                    outcome_selection,validate_quotes)
from train_fusion import SINGLE_WIN_PROXY,first_proxy_from_3rentan
from predict_g3_ordered_triples import all_ordered_triples
G3DIR=Path(__file__).resolve().parent.parent

def convert(row,second_model,third_model):
    if row.get("scope")!="G3 S級":raise ValueError("Only G3 S級 historical records")
    if row.get("fundamental_provenance")!="G3_only_prior_period_independent":
        raise ValueError("Every model probability must be fixed before the historical G3 race")
    if row.get("price_phase")!="final":
        raise ValueError("Canonical history must explicitly use finalized historical quotes")
    for k in ("race_id","race_date","card","finish","markets"):
        if k not in row:raise ValueError("Missing required join field "+k)
    finish=row["finish"]
    if set(finish)!={"first","second","third"}:
        raise ValueError("Require exactly the unique actual ordered podium")
    triple=tuple(int(finish[k]) for k in ("first","second","third"))
    if len(set(triple))!=3:raise ValueError("Finishing order not unique")
    if set(row["card"])&{"first","second","third","result","results","odds"}:
        raise ValueError("Historical winning result cannot be a predictor")
    predictions=all_ordered_triples(row["card"],second_model,third_model)
    first_probs={str(x["car_no"]):x["probability"] for x in predictions["first_marginal"]}
    cars={int(x) for x in first_probs}
    if not set(triple).issubset(cars):
        raise ValueError("Finishers outside starters")
    frames=frame_mapping(row["card"],cars)
    out={}
    for market,odds in row["markets"].items():
        if market not in EXCLUSIVE:
            if market=="ワイド":continue
            raise ValueError("Unknown or not-yet-supported historic market "+market)
        f=aggregate_probability(predictions["ordered_triples"],market,frames)
        prices=validate_quotes(odds,f,market)
        actual=outcome_selection(triple,market,frames)
        record={
            "scope":"G3 S級",
            "market":market,
            "race_id":str(row["race_id"]),
            "race_date":str(row["race_date"]),
            "fundamental_provenance":"G3_only_prior_period_independent",
            "price_phase":"final",
            "fundamental_probabilities":f,
            "odds":prices,
            "actual_winning_selection":actual,
            "original_asof_card_certified":row.get("original_asof_card_certified") is True
        }
        out[market]=record
        if market=="3連単":
            proxy={**record,
                   "market":SINGLE_WIN_PROXY,
                   "fundamental_probabilities":first_probs,
                   "actual_winning_selection":str(triple[0]),
                   "entrant_cars":sorted(cars)}
            # Verify win-proxy marginal is derived from full 3連単, not a fabricated single-win pool.
            first_proxy_from_3rentan(prices,cars)
            out[SINGLE_WIN_PROXY]=proxy
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True,help="Matched G3 race JSONL (odds + historical first forecast + podium)")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--second",type=Path,default=G3DIR/"models/g3_second_conditional_v1.json")
    p.add_argument("--third",type=Path,default=G3DIR/"models/g3_third_conditional_v1.json")
    args=p.parse_args()
    second_model=json.loads(args.second.read_text(encoding="utf-8"))
    third_model=json.loads(args.third.read_text(encoding="utf-8"))
    seen=set();lines=collections.defaultdict(list);n=0
    for line in args.input.open(encoding="utf-8"):
        if not line.strip():continue
        n+=1
        record=json.loads(line)
        rid=str(record.get("race_id",""))
        if not rid or rid in seen:raise ValueError("Missing or duplicated source G3 race id")
        seen.add(rid)
        rows=convert(record,second_model,third_model)
        if not rows:raise ValueError("No usable exhaustive G3 odds for race "+rid)
        for name,data in rows.items():
            lines[name].append(data)
    if not n:raise ValueError("Empty G3 historical source")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={
        "status":"G3_HISTORICAL_MARKET_RECORDS_STAGED_NOT_CALIBRATED",
        "input_races":n,
        "source_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "market_records":{},
        "missing_markets":sorted(set(EXCLUSIVE)-set(lines)),
        "no_invented_odds_or_results":True,
        "only_G3_prior_period_first_probs":True,
        "historical_quote_phase":"final",
    }
    for name,rr in sorted(lines.items()):
        path=args.output_dir/(name+".jsonl")
        with path.open("w",encoding="utf-8") as o:
            for r in rr:o.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
        manifest["market_records"][name]={
            "races":len(rr),"path":path.name,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()
        }
    (args.output_dir/"manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

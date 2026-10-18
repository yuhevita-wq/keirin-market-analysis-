#!/usr/bin/env python3
"""Join G3 earlier-year independent first q and observed 2車単 price snapshots.

No missing odds can be inferred from the other six markets, and no missing
horse (car) winner is inferred from public odds. Only authenticated complete
market quotes can enter Benter second-stage estimation.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
from combination import public_first_from_complete_exacta

HERE=Path(__file__).resolve().parent
QFILE=HERE.parent/"data/g3_prior_year_first_q_2023_2026h1.jsonl.gz"

def load_jsonl(p):
    if p.suffix==".gz":
        with gzip.open(p,"rt",encoding="utf-8") as f:
            yield from (json.loads(x) for x in f if x.strip())
    else:
        with p.open(encoding="utf-8") as f:
            yield from (json.loads(x) for x in f if x.strip())

def join(g3_q_records,odds_records):
    q={}
    for r in g3_q_records:
        if r.get("first_win_training_window")!="prior_year_only_G3" or r.get("meeting_grade")!="G3":
            raise ValueError("Only verified G3 past-year independent f may enter market fit")
        if r["race_id"] in q:raise ValueError("Duplicate fundamental G3 race id")
        q[r["race_id"]]=r
    market={}
    for x in odds_records:
        if x["race_id"] in market:
            raise ValueError("Ambiguous repeated market snapshots; deduplicate by a documented policy before joining")
        market[x["race_id"]]=x
    joined=[];reasons={"no_matching_exacta_snapshot":0,"matching_complete":0}
    for rid,f in q.items():
        x=market.get(rid)
        if x is None:
            reasons["no_matching_exacta_snapshot"]+=1
            continue
        if x.get("odds_phase") not in ("final","T-15"):
            raise ValueError("No unverified/mixed odds phase")
        if not x.get("source_url") or not x.get("captured_at_utc"):
            raise ValueError("Historical market must cite observed source/time")
        if x.get("market_type")!="2車単":
            raise ValueError("The Benter G3 first market proxy requires observed 2車単 quotes")
        qdict={int(p["car_no"]):float(p["probability"]) for p in f["first_win"]}
        public_first_from_complete_exacta(qdict,x["exacta_odds"])
        item={
            "race_id":rid,"race_date":f["race_date"],
            "race_type":f["race_type"],"meeting_grade":"G3",
            "first_win_training_window":"prior_year_only_G3",
            "first_win":f["first_win"],
            "actual_first":f["actual_first"], # label only, not passed to live inference
            "exacta_odds":x["exacta_odds"],
            "odds_phase":x["odds_phase"],
            "source_url":x["source_url"],
            "captured_at_utc":x["captured_at_utc"],
        }
        joined.append(item)
        reasons["matching_complete"]+=1
    joined.sort(key=lambda v:(v["race_date"],v["race_id"]))
    return joined,{"eligible_g3_fundamental_races":len(q),
                   "historical_exacta_snapshot_races":len(market),
                   "matched_full_2車単_G3_races":len(joined),
                   "count_by_phase":{phase:sum(x["odds_phase"]==phase for x in joined)
                                     for phase in ("final","T-15")},
                   "count_by_year":{str(y):sum(x["race_date"].startswith(str(y)) for x in joined)
                                    for y in (2023,2024,2025,2026)},
                   "unmatched":reasons,
                   "note":"G3 historical labels occur only in calibration dataset; inference never ingests observed finish."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--independent",type=Path,default=QFILE)
    ap.add_argument("--market",required=True,type=Path,
                    help="Actual complete G3 2車単 books JSONL with recorded original source/phase")
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    if not args.independent.exists():
        raise RuntimeError("G3 prior-year first probability archive must be generated first")
    result,audit=join(load_jsonl(args.independent),load_jsonl(args.market))
    if not result:
        raise RuntimeError("No G3 races join to a full real 2車単 historical market")
    payload="".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(payload,encoding="utf-8")
    audit["odds_input_sha256"]=hashlib.sha256(args.market.read_bytes()).hexdigest()
    audit["joined_history_sha256"]=hashlib.sha256(payload.encode("utf-8")).hexdigest()
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_BENTER_HISTORY_JOIN",json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=="__main__":main()

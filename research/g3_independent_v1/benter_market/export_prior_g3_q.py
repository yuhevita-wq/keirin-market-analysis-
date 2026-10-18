#!/usr/bin/env python3
"""Regenerate prior-year-only G3 fundamental 1st predictions as permanent market-fit input.

Only G3 2023..2026H1 qualify. 2022 in-sample training-reference q EXCLUDED.
Each record carries historical actual first exclusively as a label.
NO odds are inserted: a separate verified 2車単 book is required for Benter
stage-two fitting. Enforce once-per-race and freeze SHA256 audit.
"""
from __future__ import annotations
import gzip,hashlib,json,collections
from pathlib import Path
from train_g3_first import load,OUT
from train_g3_second import first_by_calendar

DEST=Path(__file__).resolve().parent/"data"
OUTPUT=DEST/"g3_prior_year_first_q_2023_2026h1.jsonl.gz"
AUDIT=OUT/"g3_prior_year_first_market_input_audit.json"

def main():
    records,source=load()
    qs,fits=first_by_calendar(records)
    selected=[r for r in records if "2023-01-01"<=r["date"]<"2026-07-01"]
    if not selected:
        raise ValueError("No G3 independent first q available for later years")
    payload=[]
    yearly=collections.Counter()
    for r in selected:
        i=r["id"];cars=r["cars"];q=qs[i]
        if len(q)!=len(cars):
            raise ValueError("G3 winner distribution field mismatch")
        record={
            "race_id":i,"race_date":r["date"],
            "meeting_grade":"G3","race_type":r["rtype"],
            "first_win_training_window":"prior_year_only_G3",
            "actual_first":cars[r["winner"]],
            "first_win":[{"car_no":car,"probability":float(prob)}
                         for car,prob in zip(cars,q)],
            "predicted_line_formation":[list(line) for line in r["line"]],
            "odds_join_status":"NOT_JOINED",
            "odds_phase":None,
            "exacta_odds":None
        }
        yearly[r["date"][:4]]+=1
        payload.append(record)
    if len({r["race_id"] for r in payload})!=len(payload):
        raise ValueError("Source race collision")
    data=("\n".join(json.dumps(z,ensure_ascii=False,separators=(",",":"))
                       for z in payload)+"\n").encode("utf-8")
    DEST.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open("wb") as file:
        # Fixed gzip timestamp for reproducibility, no volatile filename metadata.
        with gzip.GzipFile(filename="",mode="wb",fileobj=file,mtime=0,compresslevel=9) as gz:
            gz.write(data)
    audit={
        "status":"G3_PRIOR_YEAR_Q_READY_WAITING_FOR_AUTHENTIC_2車単_ODDS",
        "scope":"G3 S級 only",
        "records":len(payload),"year_counts":dict(sorted(yearly.items())),
        "g3_first_training_provenance":fits,
        "2022_training_reference_excluded":True,
        "original_race_ids_sha256":source["source_race_ids_sha256"],
        "uncompressed_jsonl_sha256":hashlib.sha256(data).hexdigest(),
        "gzip_archive_sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "gzip_archive_bytes":OUTPUT.stat().st_size,
        "archive_path":"research/g3_independent_v1/data/g3_prior_year_first_q_2023_2026h1.jsonl.gz",
        "contains_historical_result_label_not_a_live_prediction_input":True,
        "odds_present":False,
        "missing_inputs":"Authentic G3 2車単 complete market quotes and true odds snapshot time",
        "source_snaps_predeadline_certified":False
    }
    AUDIT.parent.mkdir(parents=True,exist_ok=True)
    AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_WIN_Q_MARKET_STAGE",json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

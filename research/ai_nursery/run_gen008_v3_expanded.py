"""Run extended November audit using newly recovered full original odds.

The V3 architecture, calibration choice and inherited Sep14 checkpoints
MUST match the already completed first November 308-race evaluation.
The November outcomes are never used to select parameter values.
"""
from __future__ import annotations
import gzip,hashlib,json

from train_week import ROOT,eligible
from source_bridge import load_cards_year,stream_podium
import run_gen008_v3 as run_v3
import verify_gen008_v3 as verify_v3
from recover_gen008_v3_nov_odds import (PRICES,REPORT,FIRST,LAST,validate_books,fingerprint)

BASE=ROOT/"research/ai_nursery/outputs/GEN-008-V3"
ORIGINAL=BASE/"first-independent-november-2023"
EXPANDED=BASE/"november-2023-expanded-original-odds"
WEB=ROOT/"docs/ai-lab/gen008-v3-november-expanded.json"
STAGE="GEN-008-V3-NOVEMBER-EXPANDED-2023"

original_rows_in=run_v3.rows_in
original_calibrate=run_v3.calibrate


def expanded_rows_in(first,last):
    rows,report=original_rows_in(first,last)
    if (first,last)!=(FIRST,LAST):
        return rows,report
    archive=json.loads(REPORT.read_text(encoding="utf-8"))
    assert archive["target_dates"]==[FIRST,LAST]
    assert archive["no_result_labels_read"] and archive["no_imputed_odds"]
    assert fingerprint(PRICES)==archive["books_sha256"]
    cards,_,_=load_cards_year(2023,FIRST,LAST)
    labels,_=stream_podium(2023,set(cards))
    existing={x["race_id"] for x in rows}
    recovered={}
    with gzip.open(PRICES,"rt",encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():continue
            o=json.loads(line)
            rid=o["race_id"]
            assert rid in cards and rid not in recovered and rid not in existing
            assert o["phase"]=="FINAL"
            validate_books(rid,sorted(cards[rid]["entries"]),o["odds"])
            recovered[rid]=o
    assert len(recovered)==archive["recovered_total"]
    new_valid=0
    no_result=0
    invalid=0
    for rid,obj in sorted(recovered.items()):
        if rid not in labels:
            no_result+=1
            continue
        row={"race_id":rid,**cards[rid],"odds":obj["odds"],"outcome":labels[rid]}
        if not eligible(row):
            invalid+=1
            continue
        rows.append(row)
        new_valid+=1
    rows.sort(key=lambda r:(r["race"]["race_date"],r["race"].get("start_time",""),
             r["race"].get("track",""),int(r["race"].get("race_no") or 0),r["race_id"]))
    assert len(rows)==len(existing)+new_valid
    assert len({r["race_id"] for r in rows})==len(rows)
    report={**report,
        "join_status":{**report["join_status"],"joined_and_eligible":len(rows)},
        "complete_races":len(rows),
        "joined_race_ids_sha256":hashlib.sha256("\n".join(r["race_id"] for r in rows).encode()).hexdigest(),
        "additional_verified_books":{
           "original_exam_races":len(existing),
           "recovered_original_final_prices":len(recovered),
           "newly_evaluable_races":new_valid,
           "missing_results_among_recovered":no_result,
           "ineligible_with_recovered_odds":invalid,
           "remaining_market_parse_failures":archive["remaining_unrecovered"],
           "source_manifest":str(REPORT.relative_to(ROOT)),
           "no_estimated_odds":True}}
    print("GEN008_V3_NOV_PRICES_REJOINED",json.dumps(report["additional_verified_books"]),flush=True)
    return rows,report


def fit_same_alpha(train,val,weights,temperature):
    fit=original_calibrate(train,val,weights,temperature)
    prior=json.loads((ORIGINAL/"v3_checkpoint.json").read_text(encoding="utf-8"))
    assert prior["calibration"]==fit, "V3 calibration unexpectedly changed since first November exam"
    return fit


def run():
    if not (ORIGINAL/"summary.json").exists():
        raise RuntimeError("Original 308-race exam must already be archived")
    if not REPORT.exists() or not PRICES.exists():
        raise RuntimeError("November recovered odds and manifest not present")
    run_v3.OUT=EXPANDED
    run_v3.WEB=WEB
    run_v3.STAGE=STAGE
    run_v3.rows_in=expanded_rows_in
    run_v3.calibrate=fit_same_alpha
    run_v3.run()
    audit()


def audit():
    verify_v3.OUT=EXPANDED
    verify_v3.WEB=WEB
    verify_v3.STAGE=STAGE
    verify_v3.check()
    summary=json.loads((EXPANDED/"summary.json").read_text(encoding="utf-8"))
    earlier=json.loads((ORIGINAL/"summary.json").read_text(encoding="utf-8"))
    assert summary["v3_alpha_fit"]==earlier["v3_alpha_fit"]
    assert summary["v3_algorithm_digest"]==earlier["v3_algorithm_digest"]
    assert summary["frozen_parent_start"]==earlier["frozen_parent_start"]
    assert summary["november_exam_races"]>earlier["november_exam_races"]
    assert summary["source_report"]["additional_verified_books"]["no_estimated_odds"]
    print("GEN008_V3_EXPANDED_AUDIT_PASS",json.dumps({
        "races":summary["november_exam_races"],
        "newly_evaluable":summary["source_report"]["additional_verified_books"]["newly_evaluable_races"],
        "same_v3_algorithm":True,"comparison":summary["comparison"]},
        ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()

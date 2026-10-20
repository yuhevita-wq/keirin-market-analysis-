"""Expanded frozen December study using original market error coefficients.

The original 285-race December audit stays immutable. Newly recovered exact-ID
FINAL prices are joined with podium labels only for settlement. There is
NO fitting, re-fitting, promotion or threshold optimization on December.
"""
from __future__ import annotations

import gzip,hashlib,json
from pathlib import Path

from train_week import ROOT,eligible
from source_bridge import load_cards_year,stream_podium
import run_gen008_v3_market_error as study
import verify_gen008_v3_market_error as auditor
from gen008_frozen_exam import load_frozen
from gen008_v3_market_error import logloss_report
from recover_gen008_market_error_dec_odds import (
 PRICES,REPORT,FIRST,LAST,validate_books,fingerprint
)

ROOT_OLD=ROOT/"research/ai_nursery/outputs/GEN-008-V3-MARKET-ERROR/frozen-dec-01-to-14-2023"
ROOT_NEW=ROOT/"research/ai_nursery/outputs/GEN-008-V3-MARKET-ERROR/dec-2023-expanded-authentic-prices"
NEW_WEB=ROOT/"docs/ai-lab/gen008-v3-market-error-dec-2023-expanded.json"
STAGE="GEN008-V3-MARKET-ERROR-DEC-2023-EXPANDED"
BASE_SOURCE_ROWS=study.source_rows


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def expanded_source_rows():
    train,diagnostic,races,reports=BASE_SOURCE_ROWS()
    archive=read(REPORT)
    assert archive["target_dates"]==[FIRST,LAST]
    assert archive["no_result_labels_read"] and archive["no_imputed_odds"]
    assert fingerprint(PRICES)==archive["books_sha256"]
    original={r["race_id"] for r in races}
    cards,_,_=load_cards_year(2023,FIRST,LAST)
    outcome,_=stream_podium(2023,set(cards))
    newer={}
    with gzip.open(PRICES,"rt",encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():continue
            rec=json.loads(line)
            rid=rec["race_id"]
            assert rid in cards and rid not in newer and rid not in original
            assert rec["phase"]=="FINAL"
            validate_books(rid,sorted(cards[rid]["entries"]),rec["odds"])
            newer[rid]=rec
    assert len(newer)==archive["recovered_total"]
    extra=0
    missing_results=0
    unfit=0
    for rid,rec in sorted(newer.items()):
        if rid not in outcome:
            missing_results+=1
            continue
        entry={"race_id":rid,**cards[rid],"odds":rec["odds"],"outcome":outcome[rid]}
        if not eligible(entry):
            unfit+=1
            continue
        races.append(entry)
        extra+=1
    races.sort(key=lambda r:(r["race"]["race_date"],r["race"].get("start_time",""),
              r["race"].get("track",""),int(r["race"].get("race_no") or 0),r["race_id"]))
    assert len(races)==len(original)+extra
    assert len({r["race_id"] for r in races})==len(races)
    d=dict(reports["december"])
    d["join_status"]={**d["join_status"],"joined_and_eligible":len(races)}
    d["complete_races"]=len(races)
    d["joined_race_ids_sha256"]=hashlib.sha256("\n".join(r["race_id"] for r in races).encode()).hexdigest()
    d["expanded_verified_final_prices"]={
       "old_exam_races":len(original),"verified_original_price_books":len(newer),
       "new_evaluable_races":extra,"recovered_without_result":missing_results,
       "recovered_ineligible":unfit,"parse_failures_left":archive["remaining_unrecovered"],
       "original_price_archive_sha256":archive["books_sha256"],"no_prices_imputed":True}
    reports={**reports,"december":d}
    print("DEC_MARKET_ODDS_REJOINED",json.dumps(d["expanded_verified_final_prices"]),flush=True)
    return train,diagnostic,races,reports


def fixed_fit(*args):
    """NO re-training on enlarged datasets; only load previously frozen parameters."""
    old=read(ROOT_OLD/"market_error_fit.json")
    assert old["status"]=="FIT_COMPLETE" and not old["future_december_labels_used_in_fit"]
    return dict(old)


def run():
    if not REPORT.exists() or not PRICES.exists():
        raise RuntimeError("Immutable authentic December price recovery required")
    if not (ROOT_OLD/"summary.json").exists():
        raise RuntimeError("First December market mistake evaluation not yet archived")
    old=read(ROOT_OLD/"summary.json")
    older_model=read(ROOT_OLD/"market_error_fit.json")
    study.OUT=ROOT_NEW
    study.WEB=NEW_WEB
    study.STAGE=STAGE
    study.source_rows=expanded_source_rows
    study.fit=fixed_fit
    if ROOT_NEW.exists() or NEW_WEB.exists():
        raise RuntimeError("Expanded evidence already exists; refuse overwrite")
    study.run()
    auditor.OUT=ROOT_NEW
    auditor.WEB=NEW_WEB
    auditor.STAGE=STAGE
    auditor.source_rows=expanded_source_rows
    auditor.audit()
    current=read(ROOT_NEW/"summary.json")
    fit=read(ROOT_NEW/"market_error_fit.json")
    assert current["fitted_model_sha256"]==old["fitted_model_sha256"]
    assert fit["coefficients"]==older_model["coefficients"]
    assert fit["algorithm_sha256"]==older_model["algorithm_sha256"]
    assert current["frozen_parent_start"]==old["frozen_parent_start"]
    assert current["test_races"]>old["test_races"]
    assert current["source_reports"]["december"]["expanded_verified_final_prices"]["no_prices_imputed"]
    _,_,races,_=expanded_source_rows()
    parent,_=load_frozen()
    market_comparison=logloss_report(
      races,older_model["coefficients"],
      parent["new"].weights,parent["new"].temperature,older_model["frozen_v3_alpha"])
    analysis={
       "kind":"POST_SEALED_DIAGNOSTIC_ONLY","does_not_change_model":True,
       "old_model_digest":old["fitted_model_sha256"],"new_model_digest":current["fitted_model_sha256"],
       "new_december_exam_races":current["test_races"],
       "recovered_additional_evaluable":current["source_reports"]["december"]["expanded_verified_final_prices"]["new_evaluable_races"],
       "out_of_sample_market_vs_corrected_logloss":market_comparison,
       "test_paper_comparison":current["comparison"],
       "no_refit":True,"no_stake_decisions_adjusted_after_outcomes":True,
    }
    (ROOT_NEW/"blind_market_diagnostic.json").write_text(
        json.dumps(analysis,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN008_V3_MARKET_ERROR_EXPANDED_AUDIT_SUCCESS",json.dumps(analysis,ensure_ascii=False),flush=True)


if __name__=="__main__":run()

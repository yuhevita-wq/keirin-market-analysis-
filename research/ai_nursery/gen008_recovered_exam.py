"""Expanded frozen GEN-008 October exam with genuinely recovered historical full odds.

Reuse the exact prior scoring/decision code and never alter the 267-race exam.
"""
from __future__ import annotations
import gzip,hashlib,json
from pathlib import Path
from train_week import ROOT,eligible
from source_bridge import joined_races as original_joined_races,load_cards_year,stream_podium
import gen008_frozen_exam as exam
import verify_gen008_frozen_exam as verifier
from recover_gen008_oct_odds import BASE,PRICES,REPORT,validate_books,fingerprint,FIRST,LAST

EXPANDED_OUT=exam.BASE/"frozen-exam-2023-10-01-to-14-odds-recovered"
EXPANDED_WEB=ROOT/"docs/ai-lab/gen008-frozen-exam-2023-10-recovered.json"
STAGE="GEN-008-PARALLEL-FROZEN-2023-10-01-14-RECOVERED-ODDS"

def expanded_joined_races(year,first,last):
    assert (year,first,last)==(2023,FIRST,LAST)
    prior,source=original_joined_races(year,first,last)
    old_ids={r["race_id"] for r in prior}
    cards,_,_=load_cards_year(year,first,last)
    labels,_=stream_podium(year,set(cards))
    audit=json.loads(REPORT.read_text(encoding="utf-8"))
    assert audit["stage"]=="GEN-008-OCT-2023-ODDS-RECOVERY"
    assert audit["target_dates"]==[FIRST,LAST]
    assert audit["no_result_labels_read"] and audit["no_imputed_odds"]
    assert fingerprint(PRICES)==audit["books_sha256"]
    recovered={}
    with gzip.open(PRICES,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            obj=json.loads(line)
            rid=obj["race_id"]
            assert rid in cards and rid not in recovered and rid not in old_ids
            assert obj["phase"]=="FINAL"
            validate_books(rid,sorted(cards[rid]["entries"]),obj["odds"])
            recovered[rid]=obj
    assert len(recovered)==audit["recovered_total"]
    rows=list(prior)
    extra={"no_outcome":0,"ineligible":0,"eligible":0}
    for rid,obj in sorted(recovered.items()):
        if rid not in labels:
            extra["no_outcome"]+=1
            continue
        row={"race_id":rid,**cards[rid],"odds":obj["odds"],"outcome":labels[rid]}
        if not eligible(row):
            extra["ineligible"]+=1
            continue
        rows.append(row)
        extra["eligible"]+=1
    rows.sort(key=lambda r:(r["race"]["race_date"],r["race"].get("start_time",""),
                            r["race"].get("track",""),int(r["race"].get("race_no") or 0),r["race_id"]))
    ids=[r["race_id"] for r in rows]
    assert len(set(ids))==len(ids)
    assert len(rows)==len(prior)+extra["eligible"]
    source={**source,
           "historical_odds_backfill":{
               "prior_eligible":len(prior),
               "recovered_price_books":len(recovered),
               "new_eligible":extra["eligible"],
               "recovered_without_outcome":extra["no_outcome"],
               "recovered_ineligible":extra["ineligible"],
               "remaining_odds_missing":audit["remaining_unrecovered"],
               "no_imputation":True,
               "books_sha256":audit["books_sha256"],
               "recovery_manifest":str(REPORT.relative_to(ROOT))
            },
            "complete_races":len(rows),
            "join_status":{**source["join_status"],"joined_and_eligible":len(rows)},
            "joined_race_ids_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    print("GEN008_HOLDOUT_ODDS_EXPANDED",json.dumps(source["historical_odds_backfill"]),flush=True)
    return rows,source

def run():
    if not REPORT.is_file() or not PRICES.is_file():
        raise RuntimeError("Genuinely recovered archived prices not ready")
    exam.OUT=EXPANDED_OUT
    exam.WEB=EXPANDED_WEB
    exam.STAGE=STAGE
    exam.joined_races=expanded_joined_races
    if EXPANDED_OUT.exists() or EXPANDED_WEB.exists():
        raise RuntimeError("Expanded frozen exam already exists; refuse overwrite")
    exam.run()
    audit()

def audit():
    verifier.OUT=EXPANDED_OUT
    verifier.WEB=EXPANDED_WEB
    verifier.STAGE=STAGE
    verifier.check()
    summary=json.loads((EXPANDED_OUT/"summary.json").read_text(encoding="utf-8"))
    src=summary["source_report"]["historical_odds_backfill"]
    assert summary["races"]==src["prior_eligible"]+src["new_eligible"]
    assert summary["races"]>src["prior_eligible"],"Recovery provided no new evaluable races"
    assert summary["no_training_on_exam"] and not summary["weights_or_policy_updated"]
    assert src["books_sha256"]==fingerprint(PRICES)
    print("GEN008_EXPANDED_FROZEN_EXAM_AUDIT_PASS",json.dumps(
        {"races":summary["races"],"added":src["new_eligible"],
         "old":summary["comparison"]["old"],"new":summary["comparison"]["new"]},
        ensure_ascii=False),flush=True)

if __name__=="__main__":
    run()

#!/usr/bin/env python3
"""Reconcile formerly blocked race-result pages using the saved publisher audit.

A missing/abnormal standard podium (possibly cancellation, dead heat or
publication format) is NOT a training label. Reconcile only known, explicit
published-podium failures. HTTP, identity and source failures remain pending.
Read-only against odds and existing valid race results.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date,datetime,timedelta,timezone
from pathlib import Path

from collect_result_days import (
    ROOT, BEGIN, END, CURSOR_DIR, LEDGER_DIR,
    accept_previous_exclusions, file_paths, load_json,
)


def reconcile():
    summaries=Counter()
    ids={}
    for ledger in sorted(LEDGER_DIR.glob("20??/20??-??-??.json")):
        day=date.fromisoformat(ledger.stem)
        if not BEGIN<=day<=END:
            continue
        state=json.loads(ledger.read_text(encoding="utf-8"))
        if state.get("day")!=day.isoformat():
            raise ValueError("Result ledger date mismatch: "+str(ledger))
        if accept_previous_exclusions(state,day):
            # Previously archived valid results must exist before a day is
            # accepted complete; an exception must never erase these labels.
            if state.get("complete") and state.get("podiums",0)>0:
                saved,_=file_paths(day)
                if not saved.is_file():
                    raise ValueError("Cannot complete day with missing result archive: "+str(saved))
            ledger.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            summaries["ledgers_reconciled"]+=1
        summaries["ledger_days_checked"]+=1
        summaries["completed_days"]+=int(bool(state.get("complete")))
        if not state.get("complete"):
            summaries["unfinished_days"]+=1
        for rid,info in state.get("excluded_nonstandard_races",{}).items():
            if rid in ids and ids[rid]!=info:
                raise ValueError("Conflicting source exclusion for "+rid)
            ids[rid]=info
    for item in ids.values():
        summaries["reason_"+item["reason"]]+=1
    summaries["published_podium_exclusions"]=len(ids)

    span=(END-BEGIN).days+1
    shard_reports={}
    for shard in range(8):
        cursor=CURSOR_DIR/f"shard_{shard:02d}.json"
        if not cursor.is_file():
            continue
        state=json.loads(cursor.read_text(encoding="utf-8"))
        past=min(span,int(state.get("next_offset",shard)))
        pending,exhausted=[],[]
        for offset in range(shard,past,8):
            day=BEGIN+timedelta(days=offset)
            _,ledger=file_paths(day)
            record=load_json(ledger,{"attempts":0,"complete":False})
            if record.get("complete"):
                continue
            (pending if int(record.get("attempts",0))<3 else exhausted).append(day.isoformat())
        state["pending_repair_days"]=len(pending)
        state["exhausted_unresolved_days"]=len(exhausted)
        state["finished_all"]=bool(state.get("finished_forward_pass")) and not pending
        cursor.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        shard_reports[str(shard)]={
            "pending_non_excludable_days":len(pending),
            "exhausted_non_excludable_days":len(exhausted),
            "finished_forward_pass":bool(state.get("finished_forward_pass")),
            "finished_all":bool(state.get("finished_all")),
            "paused_for_source_failure":bool(state.get("paused_for_source_failure")),
        }
    report={
        "schema":"nonstandard_published_podium_exclusions_v1",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "date_scope":[BEGIN.isoformat(),END.isoformat()],
        "scope_excludes":"Published race summaries without uniquely verifiable standard 1st/2nd/3rd",
        "do_not_assume_cancellation":True,
        "excluded_are_not_training_labels":True,
        "no_http_or_identity_failure_waiver":True,
        "statistics":dict(summaries),
        "shards":shard_reports,
        "note":"Excluded race-ID list persists with reason and source URL in each daily ledger. A verified outcome from an independent archive is retained separately.",
    }
    dest=ROOT/"data/audits/result_nonstandard_exclusions.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("NONSTANDARD_RESULT_RECONCILIATION",json.dumps({
        "previously_blocked_days_reconciled":summaries["ledgers_reconciled"],
        "excluded_race_ids":len(ids),
        "completed_days":summaries["completed_days"],
        "unfinished_days":summaries["unfinished_days"],
        "pending_repair_days":sum(x["pending_non_excludable_days"] for x in shard_reports.values()),
        "exhausted_repair_days":sum(x["exhausted_non_excludable_days"] for x in shard_reports.values()),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    reconcile()

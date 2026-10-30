#!/usr/bin/env python3
"""Audit every eligible 2023 race and every ordered trifecta under frozen V6.

V6 is fit identically to the previous audits on 2023 Jan-Oct score-valid cards.
2023 Jan-Oct = IN SAMPLE, Nov-Dec = development / not used for fit, but
previously inspected. Full 2023 includes both, and is NOT a full holdout.
No odds, no payout filters, no new fitting, no rewriting V6.
"""
from __future__ import annotations
import json,sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
import v6_probability_donor_recipient_structural_audit as audit
from v8_targeted_weak_podium_calibration import row_probs
prior=audit.prior
OUT=ROOT/"data/audits/v6_all_2023_donor_receiver_structural.json"

def merge_cells(dst,src):
    for k,v in src.items():
        c=dst[k]
        c.tickets+=v.tickets
        c.expected+=v.expected
        c.actual+=v.actual
        c.variance+=v.variance
        c.races_with_mass+=v.races_with_mass

def compact(sub,axis):
    return {k:{"predicted":v["expected"],"actual":v["actual"],
               "actual_minus_predicted":v["observed_minus_expected"],
               "per_race_probability_mass_error_pp":v["mass_error_pp_per_race"],
               "diagnostic_z":v["race_binomial_z_diagnostic"]}
            for k,v in sub["cells"][axis].items()}

def main():
    print("V6_2023_FULL_YEAR_AUDIT_START",flush=True)
    events,src=prior.obs.ingest()
    train={rid:e for rid,e in events.items()
           if e["date"]<="2023-10-31" and prior.weak.score_gate(e)=="ELIGIBLE"}
    assert len(train)==19017,("V6 train cohort drift",len(train))
    wg,wr=prior.v2.fit_structural(train)
    pw,_=prior.v3.train_pattern_gate(train)
    d5=prior.v5.fit_residual(train,wg,wr)
    wl,wc=prior.v6.fit_opponent(train,wg,wr,d5)
    args=(wg,wr,d5,wl,wc,pw)
    del train
    print("V6_2023_FROZEN_MODEL_REFIT_IDENTICAL",flush=True)
    cell_sets={name:defaultdict(audit.Cell) for name in ("2023_train","2023_dev")}
    counts={name:Counter() for name in ("2023_train","2023_dev")}
    tickets=Counter()
    for rid,e in events.items():
        period="2023_train" if e["date"]<="2023-10-31" else "2023_dev"
        count=counts[period]
        count["source"]+=1
        if prior.weak.score_gate(e)!="ELIGIBLE":
            count["invalid_pre_race_card"]+=1;continue
        dist=row_probs(e,*args)
        tickets[period]+=audit.race(cell_sets[period],e,dist)
        count["evaluated"]+=1
        if count["evaluated"]%3000==0:
            print("V6_2023_PROGRESS",period,count["evaluated"],"tickets",tickets[period],flush=True)
    outputs={}
    full=defaultdict(audit.Cell)
    for period in ("2023_train","2023_dev"):
        c=counts[period]
        assert c["source"]==sum(v for k,v in c.items() if k!="source"),(period,dict(c))
        outputs[period]={"races":c["evaluated"],"tickets":tickets[period],
                         "cells":audit.pack(cell_sets[period],c["evaluated"])}
        merge_cells(full,cell_sets[period])
        print("V6_2023_SECTION_RESULT",period,json.dumps({
            "coverage":dict(c),
            "patterns":compact(outputs[period],"PAT"),
            "probability_bands":compact(outputs[period],"P")},
            ensure_ascii=False),flush=True)
    total_races=sum(c["evaluated"] for c in counts.values())
    total_tickets=sum(tickets.values())
    assert total_races==23148,("Unexpected 2023 full score-valid population",total_races)
    assert total_tickets==5388222,("Unexpected full 2023 all-ticket count",total_tickets)
    assert counts["2023_dev"]["evaluated"]==4131
    outputs["2023_full"]={"races":total_races,"tickets":total_tickets,
                          "cells":audit.pack(full,total_races)}
    for label,z in outputs.items():
        for axis,sub in z["cells"].items():
            assert sum(x["actual"] for x in sub.values())==z["races"],(label,axis,"Actual")
            assert sum(x["tickets"] for x in sub.values())==z["tickets"],(label,axis,"Tickets")
            assert abs(sum(x["expected"] for x in sub.values())-z["races"])<.1,(label,axis,"Predicted")
            assert abs(sum(x["observed_minus_expected"] for x in sub.values()))<.1,(label,axis,"Residual")
    def top(axis):
        data=outputs["2023_full"]["cells"][axis]
        x=[(k,v) for k,v in data.items() if v["expected"]>=40]
        return {"overallocated":sorted(
            ({"key":k,"expected":v["expected"],"actual":v["actual"],
              "diff":v["observed_minus_expected"]}
             for k,v in x if v["observed_minus_expected"]<0),
             key=lambda x:x["diff"])[:25],
            "underallocated":sorted(
            ({"key":k,"expected":v["expected"],"actual":v["actual"],
              "diff":v["observed_minus_expected"]}
             for k,v in x if v["observed_minus_expected"]>0),
             key=lambda x:-x["diff"])[:25]}
    results={
        "state":"2023_FULL_YEAR_FROZEN_V6_ALL_TICKET_PROBABILITY_STRUCTURAL_AUDIT_COMPLETE",
        "coverage":{p:dict(c) for p,c in counts.items()},
        "training_overlap":"2023 Jan-Oct was used for V6 fitting; never claim all-2023 unknown",
        "2023_Nov_Dec":"Not used for fitting but already inspected in previous studies",
        "model_frozen":True,
        "odds_as_teacher":False,
        "population":"Every 2023 source race with valid pre-race score/card and recorded top-three outcome; no payout files required",
        "sum_p_equals_1_per_race":True,
        "periods":outputs,
        "top_structural_donor_recipient":{axis:top(axis) for axis in (
            "PAT","R1","RK1","P_PAT","P_PAT_R1","P_PAT_R1_R2","P_PAT_R1_RK1")},
        "source":src,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V6_2023_FULL_YEAR_FINAL_RESULT",json.dumps({
        "races":total_races,"tickets":total_tickets,
        "periods":{k:{"races":v["races"],"tickets":v["tickets"]}
                   for k,v in outputs.items()},
        "patterns":compact(outputs["2023_full"],"PAT"),
        "probability_bands":compact(outputs["2023_full"],"P"),
        "roles":compact(outputs["2023_full"],"R1"),
        "top_intersections":{axis:results["top_structural_donor_recipient"][axis]
                             for axis in ("P_PAT","P_PAT_R1")},
        },ensure_ascii=False),flush=True)
    print("V6_2023_FULL_YEAR_COMPLETE_AND_PROBABILITY_CONSERVED",flush=True)

if __name__=="__main__":
    main()

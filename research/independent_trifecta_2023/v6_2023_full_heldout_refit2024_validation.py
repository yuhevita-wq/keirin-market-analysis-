#!/usr/bin/env python3
"""2023 COMPLETE-YEAR V6 STRUCTURAL VALIDATION -- no partial 2023 season.

The user's test year is ALL of 2023, Jan 1-Dec 31; no 2023 outcomes or
features are included in fitting V6's weights, even incidentally.
Train the SAME *algorithm family* V2/V3(.15)/V5/V6 on PREDECLARED 2024
Jan-Oct, strictly disjoint, then predict 2023. This is reverse-time historical
evaluation, NOT realistic ex-ante training; and it is NOT the previously
2023-trained V6's immutable set of weights. The previously trained V6
continues unchanged on main / other research branches.

The label coverage is ALL 23440 2023 source races; unusable cards (non-
positive or absent source scores) are excluded transparently, never quietly
called evaluated. Crucially NO payout, odds, winner-payout matching or
winner-dependent selection criteria restrict the 2023 validation cohort.
All ordered 3ren-tan combinations predicted for every evaluable race.
No fitted repair, reweighting, calibration teacher or market prices.
"""
from __future__ import annotations
import json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as obs
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import v5_rider_score_role_size_interactions as v5
import v6_opponent_competition_structure as v6
import calibration_audit as r24
import weak_rider_podium_zero_excluded as weak
from v8_targeted_weak_podium_calibration import row_probs
from v6_probability_donor_recipient_structural_audit import (
    Cell, AXES, race as structural_race, pack as structural_pack
)

OUT=ROOT/"data/audits/v6_2023_full_calendar_heldout_from_2024_fitting.json"
EXPECTED_2023=23440
EXPECTED_2024=25348

def summarize(cells, nr):
    packed=structural_pack(cells,nr)
    pattern=packed["PAT"]
    key="AAA"
    aaa=pattern[key]["expected"];aab=pattern["AAB"]["expected"]
    aa=pattern[key]["actual"];ab=pattern["AAB"]["actual"]
    return packed,{
       "prediction_AAA_given_first_two_same_pct":round(100*aaa/(aaa+aab),8),
       "observed_AAA_given_first_two_same_pct":round(100*aa/(aa+ab),8),
       "first_two_same_expected":round(aaa+aab,6),
       "first_two_same_observed":aa+ab,
       "AAA_observed_minus_expected":round(aa-aaa,6),
       "AAB_observed_minus_expected":round(ab-aab,6),
       "models_are_reverse_time_2024_refitted_V6":True,
    }

def main():
    # Fit only on 2024, absolutely no 2023 input allowed until all weights
    # fixed. Keeps full 2023 independent of fitting.
    train_source,src24=r24.ingest_2024()
    if len(train_source)!=EXPECTED_2024:raise AssertionError(("2024 historical population changed",len(train_source)))
    train={rid:e for rid,e in train_source.items()
           if "2024-01-01"<=e["date"]<="2024-10-31"
           and weak.score_gate(e)=="ELIGIBLE"}
    assert len(train)>16000
    print("V6_FULL_2023_FIT_USES_2024_ONLY",json.dumps({
       "2024_source_records":len(train_source),"2024_training":len(train),
       "no_2023_records_in_training":all(e["date"].startswith("2024") for e in train.values())
       }),flush=True)
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    args=(wg,wr,d5,wl,wc,pw)
    # Result labels never accessed while model weights fitted.
    del train,train_source
    print("V6_FULL_2023_REVERSE_HOLDOUT_WEIGHTS_FROZEN",flush=True)
    test_source,source_meta=obs.ingest()
    if len(test_source)!=EXPECTED_2023:
        raise AssertionError(("2023 source population changed",len(test_source)))
    all_cells=defaultdict(Cell)
    months=defaultdict(lambda:{"source":0,"evaluated":0,
                               "excluded":Counter(),"cells":defaultdict(Cell),
                               "tickets":0})
    coverage=Counter()
    truth_rank=Counter()
    score=Counter()
    nll=0.;brier=0.
    for rid,e in test_source.items():
        if not ("2023-01-01"<=e["date"]<="2024-01-01"):
            raise AssertionError(("Non-2023 race in 2023 test",rid,e["date"]))
        m=e["date"][:7]
        coverage["all_2023_source_races"]+=1
        months[m]["source"]+=1
        gate=weak.score_gate(e)
        if gate!="ELIGIBLE":
            coverage["excluded_"+gate]+=1
            months[m]["excluded"][gate]+=1
            continue
        dist=row_probs(e,*args)
        # All 2023 score-positive races receive full model distribution.
        n=structural_race(all_cells,e,dist)
        # Restrict month-specific monitoring to mutually exclusive partitions
        # to avoid repeating the expensive full 13-axis computation.
        month_cell=months[m]["cells"]
        roles={car:v2.rlabel(e,car) for car in e["scores"]}
        qby=defaultdict(lambda:[0,0.])
        winner=e["order"][:3]
        for (a,b,c),p in dist.items():
            g1,g2,g3=(e["positions"][x][0] for x in (a,b,c))
            pat=("AAA" if g1==g2==g3 else
                 "AAB" if g1==g2 else
                 "ABA" if g1==g3 else
                 "ABB" if g2==g3 else "ABC")
            qby[pat][0]+=1;qby[pat][1]+=p
        a,b,c=winner;ga,gb,gc=(e["positions"][x][0] for x in winner)
        target=("AAA" if ga==gb==gc else
                "AAB" if ga==gb else
                "ABA" if ga==gc else
                "ABB" if gb==gc else "ABC")
        for pat,(tickets,q) in qby.items():
            month_cell[("PAT",pat)].add(tickets,q,int(pat==target))
        coverage["evaluated"]+=1
        coverage["all_trifecta_probabilities"]+=n
        months[m]["evaluated"]+=1
        months[m]["tickets"]+=n
        nll-=math.log(max(1e-300,dist[winner]))
        brier+=1-2*dist[winner]+sum(p*p for p in dist.values())
        if coverage["evaluated"]%3000==0:
            print("V6_2023_FULL_EVALUATED",coverage["evaluated"],"tickets",coverage["all_trifecta_probabilities"],flush=True)
    assert coverage["all_2023_source_races"]==EXPECTED_2023
    assert coverage["all_2023_source_races"]==coverage["evaluated"]+sum(
           v for k,v in coverage.items() if k.startswith("excluded_"))
    assert len(months)==12
    n=coverage["evaluated"]
    assert n>22000,("too few valid 2023 race cards",n)
    struct,conditional=summarize(all_cells,n)
    all_month={}
    for mon,v in sorted(months.items()):
        cats=v["cells"]
        assert sum(c.actual for c in cats.values())==v["evaluated"]
        assert abs(sum(c.expected for c in cats.values())-v["evaluated"])<.05
        all_month[mon]={
            "source":v["source"],"evaluated":v["evaluated"],
            "excluded":dict(v["excluded"]),"tickets":v["tickets"],
            "predicted_vs_observed_patterns":{
                 k:c.output() for (axis,k),c in sorted(cats.items())},
        }
    for axis,sections in struct.items():
        assert sum(x["actual"] for x in sections.values())==n
        assert abs(sum(x["expected"] for x in sections.values())-n)<.03
        assert sum(x["tickets"] for x in sections.values())==coverage["all_trifecta_probabilities"]
    report={
      "state":"COMPLETE_2023_HISTORICAL_HELDOUT_FROM_2024_REFIT_V6_VALIDATION",
      "tested_2023_full_year":True,"source_races_2023":23440,
      "same_V6_algorithm_as_previous_study":True,
      "original_2023_trained_V6_weights_unchanged":True,
      "this_refit_model_training":"2024 Jan-Oct score-valid; NO 2023 training",
      "reverse_time_warning":"2024-trained fit evaluated BACKWARD on 2023. No 2023 training overlap but cannot call realistic 2023 forward-time OOS.",
      "score_zero_excluded_by_original_v6_data_contract":True,
      "no_winning_payout_gate":True,
      "no_odds_teacher":True,
      "no_ticket_selection":True,
      "coverage":dict(coverage),
      "mean_log_loss":round(nll/n,9),
      "mean_multiclass_brier":round(brier/n,9),
      "conditional_same_line_third":conditional,
      "cells":struct,
      "monthly_pattern_validation":all_month,
      "source_metadata":source_meta,
      "caveats":[
        "Reverse chronology from 2024 model fitting to 2023 test is a retrospective stress test, NOT an ex-ante 2023 forecast.",
        "Weights fitted on 2024 are a DIFFERENT fitted model; the original 2023-fitted V6 cannot honestly be evaluated all year out-of-sample.",
        "2023 clean score coverage is not 100%, and excludes full races with zero/invalid scores; no silently invented scores.",
        "Observed-expected structural gaps are descriptive and depend on the chosen trained model; no V6 correction or probability transfer was fitted.",
        "Do not claim unseen 2023 results if used to decide the structural hypotheses already.",
      ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V6_2023_FULL_YEAR_FINAL_RESULT",json.dumps({
       "model_training":report["this_refit_model_training"],
       "coverage":report["coverage"],
       "metrics":{"logloss":report["mean_log_loss"],"Brier":report["mean_multiclass_brier"]},
       "conditional_AAA_vs_AAB":conditional,
       "pattern":struct["PAT"],
       "winner_role":struct["R1"],
       "winner_rank":struct["RK1"],
       "probability_bands":struct["P"],
       "key_intersections":{axis:{
           "top_donors":sorted(((k,z["observed_minus_expected"]) for k,z in struct[axis].items()),
                              key=lambda x:x[1])[:12],
           "top_receivers":sorted(((k,z["observed_minus_expected"]) for k,z in struct[axis].items()),
                                 key=lambda x:-x[1])[:12]}
           for axis in ("P_PAT","P_PAT_R1","P_PAT_R1_RK1")}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

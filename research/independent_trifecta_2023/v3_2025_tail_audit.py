#!/usr/bin/env python3
"""Forward-year 2025 calibration audit for FROZEN V3 independent 3ren-tan AI.

Research ONLY: NEVER fit on 2024 or 2025 outcomes, and never read market odds.
Refit V2 2023 Jan-Oct and V3 pattern gate 2023 Jan-Oct EXACTLY as fixed code.
V3 mixing weight 0.15 was previously selected using 2023 Nov-Dec development
and validated on 2024. This fixed model is now scored on 2025 ONCE.

Probability bands use the model's OWN predictions, and separately hold the
V2-defined ticket populations fixed to avoid misleading bin-switch comparisons.
All 2025 ordered triples are evaluated (not just those that won or top N).
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as original
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import validate_2025 as year25

OUTPUT=ROOT/"data/audits/independent_trifecta_v3_2025_tail_calibration.json"
FROZEN_V3_WEIGHT=0.15
EXPECTED_2023_TRAIN=19248
EXPECTED_2025_TEST=24635
VERSIONS=("frozen_v2","frozen_v3")
PATTERNS=("AAA","AAB","ABA","ABB","ABC")
BINS=(
    (0.,0.0001,"<0.01%"),
    (0.0001,0.0005,"0.01-0.05%"),
    (0.0005,0.001,"0.05-0.1%"),
    (0.001,0.005,"0.1-0.5%"),
    (0.005,0.01,"0.5-1%"),
    (0.01,0.02,"1-2%"),
    (0.02,0.05,"2-5%"),
    (0.05,0.1,"5-10%"),
    (0.1,0.2,"10-20%"),
    (0.2,1.000001,"20%+"),
)
LOW_BINS=frozenset(("<0.01%","0.01-0.05%","0.05-0.1%"))


def prob_bin(p):
    if not 0<=p<=1:
        raise AssertionError(("Probability out of range",p))
    for lo,hi,label in BINS:
        if lo<=p<hi:
            return label
    raise AssertionError(("No bin",p))


class Tally:
    def __init__(self):
        self.p=0.
        self.hits=0
        self.tickets=0
        self.races=0
        self.var=0.
        self.by_day=defaultdict(float)

    def add(self, mass, hit, n, day):
        if not -1e-9<=mass<=1+1e-9 or hit not in (0,1):
            raise AssertionError(("Bad race-level cohort",mass,hit))
        mass=max(0.,min(1.,mass))
        self.p+=mass
        self.hits+=hit
        self.tickets+=n
        self.races+=1
        self.var+=mass*(1-mass)
        self.by_day[day]+=hit-mass

    def pack(self):
        d=self.hits-self.p
        days=list(self.by_day.values())
        nd=len(days)
        mean=sum(days)/nd if nd else 0.
        se=math.sqrt(nd/(nd-1)*sum((x-mean)**2 for x in days)) if nd>1 else 0.
        sd=math.sqrt(self.var)
        return {
            "trifecta_ticket_count":self.tickets,
            "race_exposures":self.races,
            "expected_winning_events":round(self.p,5),
            "actual_winning_events":self.hits,
            "actual_minus_expected":round(d,5),
            "actual_over_expected":round(self.hits/self.p,6) if self.p else None,
            "mean_predicted_probability_pct":round(100*self.p/self.tickets,8) if self.tickets else None,
            "empirical_winning_frequency_pct":round(100*self.hits/self.tickets,8) if self.tickets else None,
            "race_level_z":round(d/sd,4) if sd>0 else None,
            "day_cluster_z_approx":round(d/se,4) if se>0 else None,
            "day_cluster_approx_95pct_residual_interval":[round(d-1.96*se,3),round(d+1.96*se,3)]
                if se>0 else None,
            "days":nd,
            "rare_event_warning":self.p<30,
        }


class Scores:
    def __init__(self):
        self.n=0
        self.tickets=0
        self.sumprob=0.
        self.nll=0.
        self.brier=0.
        self.ptrue=0.
        self.byday=defaultdict(lambda:[0.,0.,0])
        self.month=defaultdict(lambda:[0,0.,0.])

    def add(self,e,dist):
        truth=e["order"]
        if truth not in dist:
            raise AssertionError("2025 truth absent from complete trifecta distribution")
        norm=sum(dist.values())
        if abs(norm-1)>1e-9:
            raise AssertionError(("2025 probability conservation",norm))
        pp=dist[truth]
        nll=-math.log(max(pp,1e-15))
        brier=1-2*pp+sum(p*p for p in dist.values())
        self.n+=1
        self.tickets+=len(dist)
        self.sumprob+=norm
        self.nll+=nll
        self.brier+=brier
        self.ptrue+=pp
        d=self.byday[e["date"]]
        d[0]+=nll;d[1]+=brier;d[2]+=1
        mm=self.month[e["date"][:7]]
        mm[0]+=1;mm[1]+=nll;mm[2]+=brier

    def pack(self):
        n=self.n
        return {
            "races":n,
            "all_three_way_tickets":self.tickets,
            "mean_probability_sum":round(self.sumprob/n,10),
            "mean_full_distribution_logloss":round(self.nll/n,8),
            "mean_full_distribution_brier":round(self.brier/n,8),
            "mean_p_of_actual_ticket_pct":round(100*self.ptrue/n,7),
            "month_scores":{
                k:{"races":v[0],"mean_logloss":round(v[1]/v[0],8),
                   "mean_brier":round(v[2]/v[0],8)}
                for k,v in sorted(self.month.items())
            },
        }


def paired_day_ci(old,new,index):
    days=sorted(set(old.byday)|set(new.byday))
    n=old.n
    diff=sum(old.byday[d][index]-new.byday[d][index] for d in days)/n
    se=math.sqrt(sum((
        old.byday[d][index]-new.byday[d][index]-diff*old.byday[d][2]
    )**2 for d in days))/n
    return {
        "V2_minus_V3_improvement_positive_is_better":round(diff,8),
        "cluster_robust_se":round(se,8),
        "approx_95pct_ci":[round(diff-1.96*se,8),round(diff+1.96*se,8)],
        "calendar_days":len(days),
    }


def report_axis(agg,version,axis):
    return {key:t.pack() for (m,a,key),t in sorted(agg.items())
            if m==version and a==axis}


def group_add(local,axis,key,prob,hit):
    z=local[(axis,key)]
    z[0]+=prob;z[1]+=int(hit);z[2]+=1


def audit_one(e,base,new,all_tallies):
    if base.keys()!=new.keys():
        raise AssertionError("Different ticket universes V2 and V3")
    if abs(sum(base.values())-1)>1e-9 or abs(sum(new.values())-1)>1e-9:
        raise AssertionError("All trifecta outcomes must sum to exactly 100%")
    shape=v2.line_shape(e)
    truth=e["order"]
    labels={}
    for t in base:
        pat=v2.pattern(e,t)
        labels[t]=(pat,v2.rlabel(e,t[2]),prob_bin(base[t]))
        if pat=="AAA" and abs(base[t]-new[t])>1e-13:
            raise AssertionError("Frozen V3 altered an AAA ticket")

    # Each local grouping is sum of relevant ticket probabilities in this race,
    # not treating millions of mutually-exclusive tickets as independent rows.
    for model,dist in (("frozen_v2",base),("frozen_v3",new)):
        local=defaultdict(lambda:[0.,0,0])
        for t,p in dist.items():
            pat,r3,v2bin=labels[t]
            ownbin=prob_bin(p)
            iswin=(t==truth)
            group_add(local,"self_probability_band",ownbin,p,iswin)
            # Stable cohort: bucket membership defined by V2, same ticket set.
            group_add(local,"V2_fixed_probability_band",v2bin,p,iswin)
            group_add(local,"line_pattern",pat,p,iswin)
            group_add(local,"line_shape_x_pattern",shape+"|"+pat,p,iswin)
            if ownbin in LOW_BINS:
                group_add(local,"low_tail_bin_x_pattern",ownbin+"|"+pat,p,iswin)
                group_add(local,"low_tail_bin_x_shape",ownbin+"|"+shape,p,iswin)
                group_add(local,"low_tail_bin_x_third_role",ownbin+"|"+r3,p,iswin)
        ch=defaultdict(lambda:[0.,0,0])
        for (axis,key),(mass,hit,n) in local.items():
            all_tallies[(model,axis,key)].add(mass,hit,n,e["date"])
            z=ch[axis]
            z[0]+=mass;z[1]+=hit;z[2]+=n
        for axis in ("self_probability_band","V2_fixed_probability_band",
                     "line_pattern","line_shape_x_pattern"):
            mass, hits, tickets=ch[axis]
            if abs(mass-1)>1e-8 or hits!=1 or tickets!=len(dist):
                raise AssertionError(("Incomplete race distribution partition",axis,model,mass,hits,tickets))


def main():
    assert FROZEN_V3_WEIGHT==0.15, "V3 weight must be the one tested in 2024"
    assert original.TRAIN_END=="2023-10-31"
    train_events,audit23=original.ingest()
    train={rid:e for rid,e in train_events.items() if e["date"]<=original.TRAIN_END}
    assert len(train)==EXPECTED_2023_TRAIN
    # Fixed parameters learned ONLY on 2023 Jan-Oct. No 2023 Nov-Dec
    # required for training (already inspected during V3 development).
    wg,wr=v2.fit_structural(train)
    pattern_w,nonAAA=v3.train_pattern_gate(train)
    del train,train_events
    print("FROZEN_2023_TRAINING_REPRODUCED",EXPECTED_2023_TRAIN,
          "V3_PREAPPROVED_WEIGHT",FROZEN_V3_WEIGHT,flush=True)

    # 2025 results enter for the very first time here, after training.
    test,coverage=year25.read_2025()
    if len(test)!=EXPECTED_2025_TEST:
        raise AssertionError(("2025 cohort changed: investigate, never silently shrink",len(test)))
    print("2025_TRUE_FORWARD_ALL_ELIGIBLE",len(test),flush=True)

    scores={name:Scores() for name in VERSIONS}
    tallies=defaultdict(Tally)
    for index,(rid,e) in enumerate(test.items(),1):
        base=v3.base_distribution(e,wg,wr)
        gate=v3.predict_non_aaa(e,pattern_w)
        variants,pats=v3.variants(e,base,gate)
        proposed=variants[FROZEN_V3_WEIGHT][0]
        for model,dist in (("frozen_v2",base),("frozen_v3",proposed)):
            scores[model].add(e,dist)
        audit_one(e,base,proposed,tallies)
        if index%1000==0:
            print("2025_FULL_TRIFECTA_LOW_TAIL_CHECK",index,
                  "RACES",scores["frozen_v3"].tickets,"TICKETS",flush=True)

    for model in VERSIONS:
        s=scores[model]
        if s.n!=EXPECTED_2025_TEST:
            raise AssertionError("Year cohort incomplete")
        for axis in ("self_probability_band","V2_fixed_probability_band","line_pattern"):
            d=report_axis(tallies,model,axis)
            exp=sum(z["expected_winning_events"] for z in d.values())
            hit=sum(z["actual_winning_events"] for z in d.values())
            if abs(exp-s.n)>0.01 or hit!=s.n:
                raise AssertionError(("Disjoint group total failed",model,axis,exp,hit))
        if abs(s.sumprob-s.n)>1e-6:
            raise AssertionError("Total 2025 probability sum changed")
    checka=report_axis(tallies,"frozen_v2","line_pattern")["AAA"]
    checkb=report_axis(tallies,"frozen_v3","line_pattern")["AAA"]
    if abs(checka["expected_winning_events"]-checkb["expected_winning_events"])>1e-7:
        raise AssertionError("V3 AAA conservation violated")

    by_model={
        m:{
            "overall":scores[m].pack(),
            "self_probability_bands":report_axis(tallies,m,"self_probability_band"),
            "same_tickets_grouped_by_V2_probability_bands":report_axis(tallies,m,"V2_fixed_probability_band"),
            "line_patterns":report_axis(tallies,m,"line_pattern"),
            "line_shape_x_pattern":report_axis(tallies,m,"line_shape_x_pattern"),
            "low_tail_band_x_pattern":report_axis(tallies,m,"low_tail_bin_x_pattern"),
            "low_tail_band_x_line_shape":report_axis(tallies,m,"low_tail_bin_x_shape"),
            "low_tail_band_x_3rd_place_role":report_axis(tallies,m,"low_tail_bin_x_third_role"),
        } for m in VERSIONS
    }
    rare={}
    for model in VERSIONS:
        bins=by_model[model]["self_probability_bands"]
        out={"expected":0.,"actual":0,"tickets":0}
        for key in LOW_BINS:
            o=bins.get(key)
            if o:
                out["expected"]+=o["expected_winning_events"]
                out["actual"]+=o["actual_winning_events"]
                out["tickets"]+=o["trifecta_ticket_count"]
        out["actual_over_expected"]=round(out["actual"]/out["expected"],6) if out["expected"] else None
        rare[model]=out

    result={
        "state":"COMPLETE_UNTOUCHED_2025_V2_V3_FULL_TRIFECTA_TAIL_AUDIT",
        "question":"Does frozen V3 estimate low-probability 3ren-tan outcomes appropriately in forward 2025, without losing full probability precision?",
        "source_2024_v3_AAB_guarded_Actions_run":40832847580,
        "V3_weight_frozen_at_2024_acceptance":FROZEN_V3_WEIGHT,
        "train_period":"2023-01-01..2023-10-31",
        "development_period_previously_inspected":"2023-11-01..2023-12-31",
        "prior_2024_check_period":"2024-01-01..2024-12-31",
        "independent_2025_test_period":"2025-01-01..2025-12-31",
        "2025_races":len(test),
        "trifecta_tickets_per_model":scores["frozen_v3"].tickets,
        "market_odds_used":False,
        "2024_or_2025_outcomes_used_to_fit_or_select":False,
        "2025_period_touched_before_fitting":False,
        "AAA_probabilities_held_fixed_ticket_by_ticket":True,
        "probability_bands_percentage":{
            label:[lo*100,hi*100] for lo,hi,label in BINS
        },
        "same_cohort_comparison_note":"V2 fixed probability bands report exactly the same sets of tickets under both variants, unlike model-self bins where ticket membership changes.",
        "V2_V3_all_race_proper_score_day_cluster_comparison":{
            "NLL":paired_day_ci(scores["frozen_v2"],scores["frozen_v3"],0),
            "Brier":paired_day_ci(scores["frozen_v2"],scores["frozen_v3"],1)
        },
        "under_0_1_percent_summary_model_own_bins":rare,
        "models":by_model,
        "source_2025_exact_ID_matching_and_coverage":coverage,
        "cautions":[
            "2025 is independent from 2023 fitting and previously used 2024 acceptance; 2025 must NOT be used to refit V3 after this audit.",
            "Low probability buckets are defined by AI prediction, NEVER by observed win, price or odds.",
            "For each event bucket, compute a RACE-level total probability mass; individual tickets in same race are mutually exclusive.",
            "Day-cluster residual intervals are approximate and not multiple-comparison adjusted; tiny expected-event cells are diagnostic only.",
            "Probability <0.1% does NOT imply market high odds: actual prices were not accessed.",
            "AAB/AAA pattern frequencies need not imply each individual exact triple is calibrated.",
            "Retrospectively archived predicted lines and score cutoff times not independently certified.",
            "No market pricing or return-on-investment analysis and no model promotion."
        ]
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("2025_FROZEN_V3_TAIL_AUDIT_COMPLETE",json.dumps({
        "races":result["2025_races"],
        "tickets_per_model":result["trifecta_tickets_per_model"],
        "overall":{k:v["overall"] for k,v in by_model.items()},
        "rare_bands":{k:{b:v for b,v in z["self_probability_bands"].items()
                           if b in LOW_BINS} for k,z in by_model.items()},
        "rare_aggregate":rare,
        "line_patterns":{k:z["line_patterns"] for k,z in by_model.items()},
        "paired_score_CI":result["V2_V3_all_race_proper_score_day_cluster_comparison"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

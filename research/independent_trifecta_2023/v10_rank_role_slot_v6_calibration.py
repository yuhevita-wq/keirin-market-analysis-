#!/usr/bin/env python3
"""V10: audit baseline V6's EX-ANTE top-three marginal calibration,
conditioned on actual scored rank, line role, score gap and field size.

All score=0/missing/invalid-card races are excluded before model fitting and
all evaluations. On positive-score 2023 Jan-Oct, fit the same V2/V3/V5/V6
algorithms as V9, without changing their underlying structure. V6 P(triple)
is post V3(.15) pattern gate and normalized across ALL ordered tickets.

For every eligible starter and slot:
   predicted expected count = sum_{races,riders in cohort} P_V6(rider in slot)
   observed count = sum 1(actual rider in slot)
Use all starters in denominators, including losers, never only podium
finishers; the rank/role is assigned pre-race from original data.
top3 is sum of 3 disjoint slot probabilities per rider.
No calibration correction, no model fitting on evaluation outcomes.
2023 Nov/Dec are temporal checks; 2024-2026 were already examined
by other research, NOT untouched holdout verification.
Line formation is the published pre-race forecast, not observed tactics.

Date-cluster residual-based standard error is a DIAGNOSTIC only. This is a
non-randomized multi-cell exploratory audit; avoid strong significance
conclusions from any one apparent opportunity or multiple comparisons.
No prices, odds, payout, return or probability floors.
"""
from __future__ import annotations
import json,math,sys
from collections import defaultdict,Counter
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
import validate_2025 as r25
import v4_pattern_floor_2026 as r26
import weak_rider_podium_zero_excluded as weak
from v8_targeted_weak_podium_calibration import row_probs
OUT=ROOT/"data/audits/v10_v6_rider_rank_role_slot_calibration_nozero_2023_2026.json"
YEARS={"2024":25348,"2025":24635,"2026":11797}
SLOTS=("first","second","third","top3")
AXES=("weak_all","weak_by_rank","weak_by_role","weak_by_rank_role",
      "weak_by_score_gap","weak_by_role_gap","weak_by_line_size",
      "weak_by_role_line_size","weak_by_field_size","weak_by_race_grade",
      "weak_by_line_shape","all_by_rank","all_by_role","all_riders")

def make_groups(z,e):
    c=[]
    rank=z["rank"];n=z["field_n"]
    if z["weak"]:
        c.extend([
          ("weak_all","lower_half"),
          ("weak_by_rank",str(rank)),
          ("weak_by_role",z["role"]),
          ("weak_by_rank_role",f'{rank}|{z["role"]}'),
          ("weak_by_score_gap",z["gap_bin"]),
          ("weak_by_role_gap",f'{z["role"]}|{z["gap_bin"]}'),
          ("weak_by_line_size",str(z["line_size"])),
          ("weak_by_role_line_size",f'{z["role"]}|{z["line_size"]}'),
          ("weak_by_field_size",str(n)),
          ("weak_by_race_grade",e.get("race_class") or "UNKNOWN"),
          ("weak_by_line_shape",weak.shape(e)),
        ])
    c.extend([("all_by_rank",str(rank)),
              ("all_by_role",z["role"]),
              ("all_riders","ALL")])
    return c

class Cell:
    __slots__=("exposures","actual","expected","pred_sq","day_residual","day_exposures")
    def __init__(self):
        self.exposures=0
        self.actual=[0.,0.,0.,0.]
        self.expected=[0.,0.,0.,0.]
        self.pred_sq=[0.,0.,0.,0.]
        self.day_residual=defaultdict(lambda:[0.,0.,0.,0.])
        self.day_exposures=defaultdict(int)
    def add(self,dt,p,actual):
        if len(p)!=4 or len(actual)!=4:
            raise AssertionError("Four positions required")
        self.exposures+=1
        self.day_exposures[dt]+=1
        d=self.day_residual[dt]
        for i in range(4):
            self.expected[i]+=p[i]
            self.actual[i]+=actual[i]
            self.pred_sq[i]+=p[i]**2
            d[i]+=actual[i]-p[i]
    def pack(self):
        n=self.exposures
        if not n:return {}
        out={"rider_exposures":n,"dates":len(self.day_exposures)}
        G=len(self.day_residual)
        for i,slot in enumerate(SLOTS):
            exp=self.expected[i];y=self.actual[i];diff=y-exp
            if G>1:
                m=diff/G
                var=G/(G-1)*sum((v[i]-m)**2 for v in self.day_residual.values())
                se=math.sqrt(max(0.,var))/n
                z=diff/n/se if se>0 else None
            else:se=None;z=None
            out[slot]={
              "observed":round(y,6),"expected":round(exp,6),
              "observed_pct":round(100*y/n,6),
              "predicted_pct":round(100*exp/n,6),
              "observed_minus_pred_pct_points":round(100*diff/n,6),
              "observed_to_expected":round(y/exp,6) if exp else None,
              "day_cluster_se_pct_points":round(100*se,6) if se is not None else None,
              "day_cluster_z_diagnostic":round(z,4) if z is not None else None,
              "direction":"underpredicted" if diff>0 else "overpredicted" if diff<0 else "matched"
            }
        return out

class Report:
    def __init__(self):
        self.cells=defaultdict(Cell)
        self.n_races=0
        self.exclusions=Counter()
        self.fields=Counter()
        self.score_mass_errors=0
        self.examples=defaultdict(list)
    def race(self,rid,e,dist):
        self.n_races+=1
        self.fields[len(e["scores"])]+=1
        n=len(e["scores"])
        m=defaultdict(lambda:[0.,0.,0.])
        for (a,b,c),p in dist.items():
            m[a][0]+=p
            m[b][1]+=p
            m[c][2]+=p
        if any(abs(sum(m[c][i] for c in e["scores"])-1)>1e-8
               for i in range(3)):
            self.score_mass_errors+=1
            raise AssertionError(("Slot probability normalization error",rid))
        dt=e["date"]
        for z in weak.score_features(e):
            c=z["car"];slotprob=m[c]
            p=slotprob+[sum(slotprob)]
            targets=[
                int(e["order"][0]==c),
                int(e["order"][1]==c),
                int(e["order"][2]==c),
                int(c in e["order"][:3])
            ]
            if abs(sum(p[:3])-p[3])>1e-8 or p[3]>1+1e-8:
                raise AssertionError("Rider top3 marginal invalid")
            for ax,grp in make_groups(z,e):
                self.cells[(ax,grp)].add(dt,p,targets)
    def pack(self):
        return {
          "eligible_races":self.n_races,
          "excluded_unknown_score_races":dict(self.exclusions),
          "field_sizes":dict(self.fields),
          "slot_sum_errors":self.score_mass_errors,
          "cells":{ax:{g:cell.pack()
                       for (a,g),cell in sorted(self.cells.items()) if a==ax}
                   for ax in AXES},
        }

def evaluation(events,wg,wr,d5,wl,wc,pw,tag):
    result=Report()
    for rid,e in events.items():
        reason=weak.score_gate(e)
        if reason!="ELIGIBLE":
            result.exclusions[reason]+=1
            continue
        p=row_probs(e,wg,wr,d5,wl,wc,pw)
        result.race(rid,e,p)
        if result.n_races%5000==0:
            w=result.cells["weak_all","lower_half"]
            print("V10_CENSUS_PROGRESS",tag,result.n_races,
                  "weak_3rd_observed",int(w.actual[2]),
                  "weak_3rd_expected",round(w.expected[2],3),flush=True)
    d=result.pack()
    d["period"]=tag
    d["source_races"]=len(events)
    if d["eligible_races"]+sum(d["excluded_unknown_score_races"].values())!=len(events):
        raise AssertionError("Source count corrupted")
    # Exactly N actual and N expected for each of the three slots in all-rider cohort.
    allr=d["cells"]["all_riders"]["ALL"]
    for slot in SLOTS[:3]:
        if abs(allr[slot]["observed"]-result.n_races)>1e-6 or abs(
               allr[slot]["expected"]-result.n_races)>1e-2:
            raise AssertionError(("Slot doesn't sum to one",tag,slot))
    w=d["cells"]["weak_all"]["lower_half"]
    print("V10_CENSUS_PERIOD_COMPLETE",json.dumps({
        "period":tag,"races":result.n_races,
        "excluded":d["excluded_unknown_score_races"],
        "weak":{
            s:{"actual_pct":w[s]["observed_pct"],
               "pred_pct":w[s]["predicted_pct"],
               "diff_pp":w[s]["observed_minus_pred_pct_points"],
               "day_z":w[s]["day_cluster_z_diagnostic"]}
            for s in SLOTS
        },
        "by_rank_role":{
            k:{"n":v["rider_exposures"],
               "first_diff_pp":v["first"]["observed_minus_pred_pct_points"],
               "second_diff_pp":v["second"]["observed_minus_pred_pct_points"],
               "third_diff_pp":v["third"]["observed_minus_pred_pct_points"]}
            for k,v in d["cells"]["weak_by_rank_role"].items()
        }
    },ensure_ascii=False),flush=True)
    return d

def recurring_patterns(year_results):
    """Only simple descriptive stability: sign same three years and >=300 rows each."""
    out=[]
    axes=("weak_by_rank_role","weak_by_role","weak_by_rank",
          "weak_by_role_gap","weak_by_score_gap")
    for ax in axes:
        labels=set.intersection(*(set(year_results[y]["cells"][ax]) for y in YEARS))
        for g in sorted(labels):
            for slot in SLOTS:
                d=[year_results[y]["cells"][ax][g] for y in YEARS]
                if min(x["rider_exposures"] for x in d)<300:continue
                signs=[x[slot]["observed_minus_pred_pct_points"] for x in d]
                if not (all(q>0 for q in signs) or all(q<0 for q in signs)):continue
                n=sum(x["rider_exposures"] for x in d)
                obs=sum(x[slot]["observed"] for x in d)
                ex=sum(x[slot]["expected"] for x in d)
                out.append({
                    "axis":ax,"group":g,"slot":slot,
                    "exposures_2024_2025_2026":[x["rider_exposures"] for x in d],
                    "errors_pct_points_by_year":signs,
                    "aggregate_error_pct_points":round(100*(obs-ex)/n,6),
                    "aggregate_observed_pct":round(100*obs/n,6),
                    "aggregate_predicted_pct":round(100*ex/n,6),
                    "direction":"underpredicted" if all(q>0 for q in signs) else "overpredicted",
                    "not_confirmed_out_of_sample":True
                })
    return sorted(out,key=lambda x:(x["axis"],x["group"],x["slot"]))

def main():
    all23,src23=obs.ingest()
    train={rid:e for rid,e in all23.items()
           if e["date"]<="2023-10-31" and weak.score_gate(e)=="ELIGIBLE"}
    dev={rid:e for rid,e in all23.items() if e["date"]>"2023-10-31"}
    assert len(train)==19017,("zero-free train drift",len(train))
    print("V10_TRAIN_BEGIN",len(train),flush=True)
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    print("V10_FROZEN_V6_TRAINED_NO_ZEROS",flush=True)
    del train,all23
    results={"2023_DEV":evaluation(dev,wg,wr,d5,wl,wc,pw,"2023_DEV")}
    del dev
    audits={"2023":src23}
    for yr,fn in (("2024",r24.ingest_2024),
                  ("2025",r25.read_2025),
                  ("2026",r26.ingest_2026)):
        events,src=fn()
        if len(events)!=YEARS[yr]:raise AssertionError((yr,len(events),YEARS[yr]))
        results[yr]=evaluation(events,wg,wr,d5,wl,wc,pw,yr)
        audits[yr]=src
        del events
    recurring=recurring_patterns(results)
    out={
      "state":"V10_V6_SCORE_RANK_LINE_ROLE_SLOT_CALIBRATION_COMPLETE",
      "question":"Does V6's actual pre-race probability for lower-scored riders match finishing position by score-rank, line-role and slot?",
      "baseline":"V6 post .15 V3 pattern gate trained on zero-free 2023 Jan-Oct",
      "retrained_2023_clean":"model architecture frozen, original model code untouched, recalculated on clean score subset",
      "race_level_zero_exclusion":True,
      "no_probability_correction_or_floor":True,
      "no_result_leakage_into_predictions":True,
      "no_market_odds_or_profit":True,
      "2024_2026_are_previously_inspected":True,
      "weak_rider_definition":"score rank > ceil(n/2), i.e. 7-car ranks 5-7, 9-car ranks 6-9",
      "score_ties":"tie order by ascending car number, not post-race results",
      "line_role_definition":"published pre-race line, not track actual",
      "ranking":"exact field score ranks, NOT V6 probability ranks",
      "all_trifecta_tickets_sum_to_one_per_race":True,
      "all_slot_marginals_normalized_per_race":True,
      "day_cluster_SE":"descriptive residual-based SE across calendar days, conditional on observed source days; not multiplicity corrected",
      "candidate_repeated_patterns_exploratory_only":recurring,
      "results_by_period":results,
      "source_audits":audits,
      "limitations":[
        "Multi-way rank and role splits produce small cells; a wide sample and stable repeat across years are necessary but insufficient for proven predictability.",
        "2024-2026 previously inspected; reporting persistent sign is discovery not truly independent confirmation.",
        "No hypothesis that all riders have same minimum probabilities.",
        "Calibration correctness does not imply odds mispricing or profit.",
        "2026 source grade may be unknown; grade-group conclusions can only be made on years with grade labels.",
        "Only fully completed and result matched race cards with published predicted lines, not all past competition races.",
        "Even when predicted counts match observed, rider-level forecasts can still be incorrect. Aggregation does not establish sharpness.",
        "Rank ties broken by car number for deterministic grouping, not evidence of strength."
      ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V10_FINAL_RESEARCH_RESULT",json.dumps({
      "all_years_weak":{yr:{
          slot:{
           "observed_pct":v["cells"]["weak_all"]["lower_half"][slot]["observed_pct"],
           "predicted_pct":v["cells"]["weak_all"]["lower_half"][slot]["predicted_pct"],
           "difference_pp":v["cells"]["weak_all"]["lower_half"][slot]["observed_minus_pred_pct_points"]
          } for slot in SLOTS} for yr,v in results.items()},
      "repeated_sign_count":len(recurring),
      "selected":recurring[:30]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

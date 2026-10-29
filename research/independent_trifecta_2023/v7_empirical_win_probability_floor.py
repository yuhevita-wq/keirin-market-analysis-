#!/usr/bin/env python3
"""V7 - quantify a candidate 'minimum chance of winning' in keirin.

Observation: every rider may retain a practically nonzero chance of winning
despite very low score/prediction. Distinguish 3 separate claims:
 (A) softmax scores are mathematically positive (trivial);
 (B) cohorts of extremely low-probability riders win more often than V6 thinks;
 (C) a *universal individual* positive lower bound can be established.
Only (B) is testable with currently available finite data; (C) is NOT.

Freeze existing V2/V3/V5/V6 training algorithms. Refit them using Jan-Oct
2023 only, and evaluate V6 P1 = P(selected first rider BEFORE V3 line-pattern
gate) on every actual starter. 2023 Nov-Dec is development; test 2024,
2025, and available 2026 are historical recurrence, previously inspected.

For a transparent NULL benchmark only, test a race-specific chance floor:
   q_i=(1-epsilon)*p_i + epsilon/n_riders.
This isn't an evidence-based tactical mechanism, and is not promoted
unless it improves 2023 dev proper-scoring NLL and Brier. Epsilon chosen
only on 2023 dev; 2024+ not used to select epsilon. If epsilon=0 the
hypothesis is not supported even if empirically low-p bins contain wins.
The uniform floor here is solely a falsifiable benchmark, *not* a
recommendation or an automatic replacement of V6.

Do NOT claim that V6 P1 matches V6 full-trifecta winner marginal:
V3's subsequent pattern gate may alter the marginal. This audit studies
the first-stage rider-choice head. 2024-2026 are retrospectively viewed
and not truly untouched confirmation.

No market odds, payout, betting tickets, favored formation, or model merge.
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
import calibration_audit as year24
import validate_2025 as year25
import v4_pattern_floor_2026 as year26

OUT=ROOT/"data/audits/empirical_rider_win_probability_floor_v7_2023_2026.json"
EPSILONS=(0.,.001,.003,.01,.03,.06,.10)
P_BINS=(
    (0.,.0001,"under_0.01pct"),
    (.0001,.0005,"0.01_to_0.05pct"),
    (.0005,.001,"0.05_to_0.1pct"),
    (.001,.005,"0.1_to_0.5pct"),
    (.005,.01,"0.5_to_1pct"),
    (.01,.02,"1_to_2pct"),
    (.02,.05,"2_to_5pct"),
    (.05,.1,"5_to_10pct"),
    (.1,.2,"10_to_20pct"),
    (.2,1.0000001,"20pct_plus"),
)
EXPECTED={"2023_DEV":4192,"2024":25348,"2025":24635,"2026":11797}
COVERED_CLASSES=("A1_A2","A3","S_F1","S_grade")

def bbin(p):
    for lo,hi,lab in P_BINS:
        if lo<=p<hi:return lab
    raise AssertionError(("prob bin missing",p))

def gapbin(x):
    if x<3:return "0to3"
    if x<6:return "3to6"
    if x<10:return "6to10"
    if x<15:return "10to15"
    if x<22:return "15to22"
    return "22plus"

def wilson(y,n):
    # Only marginal descriptive interval; rider exposures within a race
    # are correlated and the time series may cluster by venue/day.
    if n<=0:return None
    z=1.95996398454005
    p=y/n;v=z*z/n
    center=(p+v/2)/(1+v)
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+v)
    return [round(100*(center-delta),6),round(100*(center+delta),6)]

class Cell:
    def __init__(self):
        self.n=0;self.actual=0
        self.expected=0.;self.qexpected=0.
        self.races=set()
    def add(self,p,q,w,raceid):
        if not (0<=p<=1 and 0<=q<=1):raise AssertionError("invalid p")
        self.n+=1;self.actual+=int(w);self.expected+=p;self.qexpected+=q
        self.races.add(raceid)
    def pack(self):
        return {
           "rider_exposures":self.n,"unique_races":len(self.races),
           "observed_wins":self.actual,
           "V6_first_stage_expected_wins":round(self.expected,6),
           "candidate_expected_wins":round(self.qexpected,6),
           "observed_win_pct":round(self.actual/self.n*100,7) if self.n else None,
           "V6_mean_pred_pct":round(self.expected/self.n*100,7) if self.n else None,
           "candidate_mean_pred_pct":round(self.qexpected/self.n*100,7) if self.n else None,
           "observed_over_pred":round(self.actual/self.expected,6) if self.expected else None,
           "raw_wilson_95pct_descriptive":[*wilson(self.actual,self.n)] if self.n else None,
           "no_strict_individual_floor_identified":True,
        }

class Eval:
    def __init__(self,epsilons):
        self.n=0
        self.epsilons=tuple(epsilons)
        self.nll=defaultdict(float)
        self.brier=defaultdict(float)
        self.cells=defaultdict(Cell)
        self.groupcounts=Counter()
        self.minp_by_n=defaultdict(list)
        self.lowp_actual=defaultdict(int)
    def add(self,rid,e,priors):
        self.n+=1
        cars=sorted(e["scores"],key=int);n=len(cars)
        p=priors
        if set(p)!=set(cars) or abs(sum(p.values())-1)>1e-8:
            raise AssertionError(("first-stage distribution invalid",rid))
        truth=e["order"][0]
        if truth not in p:raise AssertionError("winner missing")
        selected={}
        for eps in self.epsilons:
            q={c:(1-eps)*p[c]+eps/n for c in cars}
            if abs(sum(q.values())-1)>1e-8:raise AssertionError("floor normalizer")
            selected[eps]=q
            self.nll[eps]-=math.log(max(q[truth],1e-300))
            self.brier[eps]+=1-2*q[truth]+sum(x*x for x in q.values())
        chosen=selected[self.epsilons[-1]]
        order=sorted(cars,key=lambda c:(p[c],int(c)))
        least=order[0]
        gap=max(e["scores"].values())-e["scores"][least]
        grade=e.get("race_class","UNKNOWN")
        self.minp_by_n[n].append(p[least])
        self.groupcounts[v2.line_shape(e)]+=1
        for c in cars:
            rank_by_score=sorted(cars,key=lambda x:(-e["scores"][x],int(x))).index(c)+1
            role=v2.rlabel(e,c)
            line_size=e["positions"][c][2]
            owned=max(e["scores"][x] for x in e["groups"][e["positions"][c][0]-1])
            gap_to_best=max(e["scores"].values())-e["scores"][c]
            won=int(c==truth)
            groups=(
                ("all_candidates","all"),
                ("V6_pred_bin",bbin(p[c])),
                ("score_rank","last" if rank_by_score==n else "not_last"),
                ("V6_prediction_order","least" if c==least else ("second_least" if c==order[1] else "other")),
                ("score_rank4plus","yes" if rank_by_score>=4 else "no"),
                ("score_rank4plus_by_role",role if rank_by_score>=4 else "not_rank4plus"),
                ("score_rank4plus_by_field_size",str(n) if rank_by_score>=4 else "not_rank4plus"),
            )
            for axis,key in groups:
                self.cells[axis,key].add(p[c],chosen[c],won,rid)
            if c==least:
                extra=(
                    ("least_by_field_size",str(n)),
                    ("least_by_role",role),
                    ("least_by_line_size",str(line_size)),
                    ("least_by_score_gap",gapbin(gap)),
                    ("least_by_grade",grade),
                    ("least_by_grade_and_gap",grade+"|"+gapbin(gap)),
                    ("least_by_own_line_gap",gapbin(owned-e["scores"][c])),
                    ("least_by_probability_band",bbin(p[c])),
                )
                for axis,key in extra:
                    self.cells[axis,key].add(p[c],chosen[c],won,rid)
            if p[c]<.01 and won:self.lowp_actual["under_1pct"]+=1
            if p[c]<.001 and won:self.lowp_actual["under_0.1pct"]+=1
            if p[c]<.0001 and won:self.lowp_actual["under_0.01pct"]+=1
    def pack(self):
        return {
         "races":self.n,
         "first_stage_V6_NLL":round(self.nll[0.]/self.n,9),
         "first_stage_V6_Brier":round(self.brier[0.]/self.n,9),
         "epsilon_scores":{str(a):{
              "mean_NLL":round(self.nll[a]/self.n,9),
              "mean_Brier":round(self.brier[a]/self.n,9),
              "floor_pct_for_7_riders":round(100*a/7,8),
              "floor_pct_for_9_riders":round(100*a/9,8),
            } for a in self.epsilons},
         "all_line_shapes":dict(sorted(self.groupcounts.items())),
         "distinct_line_shapes":len(self.groupcounts),
         "raw_counts":dict(self.lowp_actual),
         "by_axis":{
            ax:{lab:z.pack() for (axis,lab),z in sorted(self.cells.items()) if axis==ax}
            for ax in sorted({axis for axis,lab in self.cells})},
         "least_predicted_one_per_race_by_field_size":{
            str(n):{
               "min":round(min(v),11),
               "max":round(max(v),11),
               "mean":round(sum(v)/len(v),11),
               "races":len(v),
            } for n,v in sorted(self.minp_by_n.items())},
        }

def main():
    whole,src=obs.ingest()
    train={k:v for k,v in whole.items() if v["date"]<="2023-10-31"}
    dev={k:v for k,v in whole.items() if v["date"]>"2023-10-31"}
    assert (len(train),len(dev))==(19248,4192)
    print("V7_TRAIN_FROZEN_HEADS",flush=True)
    wg,wr=v2.fit_structural(train)
    delta5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,delta5)
    del train,whole
    result={}
    audits={"2023_DEV":src}
    for yr,readfn in (("2023_DEV",lambda:(dev,{"previously_inspected":True})),
                     ("2024",year24.ingest_2024),
                     ("2025",year25.read_2025),
                     ("2026",year26.ingest_2026)):
        events,cov=readfn()
        if len(events)!=EXPECTED[yr]:
            raise AssertionError(("cohort drift",yr,len(events)))
        er=Eval(EPSILONS)
        for k,(rid,e) in enumerate(events.items(),1):
            p=v6.choice(e,(),wg,wr,delta5,wl,wc,1.,"V6")
            er.add(rid,e,p)
            if k%5000==0:
                print("V7_ALL_RIDERS",yr,k,
                      "least_win",er.cells["V6_prediction_order","least"].actual,flush=True)
        result[yr]=er.pack()
        audits[yr]=cov
        z=result[yr]["by_axis"]["V6_prediction_order"]["least"]
        print("V7_ONE_YEAR_FINISHED",json.dumps({
            "year":yr,"races":len(events),"least_rider":{
                k:v for k,v in z.items() if k in
                    ("rider_exposures","observed_wins","V6_first_stage_expected_wins",
                     "observed_win_pct","V6_mean_pred_pct","raw_wilson_95pct_descriptive")},
            "NLLs":{eps:o["mean_NLL"] for eps,o in result[yr]["epsilon_scores"].items()},
            "rare_bins":{k:v for k,v in result[yr]["by_axis"]["V6_pred_bin"].items()
                         if k in ("under_0.01pct","0.01_to_0.05pct",
                                  "0.05_to_0.1pct","0.1_to_0.5pct","0.5_to_1pct")},
        },ensure_ascii=False),flush=True)
        del events

    devs=result["2023_DEV"]["epsilon_scores"]
    v3=devs["0.0"]
    candidates=[a for a in EPSILONS if a>0 and
                devs[str(a)]["mean_NLL"]<v3["mean_NLL"]-1e-7 and
                devs[str(a)]["mean_Brier"]<=v3["mean_Brier"]+1e-8]
    chosen=min(candidates,key=lambda x:(devs[str(x)]["mean_NLL"],x)) if candidates else 0.
    for y in ("2024","2025","2026"):
        cand=result[y]["epsilon_scores"][str(chosen)]
        baseline=result[y]["epsilon_scores"]["0.0"]
        result[y]["candidate_selected_2023_only"]={
            "epsilon":chosen,"minimum_7_rider_p1_pct":round(100*chosen/7,8),
            "minimum_9_rider_p1_pct":round(100*chosen/9,8),
            "NLL_change_V6_minus_floor":round(baseline["mean_NLL"]-cand["mean_NLL"],9),
            "Brier_change_V6_minus_floor":round(baseline["mean_Brier"]-cand["mean_Brier"],9),
        }
    doc={
      "state":"PRACTICAL_MINIMUM_RIDER_FIRST_STAGE_WIN_PROBABILITY_AUDIT_COMPLETE",
      "research_question":"How often do the lowest-predicted riders win and is there evidence for a probability floor?",
      "model":"V6 frozen first-stage rider selection before subsequent V3 pattern gate",
      "model_fit_period":"2023-01 through 2023-10 ONLY",
      "development_for_epsilon":"2023-11 and 2023-12, previously inspected",
      "historical_recurrence":"2024/2025/2026, all previously inspected",
      "selected_uniform_floor_benchmark_epsilon_2023_only":chosen,
      "uniform_floor_benchmark_is_not_adopted":True,
      "no_ticket_or_payout_or_market_data":True,
      "no_line_shape_preference":True,
      "no_modification_of_V3_V5_V6":True,
      "universal_positive_floor_per_individual_not_identifiable_from_finite_data":True,
      "effective_floor_estimated_only_for_conditioned_groups":True,
      "no_claim_of_inevitability":True,
      "2026_grade_unknown":True,
      "limitations":[
        "Lowest predicted rider per race: one observation per race, but same-day outcomes and competitor effects remain correlated.",
        "Wilson intervals are descriptive marginal intervals, not day-cluster-robust nor multiple-comparison corrected.",
        "Observed nonzero winning frequency in a cohort does not imply every individual member's true win probability exceeds that frequency.",
        "V6 P1 is before V3 non-AAA pattern gate; full 3ren-tan probability first-marginal can differ.",
        "Floor benchmark blends uniform first-choice probability; it is a competing hypothesis, not an established tactical shock process.",
        "Historic 2024-2026 outcomes already inspected: no independent unseen acceptance claims.",
        "2026 sample incomplete and lacks trusted race-class labels.",
        "No odds, market, payout, profits, production promotion or automatically purchased tickets."
      ],
      "years":result,
      "coverage":audits
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V7_WIN_FLOOR_CENSUS_COMPLETE",json.dumps({
        "dev_floor_epsilon":chosen,
        "per_year":{
            y:{"least":v["by_axis"]["V6_prediction_order"]["least"],
               "epsilon_comparison":v.get("candidate_selected_2023_only",{}),
               "weak_rank4plus":v["by_axis"]["score_rank4plus"]["yes"]}
            for y,v in result.items()}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()

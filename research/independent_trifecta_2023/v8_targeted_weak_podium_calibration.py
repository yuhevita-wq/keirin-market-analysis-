#!/usr/bin/env python3
"""V8 targeted, score-verified, lower-half 1/2/3 place calibration.

Question: Can correcting the *specific underpredicted lower-scored rider/slot
cohorts* improve full-race V6 ordered-triple likelihood, not merely increase
their own hit count?

RACE-LEVEL exclusion: any score=0, nonfinite, missing or nonpositive -> NONE
of that race contributes to any fit/evaluation. Original raw data untouched.
Fit baseline V2/V3/V5/V6 ALGORITHMS on CLEAN 2023 Jan-Oct, unlike historical
V6 baseline which trained with zero scores. Calibrate only 2023 November,
evaluate 2023 December and previously inspected 2024/25/26. No result leakage.

Slot-specific subcohorts = pre-race bottom-half rider, role (leader/second/
third/solo/fourth_plus) and deficit to strongest score (<3/3-6/6-10/
10-15/15-22/22+). Calibration is only UPWARD for cells significantly
underpredicted in November, minimum 60 exposures/3 predicted hits.
No floor imposed on all low-scored riders, and no correction derived from
future results. The correction uses a log-linear tilt:
  Q(a,b,c) = P_V6(a,b,c)*exp(alpha*(d1(a)+d2(b)+d3(c))) / Z
Z is EXACT all-ticket sum, so all 1/2/3 marginals remain compatible and
the total ordered-triple probability is exactly one.
Comparison alpha=0/.25/.5/1.0 is a report-only sensitivity, alpha=.5 is
PRE-FIXED candidate, no post-hoc selection on 2024/25/26.

Calibrator operates only on known pre-race score card features.
It corrects neither UNKNOWN race outcomes nor market price residuals.
No odds, payouts or ROI used. Historical data is already inspected; no
genuinely independent unseen-period or causal claim.
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
import validate_2025 as r25
import v4_pattern_floor_2026 as r26
import weak_rider_podium_zero_excluded as weak

OUT=ROOT/"data/audits/v8_targeted_low_rider_podium_calibration_zero_excluded_2023_2026.json"
EXPECTED={"2024":25348,"2025":24635,"2026":11797}
ALPHAS=(0.,.25,.5,1.)
CANDIDATE_ALPHA=.5
CAL_MIN_EXPOSURES=60
CAL_MIN_EXPECTED=3.
CAL_MIN_OBSERVED=3
CAL_MIN_SURPLUS_Z=1.
CAL_SMOOTH=20.
CAL_MAX_RATIO=2.
MINP=1e-300

def features(e):
    return {z["car"]:z for z in weak.score_features(e)}

def key(z,slot):
    return (slot,z["role"],z["gap_bin"])

def row_probs(e,wg,wr,d5,wl,wc,pw):
    """Full ordered triple V6 distribution, keeping its V3=.15 pattern gate."""
    cars=sorted(e["scores"],key=int)
    ch0=v6.choice(e,(),wg,wr,d5,wl,wc,1.,"V6")
    ch1={a:v6.choice(e,(a,),wg,wr,d5,wl,wc,1.,"V6") for a in cars}
    ch2={(a,b):v6.choice(e,(a,b),wg,wr,d5,wl,wc,1.,"V6")
          for a in cars for b in cars if a!=b}
    raw={(a,b,c):ch0[a]*ch1[a][b]*ch2[a,b][c]
          for a in cars for b in cars if a!=b for c in cars if c!=a and c!=b}
    if abs(sum(raw.values())-1)>1e-8:
        raise AssertionError("V6 initial ticket probability conservation")
    gate=v3.predict_non_aaa(e,pw)
    gated,_=v3.variants(e,raw,gate)
    dist=gated[.15][0]
    if abs(sum(dist.values())-1)>1e-8:raise AssertionError("V6 V3 gate nonnormal")
    return dist

def marginals(dist):
    p=defaultdict(float)
    for triple,prob in dist.items():
        for k,car in enumerate(triple,1):p[k,car]+=prob
    return p

def fit_cells(data,wg,wr,d5,wl,wc,pw):
    accum=defaultdict(lambda:[0,0.,0])
    n=0
    for rid,e in data.items():
        dist=row_probs(e,wg,wr,d5,wl,wc,pw)
        p=marginals(dist)
        f=features(e)
        for c,z in f.items():
            if not z["weak"]:continue
            for slot in (1,2,3):
                a=accum[key(z,slot)]
                a[0]+=1
                a[1]+=p[slot,c]
                a[2]+=int(c==e["order"][slot-1])
        n+=1
        if n%1000==0:print("V8_CAL_TRAIN_RACES",n,flush=True)
    deltas={}
    cohort=[]
    for k,(cnt,pred,y) in sorted(accum.items()):
        selected=(cnt>=CAL_MIN_EXPOSURES and pred>=CAL_MIN_EXPECTED
                  and y>=CAL_MIN_OBSERVED and y>pred+CAL_MIN_SURPLUS_Z*math.sqrt(pred))
        ratio=(y+CAL_SMOOTH)/(pred+CAL_SMOOTH)
        delta=(min(math.log(CAL_MAX_RATIO),max(0.,math.log(ratio))) if selected else 0.)
        deltas[k]=delta
        cohort.append({"slot":k[0],"role":k[1],"gap_bin":k[2],
                       "exposures":cnt,"v6_expected":round(pred,5),
                       "actual":y,"ratio":round(y/pred,5) if pred else None,
                       "selected":bool(delta>0),"delta":round(delta,6)})
    return deltas,cohort,n

def tilt(dist,f,deltas,alpha):
    if not alpha:return dist
    weights={}
    for t,p in dist.items():
        additive=0.
        for i,c in enumerate(t,1):
            z=f[c]
            if z["weak"]:additive+=deltas.get(key(z,i),0.)
        weights[t]=p*math.exp(alpha*additive)
    norm=sum(weights.values())
    q={t:x/norm for t,x in weights.items()}
    if abs(sum(q.values())-1)>1e-9:raise AssertionError("Nonconserved mass")
    return q

def evaluate(data,wg,wr,d5,wl,wc,pw,deltas,tag):
    score={str(a):{"nll":0.,"brier":0.,"weak_predicted":[0.,0.,0.],
                   "weak_actual":[0,0,0],"selected_predicted":0.,
                   "selected_actual":0,
                   "selected_expected_by_slot":[0.,0.,0.],
                   "selected_actual_by_slot":[0,0,0],
                   "tail_expected":0.,"tail_actual":0}
           for a in ALPHAS}
    n=0; zero=0
    for rid,e in data.items():
        if weak.score_gate(e)!="ELIGIBLE":
            zero+=1
            continue
        dist=row_probs(e,wg,wr,d5,wl,wc,pw)
        f=features(e)
        actual=e["order"]
        tailkeys={t for t,p in dist.items() if p<.0001}
        tailwin=actual in tailkeys
        for a in ALPHAS:
            q=tilt(dist,f,deltas,a)
            z=score[str(a)]
            n+=int(a==0.)
            z["nll"]-=math.log(max(MINP,q[actual]))
            brier=1.-2*q[actual]+sum(x*x for x in q.values())
            z["brier"]+=brier
            marginal=marginals(q)
            for c,v in f.items():
                if not v["weak"]:continue
                for slot in (1,2,3):
                    z["weak_predicted"][slot-1]+=marginal[slot,c]
                    z["weak_actual"][slot-1]+=int(actual[slot-1]==c)
                    if deltas.get(key(v,slot),0.)>0:
                        z["selected_predicted"]+=marginal[slot,c]
                        z["selected_actual"]+=int(actual[slot-1]==c)
                        z["selected_expected_by_slot"][slot-1]+=marginal[slot,c]
                        z["selected_actual_by_slot"][slot-1]+=int(actual[slot-1]==c)
            z["tail_expected"]+=sum(q[t] for t in tailkeys)
            z["tail_actual"]+=int(tailwin)
        if n%5000==0:
            print("V8_YEAR_PROGRESS",tag,n,"candidate_NLL",round(score[str(CANDIDATE_ALPHA)]["nll"]/n,6),flush=True)
    if not n:raise ValueError("no usable test races")
    report={}
    for a,z in score.items():
        report[a]={
          "races":n,
          "full_trifecta_NLL":round(z["nll"]/n,9),
          "full_trifecta_Brier":round(z["brier"]/n,9),
          "weak_podium_predicted_by_slot":[round(v,4) for v in z["weak_predicted"]],
          "weak_podium_actual_by_slot":z["weak_actual"],
          "identified_underprediction_cohorts_predicted":round(z["selected_predicted"],4),
          "identified_underprediction_cohorts_actual":z["selected_actual"],
          "identified_cohorts_by_slot":{
               str(i+1):{"expected":round(z["selected_expected_by_slot"][i],4),
                         "actual":z["selected_actual_by_slot"][i]} for i in range(3)},
          "fixed_baseline_tiny_ticket_group_expected":round(z["tail_expected"],5),
          "fixed_baseline_tiny_ticket_group_actual":z["tail_actual"],
        }
    base=report[str(0.)]
    b=report[str(CANDIDATE_ALPHA)]
    result={
        "period":tag,"source_races":len(data),
        "score_eligible_races":n,"excluded_unknown_score_races":zero,
        "evaluations":report,
        "predeclared_half_tilt_candidate":{
            "alpha":CANDIDATE_ALPHA,
            "NLL_improvement_positive_is_better":round(base["full_trifecta_NLL"]-b["full_trifecta_NLL"],9),
            "Brier_improvement_positive_is_better":round(base["full_trifecta_Brier"]-b["full_trifecta_Brier"],9),
            "identified_cohort_expected_before":base["identified_underprediction_cohorts_predicted"],
            "identified_cohort_expected_after":b["identified_underprediction_cohorts_predicted"],
            "identified_cohort_actual":b["identified_underprediction_cohorts_actual"],
            "pass":bool(b["full_trifecta_NLL"]<base["full_trifecta_NLL"]
                       and b["full_trifecta_Brier"]<=base["full_trifecta_Brier"]),
        }}
    if n+zero!=len(data):raise AssertionError("Race exclusion conservation")
    print("V8_ONE_PERIOD_FINISHED",json.dumps({
        "tag":tag,"races":n,"excluded":zero,"candidate":result["predeclared_half_tilt_candidate"],
        "baseline":base,"tilted":b},ensure_ascii=False),flush=True)
    return result

def main():
    events,source=r23.ingest()
    rawtrain={k:v for k,v in events.items() if v["date"]<="2023-10-31"}
    cleantrain={k:v for k,v in rawtrain.items() if weak.score_gate(v)=="ELIGIBLE"}
    november={k:v for k,v in events.items() if v["date"].startswith("2023-11")
              and weak.score_gate(v)=="ELIGIBLE"}
    december={k:v for k,v in events.items() if v["date"].startswith("2023-12")
              and weak.score_gate(v)=="ELIGIBLE"}
    if len(rawtrain)!=19248 or len(cleantrain)!=19017:raise AssertionError("2023 source drift")
    print("V8_CLEAN_TRAIN_V2",len(cleantrain),flush=True)
    wg,wr=v2.fit_structural(cleantrain)
    pw,_=v3.train_pattern_gate(cleantrain)
    d5=v5.fit_residual(cleantrain,wg,wr)
    wl,wc=v6.fit_opponent(cleantrain,wg,wr,d5)
    print("V8_CLEAN_TRAIN_COMPLETED",flush=True)
    del rawtrain,cleantrain,events
    deltas,cohorts,cal_n=fit_cells(november,wg,wr,d5,wl,wc,pw)
    chosen=[x for x in cohorts if x["selected"]]
    print("V8_UNDERPREDICTED_COHORTS_SELECTED",json.dumps({
         "calibration_races":cal_n,"total_cells":len(cohorts),
         "selected":chosen},ensure_ascii=False),flush=True)
    periods={"2023_DEC":evaluate(december,wg,wr,d5,wl,wc,pw,deltas,"2023_DEC_AFTER_2023_NOV_FIT")}
    del november,december
    source_audits={}
    for year,reader in (("2024",r24.ingest_2024),("2025",r25.read_2025),("2026",r26.ingest_2026)):
        data,audit=reader()
        if len(data)!=EXPECTED[year]:raise AssertionError((year,len(data),EXPECTED[year]))
        periods[year]=evaluate(data,wg,wr,d5,wl,wc,pw,deltas,year)
        source_audits[year]=audit
        del data
    report={
      "state":"V8_TARGETED_LOWER_HALF_PODIUM_CALIBRATION_EVALUATED",
      "source":"published pre-race line predictions, score, actual top3 results",
      "baseline_training":"2023-01-01 to 2023-10-31, exclusively verified positive scores",
      "calibration_fitting":"2023-11 only, verified scores",
      "calibration_selection_rule":"weak-only pre-race role+score-gap-by-slot, >=60 candidates and >=3 expected/actual wins, and observed>expected+sqrt(expected), uplift ratio smoothed 20, capped 2x",
      "development_evaluation":"2023-12, no fitting of correction",
      "historical_replay":"2024-2026 already inspected, no independent untouched validation claim",
      "probability_model":"Frozen V6 and V3 gate trained on zero-excluded 2023 Jan-Oct, every triple normalized; correction exponential marginal slot tilt",
      "candidate_alpha_fixed_before_evaluation":CANDIDATE_ALPHA,
      "no_zero_score_race_in_fit_or_evaluation":True,
      "no_payout_market_or_odds_inputs":True,
      "does_not_modify_existing_haruq_prediction":True,
      "all_triple_probabilities_normalized":True,
      "calibration_races_2023_nov":cal_n,
      "calibration_candidate_cells":cohorts,
      "calibration_selected_count":len(chosen),
      "periods":periods,
      "source_audits":source_audits,
      "limitations":[
         "Historical 2024-2026 data were previously observed by the researcher; do not claim external novelty.",
         "No model adoption unless full-ticket NLL and Brier both improve, not merely weak-rider captured outcomes.",
         "Selected cohorts are based on 2023 November residuals; small calibration period and correlated riders risk overfitting.",
         "Weak rider defined by relative score-rank lower half; not an independently validated measure of true ability.",
         "V6 baseline is retrained on score-verified historical races, so its scores will not equal earlier zero-contaminated V6 published figures.",
         "Only observed complete published predicted lines; cannot generalize to excluded race population.",
         "The ticket-probability adjustments alter ALL slot marginals; no independent per-rider floor is imposed."
      ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V8_TARGETED_PODIUM_CALIBRATION_ALL_PERIODS_COMPLETE",json.dumps({
        "november_calibration_n":cal_n,"selected":chosen,
        "year_results":{k:v["predeclared_half_tilt_candidate"] for k,v in periods.items()},
        "excluded":{k:v["excluded_unknown_score_races"] for k,v in periods.items()}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":main()

#!/usr/bin/env python3
"""V9: 'Unexplained' winning/finishing possibilities as race-level mixture.

Axiom UNDER TEST, not a fact: each known-score rider retains some chance to
arrive regardless of observable skill/line context. This is epistemic
uncertainty, not a universal measured minimum individual success probability.

Q(triple) = (1-epsilon)*P_V6(triple) + epsilon*R(triple), evaluated on the
FULL 3ren-tan state-space, exactly normalized. NO probability 'bonuses'
outside this mixture. Compare two precisely specified shock mechanisms:
  whole-race-uniform : R(t)=1/(N*(N-1)*(N-2)); all slots uncertain.
  uniform-winner-only : R(a,b,c)=P_V6(b,c|a)/N,
                       random first finisher with baseline remaining order.
Both give every rider a winner probability of at least epsilon/N, but not
a universal 2nd/3rd floor for the winner-only mode. They do NOT assert
physical tactics or necessary causes.

Exclude ENTIRE races if any score <=0, nonfinite, invalid or missing.
Retrain frozen structural V2/V3/V5/V6 ALGORITHMS on only Jan-Oct 2023
zero-free events. 2023 Nov is DEVELOPMENT and selects mode+epsilon by full-
trifecta NLL with no Brier worsening. No further adjustments. 2023 Dec
is held until after selection; 2024/25/26 are retrospective historical
replications already seen in other research, NOT pristine blind holdouts.
The 2023-11 selection itself may be unstable at sample size ~2049 races.

Proper-scoring NLL and multiclass Brier account for the *entire* race
probability allocation. The uniform modes are negative-control hypotheses,
not evidence of true randomness. No odds, payout, market support, ROI or
operational production model modification.
"""
from __future__ import annotations
import math,json,sys
from collections import defaultdict
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

OUT=ROOT/"data/audits/v9_unexplained_probability_mixture_nozero_2023_2026.json"
EPS=(0.,.001,.003,.01,.03,.06,.10,.20)
MODES=("whole-race-uniform","uniform-winner-only")
EXPECTED={"2024":25348,"2025":24635,"2026":11797}

class Scores:
    def __init__(self):
        self.n=0
        self.m={mode:{str(e):{
            "nll":0.,"brier":0.,"ptrue":0.,
            "weak_expected":[0.,0.,0.],
            "weak_actual":[0,0,0],
            "least_winner_expected":0.,"least_winner_actual":0,
            "rare_baseline_ticket_expected":0.,"rare_baseline_ticket_actual":0,
         } for e in EPS} for mode in MODES}
    def add(self,e,p):
        n=len(e["scores"])
        k=n*(n-1)*(n-2)
        if len(p)!=k or abs(sum(p.values())-1)>1e-8:
            raise AssertionError(("probability count/mass invalid",n,len(p),sum(p.values())))
        t=tuple(e["order"][:3])
        if t not in p:raise AssertionError("Actual top 3 absent")
        cars=sorted(e["scores"],key=lambda car:(-e["scores"][car],int(car)))
        weakset=set(cars[math.ceil(n/2):])
        least=cars[-1]
        p1=defaultdict(float)
        base_weak=[0.,0.,0.]
        base_rare=0.
        for ticket,prob in p.items():
            p1[ticket[0]]+=prob
            if prob<.0001:base_rare+=prob
            for i,c in enumerate(ticket):
                if c in weakset:base_weak[i]+=prob
        if min(p1.values())<=0:raise AssertionError("baseline first marginal zero")
        psq=sum(x*x for x in p.values())
        base_least=p1[least]
        actual_weak=[int(c in weakset) for c in t]
        actual_rare=int(p[t]<.0001)
        # Compute all moment/marginal statistics ONCE; mix analytically
        # for each epsilon. This preserves full-ticket Brier exactly.
        shocks={
          "whole-race-uniform":{t:1./k for t in p},
          "uniform-winner-only":{t:prob/(n*p1[t[0]]) for t,prob in p.items()}
        }
        for mode,r in shocks.items():
            if abs(sum(r.values())-1)>1e-8:raise AssertionError("Shock mass not one")
            rsq=0.;dot=0.;shock_weak=[0.,0.,0.]
            shock_least=0.;shock_rare=0.
            for xyz,prob in p.items():
                rprob=r[xyz]
                rsq+=rprob*rprob
                dot+=prob*rprob
                if xyz[0]==least:shock_least+=rprob
                if prob<.0001:shock_rare+=rprob
                for i,c in enumerate(xyz):
                    if c in weakset:shock_weak[i]+=rprob
            for epsilon in EPS:
                z=self.m[mode][str(epsilon)]
                qtrue=(1-epsilon)*p[t]+epsilon*r[t]
                qsq=(1-epsilon)**2*psq+2*epsilon*(1-epsilon)*dot+epsilon**2*rsq
                z["nll"]-=math.log(max(1e-300,qtrue))
                z["brier"]+=1-2*qtrue+qsq
                z["ptrue"]+=qtrue
                for i in range(3):
                    z["weak_expected"][i]+=(1-epsilon)*base_weak[i]+epsilon*shock_weak[i]
                    z["weak_actual"][i]+=actual_weak[i]
                z["least_winner_expected"]+=(1-epsilon)*base_least+epsilon*shock_least
                z["least_winner_actual"]+=int(t[0]==least)
                z["rare_baseline_ticket_expected"]+=(1-epsilon)*base_rare+epsilon*shock_rare
                z["rare_baseline_ticket_actual"]+=actual_rare
        self.n+=1
    def pack(self):
        if not self.n:raise AssertionError("zero racing observations")
        return {
            "n_races":self.n,
            "by_mode":{
                mode:{
                    k:{
                      "nll":round(s["nll"]/self.n,10),
                      "brier":round(s["brier"]/self.n,10),
                      "mean_correct_ticket_probability":round(s["ptrue"]/self.n,10),
                      "weak_rider_slot_expected":[round(x,4) for x in s["weak_expected"]],
                      "weak_rider_slot_actual":s["weak_actual"],
                      "least_rank_rider_first_expected":round(s["least_winner_expected"],4),
                      "least_rank_rider_first_actual":s["least_winner_actual"],
                      "low_V6_prob_ticket_group_expected":round(s["rare_baseline_ticket_expected"],4),
                      "low_V6_prob_ticket_group_actual":s["rare_baseline_ticket_actual"]
                    } for k,s in self.m[mode].items()
                } for mode in MODES
            }
        }

def run_period(events,wg,wr,d5,wl,wc,pw,label):
    audit=defaultdict(int)
    score=Scores()
    for rid,e in events.items():
        gate=weak.score_gate(e)
        if gate!="ELIGIBLE":
            audit[gate]+=1
            continue
        p=row_probs(e,wg,wr,d5,wl,wc,pw)
        score.add(e,p)
        if score.n%5000==0:
            b=score.m[MODES[0]]["0.0"]
            s=score.m[MODES[0]]["0.03"]
            print("V9_EVALUATION_PROGRESS",label,score.n,
                  "V6_NLL",round(b["nll"]/score.n,7),
                  "uniform3pct_NLL",round(s["nll"]/score.n,7),flush=True)
    report=score.pack()
    report["period"]=label
    report["source_races"]=len(events)
    report["excluded_unknown_score_races"]=dict(audit)
    report["eligible_races"]=score.n
    if score.n+sum(audit.values())!=len(events):raise AssertionError("exclusion mismatch")
    ref=report["by_mode"][MODES[0]]["0.0"]
    if abs(report["by_mode"][MODES[1]]["0.0"]["nll"]-ref["nll"])>1e-9:
        raise AssertionError("different baseline distributions")
    print("V9_PERIOD_COMPLETE",json.dumps({
      "period":label,"n":score.n,"excluded":dict(audit),
      "baseline":{k:ref[k] for k in ("nll","brier","weak_rider_slot_expected","weak_rider_slot_actual")},
      "all_competitors":{
         m:{e:{"nll_delta":round(ref["nll"]-q["nll"],10),
               "brier_delta":round(ref["brier"]-q["brier"],10)}
            for e,q in report["by_mode"][m].items()}
         for m in MODES}
    },ensure_ascii=False),flush=True)
    return report

def choose(dev):
    base=dev["by_mode"][MODES[0]]["0.0"]
    valid=[]
    for mode in MODES:
        for e in EPS:
            s=dev["by_mode"][mode][str(e)]
            if s["nll"]<base["nll"]-1e-9 and s["brier"]<=base["brier"]+1e-9:
                valid.append((s["nll"],e,mode))
    if valid:
        _,epsilon,mode=min(valid)
        return {"mode":mode,"epsilon":epsilon,"nonzero_candidate":True}
    return {"mode":MODES[0],"epsilon":0.,"nonzero_candidate":False}

def comparison(report,choice):
    base=report["by_mode"][MODES[0]]["0.0"]
    cand=report["by_mode"][choice["mode"]][str(choice["epsilon"])]
    return {
       "n_races":report["n_races"],
       "baseline_NLL":base["nll"],"candidate_NLL":cand["nll"],
       "NLL_gain":round(base["nll"]-cand["nll"],10),
       "baseline_Brier":base["brier"],"candidate_Brier":cand["brier"],
       "Brier_gain":round(base["brier"]-cand["brier"],10),
       "passes_both":cand["nll"]<base["nll"]-1e-9 and
           cand["brier"]<=base["brier"]+1e-9,
       "lowest_score_winner_actual":base["least_rank_rider_first_actual"],
       "lowest_score_winner_expected_before":base["least_rank_rider_first_expected"],
       "lowest_score_winner_expected_after":cand["least_rank_rider_first_expected"],
       "lower_half_first_second_third_actual":base["weak_rider_slot_actual"],
       "lower_half_first_second_third_expected_before":base["weak_rider_slot_expected"],
       "lower_half_first_second_third_expected_after":cand["weak_rider_slot_expected"],
    }

def main():
    all23,audit23=obs.ingest()
    train={k:e for k,e in all23.items() if e["date"]<="2023-10-31"
           and weak.score_gate(e)=="ELIGIBLE"}
    november={k:e for k,e in all23.items() if e["date"].startswith("2023-11")}
    december={k:e for k,e in all23.items() if e["date"].startswith("2023-12")}
    if len(train)!=19017:raise AssertionError(("2023 clean training drift",len(train)))
    print("V9_FIXED_V6_TRAINING_BEGIN",len(train),flush=True)
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    del train,all23
    print("V9_FIXED_V6_TRAINING_DONE",flush=True)
    dev=run_period(november,wg,wr,d5,wl,wc,pw,"2023_NOV_CHOICE")
    selection=choose(dev)
    print("V9_SELECTED_BY_NOVEMBER_ONLY",json.dumps(selection),flush=True)
    dec=run_period(december,wg,wr,d5,wl,wc,pw,"2023_DEC_CHRONOLOGICAL")
    del november,december
    periods={"2023_NOV_DEVELOPMENT":dev,"2023_DEC_CHECK":dec}
    sources={"2023":audit23}
    for y,loader in (("2024",r24.ingest_2024),
                     ("2025",r25.read_2025),
                     ("2026",r26.ingest_2026)):
        data,audit=loader()
        if len(data)!=EXPECTED[y]:raise AssertionError(("source count drift",y,len(data)))
        periods[y]=run_period(data,wg,wr,d5,wl,wc,pw,y)
        sources[y]=audit
        del data
    tests={k:comparison(v,selection) for k,v in periods.items()}
    # Honest acceptance: one-size random shock may improve NLL without Brier;
    # do not automatically claim adoption even if all four inspected sets pass.
    doc={
      "state":"V9_UNEXPLAINED_CHANCE_MIXTURE_COMPLETE",
      "hypothesis":"Some winner/order outcomes are not explained by measured rider skill or pre-race line context.",
      "not_asserting_true_uniform_randomness":True,
      "race_level_unknown_score_exclusion":True,
      "zero_score_races_excluded_from_all_fit_and_evaluation":True,
      "V6_2023_training_nozero":"2023 Jan-Oct; 19017 valid races",
      "mixture":"Q(triple)=(1-epsilon)*P_V6(triple)+epsilon*R(triple)",
      "modes":{
        "whole-race-uniform":"R=1/[n(n-1)(n-2)]; all 1/2/3 slots random",
        "uniform-winner-only":"R(a,b,c)=(1/n)*P_V6(b,c|a); first rider random, original conditional 2nd/3rd"
      },
      "epsilon_grid":list(EPS),
      "choice_period":"November 2023 only; 2023 December checked later",
      "strict_selection":"Improved full trifecta NLL and non-worse full multiclass Brier on November, else no change",
      "selected_november_only":selection,
      "evaluate_historical_2024_2026_not_untouched":True,
      "model_production_unchanged":True,
      "no_market_odds_or_payouts":True,
      "all_outcomes_summed_to_one":True,
      "results":tests,"all_epsilon_sensitivity":periods,"source_audits":sources,
      "limitations":[
        "Uniform unexplained chance is a falsifiable alternative, not a derived causal or empirical individual minimum.",
        "All 2024-26 result periods were previously inspected in this project, so improvement would not constitute untouched forward validation.",
        "Score-zero race exclusion leaves an incomplete source population; score zero cannot be relabeled as low ability.",
        "Model is reconstructed by retraining same V6/V5/V3/V2 algorithms on clean 2023, so reference NLL may differ from original V6 historical report.",
        "November selection may be noisy and is not supported by a fresh unobserved period.",
        "Neither NLL nor Brier establishes expected value or profit without market odds.",
        "All 1/2/3 probabilities must sum to one per slot; this property is preserved by the full-ticket mixture."
      ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V9_UNEXPLAINED_CHANCE_COMPLETE",json.dumps({
      "selected":selection,
      "period_results":tests,
      "nozero_exclusions":{k:v["excluded_unknown_score_races"] for k,v in periods.items()}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

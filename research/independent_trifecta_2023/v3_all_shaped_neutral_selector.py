#!/usr/bin/env python3
"""UNBIASED replay of independent 3ren-tan V3: no preferred line formation.

Reuses the EXACT frozen V2/V3 trainers and ticket probability generators.
Unlike the old V3 research run, this selector has NO hard-coded line shape
in its acceptance criteria, diagnostics, training sample weights or features.

Train V2 and pattern weights ONLY from Jan-Oct 2023. Select lambda on the
previously inspected Nov-Dec 2023 development period. Re-evaluate on 2024,
and optionally verify the previously inspected 2025 scores. Neither year
refits models or selects tuning parameters. All predicted line shapes are
treated equally, counted per observed shape for descriptive diagnostics.
NO odds or payout inputs are consumed. Existing scripts remain untouched.
"""
from __future__ import annotations
import json, math, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as source
import line_dependence_v2 as v2
import aab_guarded_v3 as fixed_v3
import calibration_audit as source24
import validate_2025 as source25

OUT=ROOT/"data/audits/trifecta_v3_neutral_selection_2023_2025.json"
PATS=("AAA","AAB","ABA","ABB","ABC")
LAMBDAS=(0.,0.15,0.30,0.50,0.75,1.0)

class Metrics:
    def __init__(self):
        self.n=0;self.nll=0.;self.brier=0.;self.ptrue=0.;self.tickets=0
        self.expected={k:0. for k in PATS}
        self.actual={k:0 for k in PATS}
        self.shapes=defaultdict(lambda:{"races":0,"NLL_sum":0.,"Brier_sum":0.,
            "AAB_expected":0.,"AAB_actual":0,"pattern_expected":{k:0. for k in PATS},
            "pattern_actual":{k:0 for k in PATS}})
        self.day=defaultdict(lambda:[0.,0])

    def add(self,e,dist,pats):
        if abs(sum(dist.values())-1)>1e-9:raise AssertionError("probability != one")
        truth=e["order"]
        p=dist[truth]
        loss=-math.log(max(1e-15,p))
        brier=1-2*p+sum(q*q for q in dist.values())
        local={k:0. for k in PATS}
        for ticket,q in dist.items():
            local[pats[ticket]]+=q
        self.n+=1;self.tickets+=len(dist);self.nll+=loss;self.brier+=brier;self.ptrue+=p
        sh=self.shapes[v2.line_shape(e)];sh["races"]+=1
        sh["NLL_sum"]+=loss;sh["Brier_sum"]+=brier
        sh["AAB_expected"]+=local["AAB"]
        sh["AAB_actual"]+=int(pats[truth]=="AAB")
        for k in PATS:
            self.expected[k]+=local[k]
            sh["pattern_expected"][k]+=local[k]
            obs=int(pats[truth]==k)
            self.actual[k]+=obs;sh["pattern_actual"][k]+=obs
        self.day[e["date"]][0]+=loss
        self.day[e["date"]][1]+=1

    def pack(self):
        return {
          "races":self.n,"trifecta_tickets":self.tickets,
          "NLL":round(self.nll/self.n,8),
          "Brier":round(self.brier/self.n,8),
          "mean_true_ticket_probability_pct":round(100*self.ptrue/self.n,7),
          "five_pattern_distribution":{
             k:{"actual":self.actual[k],"predicted":round(self.expected[k],5),
                "absolute_error":round(abs(self.actual[k]-self.expected[k]),5)}
             for k in PATS},
          "all_line_shapes":{
            sh:{"races":x["races"],"NLL":round(x["NLL_sum"]/x["races"],8),
                "Brier":round(x["Brier_sum"]/x["races"],8),
                "AAB_actual":x["AAB_actual"],
                "AAB_predicted":round(x["AAB_expected"],5),
                "patterns":{
                   k:{"actual":x["pattern_actual"][k],
                      "predicted":round(x["pattern_expected"][k],5)}
                   for k in PATS}}
            for sh,x in sorted(self.shapes.items())}
        }

def neutral_guard(base,candidate):
    """ONLY global, shape-symmetric criteria, no geometry-dependent exceptions."""
    b=base.pack(); a=candidate.pack()
    errors=[]
    if abs(a["five_pattern_distribution"]["AAA"]["predicted"]-
           b["five_pattern_distribution"]["AAA"]["predicted"])>1e-6:
        errors.append("AAA_total_changed")
    if not a["five_pattern_distribution"]["AAB"]["absolute_error"]+1e-6 < \
           b["five_pattern_distribution"]["AAB"]["absolute_error"]:
        errors.append("AAB_not_improved")
    for pat in ("ABA","ABB","ABC"):
        if a["five_pattern_distribution"][pat]["absolute_error"] > \
           b["five_pattern_distribution"][pat]["absolute_error"]+1e-6:
            errors.append(pat+"_worsened")
    if a["NLL"]>b["NLL"]+1e-8:errors.append("NLL_worsened")
    if a["Brier"]>b["Brier"]+1e-8:errors.append("Brier_worsened")
    return {"pass":not errors,"failures":errors}

def day_cluster_ci(old,alt):
    days=sorted(set(old.day)|set(alt.day))
    n=old.n
    diff=sum(old.day[d][0]-alt.day[d][0] for d in days)/n
    se=math.sqrt(sum((
        old.day[d][0]-alt.day[d][0]-diff*old.day[d][1]
    )**2 for d in days))/n
    return {"NLL_gain":round(diff,8),
            "approx_95pct_CI":[round(diff-1.96*se,8),round(diff+1.96*se,8)],
            "days":len(days)}

def evaluate(events,wg,wr,pw,lambdas,phase):
    stats={a:Metrics() for a in lambdas}
    for i,(rid,e) in enumerate(events.items(),1):
        old=fixed_v3.base_distribution(e,wg,wr)
        gate=fixed_v3.predict_non_aaa(e,pw)
        all_options,ticket_patterns=fixed_v3.variants(e,old,gate)
        for lam in lambdas:
            stats[lam].add(e,all_options[lam][0],ticket_patterns)
        if i%2500==0:print("NEUTRAL_ALL_SHAPES",phase,i,flush=True)
    return stats

def selftest_multiple_formations():
    src={"scores":{str(k):float(65+k) for k in range(1,8)},
         "ranks":{str(k):min(k,4) for k in range(1,8)},
         "order":("1","2","3"),"date":"2023-11-01"}
    for groups in (
       [["1","2","3"],["4","5"],["6","7"]],
       [["1","2","3"],["4","5","6"],["7"]],
       [["1","2"],["3","4"],["5","6"],["7"]],
       [["1","2","3","4"],["5","6"],["7"]],
       [["1"],["2"],["3"],["4"],["5"],["6"],["7"]],
       [["1","2","3","4","5","6","7"]],
    ):
        e=dict(src);e["groups"]=groups
        e["positions"]={c:(j+1,i+1,len(g))
              for j,g in enumerate(groups) for i,c in enumerate(g)}
        old=fixed_v3.base_distribution(e,{}, {})
        gate=fixed_v3.predict_non_aaa(e,{k:{} for k in fixed_v3.NONAAA})
        options,pat=fixed_v3.variants(e,old,gate)
        for a in LAMBDAS:
            pred=options[a][0]
            assert abs(sum(pred.values())-1)<1e-9
            assert all(pred[t]==old[t] for t in old if pat[t]=="AAA")
    print("ALL_FORMATIONS_UNIT_TESTS_PASS",flush=True)

def main():
    selftest_multiple_formations()
    assert source.TRAIN_END=="2023-10-31"
    entire,detail=source.ingest()
    train={k:v for k,v in entire.items() if v["date"]<=source.TRAIN_END}
    dev={k:v for k,v in entire.items() if v["date"]>source.TRAIN_END}
    assert (len(train),len(dev))==(19248,4192)
    wg,wr=v2.fit_structural(train)
    pw,npat=fixed_v3.train_pattern_gate(train)
    del train,entire

    develop=evaluate(dev,wg,wr,pw,LAMBDAS,"2023_DEVELOPMENT")
    base=develop[0.]
    eligible={str(x):neutral_guard(base,develop[x]) for x in LAMBDAS if x}
    accepted=[x for x in LAMBDAS if x and eligible[str(x)]["pass"]]
    chosen=min(accepted,key=lambda x:(develop[x].nll,x)) if accepted else 0.
    print("NEUTRAL_SELECTION_NO_SPECIAL_SHAPE",chosen,"eligibility",eligible,flush=True)

    forward,cov24=source24.ingest_2024()
    assert len(forward)==25348
    test24=evaluate(forward,wg,wr,pw,(0.,chosen) if chosen else (0.,),"2024_VERIFICATION")
    del forward

    seen25,cov25=source25.read_2025()
    assert len(seen25)==24635
    test25=evaluate(seen25,wg,wr,pw,(0.,chosen) if chosen else (0.,),"2025_REPLAY")
    del seen25
    if chosen:
        assert abs(test24[chosen].nll/test24[chosen].n-4.31982650)<1e-5, \
            "Unexpected drift from original 2024 V3"
        assert abs(test25[chosen].nll/test25[chosen].n-4.36613863)<1e-5, \
            "Unexpected drift from original 2025 V3"
    doc={
      "state":"NEUTRAL_LINE_SHAPE_BIAS_REMOVED_V3_VALIDATED",
      "line_shape_special_selection_rules":[],
      "line_shape_in_training":"line_shape is one of several symmetric observed covariates, not a favored formation",
      "bias_removal": "Deleted singled-out formation safeguard from V3 selection; kept every formation in equal-weight global NLL/Brier and AAA/AAB/ABA/ABB/ABC guards.",
      "current_production_or_original_AI_changed":False,
      "frozen_V2_V3_training_code_changed":False,
      "market_odds_read":False,
      "2024_or_2025_used_in_parameter_selection":False,
      "2023_development_races":4192,
      "2024_evaluation_races":25348,
      "2025_replay_races":24635,
      "neutral_candidate_comparison":{str(x):z.pack() for x,z in develop.items()},
      "neutral_guard_results":eligible,
      "neutral_selected_lambda":chosen,
      "historical_selected_lambda":0.15,
      "historical_selection_preserved":chosen==0.15,
      "2024_base":test24[0.].pack(),
      "2024_neutral_candidate":test24[chosen].pack(),
      "2024_neutral_guard":neutral_guard(test24[0.],test24[chosen]),
      "2024_day_cluster_NLL_ci":day_cluster_ci(test24[0.],test24[chosen]),
      "2025_base":test25[0.].pack(),
      "2025_neutral_candidate":test25[chosen].pack(),
      "2025_neutral_guard":neutral_guard(test25[0.],test25[chosen]),
      "2025_day_cluster_NLL_ci":day_cluster_ci(test25[0.],test25[chosen]),
      "line_shape_weights_and_selection":"Each eligible race has weight one; no pattern or formation is singled out for model selection.",
      "cautions":[
         "2023 Nov-Dec was previously inspected; neutral replay is a bias-remediation audit, not a new blind model-selection experiment.",
         "2024/2025 outcomes are previously inspected; this is a reconstruction verifying absence of a specific geometric bias.",
         "All-shapes aggregate fit does not guarantee uniform accuracy within each line geometry; per-shape diagnostics are descriptive only.",
         "The frozen features include line shape as an ordinary predictive attribute; that is not equivalent to a hand-selected favored geometry.",
         "No future, market, odds, betting or ROI validation implied."
      ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("NEUTRAL_V3_ALL_SHAPES_RESEARCH_COMPLETE",json.dumps({
       "lambda":chosen,
       "same_as_historical":chosen==0.15,
       "shape_count_2024":len(test24[chosen].shapes),
       "shape_count_2025":len(test25[chosen].shapes),
       "2024":{k:doc[k] for k in ("2024_neutral_guard","2024_day_cluster_NLL_ci")},
       "2025":{k:doc[k] for k in ("2025_neutral_guard","2025_day_cluster_NLL_ci")}
    }),flush=True)

if __name__=="__main__":
    main()

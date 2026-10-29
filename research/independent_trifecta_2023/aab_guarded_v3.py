#!/usr/bin/env python3
"""Independent 3ren-tan V3: AAB structural allocation without harming AAA.

FROZEN reference: original 2023 observe.py and 2023-trained line_dependence_v2.py.
Only Jan-Oct 2023 fits an odds-free, conditional NON-AAA structural pattern model.
Nov-Dec 2023 is a DEVELOPMENT set already inspected in V2; not a blind holdout.
Weights/acceptance are frozen using development before evaluating 2024, a new year.
AAB cannot gain probability from AAA, which is held exactly at its V2 ticket p.
AAB must improve WITHOUT worsening any of ABA, ABB, ABC residuals, and without
worsening proper full-trifecta NLL/Brier, to pass the selection/acceptance gates.
2024 is evaluated ONCE after development selection. Failure => NO ADOPTION.
"""
from __future__ import annotations
import json, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as frozen
import line_dependence_v2 as v2
import calibration_audit as source2024

OUTPUT=ROOT/"data/audits/independent_trifecta_aab_guarded_v3_2024.json"
PATTERNS=("AAA","AAB","ABA","ABB","ABC")
OTHERS=("ABA","ABB","ABC")
NONAAA=("AAB","ABA","ABB","ABC")
LAMBDAS=(0.,0.15,0.30,0.50,0.75,1.0)
EPOCHS=6
SEED=20231031
LR=0.035
EPOCH_SHRINK=0.99
MAX_SCORE_OFFSET=3.
REQUIRED_2024=20000
DIAGNOSTIC_SHAPE="3-2-2"


def pattern_counts(e):
    sizes=[len(g) for g in e["groups"]]
    n=sum(sizes)
    aaa=sum(z*(z-1)*(z-2) for z in sizes)
    pair=sum(z*(z-1)*(n-z) for z in sizes)
    abc=n*(n-1)*(n-2)-aaa-3*pair
    out={"AAA":aaa,"AAB":pair,"ABA":pair,"ABB":pair,"ABC":abc}
    if min(out.values())<0 or sum(out.values())!=n*(n-1)*(n-2):
        raise AssertionError("Analytic line path enumeration failed")
    return out


def shape(e):
    return v2.line_shape(e)


def features(e):
    """Only pre-race scores and line structure, never outcome or odds."""
    groups=e["groups"]
    all_scores=list(e["scores"].values())
    n=len(all_scores)
    sizes=[len(g) for g in groups]
    sorted_sizes=sorted(sizes,reverse=True)
    avg=sum(all_scores)/n
    spread=max(all_scores)-min(all_scores)
    max_per_line=[max(e["scores"][c] for c in g) for g in groups]
    mean_per_line=[sum(e["scores"][c] for c in g)/len(g) for g in groups]
    best_line=max(range(len(groups)),key=lambda i:(max_per_line[i],-i))
    car_by_score=sorted(e["scores"],key=lambda c:(-e["scores"][c],int(c)))
    topg=[e["positions"][c][0] for c in car_by_score[:3]]
    strongest_role=v2.rlabel(e,car_by_score[0])
    f={
        "bias":1.,
        "line_shape="+shape(e):1.,
        "field_size="+str(n):1.,
        "number_lines="+str(min(len(groups),8)):1.,
        "max_line_size="+str(min(sorted_sizes[0],5)):1.,
        "second_line_size="+str(min(sorted_sizes[1] if len(sizes)>1 else 0,5)):1.,
        "three_plus_lines="+str(sum(x>=3 for x in sizes)):1.,
        "single_lines="+str(min(sum(x==1 for x in sizes),6)):1.,
        "best_rider_line_size="+str(min(len(groups[best_line]),5)):1.,
        "best_rider_role="+strongest_role:1.,
        "top2_score_same_line="+str(int(topg[0]==topg[1])):1.,
        "top3_score_all_same_line="+str(int(len(set(topg))==1)):1.,
        "top3_score_three_different_lines="+str(int(len(set(topg))==3)):1.,
        "top_score_gap":min(2.,(e["scores"][car_by_score[0]]-e["scores"][car_by_score[1]])/15.),
        "field_score_spread":min(2.,spread/25.),
        "strongest_line_gap":max(-2.,min(2.,(max(max_per_line)-sum(max_per_line)/len(groups))/12.)),
        "strongest_line_avg_gap":max(-2.,min(2.,(max(mean_per_line)-sum(mean_per_line)/len(groups))/12.)),
        "leader_average_advantage":max(-2.,min(2.,(
           sum(e["scores"][g[0]] for g in groups)/len(groups)-avg)/12.)),
    }
    return f


def softmax(scores):
    m=max(scores.values())
    exps={k:math.exp(v-m) for k,v in scores.items()}
    z=sum(exps.values())
    return {k:v/z for k,v in exps.items()}


def predict_non_aaa(e,w):
    counts=pattern_counts(e)
    f=features(e)
    logits={}
    for k in NONAAA:
        if counts[k]:
            logits[k]=math.log(counts[k])+sum(w[k].get(key,0.)*value for key,value in f.items())
    if not logits:
        return {}
    return softmax(logits)


def train_pattern_gate(train):
    """Fit event-level structural choices by multinomial conditional likelihood.

    This is a DIFFERENT generative level P(pattern | race features, not AAA);
    unlike post-hoc AAB correction, predictions vary with full line geometry,
    score and role context; all coefficients are fit on JAN-OCT 2023 ONLY.
    """
    w={k:defaultdict(float) for k in NONAAA}
    examples=[]
    for e in train.values():
        target=v2.pattern(e,e["order"])
        if target=="AAA":
            continue
        f=features(e)
        c=pattern_counts(e)
        logits={k:math.log(c[k]) for k in NONAAA if c[k]>0}
        if target not in logits:
            raise AssertionError(("Observed pattern structurally impossible",target))
        examples.append((f,logits,target))
    rng=random.Random(SEED)
    for ep in range(EPOCHS):
        rng.shuffle(examples)
        lr=LR/math.sqrt(ep+1)
        loss=0.
        for f,prior,target in examples:
            pred=softmax({k:off+sum(w[k].get(feature,0.)*v for feature,v in f.items())
                          for k,off in prior.items()})
            loss-=math.log(max(pred[target],1e-15))
            for k in prior:
                delta=lr*((1. if k==target else 0.)-pred[k])
                if delta:
                    for feat,value in f.items():
                        w[k][feat]+=delta*value
        for z in w.values():
            for key in z:
                z[key]*=EPOCH_SHRINK
        print("AAB_STRUCTURAL_GATE_EPOCH",ep+1,"EVENTS",len(examples),
              "NLL",round(loss/len(examples),7),flush=True)
    return w,len(examples)


def base_distribution(e,wg,wr):
    """Exactly V2 NEW_LINE_AND_RIDER, without re-fitting or post-hoc tuning."""
    cars=sorted(e["scores"],key=int)
    def single(prefix):
        gp,cp=v2.struct_components(e,prefix,wg,wr)
        return {c:gp[e["positions"][c][0]]*cp[c] for c in cars if c not in prefix}
    p1=single(())
    dist={}
    for a in cars:
        p2=single((a,))
        for b in cars:
            if b==a:
                continue
            p3=single((a,b))
            for c in cars:
                if c in (a,b):
                    continue
                dist[a,b,c]=p1[a]*p2[b]*p3[c]
    if len(dist)!=len(cars)*(len(cars)-1)*(len(cars)-2):
        raise AssertionError("Incomplete frozen V2 3ren-tan distribution")
    if abs(sum(dist.values())-1)>1e-9:
        raise AssertionError("Frozen V2 probability not 100%")
    return dist


def variants(e,base,gate):
    pats={t:v2.pattern(e,t) for t in base}
    masses={k:0. for k in PATTERNS}
    for t,p in base.items():
        masses[pats[t]]+=p
    nonaaa_mass=1.-masses["AAA"]
    if abs(sum(masses.values())-1)>1e-9:
        raise AssertionError("Invalid five-pattern mass partition")
    fitted=gate
    if set(fitted)!={k for k in NONAAA if masses[k]>1e-13}:
        raise AssertionError(("Pattern gate availability differs",fitted,masses))
    targets={"AAA":masses["AAA"]}
    targets.update({k:nonaaa_mass*fitted.get(k,0.) for k in NONAAA})
    if abs(sum(targets.values())-1)>1e-9:
        raise AssertionError("New nonAAA gate fails normalization")
    result={}
    for lam in LAMBDAS:
        factors={"AAA":1.}
        for k in NONAAA:
            old=masses[k];new=targets[k]
            factors[k]=(1.-lam)+lam*new/old if old>0 else 1.
        p={t:old*factors[pats[t]] for t,old in base.items()}
        if abs(sum(p.values())-1)>1e-9:
            raise AssertionError(("Total three-way mass not 100%",lam))
        for t,pat in pats.items():
            if pat=="AAA" and abs(p[t]-base[t])>1e-13:
                raise AssertionError("Individual AAA ticket changed")
        result[lam]=(p,factors,masses,targets)
    return result,pats


class Metrics:
    def __init__(self):
        self.n=0
        self.nll=0.
        self.brier=0.
        self.total_mass=0.
        self.ticket_count=0
        self.ptrue=0.
        self.events={k:0 for k in PATTERNS}
        self.exp={k:0. for k in PATTERNS}
        self.shape_aab={"predicted":0.,"actual":0,"races":0}
        self.daily_nll=defaultdict(lambda:[0.,0])

    def add(self,e,dist,truth,pats):
        ptrue=dist[truth]
        self.n+=1
        self.ptrue+=ptrue
        self.nll-=math.log(max(ptrue,1e-15))
        self.brier+=1.-2.*ptrue+sum(p*p for p in dist.values())
        self.total_mass+=sum(dist.values())
        self.ticket_count+=len(dist)
        exp={k:0. for k in PATTERNS}
        for t,p in dist.items():
            exp[pats[t]]+=p
        for k in PATTERNS:
            self.exp[k]+=exp[k]
        truepat=pats[truth]
        self.events[truepat]+=1
        if shape(e)==DIAGNOSTIC_SHAPE:
            self.shape_aab["races"]+=1
            self.shape_aab["predicted"]+=exp["AAB"]
            self.shape_aab["actual"]+=int(truepat=="AAB")
        z=self.daily_nll[e["date"]]
        z[0]+=-math.log(max(ptrue,1e-15))
        z[1]+=1

    def report(self):
        n=self.n
        return {
          "races":n,
          "trifecta_combinations":self.ticket_count,
          "mean_NLL":round(self.nll/n,8),
          "mean_Brier":round(self.brier/n,8),
          "mean_realized_probability_pct":round(100*self.ptrue/n,7),
          "mean_total_trifecta_probability":round(self.total_mass/n,10),
          "line_patterns":{
              k:{"actual":self.events[k],
                 "predicted":round(self.exp[k],5),
                 "actual_minus_predicted":round(self.events[k]-self.exp[k],5),
                 "absolute_error":round(abs(self.events[k]-self.exp[k]),5)}
              for k in PATTERNS
          },
          "shape_3_2_2_AAB":{
             "races":self.shape_aab["races"],
             "actual":self.shape_aab["actual"],
             "predicted":round(self.shape_aab["predicted"],5),
             "absolute_error":round(abs(self.shape_aab["actual"]-self.shape_aab["predicted"]),5)
          }
        }


def evaluate(events,wg,wr,weights,tag):
    stats={lam:Metrics() for lam in LAMBDAS}
    for i,(rid,e) in enumerate(events.items(),1):
        base=base_distribution(e,wg,wr)
        pmap,pats=variants(e,base,predict_non_aaa(e,weights))
        for lam, (dist,_,_,_) in pmap.items():
            stats[lam].add(e,dist,e["order"],pats)
        if i%2000==0:
            print("AAB_GUARDED_EVALUATING",tag,i,flush=True)
    return stats


def guard(base,alt,require_shape):
    b=base.report();a=alt.report()
    failures=[]
    if abs(a["line_patterns"]["AAA"]["predicted"]-b["line_patterns"]["AAA"]["predicted"])>1e-6:
        failures.append("AAA_probability_changed")
    if not a["line_patterns"]["AAB"]["absolute_error"]+1e-6 < b["line_patterns"]["AAB"]["absolute_error"]:
        failures.append("AAB_not_improved")
    for k in OTHERS:
        if a["line_patterns"][k]["absolute_error"] > b["line_patterns"][k]["absolute_error"]+1e-6:
            failures.append(k+"_worsened")
    if a["mean_NLL"]>b["mean_NLL"]+1e-8:
        failures.append("proper_log_loss_worsened")
    if a["mean_Brier"]>b["mean_Brier"]+1e-8:
        failures.append("proper_brier_worsened")
    if require_shape and a["shape_3_2_2_AAB"]["absolute_error"]>b["shape_3_2_2_AAB"]["absolute_error"]+1e-6:
        failures.append("AAB_3_2_2_worsened")
    return {"passed":not failures,"failures":failures}


def day_cluster_ci(before,after):
    days=sorted(set(before.daily_nll)|set(after.daily_nll))
    n=before.n
    gain=sum(before.daily_nll[d][0]-after.daily_nll[d][0] for d in days)/n
    se=math.sqrt(sum((before.daily_nll[d][0]-after.daily_nll[d][0]-
                      gain*before.daily_nll[d][1])**2 for d in days))/n
    return {"mean_NLL_gain":round(gain,8),
            "day_cluster_robust_se":round(se,8),
            "approx_95pct_CI":[round(gain-1.96*se,8),round(gain+1.96*se,8)],
            "days":len(days)}


def selftest():
    e={"scores":{str(i):float(65+i*2) for i in range(1,8)},
       "positions":{"1":(1,1,3),"2":(1,2,3),"3":(1,3,3),
                    "4":(2,1,2),"5":(2,2,2),"6":(3,1,2),"7":(3,2,2)},
       "groups":[["1","2","3"],["4","5"],["6","7"]],
       "ranks":{str(i):min(i,4) for i in range(1,8)},
       "order":("1","2","3"),"date":"2023-11-01"}
    assert pattern_counts(e)=={"AAA":6,"AAB":44,"ABA":44,"ABB":44,"ABC":72}
    wg,wr={},{}
    baseline=base_distribution(e,wg,wr)
    f=predict_non_aaa(e,{k:{} for k in NONAAA})
    pmap,pats=variants(e,baseline,f)
    for lam,(dist,_,_,_) in pmap.items():
        assert abs(sum(dist.values())-1)<1e-9
        assert all(dist[t]==baseline[t] for t in baseline if pats[t]=="AAA")
    print("AAB_GUARDED_V3_CONSERVATION_TEST_PASS",flush=True)


def main():
    selftest()
    assert frozen.TRAIN_END=="2023-10-31"
    entire,sourceaudit=frozen.ingest()
    train={rid:e for rid,e in entire.items() if e["date"]<=frozen.TRAIN_END}
    dev={rid:e for rid,e in entire.items() if e["date"]>frozen.TRAIN_END}
    assert len(train)==19248 and len(dev)==4192
    print("FROZEN_V2_ORIGINAL_TRAIN_2023_ONLY",len(train),flush=True)
    # Refit original V2 exactly with frozen code, no 2023 Nov-Dec in training.
    wg,wr=v2.fit_structural(train)
    # A new structurally conditional race-level event model fits Jan-Oct only.
    patweights,patn=train_pattern_gate(train)
    # Nov-Dec 2023 was already analyzed in V2 => development only, NOT blind.
    dev_results=evaluate(dev,wg,wr,patweights,"2023_DEVELOPMENT")
    olddev=dev_results[0.]
    dev_check={str(lam):guard(olddev,metric,True)
               for lam,metric in dev_results.items() if lam>0}
    options=[lam for lam in LAMBDAS if lam>0 and dev_check[str(lam)]["passed"]]
    # EXACT prespecified selection rule: best development NLL among candidates
    # passing AAB gain, all competing patterns, NLL/Brier and 322 safeguards.
    selected=(min(options,key=lambda lam:(dev_results[lam].nll,lam)) if options else 0.)
    print("V3_CHOSEN_FROM_ALREADY_USED_DEVELOPMENT_ONLY",selected,flush=True)

    # 2024 is the forward-year test. No 2024 outcome has affected parameters
    # or choice of selected weight.
    test,cov=source2024.ingest_2024()
    if len(test)<REQUIRED_2024:
        raise AssertionError(("2024 holdout insufficient",len(test)))
    print("V3_2024_FORWARD_UNSEEN",len(test),flush=True)
    # To distinguish outcome discovery from selection, evaluate only baseline
    # and ONE development-selected candidate, never choose among 2024 versions.
    out_old=Metrics();out_new=Metrics()
    for i,(rid,e) in enumerate(test.items(),1):
        base=base_distribution(e,wg,wr)
        pmap,pats=variants(e,base,predict_non_aaa(e,patweights))
        out_old.add(e,base,e["order"],pats)
        out_new.add(e,pmap[selected][0],e["order"],pats)
        if i%2000==0:
            print("V3_FORWARD_2024_EVAL",i,len(test),flush=True)
    finalguard=guard(out_old,out_new,True)
    ci=day_cluster_ci(out_old,out_new)
    # Guarded acceptance is descriptive, not a claim of statistical dominance.
    # Require substantial forward NLL evidence as a further safeguard.
    improvement=(selected>0 and finalguard["passed"] and
                 ci["approx_95pct_CI"][0]>0)
    if selected==0:
        finalguard["passed"]=False
        finalguard["failures"].append("no_development_candidate_passed_guards")
    if selected>0 and ci["approx_95pct_CI"][0]<=0:
        finalguard["passed"]=False
        finalguard["failures"].append("no_positive_day_cluster_nll_95pct_lower_bound")
    doc={
      "state":"COMPLETED_AAB_GUARDED_STRUCTURAL_V3_2024",
      "objective":"Improve underallocated AAB without deteriorating AAA or any other structural pattern, overall NLL/Brier, or 3-2-2 AAB.",
      "mechanism":"Train P(AAB,ABA,ABB,ABC | race size, full line geometry, score strength, predicted roles; not AAA) from 2023 Jan-Oct outcomes. Combine with frozen V2 probability of AAA and frozen V2 conditional ticket ordering within each pattern. Every AAA ticket keeps its exact original V2 probability.",
      "is_simple_aab_multiplier":False,
      "odds_market_payout_used":False,
      "2024_labels_used_for_fit_or_selection":False,
      "2023_NovDec_marked_as_already_inspected_development":True,
      "baseline_original_AI_changed":False,
      "baseline_V2_changed":False,
      "train_2023_JanOct_races":len(train),
      "development_2023_NovDec_races":len(dev),
      "test_2024_full_year_races":len(test),
      "new_pattern_training_nonAAA_races":patn,
      "training_settings":{"epochs":EPOCHS,"seed":SEED,"lr":LR,
                           "epoch_shrink":EPOCH_SHRINK,"candidate_mix_weights":list(LAMBDAS)},
      "trained_pattern_parameter_count":sum(len(z) for z in patweights.values()),
      "trained_pattern_parameters":{k:dict(sorted(w.items())) for k,w in patweights.items()},
      "development_comparison":{str(k):v.report() for k,v in dev_results.items()},
      "development_guard_results":dev_check,
      "preselected_weight_using_development_only":selected,
      "2024_forward_test_frozen_V2":out_old.report(),
      "2024_forward_test_preselected_V3":out_new.report(),
      "2024_forward_test_guard":finalguard,
      "2024_day_cluster_NLL_CI":ci,
      "2024_strict_accepted":bool(improvement),
      "candidate_did_not_pass_action":"KEEP_FROZEN_V2_NO_UPGRADE",
      "2024_data_coverage":cov,
      "limits":[
         "AAB and other categories are exhaustive; AAB cannot gain mass unless one or more nonAAA competitors lose mass.",
         "AAA individual ticket p values are identical by construction; all ordered tickets in each race sum to exactly 100%.",
         "No 2024 evaluation is used to select lambda, fit models, or tune the experiment.",
         "The 2023 Nov-Dec development set was previously examined and is not a fresh independent test.",
         "2024 is forward in time but historical score/line timestamp cutoff has not been independently validated.",
         "Strict no-worsening guard is observed-sample conditional, not a mathematical promise for future unseen races.",
         "No odds, market, purchases, or ROI used."
      ]
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("AAB_GUARDED_STRUCTURAL_V3_FINAL",json.dumps({
       "selected_mix_weight":selected,
       "dev_passed":bool(selected),
       "2024_races":len(test),
       "baseline":out_old.report(),
       "candidate":out_new.report(),
       "forward_guard":finalguard,
       "forward_ci":ci,
       "ADOPT":bool(improvement)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

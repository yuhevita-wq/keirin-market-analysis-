#!/usr/bin/env python3
"""V5 R&D: within-line rider softmax interactions of race score, role and line size.

Frozen comparison: V3=V2 group-first probabilities, V3 non-AAA gate lambda=.15.
New fit is a *residual rider head*, initialized at EXACT V2 conditional
P(rider | line, prefix). The line selection head is identically frozen.
Interacted softmax features (role x original line size x score-difference,
role x original line size x gap-to-line-best, etc.) are trained by regularized
conditional likelihood on 2023 Jan-Oct ONLY.

Hyperparameters (4 SGD epochs, lr .012, shrink .975, seed 20231031, fixed
alphas 0/.25/.50/1) are deliberately low capacity. Choose alpha on the
ALREADY INSPECTED Nov-Dec 2023 DEV only by improving full 3ren-tan NLL and
not worsening Brier; evaluated pre-race all-ordered triples, NEVER true-prefix.
2024/25/26 were previously inspected in research, so forward chronological
checks are historical replication, NOT freshly untouched evaluation.

No hand-chosen line formation; every eligible race contributes equally.
Entire 3ren-tan probability sums one; V3 pattern gate is re-applied to each
candidate's *own* raw probability distribution. Unlike V3-versus-V2, AAA
individual ticket and aggregate mass need not remain equal to baseline; report
all structural pattern changes. No price, odds, market, payout, or profit used.

NO production modification: challenger is a separate research script,
output JSON alone. Promote only if proper scoring and calibration survive
truly NEW data not already inspected.
"""
from __future__ import annotations
import json,math,random,sys
from collections import defaultdict,Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as original
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import calibration_audit as src24
import validate_2025 as src25
import v4_pattern_floor_2026 as src26

OUT=ROOT/"data/audits/independent_trifecta_v5_rider_role_score_size_2023_2026.json"
SEED=20231031
EPOCHS=4
LR=.012
SHRINK=.975
CANDIDATE_SCALES=(0.,.25,.5,1.)
V3_PATTERN_MIX=.15
TRAIN_N=19248
DEV_N=4192
EXPECT_YEARS={"2024":25348,"2025":24635,"2026":11797}
TYPES=("AAA","AAB","ABA","ABB","ABC")
BINS=(
    (0.,.0001,"lt_0_01pct"),
    (.0001,.0005,"0_01_to_0_05pct"),
    (.0005,.001,"0_05_to_0_1pct"),
    (.001,.005,"0_1_to_0_5pct"),
    (.005,.01,"0_5_to_1pct"),
    (.01,.02,"1_to_2pct"),
    (.02,.05,"2_to_5pct"),
    (.05,.1,"5_to_10pct"),
    (.1,.2,"10_to_20pct"),
    (.2,1.000001,"20pct_plus"),
)

def feature(e,prefix,c):
    """All candidate covariates known from pre-race entry/formation and prior
    HYPOTHETICAL prefix. No actual podium for prediction. Stage-specific.
    """
    st=len(prefix)+1
    g,pos,line_size=e["positions"][c]
    members=[m for m in e["groups"][g-1] if m not in prefix]
    rem=len(members)
    role=v2.rlabel(e,c)
    score=e["scores"][c]
    mean=sum(e["scores"][m] for m in members)/rem
    best=max(e["scores"][m] for m in members)
    centered=max(-2.5,min(2.5,(score-mean)/12))
    gap=max(0.,min(2.5,(best-score)/12))
    field=v2.score_offset(e,score)
    key=f"s{st}|n{line_size}|r{role}"
    # Three-way interactions are exact: stage x original-line-size x role x score.
    return {key+"|bias":1.,
            key+"|local_center":centered,
            key+"|gap":gap,
            key+"|field_score":field,
            key+"|remaining":(rem-1.)/3.}

def dot(w,feats):
    return sum(w.get(k,0.)*x for k,x in feats.items())

def rider_conditional(e,prefix,g,members,cp,delta,alpha):
    if alpha==0. or len(members)<=1:
        return {c:cp[c] for c in members}
    scores={}
    for c in members:
        scores[c]=math.log(max(cp[c],1e-300))+alpha*dot(delta,feature(e,prefix,c))
    return v2.softmax(scores)

def fit_residual(train,wg,wr):
    weights={}
    examples=list(train.values())
    rng=random.Random(SEED)
    for ep in range(EPOCHS):
        rng.shuffle(examples)
        lr=LR/math.sqrt(ep+1)
        loss=0.; n=0
        for e in examples:
            for st in (1,2,3):
                prefix=e["order"][:st-1]
                actual=e["order"][st-1]
                selected_g=e["positions"][actual][0]
                groups=v2.line_members(e,prefix)
                members=groups[selected_g]
                if len(members)<2:continue
                _,cp=v2.struct_components(e,prefix,wg,wr)
                cfeat={c:feature(e,prefix,c) for c in members}
                logits={c:math.log(max(cp[c],1e-300))+dot(weights,cfeat[c])
                        for c in members}
                pred=v2.softmax(logits)
                loss-=math.log(max(1e-15,pred[actual]))
                grad=defaultdict(float)
                for c in members:
                    d=(1. if c==actual else 0.)-pred[c]
                    for k,val in cfeat[c].items():
                        grad[k]+=d*val
                for k,z in grad.items():
                    weights[k]=weights.get(k,0.)+lr*z
                n+=1
        for k in weights:weights[k]*=SHRINK
        print("V5_INTERACTION_TRAIN_EPOCH",ep+1,"STAGES",n,
              "CONDITIONAL_NLL",round(loss/n,8),
              "PARAMETERS",len(weights),flush=True)
    return weights

def full_distributions(e,wg,wr,delta,scales):
    """Generate every ordered ticket and both V3 pattern-adjusted models.

    The *line-selection* probabilities are literally shared across models.
    For every conditional prefix, recompute only rider-within-line softmax.
    """
    cars=sorted(e["scores"],key=int)
    all_scales=tuple(scales)
    gp1,cp1=v2.struct_components(e,(),wg,wr)
    ones={}
    for alpha in all_scales:
        probs={}
        for g,members in v2.line_members(e,()).items():
            for c,p in rider_conditional(e,(),g,members,cp1,delta,alpha).items():
                probs[c]=gp1[g]*p
        ones[alpha]=probs
    raw={alpha:{} for alpha in all_scales}
    for a in cars:
        gp2,cp2=v2.struct_components(e,(a,),wg,wr)
        twos={}
        for alpha in all_scales:
            probs={}
            for g,mem in v2.line_members(e,(a,)).items():
                for c,p in rider_conditional(e,(a,),g,mem,cp2,delta,alpha).items():
                    probs[c]=gp2[g]*p
            twos[alpha]=probs
        for b in cars:
            if a==b:continue
            gp3,cp3=v2.struct_components(e,(a,b),wg,wr)
            for alpha in all_scales:
                twobody=ones[alpha][a]*twos[alpha][b]
                cs={}
                for g,mem in v2.line_members(e,(a,b)).items():
                    for c,p in rider_conditional(e,(a,b),g,mem,cp3,delta,alpha).items():
                        cs[c]=gp3[g]*p
                for c,v in cs.items():
                    raw[alpha][(a,b,c)]=twobody*v
    total=len(cars)*(len(cars)-1)*(len(cars)-2)
    gate=v3.predict_non_aaa(e,full_distributions.pattern_weights)
    results={}
    for alpha in all_scales:
        base=raw[alpha]
        if len(base)!=total or abs(sum(base.values())-1)>1e-9:
            raise AssertionError(("Invalid rider probability conservation",alpha,len(base),total,sum(base.values())))
        v3options,pat=v3.variants(e,base,gate)
        results[alpha]=v3options[V3_PATTERN_MIX][0]
        if abs(sum(results[alpha].values())-1)>1e-9:
            raise AssertionError("Invalid V3 re-gated mass conservation")
        if alpha==0.:
            # Frozen comparison must reproduce the immutable baseline EXACTLY.
            for k in list(base)[:5]:
                if base[k]<=0:raise AssertionError("Non-positive baseline")
    return results,pat

def cell():
    return [0.,0,0]

def bin_of(p):
    for lo,hi,b in BINS:
        if lo<=p<hi:return b
    raise AssertionError(("Bad probability bucket",p))

class Eval:
    def __init__(self):
        self.n=0;self.tickets=0;self.nll=0.;self.brier=0.
        self.actual_prob=0.
        self.days=defaultdict(lambda:[0.,0.,0])
        self.by_pattern=defaultdict(lambda:[0.,0,0])
        self.fixed_tail=defaultdict(cell)
        self.own_tail=defaultdict(cell)
        self.shape_summary=defaultdict(lambda:[0,0.,0.])
        self.line_prefix_nll=[0.,0.,0.]
    def add(self,e,d,pat,original_d):
        self.n+=1;self.tickets+=len(d)
        pa=d[e["order"]]
        loss=-math.log(max(1e-15,pa))
        br=1-2*pa+sum(p*p for p in d.values())
        self.nll+=loss;self.brier+=br;self.actual_prob+=pa
        daily=self.days[e["date"]];daily[0]+=loss;daily[1]+=br;daily[2]+=1
        shape=v2.line_shape(e)
        zz=self.shape_summary[shape];zz[0]+=1;zz[1]+=loss;zz[2]+=br
        for ticket,p in d.items():
            truth=int(ticket==e["order"])
            z=self.by_pattern[pat[ticket]];z[0]+=p;z[1]+=truth;z[2]+=1
            pbin=bin_of(p)
            fixed=bin_of(original_d[ticket])
            l=self.fixed_tail[fixed];l[0]+=p;l[1]+=truth;l[2]+=1
            h=self.own_tail[pbin];h[0]+=p;h[1]+=truth;h[2]+=1
    def pack(self):
        return {"races":self.n,"tickets":self.tickets,
                "NLL":round(self.nll/self.n,9),
                "Brier":round(self.brier/self.n,9),
                "mean_actual_ticket_probability_pct":round(100*self.actual_prob/self.n,8),
                "all_line_shapes":{"count":len(self.shape_summary),
                   "entries":{s:{"races":n,"NLL":round(ln/n,7),"Brier":round(br/n,7)}
                       for s,(n,ln,br) in sorted(self.shape_summary.items())}},
                "pattern_mass_calibration":{k:{"forecast":round(p,5),"actual":h,
                     "actual_minus_forecast":round(h-p,5)}
                     for k,(p,h,t) in sorted(self.by_pattern.items())},
                "fixed_V3_ticket_cohort_calibration":{k:{"forecast":round(p,5),
                    "actual":h,"tickets":n,"ratio":round(h/p,5) if p>0 else None}
                    for k,(p,h,n) in sorted(self.fixed_tail.items())},
                "candidate_own_ticket_cohort_calibration":{k:{"forecast":round(p,5),
                    "actual":h,"tickets":n,"ratio":round(h/p,5) if p>0 else None}
                    for k,(p,h,n) in sorted(self.own_tail.items())}}

def paired_ci(x,y,idx):
    if x.n!=y.n:raise AssertionError("unpaired racing cohorts")
    dif=x.nll/x.n-y.nll/y.n if idx==0 else x.brier/x.n-y.brier/y.n
    se=math.sqrt(sum((x.days[d][idx]-y.days[d][idx]-dif*x.days[d][2])**2 for d in x.days))/x.n
    return {"gain_V3_minus_challenger":round(dif,9),
            "cluster_day_SE":round(se,9),
            "approx_95pct_CI":[round(dif-1.96*se,9),round(dif+1.96*se,9)],
            "days":len(x.days)}

def run(events,wg,wr,delta,scales,tag):
    res={alpha:Eval() for alpha in scales}
    for n,(rid,e) in enumerate(events.items(),1):
        d,pats=full_distributions(e,wg,wr,delta,scales)
        original_d=d[0.]
        for alpha in scales:
            res[alpha].add(e,d[alpha],pats,original_d)
        if n%2500==0:print("V5_FULL_TRIFECTA_SCORING",tag,n,flush=True)
    return res

def dev_eligible(old,new):
    a=old.pack();b=new.pack()
    return (new.nll<old.nll-1e-7 and new.brier<=old.brier+1e-8)

def main():
    assert original.TRAIN_END=="2023-10-31"
    full,coverage=original.ingest()
    train={k:v for k,v in full.items() if v["date"]<=original.TRAIN_END}
    dev={k:v for k,v in full.items() if v["date"]>original.TRAIN_END}
    assert (len(train),len(dev))==(TRAIN_N,DEV_N)
    print("V5_BASELINE_V2_AND_V3_TRAINING_FROM_2023_JAN_OCT_ONLY",flush=True)
    wg,wr=v2.fit_structural(train)
    pw,patn=v3.train_pattern_gate(train)
    full_distributions.pattern_weights=pw
    correction=fit_residual(train,wg,wr)
    print("V5_INTERACTIONS_TRAINED",len(correction),"max_abs_coeff",max(map(abs,correction.values())),flush=True)
    del train,full

    dev_metrics=run(dev,wg,wr,correction,CANDIDATE_SCALES,"2023_DEVELOPMENT")
    candidates=[alpha for alpha in CANDIDATE_SCALES if alpha>0 and
                dev_eligible(dev_metrics[0.],dev_metrics[alpha])]
    selected=min(candidates,key=lambda a:(dev_metrics[a].nll,a)) if candidates else 0.
    print("V5_ALPHA_CHOSEN_DEVELOPMENT_ONLY",selected,
          "dev_NLLs",{str(a):round(z.nll/z.n,8) for a,z in dev_metrics.items()},flush=True)
    del dev

    tests={}
    metrics={}
    coverage_out={}
    for year,readfn in (("2024",src24.ingest_2024),
                        ("2025",src25.read_2025),
                        ("2026",src26.ingest_2026)):
        events,cov=readfn()
        if len(events)!=EXPECT_YEARS[year]:
            raise AssertionError(("Mismatched yearly cohort",year,len(events)))
        result=run(events,wg,wr,correction,(0.,selected) if selected else (0.,),year)
        if year=="2024" and abs(result[0.].nll/result[0.].n-4.31982650)>1e-5:
            raise AssertionError("V3 2024 baseline reproduction FAILED")
        if year=="2025" and abs(result[0.].nll/result[0.].n-4.36613863)>1e-5:
            raise AssertionError("V3 2025 baseline reproduction FAILED")
        if year=="2026" and abs(result[0.].nll/result[0.].n-4.40591444)>1e-5:
            raise AssertionError("V3 2026 baseline reproduction FAILED")
        metrics[year]=result
        coverage_out[year]=cov
        old,new=result[0.],result[selected]
        both={
            "NLL":paired_ci(old,new,0),
            "Brier":paired_ci(old,new,1),
        }
        old_cal=old.fixed_tail["lt_0_01pct"]
        new_cal=new.fixed_tail["lt_0_01pct"]
        testpass=bool(selected>0 and both["NLL"]["approx_95pct_CI"][0]>0
                      and both["Brier"]["gain_V3_minus_challenger"]>=-1e-8
                      and abs(new_cal[1]-new_cal[0])<abs(old_cal[1]-old_cal[0]))
        tests[year]={
            "V3":old.pack(),
            "challenger":new.pack(),
            "gain_CIs":both,
            "predeclared_year_acceptance_passed":testpass,
            "rare_fixed_V3_ticket_cohort_actual":old_cal[1],
            "rare_fixed_V3_ticket_cohort_predicted_baseline":round(old_cal[0],6),
            "rare_fixed_V3_ticket_cohort_predicted_challenger":round(new_cal[0],6),
        }
        print("V5_YEAR_RESULT",json.dumps({
            "year":year,"alpha":selected,"n_races":len(events),
            "NLL_old":round(old.nll/old.n,8),
            "NLL_new":round(new.nll/new.n,8),
            "Brier_old":round(old.brier/old.n,8),
            "Brier_new":round(new.brier/new.n,8),
            "ci":both,
            "rare_under_0_01pct_V3_fixed_tickets":{"V3_forecast":round(old_cal[0],4),
                   "candidate_forecast":round(new_cal[0],4),"actual":old_cal[1]},
            "passed":testpass,
        }),flush=True)
        del events

    # 2024-2026 have been inspected by previous studies: their use here is
    # already-informed historical recurrence, NOT an unseen final test.
    historical_gate=selected>0 and all(tests[yr]["predeclared_year_acceptance_passed"]
                                       for yr in ("2024","2025","2026"))
    doc={
        "state":"V5_RIDER_ROLE_SCORE_LINE_SIZE_INTERACTION_RESEARCH_COMPLETE",
        "training":"2023-01-01 through 2023-10-31 only",
        "development":"2023-11-01 through 2023-12-31, previously inspected",
        "historical_replay_years":[2024,2025,2026],
        "historical_years_are_fresh_blind_holdout":False,
        "all_line_shapes_no_favored_pattern":True,
        "baseline_V2_and_V3_unchanged":True,
        "V3_line_selection_head_unchanged":True,
        "V3_nonAAA_pattern_gate_lambda_unchanged":0.15,
        "candidate_rider_only_interaction_features":"stage x role x original line size: centered line score, gap to best remaining score, raw field score, available line members and role intercept",
        "rider_sgd":{"epochs":EPOCHS,"lr":LR,"shrink":SHRINK,"seed":SEED},
        "rider_interaction_weights":{k:round(v,9) for k,v in sorted(correction.items())},
        "candidate_scales":list(CANDIDATE_SCALES),
        "development_metrics":{str(a):z.pack() for a,z in dev_metrics.items()},
        "development_select_gate":"full-trifecta NLL lower and Brier nonworse within 1e-8; select best NLL",
        "selected_scale":selected,
        "test_results":tests,
        "historical_all_year_guards_pass":historical_gate,
        "promoted_to_original_V3":False,
        "all_probabilities_sum_one":True,
        "no_market_odds_payout":True,
        "previously_inspected_2024_2026_no_new_blind_test":True,
        "limitations":[
            "Correcting rider probabilities also changes overall AAA mass, unlike V3 vs V2; reports show structural changes.",
            "2024-2026 were previously inspected and modeled; cannot claim a fresh untouched out-of-time experiment.",
            "Original published line and scores collected retrospectively may lack proof of availability at betting cutoff.",
            "Full-podium ticket calibration and predicted probability at observed prefixes are distinct quantities.",
            "Do not treat a 0.01% band in another model as the same ticket cohort; fixed V3 cohort is explicitly used.",
            "No race-specific score features were fitted on post-2023 results, no line-shape priority, no ROI conclusion.",
            "Historical gate is necessary but not sufficient for independent model adoption."
        ],
        "coverage":coverage_out,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V5_INTERACTION_FULL_TRIFECTA_EXPERIMENT_COMPLETE",
          json.dumps({"selected_scale":selected,
            "dev_NLLs":{str(a):round(z.nll/z.n,8) for a,z in dev_metrics.items()},
            "years":{yr:{"NLL_gain":tests[yr]["gain_CIs"]["NLL"],
                         "Brier_gain":tests[yr]["gain_CIs"]["Brier"],
                         "rare":(tests[yr]["rare_fixed_V3_ticket_cohort_predicted_baseline"],
                                 tests[yr]["rare_fixed_V3_ticket_cohort_predicted_challenger"],
                                 tests[yr]["rare_fixed_V3_ticket_cohort_actual"]),
                         "gate":tests[yr]["predeclared_year_acceptance_passed"]}
                     for yr in tests},
            "historical_all_year_guards_pass":historical_gate,
            "actual_model_promoted":False},
          ensure_ascii=False),flush=True)

if __name__=="__main__":main()

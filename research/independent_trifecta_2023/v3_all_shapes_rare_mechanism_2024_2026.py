#!/usr/bin/env python3
"""UNBIASED rare-tail V3 failure-mechanism census, 2024/2025/2026.

No parameter fitting from post-2023 data, no model changes, no odds/payout.
Frozen V2 group/rider generators and frozen V3 non-AAA pattern gate are refit
with 2023 Jan-Oct only (the exact originally specified training procedure).
All observed line formations are included without favored shape/role.

Why a trifecta is below 0.01%: factor V3's P(t) into seven interpretable
conditional components: line-choice and within-line rider-choice at stages
1,2,3, plus the V3 line-pattern gate. Measure which component contributes
the largest log-surprisal relative to uniform remaining riders, and validate
that their contributions reproduce the exact V3 full-ticket probability.

IMPORTANT: attribution of a low *model* probability is not a causal
explanation of real racing upsets. We also count expected and actual events
for EVERY full ticket, grouped by ALL line shapes, patterns, rider roles and
bottleneck components; never inspect only winning tickets.

Separately audit selection probabilities at every realized podium prefix:
P(line | observed previous placings), P(rider | chosen actual line, prefix).
This audits conditional calibration with the outcome only as the label,
not as a predictor for the race. Realized-prefix conditional diagnostics
are NOT obtainable as an unconditional pre-race betting model.

2024, 2025 and 2026 outcomes have already been inspected in prior studies,
so all outputs here are exploratory diagnostics; require a fresh later
period before accepting modifications of the model.
"""
from __future__ import annotations
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as original
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import calibration_audit as source2024
import validate_2025 as source2025
import v4_pattern_floor_2026 as source2026

OUT=ROOT/"data/audits/trifecta_v3_all_shapes_rare_tail_failure_mechanism_2024_2026.json"
V3_LAMBDA=0.15
RARE=.0001
TRAIN_EXPECTED=19248
EXPECTED_COUNTS={"2024":25348,"2025":24635,"2026":11797}
PATTERNS=("AAA","AAB","ABA","ABB","ABC")
COMPONENTS=("1st_line","1st_rider","2nd_line","2nd_rider",
            "3rd_line","3rd_rider","pattern_gate")
TAIL_BINS=(
    (0.,.00001,"below_0_001pct"),
    (.00001,.00005,"0_001_to_0_005pct"),
    (.00005,.0001,"0_005_to_0_01pct"),
    (.0001,.0005,"0_01_to_0_05pct"),
    (.0005,.001,"0_05_to_0_1pct"),
    (.001,.005,"0_1_to_0_5pct"),
    (.005,.01,"0_5_to_1pct"),
    (.01,.05,"1_to_5pct"),
    (.05,1.0000001,"5pct_or_more")
)
CONDITION_BINS=(
    (0.,.01,"0_to_1pct"),
    (.01,.05,"1_to_5pct"),
    (.05,.1,"5_to_10pct"),
    (.1,.2,"10_to_20pct"),
    (.2,.5,"20_to_50pct"),
    (.5,1.0000001,"50_to_100pct")
)

def bucket(p,bins):
    for lo,hi,label in bins:
        if lo<=p<hi:return label
    raise AssertionError(("Probability bucket missing",p))

class Cell:
    def __init__(self):
        self.expected=0.;self.actual=0;self.tickets=0
        self.exposures=0;self.nullvar=0.
        self.day=defaultdict(float)
    def add(self,p,nactual,ntickets,day):
        if not (-1e-8<=p<=1+1e-8):
            raise AssertionError(("Bad race/group probability",p))
        if nactual not in (0,1):
            raise AssertionError("group cannot contain >1 winning trifecta")
        p=max(0.,min(1.,p))
        self.expected+=p;self.actual+=nactual;self.tickets+=ntickets
        self.exposures+=1
        self.nullvar+=p*(1-p)
        self.day[day]+=nactual-p
    def summary(self):
        n=len(self.day)
        residual=self.actual-self.expected
        dayvals=list(self.day.values())
        avg=sum(dayvals)/n if n else 0.
        se=math.sqrt(n/(n-1)*sum((x-avg)**2 for x in dayvals)) if n>1 else 0.
        return {
            "tickets":self.tickets,
            "race_exposures":self.exposures,
            "expected_events":round(self.expected,6),
            "actual_events":self.actual,
            "actual_minus_expected":round(residual,6),
            "observed_over_expected":round(self.actual/self.expected,6) if self.expected else None,
            "mean_ticket_probability_pct":round(100*self.expected/self.tickets,8) if self.tickets else None,
            "empirical_ticket_frequency_pct":round(100*self.actual/self.tickets,8) if self.tickets else None,
            "race_z":round(residual/math.sqrt(self.nullvar),4) if self.nullvar>0 else None,
            "day_cluster_z":round(residual/se,4) if se>0 else None,
            "day_cluster_residual_95pct_reference_interval":[
                round(residual-1.96*se,4),round(residual+1.96*se,4)] if se>0 else None,
            "calendar_days":n,
            "tiny_expected_lt_30":self.expected<30,
        }

def flush_local(local,aggregate,day):
    for key,(p,hit,n) in local.items():
        aggregate[key].add(p,hit,n,day)

def record(local,axis,label,p,hit,nt=1):
    z=local[axis,label]
    z[0]+=p;z[1]+=int(hit);z[2]+=nt

def get_stage_components(e,wg,wr):
    """All possible prefixes, computed from pre-race card alone."""
    cars=sorted(e["scores"],key=int)
    result={}
    prefixes=[()]
    prefixes += [(a,) for a in cars]
    prefixes += [(a,b) for a in cars for b in cars if a!=b]
    for prefix in prefixes:
        gp,cp=v2.struct_components(e,prefix,wg,wr)
        groupmembers=v2.line_members(e,prefix)
        result[prefix]=(gp,cp,groupmembers)
    return result

def decomposition(e,t,cached,ratio):
    """Log penalty RELATIVE to the uniform ordered triple, for comparability.

    group neutral probability is number of available riders in group /
    total riders remaining; within-line neutral=1/line's remaining riders.
    Under this normalization all 7 log penalties sum exactly to
    ln(uniform triple probability / actual frozen V3 triple probability).
    """
    p={}
    nall=len(e["scores"])
    for st,(prefix,c) in enumerate((((),t[0]),((t[0],),t[1]),((t[0],t[1]),t[2])),1):
        gp,cp,groups=cached[prefix]
        gid=e["positions"][c][0]
        rem=len(groups[gid]);total=nall-len(prefix)
        gprob=gp[gid];rprob=cp[c]
        if min(gprob,rprob)<=0:raise AssertionError("invalid conditional probabilities")
        p[f"{st}"+("st" if st==1 else "nd" if st==2 else "rd")+"_group"]=gprob
        p[COMPONENTS[2*(st-1)]]=math.log((rem/total)/gprob)
        p[COMPONENTS[2*(st-1)+1]]=math.log((1/rem)/rprob)
    if ratio<=0:raise AssertionError("invalid final V3 correction factor")
    p["pattern_gate"]=-math.log(ratio)
    return {k:p[k] for k in COMPONENTS}

def stage_conditionals(e,cache,conditional_local):
    """Calibrate all choices at ACTUAL preceding podium prefixes.

    Group of actual rider known only for rider-within-line diagnostic.
    Reports conditional distributions only; not an inference-time feature.
    """
    actual=e["order"]
    for st in (1,2,3):
        pre=actual[:st-1]
        gid=e["positions"][actual[st-1]][0]
        gp,cp,groupmembers=cache[pre]
        for g,p in gp.items():
            hit=int(g==gid)
            label=bucket(p,CONDITION_BINS)
            record(conditional_local,f"stage{st}_line_prob_bin",label,p,hit)
            record(conditional_local,"all_stages_line_prob_bin",label,p,hit)
        for c in groupmembers[gid]:
            p=cp[c]
            hit=int(c==actual[st-1])
            label=bucket(p,CONDITION_BINS)
            record(conditional_local,f"stage{st}_within_line_rider_prob_bin",label,p,hit)
            record(conditional_local,"all_stages_within_line_rider_prob_bin",label,p,hit)
        for g,members in groupmembers.items():
            for c in members:
                p=gp[g]*cp[c]
                hit=int(c==actual[st-1])
                label=bucket(p,CONDITION_BINS)
                record(conditional_local,f"stage{st}_rider_probability_bin",label,p,hit)
    for st in (1,2,3):
        for ax in (f"stage{st}_line_prob_bin",f"stage{st}_within_line_rider_prob_bin",
                   f"stage{st}_rider_probability_bin"):
            total=sum(z[0] for (a,k),z in conditional_local.items() if a==ax)
            hits=sum(z[1] for (a,k),z in conditional_local.items() if a==ax)
            if abs(total-1)>1e-8 or hits!=1:
                raise AssertionError(("bad stage probability mass",st,ax,total,hits))

def run_one(label,events,wg,wr,pattern_weights):
    agg=defaultdict(Cell)
    winners=[]
    n_races=0;n_tickets=0
    n_under=0;expected_under=0.;actual_under=0
    pattern_overall=Counter()
    for rid,e in events.items():
        n_races+=1
        cache=get_stage_components(e,wg,wr)
        old=v3.base_distribution(e,wg,wr)
        gate=v3.predict_non_aaa(e,pattern_weights)
        options,patterns=v3.variants(e,old,gate)
        prob=options[V3_LAMBDA][0]
        true=e["order"]
        local=defaultdict(lambda:[0.,0,0])
        clocal=defaultdict(lambda:[0.,0,0])
        stage_conditionals(e,cache,clocal)
        flush_local(clocal,agg,e["date"])
        if abs(sum(prob.values())-1)>1e-9:
            raise AssertionError("V3 per-race normalization violated")
        shape=v2.line_shape(e)
        pattern_overall[shape]+=1
        n_tickets+=len(prob)
        rarewinner=None
        for t,p in prob.items():
            truth=int(t==true)
            pat=patterns[t]
            ibin=bucket(p,TAIL_BINS)
            record(local,"probability_bands",ibin,p,truth)
            record(local,"all_line_shape_x_pattern",shape+"|"+pat,p,truth)
            record(local,"all_line_pattern",pat,p,truth)
            if p>=RARE:continue
            n_under+=1;expected_under+=p;actual_under+=truth
            factor=prob[t]/old[t]
            components=decomposition(e,t,cache,factor)
            winner_rank=str(e["ranks"][t[0]])
            roles=[v2.rlabel(e,c) for c in t]
            penalty=max(components,key=components.get)
            bottleneck_group="line" if penalty.endswith("_line") else (
                "rider" if penalty.endswith("_rider") else "pattern_gate"
            )
            checks=(
               ("rare_by_line_shape",shape),
               ("rare_by_line_pattern",pat),
               ("rare_by_shape_and_pattern",shape+"|"+pat),
               ("rare_by_first_score_rank",winner_rank),
               ("rare_by_first_role",roles[0]),
               ("rare_by_second_role",roles[1]),
               ("rare_by_third_role",roles[2]),
               ("rare_by_three_roles","-".join(roles)),
               ("rare_by_bottleneck",penalty),
               ("rare_by_bottleneck_type",bottleneck_group),
               ("rare_by_shape_and_bottleneck",shape+"|"+penalty),
               ("rare_by_pattern_and_bottleneck",pat+"|"+penalty),
               ("rare_finer_probability_bin",ibin),
            )
            record(local,"rare_all","ALL",p,truth)
            for ax,name in checks:
                record(local,ax,name,p,truth)
            if truth:
                n=len(e["scores"])
                uniform=1/(n*(n-1)*(n-2))
                reconstructed=sum(components.values())
                expected_log=math.log(uniform/p)
                if abs(expected_log-reconstructed)>1e-8:
                    raise AssertionError(("factorization error",rid,t,expected_log,reconstructed,components))
                rarewinner={
                    "race_id":rid,"race_date":e["date"],
                    "line_shape":shape,"trifecta":"-".join(t),
                    "line_pattern":pat,"roles_123":roles,
                    "score_ranks_123":[e["ranks"][c] for c in t],
                    "V3_probability":round(p,13),
                    "V3_probability_pct":round(100*p,11),
                    "V2_probability":round(old[t],13),
                    "pattern_gate_factor_V3_div_V2":round(factor,7),
                    "uniform_ordered_trifecta_probability":uniform,
                    "log_penalty_vs_uniform_by_component":{
                        k:round(v,8) for k,v in components.items()},
                    "largest_penalty_component":penalty,
                    "largest_penalty_vs_uniform":round(components[penalty],8),
                }
        flush_local(local,agg,e["date"])
        if rarewinner is not None:winners.append(rarewinner)
        if n_races%2000==0:
            print("V3_RARE_MECHANISM_CENSUS",label,n_races,
                  "ALL_TRIPLES",n_tickets,"RARE_ACTUAL",actual_under,flush=True)
    if len(winners)!=actual_under:
        raise AssertionError("Rare winning tickets inconsistent")
    if abs(agg["rare_all","ALL"].expected-expected_under)>0.001:
        raise AssertionError("Rare full ticket mass mismatch")
    if agg["rare_all","ALL"].actual!=actual_under:
        raise AssertionError("Rare actual hit mismatch")
    if agg["rare_all","ALL"].tickets!=n_under:
        raise AssertionError("Rare ticket count mismatch")
    if abs(sum(agg["all_line_pattern",k].expected for k in PATTERNS)-n_races)>1e-6:
        raise AssertionError("Pattern marginal mass != race count")
    if sum(agg["all_line_pattern",k].actual for k in PATTERNS)!=n_races:
        raise AssertionError("Pattern hit count != race count")
    for axis in ("rare_by_line_shape","rare_by_line_pattern","rare_by_bottleneck",
                 "rare_by_bottleneck_type","rare_finer_probability_bin"):
        selected=[c for (a,k),c in agg.items() if a==axis]
        if abs(sum(c.expected for c in selected)-expected_under)>1e-6 or \
           sum(c.actual for c in selected)!=actual_under or \
           sum(c.tickets for c in selected)!=n_under:
            raise AssertionError(("Rare disjoint group partition wrong",label,axis))
    # Selected 2024/2025/2026 tails must reproduce previous runs exactly.
    ref={"2024":(108,20.1459),"2025":(106,19.76645),"2026":(75,9.1639)}[label]
    if actual_under!=ref[0] or abs(expected_under-ref[1])>0.15:
        raise AssertionError(("2023-trained V3 rare probability not reproduced",label,actual_under,expected_under,ref))
    packed={}
    for (axis,key),cell in agg.items():
        packed.setdefault(axis,{})[key]=cell.summary()
    for axis in packed:
        packed[axis]=dict(sorted(packed[axis].items()))
    return {
        "year":label,"races":n_races,"all_trifecta_tickets":n_tickets,
        "distinct_line_shapes":len(pattern_overall),
        "line_shape_race_counts":dict(sorted(pattern_overall.items())),
        "all_eligible_races_equal_weight":True,
        "frozen_V3_rare_below_0_01pct":agg["rare_all","ALL"].summary(),
        "all_probability_bands":packed["probability_bands"],
        "all_pattern_marginals":packed["all_line_pattern"],
        "rare_breakdowns":{axis:packed.get(axis,{}) for axis in (
            "rare_by_line_shape","rare_by_line_pattern","rare_by_shape_and_pattern",
            "rare_by_first_score_rank","rare_by_first_role","rare_by_second_role",
            "rare_by_third_role","rare_by_three_roles","rare_by_bottleneck",
            "rare_by_bottleneck_type","rare_by_shape_and_bottleneck",
            "rare_by_pattern_and_bottleneck","rare_finer_probability_bin")},
        "conditional_stage_diagnostics":{k:v for k,v in packed.items() if (
            k.startswith("stage") or k.startswith("all_stages"))},
        "rare_trifecta_actual_observations":winners,
    }

def main():
    assert V3_LAMBDA==0.15
    assert original.TRAIN_END=="2023-10-31"
    source,coverage=original.ingest()
    train={rid:e for rid,e in source.items() if e["date"]<=original.TRAIN_END}
    assert len(train)==TRAIN_EXPECTED
    wg,wr=v2.fit_structural(train)
    pattern_weights,nonAAA=v3.train_pattern_gate(train)
    del train,source
    print("OLD_V3_TRAINING_FIXED_JAN_OCT_2023",flush=True)
    complete={}
    sources={}
    for yr,reader in (
        ("2024",source2024.ingest_2024),
        ("2025",source2025.read_2025),
        ("2026",source2026.ingest_2026),
    ):
        events,audit=reader()
        if len(events)!=EXPECTED_COUNTS[yr]:
            raise AssertionError(("Altered eligible sample",yr,len(events)))
        result=run_one(yr,events,wg,wr,pattern_weights)
        complete[yr]=result
        sources[yr]=audit
        del events
        print("V3_MECHANISM_YEAR_COMPLETE",json.dumps({
            "year":yr,"races":result["races"],
            "rare":result["frozen_V3_rare_below_0_01pct"],
            "strongest_bottleneck_by_events":{
                k:{"predicted":v["expected_events"],"actual":v["actual_events"]}
                for k,v in result["rare_breakdowns"]["rare_by_bottleneck"].items()},
            "shapes":len(result["line_shape_race_counts"])
        }),flush=True)
    doc={
        "state":"FROZEN_V3_ALL_LINE_SHAPES_RARE_TRIFECTA_MECHANISM_AUDIT_COMPLETE",
        "model":"V3 frozen source, 2023 Jan-Oct ONLY",
        "V3_mix_factor":0.15,
        "source_neutral_unbiased_V3_run":40837856314,
        "type":"observational model error decomposition, not causal claim or model improvement",
        "market_prices_odds_used":False,
        "original_V2_V3_code_changed":False,
        "2024_2025_2026_used_to_fit_model":False,
        "favored_line_shape_or_role":[] ,
        "all_shapes_sampled_without_prefilter":True,
        "yearly":complete,
        "source_coverage":sources,
        "methodology":{
            "full_ticket_universe":"all ordered trifectas for every eligible race",
            "rare_definition":"V3 trifecta P < 0.0001 = 0.01%, fixed from prior research",
            "bottleneck":"largest -log(P_component / uniform_component) across 3 stage line decisions, 3 stage within-line rider decisions, final V3 pattern gate",
            "stage1": "P(1st line) times P(1st rider within its line)",
            "stage2": "P(2nd line|1st rider) times P(2nd rider|line,1st rider)",
            "stage3": "P(3rd line|first two riders) times P(3rd rider|line,first two riders)",
            "pattern_gate":"V3 pattern-specific multiplier to preserve AAA and redistribute nonAAA",
            "identity":"sum of seven bottleneck contributions equals log(uniform ordered-triple probability / V3 triple probability)",
            "selection_bias_control":"Group expected and actual counts over ALL ordered triples, not only winners; conditional diagnostics are separate realized-prefix tests",
            "uncertainty":"Race-level Bernoulli variance and day-cluster robust approximations; no multiple-comparison-adjusted discovery guarantees",
            "not_a_causal_identification":"Bottleneck is a descriptive suppression factor; cause requires independently validated intervention."
        },
        "limitations":[
            "2024/2025/2026 outcomes have already been inspected in previous studies. This is exploratory, never a fresh holdout.",
            "Selection of conditional prefixes from actual podium finishes is valid only for conditional diagnostics, never as a pre-race feature.",
            "All comparisons operate on the model's own 0.01%-below cohort, not a market-implied odds band.",
            "Line and score historical availability before betting cutoff is not independently certified.",
            "Extreme cells with expected events <30 or few race exposures should not drive modeling.",
            "No automatic correction, production promotion, return model, or new bets are authorized."
        ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V3_ALL_SHAPES_RARE_MECHANISM_AUDIT_COMPLETE",json.dumps({
        "total_races":sum(complete[y]["races"] for y in complete),
        "total_rare_actual":sum(complete[y]["frozen_V3_rare_below_0_01pct"]["actual_events"] for y in complete),
        "total_rare_predicted":round(sum(complete[y]["frozen_V3_rare_below_0_01pct"]["expected_events"] for y in complete),5),
        "years":{k:{"rare":v["frozen_V3_rare_below_0_01pct"],
                    "bottlenecks":v["rare_breakdowns"]["rare_by_bottleneck_type"]}
                 for k,v in complete.items()}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

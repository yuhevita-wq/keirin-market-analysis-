#!/usr/bin/env python3
"""Observe V2/V3 within-line rider selection mistakes, all eligible race shapes.

NO correction, no new predictor, no odds; original 2023 Jan-Oct V2 training.
For each observed finishing stage, condition on the ACTUAL earlier podium prefix
and the line containing the ACTUAL finisher; compare every still-eligible rider's
P(rider | that line, podium prefix) with the observed winner.

This is a diagnostic conditional experiment, NOT an operational pre-race
predictor: actual previous placings and winning line may not be known pre-race.
The stage-internal observed finisher is NEVER used to produce probabilities.
Each candidate's feature tags derive only from pre-race card/formation and
the stage's observed prefix. Every stage's probabilities sum to 1.

2024 DISCOVERY / 2025 and 2026 historical recurrence, ALL previously inspected
in earlier research; none constitute pristine blind model evaluation. Any
hypothesis learned here must be frozen and tested against a NEW forward period.
All eligible shapes enter equally; any line-shape tables are DESCRIPTIVE only.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as original
import line_dependence_v2 as v2
import calibration_audit as src24
import validate_2025 as src25
import v4_pattern_floor_2026 as src26

OUT=ROOT/"data/audits/trifecta_all_shapes_rider_missed_arrival_structures_2024_2026.json"
RARE_P=.01
YEARS={"2024":25348,"2025":24635,"2026":11797}
AXES=(
    "p_band","role","field_rank","within_line_rank","gap_to_line_best",
    "gap_to_field_best","line_size","available_line_size",
    "role_x_field_rank","role_x_within_rank","role_x_gap_to_line_best",
    "within_rank_x_gap","role_x_line_size","prefix_line_relationship",
    "role_x_prefix_line_relationship","line_shape",
)
BINS_P=(
    (0.,.001,"below_0.1pct"),(.001,.01,"0.1_to_1pct"),
    (.01,.05,"1_to_5pct"),(.05,.1,"5_to_10pct"),
    (.1,.2,"10_to_20pct"),(.2,.5,"20_to_50pct"),
    (.5,1.000000001,"50pct_plus"),
)
FAMILY_KEYS=(
    "role_rank","role","rank","field_score","within_line_score",
    "relative_to_first","first_role","relative_to_second","second_role",
    "triple_role",
)

def bin_by(p,bins):
    for a,b,lab in bins:
        if a<=p<b:return lab
    raise AssertionError(("invalid bin",p))

def gap_bin(v):
    if v<=1e-9:return "zero"
    if v<=3:return "0_to_3"
    if v<=8:return "3_to_8"
    if v<=15:return "8_to_15"
    return "over_15"

def capped_rank(i):
    return str(i) if i<=3 else "4plus"

def family(key):
    name=key.split("|",1)[1] if "|" in key else key
    for prefix in FAMILY_KEYS:
        if name==prefix or name.startswith(prefix+"=") or name.startswith(prefix+"_"):
            return prefix
    raise AssertionError(("Unknown role feature",key))

class Cell:
    def __init__(self):
        self.expected=0.
        self.actual=0
        self.candidates=0
        self.stages=0
        self.var=0.
        self.daily=defaultdict(float)

    def add(self,mass,hit,count,day):
        if not -1e-8<=mass<=1.+1e-8 or hit not in (0,1):
            raise AssertionError(("Conditional partition invalid",mass,hit))
        mass=max(0.,min(1.,mass))
        self.expected+=mass;self.actual+=hit;self.candidates+=count
        self.stages+=1;self.var+=mass*(1-mass)
        self.daily[day]+=hit-mass

    def pack(self):
        n=len(self.daily)
        vals=list(self.daily.values())
        mean=sum(vals)/n if n else 0.
        std=math.sqrt(n/(n-1)*sum((v-mean)**2 for v in vals)) if n>1 else 0.
        delta=self.actual-self.expected
        return {
          "conditional_candidate_count":self.candidates,
          "conditional_stage_exposures":self.stages,
          "expected_winner_count":round(self.expected,5),
          "actual_winner_count":self.actual,
          "actual_minus_expected":round(delta,5),
          "actual_over_expected":round(self.actual/self.expected,5) if self.expected else None,
          "mean_predict_pct":round(self.expected/self.candidates*100,6) if self.candidates else None,
          "empirical_pct":round(self.actual/self.candidates*100,6) if self.candidates else None,
          "day_cluster_z_approx":round(delta/std,3) if std else None,
          "day_cluster_approx_95pct_residual_range":[round(delta-1.96*std,3),round(delta+1.96*std,3)]
              if std else None,
          "low_expected_lt_30":self.expected<30,
          "days":n,
        }

def add(local,axis,tag,p,won):
    data=local[axis,tag]
    data[0]+=p;data[1]+=int(won);data[2]+=1

def score_family_driver(e,prefix,c,members,wr):
    """Exact additive decomposition of rider's logit gap vs top candidate.

    Attribution describes the V2 MODEL'S internal suppression mechanism,
    NOT physical causality. The roles/ranks and scores can co-vary.
    """
    feats={x:v2.cfeatures(e,prefix,x) for x in members}
    logits={x:v2.dot(wr,feats[x]) for x in members}
    best=max(members,key=lambda x:(logits[x],-int(x)))
    keys=set(feats[best])|set(feats[c])
    vals=defaultdict(float)
    for k in keys:
        vals[family(k)]+=wr.get(k,0.)*(feats[best].get(k,0.)-feats[c].get(k,0.))
    gap=logits[best]-logits[c]
    if abs(sum(vals.values())-gap)>1e-7:
        raise AssertionError(("Cannot reconstruct logit gap",gap,sum(vals.values())))
    if gap<-1e-8:raise AssertionError("Negative best-minus-candidate logit")
    dominant=max(vals,key=vals.get) if vals else "none"
    return dominant,round(max(0.,gap),6),{
        k:round(v,6) for k,v in sorted(vals.items())
    }

def descriptors(e,prefix,c,members):
    g,pos,n=e["positions"][c]
    ordered=sorted(e["scores"],key=lambda x:(-e["scores"][x],int(x)))
    field_rank=ordered.index(c)+1
    within=sorted(members,key=lambda x:(-e["scores"][x],int(x))).index(c)+1
    r=v2.rlabel(e,c)
    line_gap=max(e["scores"][x] for x in members)-e["scores"][c]
    field_gap=max(e["scores"].values())-e["scores"][c]
    if len(prefix)==0:
        rel="first_place"
    elif len(prefix)==1:
        rel="same_first_line" if e["positions"][prefix[0]][0]==g else "different_first_line"
    else:
        ga=e["positions"][prefix[0]][0];gb=e["positions"][prefix[1]][0]
        if ga==gb:
            rel="same_as_both_previous" if g==ga else "different_from_paired_previous"
        elif g==ga:rel="returns_to_first_line"
        elif g==gb:rel="follows_second_line"
        else:rel="third_new_line"
    desc={
      "role":r,
      "field_rank":capped_rank(field_rank),
      "within_line_rank":capped_rank(within),
      "gap_to_line_best":gap_bin(line_gap),
      "gap_to_field_best":gap_bin(field_gap),
      "line_size":str(n),
      "available_line_size":str(len(members)),
      "role_x_field_rank":r+"|"+capped_rank(field_rank),
      "role_x_within_rank":r+"|"+capped_rank(within),
      "role_x_gap_to_line_best":r+"|"+gap_bin(line_gap),
      "within_rank_x_gap":capped_rank(within)+"|"+gap_bin(line_gap),
      "role_x_line_size":r+"|"+str(n),
      "prefix_line_relationship":rel,
      "role_x_prefix_line_relationship":r+"|"+rel,
      "line_shape":v2.line_shape(e),
    }
    return desc,field_rank,within,line_gap

def evaluate(year,events,wg,wr):
    totals=defaultdict(Cell)
    rare_examples=[]
    stage_winners={1:0,2:0,3:0}
    rare_events=Counter()
    conditional_nll={1:0.,2:0.,3:0.}
    conditional_brier={1:0.,2:0.,3:0.}
    all_shapes=Counter()
    for i,(rid,e) in enumerate(events.items(),1):
        all_shapes[v2.line_shape(e)]+=1
        true=e["order"]
        for st in (1,2,3):
            prefix=true[:st-1]
            winner=true[st-1]
            targetg=e["positions"][winner][0]
            gp,cp=v2.struct_components(e,prefix,wg,wr)
            groups=v2.line_members(e,prefix)
            members=groups[targetg]
            if not 0<=gp[targetg]<=1:raise AssertionError("Invalid line prob")
            if abs(sum(cp[x] for x in members)-1)>1e-9:
                raise AssertionError(("Rider conditional not normalized",year,rid,st))
            stage_winners[st]+=1
            pw=cp[winner]
            conditional_nll[st]-=math.log(max(pw,1e-15))
            conditional_brier[st]+=1-2*pw+sum(cp[x]**2 for x in members)
            local=defaultdict(lambda:[0.,0,0])
            for c in members:
                p=cp[c]
                won=int(c==winner)
                desc,field_rank,rankline,gap=descriptors(e,prefix,c,members)
                # EVERY eligible rider in observed line is an exposure,
                # not just riders who eventually won.
                add(local,f"s{st}/p_band",bin_by(p,BINS_P),p,won)
                for axis,val in desc.items():
                    add(local,f"s{st}/{axis}",val,p,won)
                if p<RARE_P:
                    rare_events[f"s{st}/candidates"]+=1
                    driver,margin,contrib=score_family_driver(e,prefix,c,members,wr)
                    add(local,f"s{st}/tail_all","all",p,won)
                    add(local,f"s{st}/tail_role",desc["role"],p,won)
                    add(local,f"s{st}/tail_field_rank",desc["field_rank"],p,won)
                    add(local,f"s{st}/tail_within_rank",desc["within_line_rank"],p,won)
                    add(local,f"s{st}/tail_role_x_field_rank",desc["role_x_field_rank"],p,won)
                    add(local,f"s{st}/tail_role_x_gap",desc["role_x_gap_to_line_best"],p,won)
                    add(local,f"s{st}/tail_logit_driver",driver,p,won)
                    add(local,f"s{st}/tail_logit_driver_x_role",driver+"|"+desc["role"],p,won)
                    add(local,f"s{st}/tail_line_shape",desc["line_shape"],p,won)
                    add(local,f"s{st}/tail_probability_bin",bin_by(p,BINS_P),p,won)
                    if won:
                        rare_events[f"s{st}/hits"]+=1
                        rare_examples.append({
                          "race_id":rid,"year":year,"date":e["date"],"stage":st,
                          "line_shape":v2.line_shape(e),
                          "winner_car_no":winner,
                          "prefix_actual_cars":list(prefix),
                          "selected_line_cars":list(members),
                          "within_line_probability_pct":round(p*100,9),
                          "line_probability_pct":round(gp[targetg]*100,7),
                          "role":desc["role"],"field_rank_uncapped":field_rank,
                          "line_rank_available_uncapped":rankline,
                          "score_gap_to_line_best":round(gap,3),
                          "score":e["scores"][c],
                          "biggest_logit_suppression_family":driver,
                          "logit_gap_to_top_rider":margin,
                          "model_internal_logit_gap_families":contrib,
                        })
            for (axis,name),(mass,hit,num) in local.items():
                totals[axis,name].add(mass,hit,num,e["date"])
            for axis in ("p_band",)+AXES[1:]:
                total_mass=sum(data[0] for (ax,t),data in local.items() if ax==f"s{st}/{axis}")
                total_hits=sum(data[1] for (ax,t),data in local.items() if ax==f"s{st}/{axis}")
                if abs(total_mass-1)>1e-8 or total_hits!=1:
                    raise AssertionError(("Non-exhaustive conditional stage axis",year,rid,st,axis,total_mass,total_hits))
            raremass=sum(v[0] for (axis,tag),v in local.items() if axis==f"s{st}/tail_all")
            if raremass>1+1e-9:raise AssertionError("Invalid rare conditional partition")
        if i%2500==0:
            print("RIDER_STRUCTURE_CENSUS",year,i,
                  "rare_wins",[rare_events.get(f"s{s}/hits",0) for s in (1,2,3)],flush=True)
    for st in (1,2,3):
        assert stage_winners[st]==len(events)
    # Historical 1% extreme-p selection stats are independent sanity checks.
    check={"2024":(44,13,14),"2025":(43,17,17),"2026":(42,13,10)}
    for st, expected_hits in enumerate(check[year],1):
        got=totals[f"s{st}/tail_all","all"].actual
        if got!=expected_hits:
            raise AssertionError(("Historical conditional-tail winner counts drifted",
                                  year,st,got,expected_hits))
    result={}
    for (axis,key),obj in totals.items():
        result.setdefault(axis,{})[key]=obj.pack()
    for ax in result:
        result[ax]=dict(sorted(result[ax].items()))
    return {
      "year":year,
      "races":len(events),
      "distinct_line_shapes":len(all_shapes),
      "all_line_shape_races":dict(sorted(all_shapes.items())),
      "conditional_NLL_by_stage":{str(s):round(conditional_nll[s]/len(events),8) for s in (1,2,3)},
      "conditional_Brier_by_stage":{str(s):round(conditional_brier[s]/len(events),8) for s in (1,2,3)},
      "per_stage_category_stats":result,
      "rare_conditional_winning_riders":sorted(
          rare_examples,key=lambda x:(x["stage"],x["within_line_probability_pct"],x["date"])),
      "total_rare_winner_examples":len(rare_examples),
    }

def shortlist(report):
    """Pre-declared filtering; these are exploratory repeated patterns only.

    A group must be independently well exposed in 2024 and 2025 and maintain
    direction in 2026. NO selection by line formation; shapes excluded here.
    """
    kinds=("role_x_field_rank","role_x_within_rank",
           "role_x_gap_to_line_best","within_rank_x_gap",
           "role_x_line_size")
    out=[]
    for st in (1,2,3):
        for kind in kinds:
            axis=f"s{st}/{kind}"
            keys=set(report["2024"]["per_stage_category_stats"][axis])
            for label in sorted(keys):
                v={yr:report[yr]["per_stage_category_stats"].get(axis,{}).get(label)
                   for yr in ("2024","2025","2026")}
                if any(item is None for item in v.values()):continue
                if not (v["2024"]["expected_winner_count"]>=15
                        and v["2025"]["expected_winner_count"]>=15
                        and v["2026"]["expected_winner_count"]>=7):
                    continue
                residuals=[v[y]["actual_minus_expected"] for y in ("2024","2025","2026")]
                if not (all(z>0 for z in residuals) or all(z<0 for z in residuals)):
                    continue
                z24=v["2024"]["day_cluster_z_approx"]
                z25=v["2025"]["day_cluster_z_approx"]
                if z24 is None or z25 is None or max(abs(z24),abs(z25))<1.96:
                    continue
                out.append({
                   "stage":st,"feature_axis":kind,"feature_value":label,
                   "yearly":v,
                   "sum_actual_minus_expected":round(sum(residuals),4),
                   "direction":"underpredicted" if residuals[0]>0 else "overpredicted",
                   "exploratory_not_bonferroni_or_BH_corrected":True,
                })
    return sorted(out,key=lambda x:-abs(x["sum_actual_minus_expected"]))

def main():
    data23,cov23=original.ingest()
    train={rid:e for rid,e in data23.items()
           if e["date"]<=original.TRAIN_END}
    assert original.TRAIN_END=="2023-10-31" and len(train)==19248
    wg,wr=v2.fit_structural(train)
    del train,data23
    print("FROZEN_V2_RIDER_WEIGHTS_TRAINED_2023_ONLY",flush=True)
    result={}
    coverage={}
    for year,reader in (("2024",src24.ingest_2024),
                        ("2025",src25.read_2025),
                        ("2026",src26.ingest_2026)):
        events,metadata=reader()
        if len(events)!=YEARS[year]:
            raise AssertionError(("Unexpected year cohort",year,len(events)))
        result[year]=evaluate(year,events,wg,wr)
        coverage[year]=metadata
        del events
        summary={
            "races":result[year]["races"],
            "shapes":result[year]["distinct_line_shapes"],
            "rare_stage_conditionals":{
                f"s{s}":result[year]["per_stage_category_stats"][f"s{s}/tail_all"]["all"]
                for s in (1,2,3)
            },
        }
        print("RIDER_STRUCTURE_YEAR_COMPLETE",year,json.dumps(summary,ensure_ascii=False),flush=True)

    recurring=shortlist(result)
    out={
      "state":"V3_RIDER_SELECTION_STRUCTURE_RESEARCH_ALL_SHAPES_COMPLETE",
      "model_trained":"Frozen V2 rider logits used inside frozen V3",
      "model_training_period":"2023-01-01..2023-10-31",
      "exploratory_discovery_2024":True,
      "historical_recurrence_2025_2026":True,
      "fresh_blind_out_of_sample_test":False,
      "market_odds_or_payout_used":False,
      "production_model_changed":False,
      "candidate_correction_performed":False,
      "specific_line_shape_prioritized":False,
      "all_eligible_line_shapes_included":True,
      "rider_conditional_diagnostic_caveat":"Actual line of eventual stage winner and previous observed placings are used to DEFINE the conditional evaluation, not as pre-race predictors.",
      "coeff_driver_caveat":"Logit feature contributions describe V2 model coefficient suppression; they do not establish real race causality.",
      "suspicion":"Overconfident within-line softmax may assign less than 1% to real winners; determine reproducible role/score-rank/score-gap structures before designing a conditional nonuniform alternative.",
      "predeclared_shortlist_rule":"No preferred line geometry. Score-rank/role/score-gap interactions only; E>=15 both 2024/25, >=7 in 2026, residual same sign all years, day-cluster |z|>1.96 in at least 2024 or 2025. Exploratory, multiple testing unadjusted.",
      "recurrent_structural_candidates":recurring,
      "years":result,
      "coverage":coverage,
      "limitations":[
        "2024-26 previously inspected; this is exploration, not independent proof.",
        "The conditional rider choice is evaluated inside the actual winner's line, not a stand-alone pre-race prediction.",
        "The previous-podium prefix is an actual outcome available for ex-post conditional diagnostics only.",
        "Observed role/score associations do not prove any mechanism causes an upset.",
        "Multiple testing and correlated features mean shortlist can contain false patterns.",
        "Field score rank is recalculated uncapped for discovery, but model training features cap rank at 4.",
        "Line histories and pre-race availability of archived forecast data require independent verification.",
        "No probability floor, model correction, ROI, wagering or production promotion."
      ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RIDER_STRUCTURE_DISCOVERY_FINISHED",json.dumps({
        "races":{y:v["races"] for y,v in result.items()},
        "recurrent_candidates":len(recurring),
        "top_candidate_signals":[{
            "stage":v["stage"],"axis":v["feature_axis"],"value":v["feature_value"],
            "direction":v["direction"],
            "residual":v["sum_actual_minus_expected"]
        } for v in recurring[:20]],
        "rare_stage_wins":{
            y:{str(st):result[y]["per_stage_category_stats"][f"s{st}/tail_all"]["all"]["actual_winner_count"]
               for st in (1,2,3)}
            for y in result
        },
        "model_changed":False,
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

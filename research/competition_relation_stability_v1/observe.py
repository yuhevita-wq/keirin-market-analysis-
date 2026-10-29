#!/usr/bin/env python3
"""競輪の競争関係は決着を固定するのか、揺らすのか: pre-race natural comparisons.

Predeclare and freeze hypotheses BEFORE looking at post-2023 records:
 H1: Two equally strong opposing *line leaders* increase underdog-third-line
     leader wins compared with clear top-two disparity, AFTER stratifying on
     that underdog's own strength gap, line shape, field and support.
 H2: Strong bante support behind the best-scored line leader increases
     same-line 1st-2nd finishes, controlling best leader lead, line shape and
     remaining leaders' strength.
 H3: Leader parity, third-line distance and bante support add out-of-year
     predictive information on race structure relative to field+line-shape.

Observed tactics, dashes, actual attacks and causes are UNKNOWN. "Race
competition" means pre-race published predicted lines + scores, not proven
causal mechanism. Do not fit V6 or modify odds/predictions. This is an
independent observational probe for mechanisms explaining V6 calibration.
2023 Jan-Oct train a shrinkage structure forecast, Nov-Dec dev; evaluate
2024-2026 without changing thresholds. Later years previously inspected.
Exclude entire races with score zero or missing. No payout, odds or refund.
All eligible field and line structures included for descriptive census;
3+ actual multi-rider lines required only for H1.
"""
from __future__ import annotations
import json,math,sys
from collections import defaultdict,Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as src23
import calibration_audit as src24
import validate_2025 as src25
import v4_pattern_floor_2026 as src26
import weak_rider_podium_zero_excluded as weak
import line_dependence_v2 as v2

OUT=ROOT/"data/audits/competition_relations_stability_2023_2026.json"
EXPECTED={"2023":23440,"2024":25348,"2025":24635,"2026":11797}
METRICS=("best_head_win","best_line_win","same_line_12",
         "three_distinct_lines","weak_score_win","best_line_head_bante_top2",
         "third_multi_head_win","third_multi_line_win")
PREDICT=("best_head_win","best_line_win","same_line_12","three_distinct_lines",
         "weak_score_win")
SHRINK_FINE=80.
SHRINK_BASE=100.

def cut(x,cuts):
    for i,t in enumerate(cuts):
        if x<t:return str(i)
    return str(len(cuts))

def describe(e):
    groups=e["groups"]
    if len(e["scores"])!=sum(map(len,groups)):
        raise AssertionError("Roster does not match published lines")
    n=len(e["scores"])
    sc=e["scores"]
    # Leader = person listed first in pre-race predicted line. Solo is
    # excluded only from the 3-multi-line H1 target, not the total census.
    heads=sorted(((sc[g[0]],len(g),g[0],i+1,g[1] if len(g)>1 else None)
                  for i,g in enumerate(groups)),key=lambda x:(-x[0],int(x[2])))
    multi=sorted((h for h in heads if h[1]>=2),key=lambda x:(-x[0],int(x[2])))
    best=heads[0]; runner=heads[1] if len(heads)>1 else None
    third_multi=multi[2] if len(multi)>=3 else None
    line_shape="-".join(map(str,sorted(map(len,groups),reverse=True)))
    winner=e["order"][0];sec=e["order"][1];third=e["order"][2]
    firstline=e["positions"][winner][0]
    rank_order=sorted(sc,key=lambda c:(-sc[c],int(c)))
    weak_cars=set(rank_order[math.ceil(n/2):])
    winner_line=firstline
    first_two_same=(firstline==e["positions"][sec][0])
    first_three_distinct=len({e["positions"][x][0] for x in (winner,sec,third)})==3
    best_line_top2=(
        len(groups[best[3]-1])>=2 and
        set(e["order"][:2])==set(groups[best[3]-1][:2])
    )
    outcomes={
        "best_head_win":int(winner==best[2]),
        "best_line_win":int(winner_line==best[3]),
        "same_line_12":int(first_two_same),
        "three_distinct_lines":int(first_three_distinct),
        "weak_score_win":int(winner in weak_cars),
        "best_line_head_bante_top2":int(best_line_top2),
        "third_multi_head_win":int(third_multi is not None and winner==third_multi[2]),
        "third_multi_line_win":int(third_multi is not None and winner_line==third_multi[3])
    }
    line_gap=best[0]-runner[0] if runner else 100.
    multi_gap=multi[0][0]-multi[1][0] if len(multi)>1 else 100.
    third_def=multi[0][0]-third_multi[0] if third_multi else 100.
    second_best=multi[0][4] if multi else None
    top_support_gap=(multi[0][0]-sc[second_best]) if second_best is not None else 100.
    if len(multi)>1:
        rival_head_gap=multi[0][0]-multi[1][0]
    else:rival_head_gap=100.
    # Relative to all riders; purposefully *not* the outcome.
    dispersion=max(sc.values())-min(sc.values())
    info={
       "n":n,"line_shape":line_shape,"num_lines":len(groups),
       "num_multi":len(multi),"best_head_gap":line_gap,
       "multi_top_head_gap":multi_gap,"third_multi_deficit":third_def,
       "best_bante_gap":top_support_gap,"rival_head_gap":rival_head_gap,
       "dispersion":dispersion,
       "third_multi_line_size":third_multi[1] if third_multi else None,
       "best_multi_line_size":multi[0][1] if multi else None,
       "race_class":e.get("race_class","UNSPECIFIED"),
       "parity_bin":cut(multi_gap,[2,4,7,12]),
       "third_deficit_bin":cut(third_def,[5,8,12,17,23]),
       "support_bin":cut(top_support_gap,[-4,0,3,6,10,15]),
       "dispersion_bin":cut(dispersion,[10,17,25,35]),
       "leader_hierarchy":"one clear leader" if line_gap>=8
                            else "two near-equal leaders" if line_gap<=3
                            else "intermediate",
       "top_multi_heads_within3":sum(h[0]>=multi[0][0]-3 for h in multi) if multi else 0,
    }
    return info,outcomes

class Simple:
    def __init__(self):
        self.n=0;self.out=Counter();self.days=set()
    def add(self,y,day):
        self.n+=1;self.days.add(day)
        for k,v in y.items():self.out[k]+=v
    def emit(self):
        if not self.n:return None
        return {"races":self.n,"days":len(self.days),
                "count":dict(self.out),
                "rates_pct":{k:round(100*self.out[k]/self.n,5) for k in METRICS}}

def descriptive(rows):
    result=defaultdict(Simple)
    for rid,e,f,y in rows:
        for k in (
            ("all","ALL"),
            ("field_size",str(f["n"])),
            ("line_shape",f["line_shape"]),
            ("leader_hierarchy",f["leader_hierarchy"]),
            ("multiple_head_close_3",str(min(4,f["top_multi_heads_within3"]))),
            ("field_dispersion",f["dispersion_bin"]),
            ("top_line_support",f["support_bin"]),
            ("multi_leader_parity",f["parity_bin"]),
            ("third_multi_gap",f["third_deficit_bin"]),
        ):result[k].add(y,e["date"])
    return {group:{key:cell.emit() for (axis,key),cell in sorted(result.items())
                   if axis==group}
            for group in ("all","field_size","line_shape","leader_hierarchy",
                          "multiple_head_close_3","field_dispersion",
                          "top_line_support","multi_leader_parity","third_multi_gap")}

def h1(r,f):
    # Require 3+ actual 2+-person racing lines; solo is not weak-line leader.
    if f["num_multi"]<3:return None
    if not 7<=f["third_multi_deficit"]<23:return None
    gap=f["multi_top_head_gap"]
    if gap<=3:arm="TOP_TWO_TIED"
    elif gap>=7:arm="TOP_TWO_SEPARATED"
    else:return None
    # Match strongest-to-weakest deficit before looking at winner; ensure the
    # weaker 3rd line is similarly disadvantaged and has comparable support.
    key=(f["n"],f["line_shape"],f["third_multi_line_size"],
         cut(f["third_multi_deficit"],[11,15,19]),
         cut(f["dispersion"],[15,25,35]))
    return arm,key

def h2(r,f):
    if f["num_multi"]<2 or f["best_multi_line_size"] is None:return None
    # Best multi-line head need not be overall best if solo has higher score;
    # skip to keep structural outcome's "best_head" fixed.
    if f["best_head_gap"]<0 or f["multi_top_head_gap"]>12:return None
    # For H2, group best by head must be an actual multi-rider line.
    if f["best_multi_line_size"]<2:return None
    if f["best_bante_gap"]<=3:arm="BANTE_CLOSE_SCORE"
    elif f["best_bante_gap"]>=9:arm="BANTE_FAR_BELOW"
    else:return None
    key=(f["n"],f["line_shape"],f["best_multi_line_size"],
         cut(f["multi_top_head_gap"],[2,5,9]),
         cut(f["third_multi_deficit"],[8,15,25]),
         cut(f["dispersion"],[15,25,35]))
    return arm,key

def matched(rows,experiment):
    groups=defaultdict(lambda:{"a":[0,Counter()],"b":[0,Counter()]})
    raw={"a":[0,Counter()],"b":[0,Counter()]}
    arms=("TOP_TWO_TIED","TOP_TWO_SEPARATED") if experiment=="H1" else (
          "BANTE_CLOSE_SCORE","BANTE_FAR_BELOW")
    target=("third_multi_head_win","third_multi_line_win","weak_score_win",
            "same_line_12") if experiment=="H1" else (
            "best_line_head_bante_top2","same_line_12","best_line_win",
            "three_distinct_lines")
    for rid,e,f,y in rows:
        entry=(h1 if experiment=="H1" else h2)(rid,f)
        if entry is None:continue
        arm,stratum=entry
        ix="a" if arm==arms[0] else "b"
        for group in (groups[stratum],raw):
            group[ix][0]+=1
            for k in target:group[ix][1][k]+=y[k]
    valid=[]
    for key,x in groups.items():
        a,b=x["a"][0],x["b"][0]
        if min(a,b)>=5:valid.append((key,x))
    # Overlap weighted empirical difference. Suppresses no-overlap strata.
    denom=0.;totals={k:0. for k in target}
    va=vb=0
    for key,x in valid:
        a,ca=x["a"];b,cb=x["b"]
        va+=a;vb+=b
        w=a*b/(a+b)
        denom+=w
        for k in target:totals[k]+=w*(ca[k]/a-cb[k]/b)
    raw_summ={arms[z]:{"races":raw[x][0],
                       "observed_pct":{k:round(100*raw[x][1][k]/raw[x][0],6)
                                if raw[x][0] else None for k in target}}
              for z,x in (("a","a"),("b","b"))}
    return {
       "hypothesis":experiment,"comparison_arms":list(arms),
       "raw_arms":raw_summ,"matched_strata_min5_each":len(valid),
       "matched_arm_counts":{"arm_a":va,"arm_b":vb},
       "matched_effect_A_minus_B_pct_points":
           {k:round(100*totals[k]/denom,7) if denom else None for k in target},
       "matched_effect_is_observational_not_causal":True,
       "thin_strata_excluded":True,
    }

def key_base(f):
    return (f["n"],f["line_shape"])
def key_fine(f):
    # Parity and support are pre-declared. A naive rank-only regression doesn't
    # see these relations among LINES.
    return key_base(f)+(f["parity_bin"],f["third_deficit_bin"],f["support_bin"])

def train_statistics(rows):
    base=defaultdict(lambda:[0,Counter()])
    fine=defaultdict(lambda:[0,Counter()])
    global_=Simple()
    for rid,e,f,y in rows:
        global_.add(y,e["date"])
        for key,tab in ((key_base(f),base),(key_fine(f),fine)):
            v=tab[key];v[0]+=1
            for k in PREDICT:v[1][k]+=y[k]
    return base,fine,global_

def probabilities(f,base,fine,total):
    k=key_base(f)
    n,c=base.get(k,(0,Counter()))
    n2,c2=fine.get(key_fine(f),(0,Counter()))
    ans={}
    for m in PREDICT:
        pglobal=(total.out[m]+1)/(total.n+2)
        qbase=(c[m]+SHRINK_BASE*pglobal)/(n+SHRINK_BASE)
        qfine=(c2[m]+SHRINK_FINE*qbase)/(n2+SHRINK_FINE)
        ans[m]={"baseline":qbase,"relation":qfine}
    return ans

def proper_score(rows,base,fine,total):
    d=defaultdict(lambda:{"n":0,"base_brier":0.,"relation_brier":0.,
                          "base_logloss":0.,"relation_logloss":0.,
                          "base_expected":0.,"relation_expected":0.,"observed":0})
    for rid,e,f,y in rows:
        p=probabilities(f,base,fine,total)
        for metric in PREDICT:
            yy=y[metric];o=d[metric];o["n"]+=1;o["observed"]+=yy
            for label in ("baseline","relation"):
                q=max(1e-12,min(1-1e-12,p[metric][label]))
                o[label+"_brier"]+=(yy-q)**2
                o[label+"_logloss"]-=yy*math.log(q)+(1-yy)*math.log(1-q)
                o[label+"_expected"]+=q
    return {metric:{"n":v["n"],
        "mean_rate_pct":round(100*v["observed"]/v["n"],6),
        "baseline_predicted_pct":round(100*v["base_expected"]/v["n"],6),
        "relation_predicted_pct":round(100*v["relation_expected"]/v["n"],6),
        "baseline_brier":round(v["base_brier"]/v["n"],9),
        "relation_brier":round(v["relation_brier"]/v["n"],9),
        "relative_brier_improvement_pct":round(
            100*(v["base_brier"]-v["relation_brier"])/v["base_brier"],6),
        "baseline_log_loss":round(v["base_logloss"]/v["n"],9),
        "relation_log_loss":round(v["relation_logloss"]/v["n"],9),
        "relative_logloss_improvement_pct":round(
            100*(v["base_logloss"]-v["relation_logloss"])/v["base_logloss"],6)
    } for metric,v in sorted(d.items())}

def prepare(events,year):
    stats=Counter();data=[]
    for rid,e in events.items():
        stats["source_races"]+=1
        gate=weak.score_gate(e)
        if gate!="ELIGIBLE":
            stats["excluded_"+gate]+=1;continue
        if len(e["groups"])<2:
            stats["one_line_excluded"]+=1;continue
        f,y=describe(e)
        data.append((rid,e,f,y))
        stats["eligible"]+=1
    assert stats["source_races"]==EXPECTED[year]
    print("COMPETITION_PREPARE",year,json.dumps(dict(stats)),flush=True)
    return data,dict(stats)

def main():
    all23,source23=src23.ingest()
    rows23,cover23=prepare(all23,"2023")
    training=[v for v in rows23 if v[1]["date"]<="2023-10-31"]
    dev=[v for v in rows23 if v[1]["date"]>"2023-10-31"]
    base,fine,total=train_statistics(training)
    print("COMPETITION_2023_FROZEN_PRIORS",
          json.dumps({"train":len(training),"dev":len(dev),
                      "coarse_groups":len(base),"fine_groups":len(fine)},flush=True))
    report={
       "state":"COMPETITOR_RELATIONSHIP_STABILITY_CAUSALITY_LIMITED_AUDIT",
       "source_definition":"Pre-race published line and scores (not actual racing moves)",
       "model_V6_and_data_unchanged":True,
       "no_odds_or_payout_used":True,
       "H1":"Top two multi-line leaders parity<=3 vs gap>=7, same third leader deficit, shape, field, line size and score spread",
       "H2":"Best multi-line leader bante gap<=3 vs >=9, same leader parity, third leader gap, shape, field and score spread",
       "H3":"2023-trained relationship score improves Brier/log-loss on 2024-2026 over field+line-shape base, NOT V6 itself",
       "training_only":"2023-01-01 to 2023-10-31",
       "historic_verification_not_genuine_unseen_holdout":True,
       "coverage":{"2023":cover23},"periods":{}
    }
    periods={"2023_train":training,"2023_dev":dev}
    del all23,rows23,training,dev
    for year,loader in (("2024",src24.ingest_2024),("2025",src25.read_2025),
                        ("2026",src26.ingest_2026)):
        events,source=loader()
        rows,cov=prepare(events,year)
        periods[year]=rows;report["coverage"][year]=cov
        del events
    periods["2024_to_2026"]=[v for y in ("2024","2025","2026") for v in periods[y]]
    for name,rows in periods.items():
        report["periods"][name]={
            "races":len(rows),
            "descriptive":descriptive(rows),
            "H1_competing_frontline_parity":matched(rows,"H1"),
            "H2_bante_support":matched(rows,"H2"),
            "H3_frozen_relational_predictive_value":proper_score(rows,base,fine,total),
        }
        out=report["periods"][name]
        print("COMPETITION_PERIOD_COMPLETE",json.dumps({
            "period":name,"races":len(rows),
            "H1":out["H1_competing_frontline_parity"],
            "H2":out["H2_bante_support"],
            "H3":out["H3_frozen_relational_predictive_value"],
            "class_warning":"2023 source has no class field, matching 2023 is within-field-and-shape but not class-controlled"
        },ensure_ascii=False),flush=True)
    report["limitations"]=[
      "H1 and H2 are observational matched associations, not causal identification or on-track dynamics",
      "2023 archive lacks race_class, H1/H2 not class-matched; 2024/25 class may be known but is also not matched here",
      "Support and leader strength are taken from pre-race listed line, not an independent measure of actual race tactics",
      "Outcomes are historically observed exact-race results, not forecasted from test outcomes",
      "Event-type forecast is a separate research tool, NOT the frozen V6 model",
      "2024-26 was viewed in other research; cannot claim fully untouched holdout",
      "No adjusted hypothesis-selection or causal effect claim; min 5/arm strata insufficient for rare exact-winner events",
    ]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    assert report["periods"]["2024_to_2026"]["races"]>55000
    print("COMPETITION_FINAL_RESULT",json.dumps({
        "races_by_period":{k:v["races"] for k,v in report["periods"].items()},
        "H1_2024_2026":report["periods"]["2024_to_2026"]["H1_competing_frontline_parity"],
        "H2_2024_2026":report["periods"]["2024_to_2026"]["H2_bante_support"],
        "H3_2024_2026":report["periods"]["2024_to_2026"]["H3_frozen_relational_predictive_value"],
        "key_descriptive":report["periods"]["2024_to_2026"]["descriptive"]["leader_hierarchy"]
        },ensure_ascii=False),flush=True)
if __name__=="__main__":main()

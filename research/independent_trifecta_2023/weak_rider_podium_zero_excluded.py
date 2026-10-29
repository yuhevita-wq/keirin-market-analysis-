#!/usr/bin/env python3
"""Research: HOW confirmed lower-scored cyclists enter the top-3.

Hard exclusion: if ANY starter has score=0, missing, negative or nonfinite,
entire race is excluded from this research, never silently treating unknown
as the weakest. Raw archives are unchanged.

Denominator is ALL eligible starters, not just actual podium finishers.
Primary 'weak' definition: lower half of positive pre-race scores, e.g.
7-rider race rank 5-7, 9-rider race rank 6-9. Secondary diagnostic rank 4+
is deliberately kept separate. Rank calculated from every starter's score
and tie-broken by rider number, not from model predictions or results.
Wins, runner-up, third, ANY top-3, and same-line companion occurrence are
tabulated separately by role, line size, exact rank, grade (where available),
field-score deficit, line-score deficit, and cross-line distribution.
All line compositions accepted. No odds, return, ticket-selection or
post-hoc profit claims.

For predictive relevance, compare two frozen 2023-trained binomial estimators
for 4 target outcomes on lower-half riders:
  - basic: field size + exact rank + field top-score deficit
  - line_context: basic + role + original line size + within-line deficit
The line model shrinks toward the basic model using 40 pseudo-observations
and the basic model shrinks toward training target frequency using 80.
No tuning on inspected 2024/25/26. 2023 Nov-Dec is development. All historic
replays are previously inspected, so NOT fresh unseen confirmation.
Scores for observed result are *never* used to form model predictors.
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
import observe as r23
import line_dependence_v2 as v2
import calibration_audit as r24
import validate_2025 as r25
import v4_pattern_floor_2026 as r26

OUT=ROOT/"data/audits/weak_rider_podium_zero_excluded_2023_2026.json"
YEARS={"2023":23440,"2024":25348,"2025":24635,"2026":11797}
LABELS=("first","second","third","top3")
SUFF_TRAIN=80
SUFF_LINE=40

def gapbin(v):
    if v<3:return "<3"
    if v<6:return "3-6"
    if v<10:return "6-10"
    if v<15:return "10-15"
    if v<22:return "15-22"
    return "22+"

def role(e,c):
    return v2.rlabel(e,c)

def shape(e):
    return "-".join(map(str,sorted((len(group) for group in e["groups"]),reverse=True)))

def pattern(e):
    groups=[e["positions"][c][0] for c in e["order"]]
    a,b,c=groups
    if a==b==c:return "AAA"
    if a==b:return "AAB"
    if a==c:return "ABA"
    if b==c:return "ABB"
    return "ABC"

def score_gate(e):
    """Hard race-level exclusion, including score zero, not rider-level filtering."""
    scores=e["scores"]
    if len(scores)<4 or len(scores)!=len(e["positions"]):
        return "INCOMPLETE_STARTER_CARD"
    if not all(isinstance(x,(float,int)) and math.isfinite(x) and x>0
               for x in scores.values()):
        return "UNKNOWN_ZERO_OR_INVALID_SCORE"
    if len(e["order"])<3 or len(set(e["order"][:3]))!=3 or any(
            x not in scores for x in e["order"][:3]):
        return "INVALID_PODIUM"
    return "ELIGIBLE"

def score_features(e):
    cars=sorted(e["scores"],key=lambda x:(-e["scores"][x],int(x)))
    n=len(cars)
    gap0=e["scores"][cars[0]]
    rows=[]
    for i,c in enumerate(cars,1):
        g,pos,sz=e["positions"][c]
        own=e["groups"][g-1]
        best_in_line=max(e["scores"][x] for x in own)
        weak=i>math.ceil(n/2)
        target={name:int(c==e["order"][j]) for j,name in enumerate(LABELS[:3])}
        target["top3"]=int(c in e["order"][:3])
        gap=gap0-e["scores"][c]
        in_gap=best_in_line-e["scores"][c]
        rows.append({
            "car":c,"rank":i,"weak":weak,"rank4plus":i>=4,
            "field_n":n,"score_gap":gap,"gap_bin":gapbin(gap),
            "own_line_gap":in_gap,"own_gap_bin":gapbin(in_gap),
            "role":role(e,c),"line_size":sz,"line_size_bin":str(min(sz,4)),
            "line_group":g,"target":target
        })
    return rows

def basic_key(z):
    return (z["field_n"],z["rank"],z["gap_bin"])

def line_key(z):
    return basic_key(z)+(z["role"],z["line_size_bin"],z["own_gap_bin"])

def apply_train(data):
    basic=defaultdict(lambda:[0,0,0,0,0])
    lines=defaultdict(lambda:[0,0,0,0,0])
    agg=[0,0,0,0,0]
    for e in data.values():
        if score_gate(e)!="ELIGIBLE":continue
        for z in score_features(e):
            if not z["weak"]:continue
            agg[0]+=1
            b=basic[basic_key(z)]
            l=lines[line_key(z)]
            b[0]+=1;l[0]+=1
            for j,label in enumerate(LABELS,1):
                hit=z["target"][label]
                agg[j]+=hit;b[j]+=hit;l[j]+=hit
    return basic,lines,agg

def prob(z,base,line,agg,label,with_line):
    j=LABELS.index(label)+1
    a,n=agg[j],agg[0]
    global_prior=a/n
    bn=base.get(basic_key(z),(0,0,0,0,0))
    q=(bn[j]+SUFF_TRAIN*global_prior)/(bn[0]+SUFF_TRAIN)
    if not with_line:return q
    ln=line.get(line_key(z),(0,0,0,0,0))
    return (ln[j]+SUFF_LINE*q)/(ln[0]+SUFF_LINE)

def safe_cell(item):
    if not item[0]:return None
    n=item[0]
    d={
        "exposures":n,
        "first":item[1],"second":item[2],"third":item[3],
        "top3":item[4],
    }
    for j,name in enumerate(LABELS,1):
        d[name+"_pct"]=round(100*item[j]/n,5)
    return d

def add_cell(x,z):
    x[0]+=1
    for j,label in enumerate(LABELS,1):
        x[j]+=z["target"][label]

def doc_stats(rows,by):
    d={}
    for axis in by:
        cells={}
        for z,e,_,_ in rows:
            label=axis(z,e)
            if label is None:continue
            if label not in cells:cells[label]=[0,0,0,0,0]
            add_cell(cells[label],z)
        d[getattr(axis,"__name__","unknown")]={k:safe_cell(v) for k,v in sorted(cells.items())}
    return d

def ax_rank(z,e):return str(z["rank"])
def ax_role(z,e):return z["role"]
def ax_field_gap(z,e):return z["gap_bin"]
def ax_own_gap(z,e):return z["own_gap_bin"]
def ax_line_size(z,e):return str(z["line_size"])
def ax_race_grade(z,e):return e.get("race_class","UNKNOWN_GRADE")
def ax_race_shape(z,e):return shape(e)
def ax_field_size(z,e):return str(z["field_n"])
def ax_role_field_gap(z,e):return z["role"]+"|"+z["gap_bin"]
def ax_role_line_size(z,e):return z["role"]+"|"+str(z["line_size"])
def ax_rank_role(z,e):return str(z["rank"])+"|"+z["role"]
def ax_own_gap_role(z,e):return z["own_gap_bin"]+"|"+z["role"]

AXES=[ax_rank,ax_role,ax_field_gap,ax_own_gap,ax_line_size,
      ax_race_grade,ax_race_shape,ax_field_size,ax_role_field_gap,
      ax_role_line_size,ax_rank_role,ax_own_gap_role]

class Score:
    def __init__(self):
        self.n=0
        self.loss=defaultdict(float)
        self.brier=defaultdict(float)
        self.measured=defaultdict(float)
        self.predicted=defaultdict(float)
    def add(self,feature,basic,line,agg):
        self.n+=1
        for label in LABELS:
            y=feature["target"][label]
            for model,w in (("basic",False),("line_context",True)):
                p=prob(feature,basic,line,agg,label,w)
                key=model+"|"+label
                self.loss[key]-=y*math.log(max(1e-12,p))+(1-y)*math.log(max(1e-12,1-p))
                self.brier[key]+=(y-p)**2
                self.predicted[key]+=p
                self.measured[key]+=y
    def pack(self):
        return {
          "evaluated_weak_riders":self.n,
          "score":{
            model:{
              label:{
                "binary_log_loss":round(self.loss[model+"|"+label]/self.n,9),
                "binary_Brier":round(self.brier[model+"|"+label]/self.n,9),
                "predicted_event_total":round(self.predicted[model+"|"+label],4),
                "actual_event_total":round(self.measured[model+"|"+label],4),
              } for label in LABELS
            } for model in ("basic","line_context")
          },
        }

def observe(events,year,base,line,agg,tag):
    excluded=Counter()
    eligible={}
    weak_rows=[]
    rank4_rows=[]
    role_outcome=defaultdict(lambda:[0,0,0,0,0])
    podium_pattern=Counter()
    podium_companions=Counter()
    weak_podium_items=Counter()
    race_patterns=Counter()
    weak_nonpodium=0
    scores=Score()
    for rid,e in events.items():
        gate=score_gate(e)
        if gate!="ELIGIBLE":
            excluded[gate]+=1
            continue
        eligible[rid]=e
        pr=pattern(e)
        race_patterns[pr]+=1
        members=score_features(e)
        for z in members:
            if z["rank4plus"]:
                rank4_rows.append((z,e,rid,year))
            if not z["weak"]:continue
            weak_rows.append((z,e,rid,year))
            scores.add(z,base,line,agg)
            add_cell(role_outcome[z["role"]],z)
            if z["target"]["top3"]:
                slot=next(j for j,c in enumerate(e["order"][:3],1) if c==z["car"])
                othercars=[x for x in e["order"][:3] if x!=z["car"]]
                companion=sum(e["positions"][a][0]==z["line_group"] for a in othercars)
                podium_pattern[f"slot{slot}|{pr}"]+=1
                podium_companions[f"slot{slot}|same_line_companions_{companion}"]+=1
                weak_podium_items[f"slot{slot}|role_{z['role']}|same_line_{companion}"]+=1
            else:weak_nonpodium+=1
    doc={
        "year_period":tag,
        "total_races_before_score_gate":len(events),
        "excluded_races":dict(excluded),
        "excluded_if_any_score_zero_or_invalid":True,
        "zero_races_cannot_contribute_results_or_denominators":True,
        "eligible_races":len(eligible),
        "line_shape_coverage":len({shape(e) for e in eligible.values()}),
        "eligible_field_sizes":dict(Counter(str(len(e["scores"])) for e in eligible.values())),
        "primary_weak_definition":"rank > ceil(n_riders / 2), score positive; 7-rider ranks 5-7, 9-rider ranks 6-9",
        "secondary_rank4plus_definition":"rank >=4, overlaps median in 7-rider fields",
        "primary_weak":safe_cell(_sum_cells(weak_rows)),
        "secondary_rank4plus":safe_cell(_sum_cells(rank4_rows)),
        "primary_weak_by":doc_stats(weak_rows,AXES),
        "primary_weak_podium_pattern":dict(sorted(podium_pattern.items())),
        "primary_weak_podium_companion":dict(sorted(podium_companions.items())),
        "primary_weak_role_and_companions":dict(sorted(weak_podium_items.items())),
        "weak_out_of_top3":weak_nonpodium,
        "race_actual_line_decision_patterns":dict(sorted(race_patterns.items())),
        "empirical_rate_model_comparison":scores.pack(),
    }
    if doc["eligible_races"]+sum(excluded.values())!=len(events):
        raise AssertionError("race exclusion accounting")
    if doc["primary_weak"]["top3"]!=sum(podium_pattern.values()):
        raise AssertionError("Podium event counting inconsistent")
    if doc["primary_weak"]["top3"]!=sum(podium_companions.values()):
        raise AssertionError("Companion event counting inconsistent")
    if doc["primary_weak"]["first"]+doc["primary_weak"]["second"]+doc["primary_weak"]["third"]!=doc["primary_weak"]["top3"]:
        raise AssertionError("Top3 cannot disagree with placement")
    print("WEAK_RIDER_PODIUM_SCORE_ZERO_EXCLUDED",json.dumps({
        "year":tag,"races":len(events),"score_eligible":len(eligible),
        "excluded":dict(excluded),"weak":doc["primary_weak"],
        "prediction":scores.pack(),
        "companion":dict(sorted(podium_companions.items()))
    },ensure_ascii=False),flush=True)
    return doc

def _sum_cells(rows):
    x=[0,0,0,0,0]
    for z,e,rid,year in rows:
        add_cell(x,z)
    return x

def main():
    data23,src23=r23.ingest()
    jan={k:v for k,v in data23.items() if v["date"]<="2023-10-31"}
    nov={k:v for k,v in data23.items() if v["date"]>"2023-10-31"}
    if (len(jan),len(nov))!=(19248,4192):
        raise AssertionError("2023 source drift")
    base,line,agg=apply_train(jan)
    print("WEAK_RIDER_PODIUM_2023_TRAIN",json.dumps({
        "eligible_weak_exposures":agg[0],
        "training_events":dict(zip(LABELS,agg[1:])),
        "training_buckets":{"basic":len(base),"line_context":len(line)}
    },ensure_ascii=False),flush=True)
    results={}
    results["2023_TRAIN"]=observe(jan,"2023",base,line,agg,"2023_TRAIN_IN_SAMPLE")
    results["2023_DEV"]=observe(nov,"2023",base,line,agg,"2023_DEV_EVALUATION")
    del data23,jan,nov
    audit={}
    for y,loader in (("2024",r24.ingest_2024),
                     ("2025",r25.read_2025),
                     ("2026",r26.ingest_2026)):
        data,stats=loader()
        if len(data)!=YEARS[y]:raise AssertionError((y,len(data)))
        results[y]=observe(data,y,base,line,agg,y+"_PREVIOUSLY_INSPECTED")
        audit[y]=stats
        del data
    def score_comparisons(name):
        x=results[name]["empirical_rate_model_comparison"]["score"]
        return {label:{
           "basic_NLL":x["basic"][label]["binary_log_loss"],
           "line_NLL":x["line_context"][label]["binary_log_loss"],
           "improvement":round(
             x["basic"][label]["binary_log_loss"]-
             x["line_context"][label]["binary_log_loss"],9)
        } for label in LABELS}
    out={
        "state":"WEAK_RIDER_PODIUM_ALL_LINE_SHAPES_WITH_ZERO_SCORE_RACES_EXCLUDED",
        "excluded_race_rule":"ANY starter score=0, nonfinite, negative or missing => exclude entire race; no rider-only exclusions",
        "meaning_of_zero":"UNKNOWN; never ranked as weak in this research",
        "rank_definition":"all-start positive-score ordinal rank; ties bike-number ordered",
        "weak_primary":"score-rank strictly in lower half of full field",
        "weak_secondary":"score-rank 4 or worse",
        "position_labels":["first","second","third","top3"],
        "study_goal":"which known lower-scored riders arrive in top 3 and where, conditioned on relative scores, line role, line size, race type, other line finishers and full race formation",
        "not_a_ticket_ev_or_market_proof":True,
        "no_betting_odds_or_payout":True,
        "model_fit_only_2023_jan_oct":True,
        "2023_dev_previously_inspected":True,
        "2024_2026_previously_inspected":True,
        "line_shapes_not_privileged":True,
        "2023_and_2026_race_grade_UNKNOWN":True,
        "time_of_data":"pre-race published predicted line, not guaranteed track actual",
        "caution":[
            "Retrospective observation not a causal mechanism or independent confirmation.",
            "A lower score relative to field is not necessarily absolute low ability, especially across race grade.",
            "Line pattern and same-line companionship are descriptions of realized podium; they are never used as pre-race predictions.",
            "Winning patterns observed after the race are not a price edge absent race-matched odds and withheld period.",
            "2023 Jan-Oct is training; all other historical results were examined before this study and cannot be treated as untouched forward holdouts.",
            "Race classes absent in 2023 and 2026 reader are labeled UNKNOWN; 2024/25 grades sourced by file type.",
            "Score nonpositive or absent could be collection issue or entrant ungraded; exclusion is a research choice, not a correction to original data."
        ],
        "training_counts":{
           "eligible_weak_exposures":agg[0],"target_counts":dict(zip(LABELS,agg[1:])),
           "basic_groups":len(base),"line_groups":len(line),
           "smoothing_basic":SUFF_TRAIN,"smoothing_line":SUFF_LINE},
        "year_results":results,
        "score_calibration_delta":{name:score_comparisons(name) for name in
          ("2023_DEV","2024","2025","2026")},
        "source_audits":audit
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("WEAK_RIDER_PODIUM_AUDIT_COMPLETE",
       json.dumps({"excluded":{k:v["excluded_races"] for k,v in results.items()},
                   "weak_podium":{k:v["primary_weak"] for k,v in results.items()},
                   "NLL_comparisons":out["score_calibration_delta"]},
                  ensure_ascii=False),flush=True)

if __name__=="__main__":main()

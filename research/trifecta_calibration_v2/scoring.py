#!/usr/bin/env python3
"""Pure pre-race V2 trifecta probability post-calibration (NO labels/odds).

For arbitrary future race, supply exactly:
    base_probabilities: all 3連単 probabilities calculated by FROZEN V1
    predicted_line_formation: e.g. "1-2-3/4-5/6-7"
    scores: {car_no: archived pre-race competition score}
    coefficients: versioned JSON calibrator learned from 2024 only
Returns a probability for EVERY ordered triple, summing to 1.
No result, winner, odds, or payout is accepted or read.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

SCHEMA="frozen_keirin_3trifecta_no_odds_postcalibrator_v2"
FEATURE_COUNT=8

def parse_line_and_scores(formation,scores):
    if not isinstance(formation,str) or not formation.strip():
        raise ValueError("Published line formation is missing")
    if not isinstance(scores,dict) or len(scores)<4 or len(scores)>9:
        raise ValueError("Need 4-9 non-girls participant score entries")
    carset={str(k) for k in scores}
    if any(not c.isdigit() or not 1<=int(c)<=9 for c in carset):
        raise ValueError("Rider keys must be valid car numbers")
    number_scores={}
    for c,v in scores.items():
        f=float(v)
        if not math.isfinite(f):raise ValueError("Non-finite competition score")
        number_scores[str(c)]=f
    parts=[];positions={}
    for group_index,chunk in enumerate(formation.strip().split("/")):
        group=chunk.strip().split("-")
        if not group or any(not c.isdigit() or not 1<=int(c)<=9 for c in group):
            raise ValueError("Invalid published line")
        parts.append(group)
        for role,car in enumerate(group):
            if car in positions:raise ValueError("Repeated car in published line")
            positions[car]=(group_index,role+1,len(group))
    if set(positions)!=carset:raise ValueError("Line does not cover all participants")
    order=sorted(number_scores,key=lambda k:(-number_scores[k],int(k)))
    rank1=order[0]
    return positions,rank1,sorted(carset,key=int)

def load_coefficients(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema")!=SCHEMA or not data.get("holdout_2025_not_used"):
        raise ValueError("Unexpected or unsealed v2 weight manifest")
    vals=data.get("coefficients")
    if not isinstance(vals,list) or len(vals)!=FEATURE_COUNT or any(not math.isfinite(float(x)) for x in vals):
        raise ValueError("Bad v2 coefficients")
    return [float(v) for v in vals]

def apply_v2(base_probabilities,formation,scores,coefficients):
    """Calculate fully normalized ticket probabilities WITHOUT any race outcomes.

    base_probabilities: dict e.g. {"1-2-3":0.01,...} over all permutations.
    Returned dict has exactly the same ticket keys.
    """
    if len(coefficients)!=FEATURE_COUNT or any(not math.isfinite(float(v)) for v in coefficients):
        raise ValueError("Expected 8 finite, versioned coefficients")
    w=[float(v) for v in coefficients]
    positions,rank1,cars=parse_line_and_scores(formation,scores)
    expected={"-".join(t) for t in itertools.permutations(cars,3)}
    if not isinstance(base_probabilities,dict) or set(base_probabilities)!=expected:
        raise ValueError(f"Incomplete/mismatched base 3連単 grid: need {len(expected)} combinations")
    values={};raw_sum=0.0
    for ticket,p in base_probabilities.items():
        p=float(p)
        if not(math.isfinite(p) and 0<p<=1):
            raise ValueError("Each base probability must be positive and finite")
        raw_sum+=p
        a,b,c=ticket.split("-")
        la=positions[a][0];lb=positions[b][0];lc=positions[c][0]
        s12=float(la==lb);s13=float(la==lc);s23=float(lb==lc)
        features=(math.log(p),s12,s13,s23,float(la==lb==lc),
                  float(len({la,lb,lc})==3),float(positions[a][1]==1),
                  float(a==rank1))
        values[ticket]=sum(u*v for u,v in zip(w,features))
    if abs(raw_sum-1)>2e-6:
        raise ValueError(f"Input base 3連単 total probability must equal 1; got {raw_sum}")
    offset=max(values.values())
    exp={k:math.exp(v-offset) for k,v in values.items()}
    z=sum(exp.values())
    result={k:v/z for k,v in exp.items()}
    if len(result)!=len(expected) or abs(sum(result.values())-1)>1e-9:
        raise AssertionError("V2 probability conservation failed")
    return result

def main():
    parser=argparse.ArgumentParser(description="Apply frozen odds-free 3連単 v2 correction to a pre-race distribution")
    parser.add_argument("--race-input",required=True,help="JSON with base_probabilities, formation, scores")
    parser.add_argument("--weights",required=True,help="Frozen v2 calibrator JSON")
    parser.add_argument("--output",required=True,help="Output JSON for all ordered 3連単 probabilities")
    args=parser.parse_args()
    payload=json.loads(Path(args.race_input).read_text(encoding="utf-8"))
    probs=apply_v2(payload["base_probabilities"],payload["predicted_line_formation"],
                   payload["scores"],load_coefficients(args.weights))
    ordered=sorted(probs.items(),key=lambda x:(-x[1],x[0]))
    output={
       "status":"PRE_RACE_INDEPENDENT_PROBABILITIES_NO_ODDS",
       "model":"v2",
       "race_id":payload.get("race_id"),
       "number_of_trifecta_outcomes":len(probs),
       "probability_sum":sum(probs.values()),
       "all_ticket_probabilities":probs,
       "top10":[{"trifecta":k,"probability":p,"probability_pct":100*p} for k,p in ordered[:10]]
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V2_SCORING_COMPLETE",len(probs),round(sum(probs.values()),12),args.output)

if __name__=="__main__":
    main()

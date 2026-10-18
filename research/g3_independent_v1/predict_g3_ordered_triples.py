#!/usr/bin/env python3
"""G3 independent 1st+2nd+3rd ordered full probability distribution.

Uses independently fitted G3 first q (input) + previously frozen G3 second
conditional v1 + newly fitted G3 third conditional v1. No F1 models.
Uses no actual results to predict, no odds and no automatic bet selection.
All distinct ordered triples n*(n-1)*(n-2) are evaluated (9-rider: 504).
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from predict_g3_first_second import infer as infer_two,parse_card

HERE=Path(__file__).resolve().parent
DEFAULT_SECOND=HERE/"models/g3_second_conditional_v1.json"
DEFAULT_THIRD=HERE/"models/g3_third_conditional_v1.json"

def third_position(first,second,third,where):
    li,pi=where[first];lj,pj=where[second];lk,pk=where[third]
    if len({first,second,third})!=3:
        raise ValueError("Each finishing place must have distinct rider")
    if lk==li==lj:
        return ("direction_ahead_of_both" if pk<min(pi,pj) else
                "direction_behind_both" if pk>max(pi,pj) else
                "direction_between_first_second")
    if lk==li:
        return ("direction_ahead_of_first" if pk<pi else
                "direction_directly_behind_first" if pk==pi+1 else
                "direction_further_behind_first")
    if lk==lj:
        return ("direction_ahead_of_second" if pk<pj else
                "direction_directly_behind_second" if pk==pj+1 else
                "direction_further_behind_second")
    return None # neither of the actual first and second rider's predicted lines

def third_role(third,where,lines):
    line,pos=where[third]
    if len(lines[line])==1:return "role_solo"
    if pos==0:return None # leader reference coefficient 0
    if pos==1:return "role_second"
    return "role_third_or_later"

def all_ordered_triples(card,second_model,third_model):
    if third_model.get("status")!="G3_THIRD_RESEARCH_REFERENCE_SELECTED_BY_2025_VALIDATION":
        raise ValueError("Missing or unapproved independent G3 third-place model")
    if third_model.get("scope")!="G3開催 S級 only 5-9 starters; no G1 G2 F1 F2 A級 girls":
        raise ValueError("G3 third model scope mismatch")
    q,lines=parse_card(card)
    seconds=infer_two(card,second_model)
    where={car:(line_idx,idx) for line_idx,line in enumerate(lines)
           for idx,car in enumerate(line)}
    par=third_model["coefficients"]
    d=float(par["log_q"])
    marg3={car:0.0 for car in q}
    outputs=[]
    for row in seconds["pairs"]:
        first=int(row["first_car_no"]);second=int(row["second_car_no"])
        joint=float(row["joint_first_second_probability"])
        scores={}
        for third,prob in q.items():
            if third in (first,second):continue
            pos=third_position(first,second,third,where)
            role=third_role(third,where,lines)
            scores[third]=d*math.log(prob)+(float(par[pos]) if pos else 0.)+(
                float(par[role]) if role else 0.)
        maxs=max(scores.values())
        weighted={k:math.exp(v-maxs) for k,v in scores.items()}
        normal=sum(weighted.values())
        all_cond=0.
        for third in sorted(weighted):
            third_given=weighted[third]/normal
            all_cond+=third_given
            p=joint*third_given
            marg3[third]+=p
            outputs.append({
                "first_car_no":first,"second_car_no":second,
                "third_car_no":third,
                "conditional_third_given_first_second":third_given,
                "joint_first_second_third_probability":p
            })
        if abs(all_cond-1)>1e-12:
            raise AssertionError("Third conditional not normalized within exacta")
    n=len(q)
    if len(outputs)!=n*(n-1)*(n-2):
        raise AssertionError("Every distinct 1-2-3 triple must be enumerated")
    mass=sum(t["joint_first_second_third_probability"] for t in outputs)
    if abs(mass-1)>1e-9 or abs(sum(marg3.values())-1)>1e-9:
        raise AssertionError("G3 ordered triple distribution not normalized")
    for pair in seconds["pairs"]:
        i,j=pair["first_car_no"],pair["second_car_no"]
        recover=sum(t["joint_first_second_third_probability"] for t in outputs
                    if t["first_car_no"]==i and t["second_car_no"]==j)
        if abs(recover-pair["joint_first_second_probability"])>1e-11:
            raise AssertionError("Third model must preserve frozen G3 two-place probabilities")
    outputs.sort(key=lambda t:(-t["joint_first_second_third_probability"],
                               t["first_car_no"],t["second_car_no"],t["third_car_no"]))
    return {
        "scope":"G3開催 S級 only",
        "model":"G3 independent first win + G3 second conditional v1 + G3 third conditional v1",
        "all_distinct_triples_count":len(outputs),
        "first_marginal":seconds["first_marginal"],
        "second_marginal":seconds["second_marginal"],
        "third_marginal":[{"car_no":x,"probability":marg3[x]} for x in sorted(marg3)],
        "ordered_triples":outputs,
        "market_odds_used":False,
        "expected_value_or_ticket_selection_performed":False,
        "no_observed_results_allowed":True
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True,help="G3-only independent q and predicted formation without known results")
    ap.add_argument("--second",type=Path,default=DEFAULT_SECOND)
    ap.add_argument("--third",type=Path,default=DEFAULT_THIRD)
    args=ap.parse_args()
    card=json.loads(args.input.read_text(encoding="utf-8"))
    second=json.loads(args.second.read_text(encoding="utf-8"))
    third=json.loads(args.third.read_text(encoding="utf-8"))
    print(json.dumps(all_ordered_triples(card,second,third),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

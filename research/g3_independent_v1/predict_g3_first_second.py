#!/usr/bin/env python3
"""Turn G3-only independent 1st probabilities into EVERY ordered 1st/2nd pair.

Independent 1st probabilities q are passed from the separate G3-first model.
Neither historical nor live results, public odds or ticket prices are inputs.
This computes full probabilities, NOT positive-EV tickets.
"""
from __future__ import annotations
import argparse,json,math,re,unicodedata
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT_MODEL=HERE/"models/g3_second_conditional_v1.json"
RE_LINE=re.compile(r"^[1-9](?:-[1-9])*$")
NOT_PRE_RACE={"result","results","winner","first","second","third",
              "finish_position","odds","payout"}

def normalized(s):
    return unicodedata.normalize("NFKC",str(s or "")).upper().strip()

def parse_card(card):
    if NOT_PRE_RACE.intersection(card):
        raise ValueError("Historical results, odds and finishes forbidden in pre-race G3-independent model")
    if normalized(card.get("meeting_grade")) not in ("G3","GIII"):
        raise ValueError("Only G3 grades are supported")
    if not normalized(card.get("race_type")).startswith("S級"):
        raise ValueError("Only G3開催S級 supported")
    if card.get("probability_model")!="G3 independent 1st v1":
        raise ValueError("First probabilities must originate from independent G3 first model")
    rows=card.get("first_win")
    if not isinstance(rows,list) or not 5<=len(rows)<=9:
        raise ValueError("Expected 5-9 G3 independent first probabilities")
    q={}
    for x in rows:
        if not isinstance(x,dict) or set(x)!={"car_no","probability"}:
            raise ValueError("Each 1st probability must be {car_no,probability}")
        car=x["car_no"]
        if isinstance(car,bool):
            raise ValueError("Invalid car")
        car=int(car)
        p=float(x["probability"])
        if not 1<=car<=9 or car in q or not math.isfinite(p) or p<=0 or p>=1:
            raise ValueError("Invalid q value, duplicate car number or nonpositive probability")
        q[car]=p
    if abs(sum(q.values())-1)>1e-7:
        raise ValueError("First probabilities must sum to 1 within tolerance")
    formation=str(card.get("predicted_line_formation","")).strip().split("/")
    if not formation or any(not RE_LINE.fullmatch(x) for x in formation):
        raise ValueError("Predicted formation invalid")
    lines=[[int(v) for v in s.split("-")] for s in formation]
    flat=[x for line in lines for x in line]
    if len(flat)!=len(set(flat)) or set(flat)!=set(q):
        raise ValueError("Predicted formation must cover all and only G3 entrants")
    return q,lines

def conditional_position(i,j,positions):
    li,pi=positions[i];lj,pj=positions[j]
    if li!=lj:return "other_line"
    if pj==pi+1:return "directly_behind_first"
    if pj<pi:return "ahead_of_first"
    if pj>pi+1:return "further_behind_first"
    raise ValueError("Duplicate position")

def role(j,lines,positions):
    li,pj=positions[j]
    group=lines[li]
    if len(group)==1:return "role_solo"
    if pj==0:return "role_leader"
    if pj==1:return "role_second"
    return "role_third_or_later"

def infer(card,model):
    if model.get("status")!="ADOPTED_AS_G3_RESEARCH_REFERENCE_NOT_EXPECTED_VALUE_BETTING":
        raise ValueError("Unknown G3 2nd model version")
    if model.get("scope")!="G3開催 S級 only, 5-9 riders; no F1 or other grades":
        raise ValueError("G3-only 2nd model scope invalid")
    q,lines=parse_card(card)
    positions={car:(a,pos) for a,line in enumerate(lines) for pos,car in enumerate(line)}
    params=model["coefficients"]
    gamma=float(params["gamma"])
    second={car:0.0 for car in q}
    pairs=[]
    for i,qi in sorted(q.items()):
        logits={}
        for j,qj in q.items():
            if i==j:continue
            d=float(params[conditional_position(i,j,positions)])
            r=float(params[role(j,lines,positions)])
            logits[j]=gamma*math.log(qj)+d+r
        maxlog=max(logits.values())
        w={j:math.exp(logit-maxlog) for j,logit in logits.items()}
        norm=sum(w.values())
        for j in sorted(w):
            conditional=w[j]/norm
            joint=qi*conditional
            second[j]+=joint
            pairs.append({
                "first_car_no":i,"second_car_no":j,
                "conditional_second_given_first":conditional,
                "joint_first_second_probability":joint
            })
    n=len(q)
    if len(pairs)!=n*(n-1):
        raise AssertionError("All ordered first-second combinations must be enumerated")
    if abs(sum(x["joint_first_second_probability"] for x in pairs)-1)>1e-9:
        raise AssertionError("Sum of ordered exacta probabilities is not 1")
    for i,v in q.items():
        if abs(sum(x["joint_first_second_probability"] for x in pairs
                   if x["first_car_no"]==i)-v)>1e-9:
            raise AssertionError("Did not preserve original independent first probability")
    if abs(sum(second.values())-1)>1e-9:
        raise AssertionError("Second marginal probabilities do not sum to 1")
    return {
        "scope":"G3 S級 only",
        "model":"independent G3 first q + fitted G3 second conditional v1",
        "market_odds_used":False,
        "ordered_first_second_pair_count":len(pairs),
        "first_marginal":[{"car_no":i,"probability":v} for i,v in sorted(q.items())],
        "second_marginal":[{"car_no":j,"probability":v} for j,v in sorted(second.items())],
        "pairs":pairs,
        "disclaimer":"Probability estimation only. No all-ticket market comparison or betting expected value."
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True,
                        help="G3-first-q and published predicted formation, without outcomes")
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    a=parser.parse_args()
    card=json.loads(a.input.read_text(encoding="utf-8"))
    model=json.loads(a.model.read_text(encoding="utf-8"))
    print(json.dumps(infer(card,model),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

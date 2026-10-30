#!/usr/bin/env python3
"""Read the saved V6-C1 calibrator and apply it to any frozen V6 triple grid.

The caller must supply P_V6 for ALL ordered top-three tuples.
This module never uses odds, payouts, or observed finishes, and never refits V6.
It uses only line identity, line role, and the supplied V6 triple probability.
"""
from __future__ import annotations
import bisect
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_MODEL=ROOT/"data/audits/v6_c1_full_data_crossfit_2023_2026/v6_c1_fitted_model.json"
PAT=("AAA","AAB","ABA","ABB","ABC")
ROLE=("leader","second","third","solo","fourth_plus")
BND=(.0001,.0005,.001,.005,.01,.02,.05,.1,.2)

def read_model(path=DEFAULT_MODEL):
    model=json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("model_name")!="V6-C1" or model.get("total_races")!=84065:
        raise ValueError("Unrecognized V6-C1 reference model")
    if model.get("no_market_odds_in_features_or_labels") is not True:
        raise ValueError("Unexpected calibration source")
    if len(model["cell_multipliers"])!=250:
        raise ValueError("V6-C1 requires 250 learned cell weights")
    if not all(math.isfinite(v) and v>0 for v in model["cell_multipliers"]):
        raise ValueError("Invalid V6-C1 calibration multipliers")
    return model

def _role(e,car):
    _,pos,size=e["positions"][car]
    if size==1:return "solo"
    if pos==1:return "leader"
    if pos==2:return "second"
    if pos==3:return "third"
    return "fourth_plus"

def _cell(e,t,p):
    a,b,c=t
    x=e["positions"][a][0];y=e["positions"][b][0];z=e["positions"][c][0]
    pattern=("AAA" if x==y==z else "AAB" if x==y
             else "ABA" if x==z else "ABB" if y==z else "ABC")
    role=_role(e,a)
    return (PAT.index(pattern)*10+bisect.bisect_right(BND,p))*5+ROLE.index(role)

def apply_calibration(card,p_v6,model=None):
    """Return V6-C1's full, normalized ordered-trifecta distribution.

    card: existing V6 pre-race card with positions; NO outcome or odds needed
    p_v6: dict[(first, second, third)] -> positive V6 probability
    model: parsed read_model() output, or None for repository saved reference
    """
    if model is None:model=read_model()
    cars=set(card["scores"])
    if not cars:raise ValueError("Empty race card")
    if len(p_v6)!=len(cars)*(len(cars)-1)*(len(cars)-2):
        raise ValueError("Incomplete ordered-triple probability grid")
    if any(len(t)!=3 or len(set(t))!=3 or not set(t)<=cars for t in p_v6):
        raise ValueError("Invalid tuple in ordered-triple probability grid")
    if not all(math.isfinite(p) and p>0 for p in p_v6.values()):
        raise ValueError("V6 probabilities must be positive finite")
    if abs(sum(p_v6.values())-1.)>1e-8:raise ValueError("V6 distribution not normalized")
    weights=model["cell_multipliers"]
    raw={t:p*weights[_cell(card,t,p)] for t,p in p_v6.items()}
    z=sum(raw.values())
    if not z>0 or not math.isfinite(z):raise ValueError("Invalid calibrated mass")
    out={t:v/z for t,v in raw.items()}
    if abs(sum(out.values())-1.)>1e-9 or not all(v>0 for v in out.values()):
        raise ValueError("V6-C1 mass conservation failed")
    return out

def _self_test():
    import itertools
    cars=("1","2","3","4","5")
    card={"scores":{c:70+i for i,c in enumerate(cars)},
          "positions":{
            "1":(1,1,3),"2":(1,2,3),"3":(1,3,3),
            "4":(2,1,2),"5":(2,2,2)}}
    t=list(itertools.permutations(cars,3))
    base={c:1./len(t) for c in t}
    identity={"cell_multipliers":[1.]*250}
    assert max(abs(v-base[k]) for k,v in apply_calibration(card,base,identity).items())<1e-12
    tilted={"cell_multipliers":[1.+i/500. for i in range(250)]}
    q=apply_calibration(card,base,tilted)
    assert len(q)==60 and abs(sum(q.values())-1)<1e-12
    print("V6_C1_APPLY_60_ORDERED_TICKETS_AND_IDENTITY_VERIFIED")

if __name__=="__main__":
    _self_test()

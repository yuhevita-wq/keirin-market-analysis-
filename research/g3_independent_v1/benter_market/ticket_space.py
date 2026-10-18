"""G3 odds ↔ exact ordered finishing outcomes (7 official Keirin bet types).

Never imply that an offered ticket exists merely because its outcome is
mathematically possible. Availability and prices come from the odds book.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from math import isfinite

BET_TYPES=("2車単","2車複","3連単","3連複","ワイド","2枠単","2枠複")
# Official 9-rider frame mapping; for withdrawals retain original car/frame.
FRAME_BY_CAR={1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:6,9:6}
LENGTH={"2車単":2,"2車複":2,"3連単":3,"3連複":3,
        "ワイド":2,"2枠単":2,"2枠複":2}
UNORDERED={"2車複","3連複","ワイド","2枠複"}

def canonical(bet_type,selection):
    if bet_type not in BET_TYPES:
        raise ValueError("Unrecognized G3 ticket type: "+str(bet_type))
    if not isinstance(selection,(list,tuple)) or len(selection)!=LENGTH[bet_type]:
        raise ValueError("Incorrect selection size")
    if any(isinstance(v,bool) or not isinstance(v,int) for v in selection):
        raise ValueError("Ticket identifiers must be integers")
    key=tuple(sorted(selection)) if bet_type in UNORDERED else tuple(selection)
    limit=6 if bet_type.startswith("2枠") else 9
    if not all(1<=v<=limit for v in key):
        raise ValueError("Invalid car or frame number")
    if bet_type not in ("2枠単","2枠複") and len(set(key))!=len(key):
        raise ValueError("A car cannot appear twice in a distinct-place ticket")
    if key!=tuple(selection):
        raise ValueError("Unordered bet keys must be in ascending canonical order")
    return key

def projection(ordered_triples):
    """Marginalize every distinct (1st,2nd,3rd) ordered tuple to seven markets.

    Wide is counted as each of the three pairs of top-three CAR entrants;
    three different WIDE tickets may win for one outcome. 2枠 effects are
    derived from FIRST TWO finishing cars' official frames.
    """
    if not ordered_triples:raise ValueError("No order distribution")
    projected={market:defaultdict(float) for market in BET_TYPES}
    total=0.0
    visited=set()
    for x in ordered_triples:
        i,j,k=(int(x[c]) for c in ("first_car_no","second_car_no","third_car_no"))
        p=float(x["probability"])
        if len({i,j,k})!=3 or any(not 1<=z<=9 for z in (i,j,k)):
            raise ValueError("Only distinct valid car numbers supported")
        if not isfinite(p) or p<=0:raise ValueError("Trio probabilities must be strictly positive")
        if (i,j,k) in visited:raise ValueError("Duplicate triple")
        visited.add((i,j,k))
        total+=p
        fi,fj=FRAME_BY_CAR[i],FRAME_BY_CAR[j]
        keys={
          "2車単":(i,j),
          "2車複":tuple(sorted((i,j))),
          "3連単":(i,j,k),
          "3連複":tuple(sorted((i,j,k))),
          "2枠単":(fi,fj),
          "2枠複":tuple(sorted((fi,fj))),
        }
        for typ,key in keys.items():projected[typ][key]+=p
        for pair in combinations((i,j,k),2):
            projected["ワイド"][tuple(sorted(pair))]+=p
    if abs(total-1)>1e-9:raise ValueError("Ordered triples must sum to one")
    for typ,probs in projected.items():
        expectation=3 if typ=="ワイド" else 1
        if abs(sum(probs.values())-expectation)>1e-8:
            raise AssertionError("Incorrect normalization for "+typ)
    return {typ:dict(v) for typ,v in projected.items()}

def market_key_to_string(typ,key):
    return typ+":"+"-".join(map(str,key))

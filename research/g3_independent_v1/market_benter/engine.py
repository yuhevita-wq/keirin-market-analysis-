"""Benter market-interface for G3 probability research, no retrospective bet picking.

This module bridges a *frozen* G3 independent ordered-triple distribution to
exhaustive pari-mutuel quote books. Actual market calibration coefficients must
be learned in a separate historical outcomes+odds fit.

Critical: normalized inverse price is an APPROXIMATION for exhaustive,
mutually-exclusive partitions. Wide (ワイド) is not such a partition: it
can have THREE winners in a single race, so its quoted price never becomes
a normalized probability across pairs here.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
G3ROOT=HERE.parent
sys.path.insert(0,str(G3ROOT))
from predict_g3_ordered_triples import all_ordered_triples

EXCLUSIVE=("3連単","3連複","2車単","2車複","2枠単","2枠複")
SUPPORTED=EXCLUSIVE+("ワイド",)
REQUIRES_FRAMES={"2枠単","2枠複"}

def _sorted_key(nums):
    return "-".join(map(str,sorted(nums)))

def outcome_selection(triple,market,frame_map=None):
    i,j,k=triple
    if market=="3連単":return f"{i}-{j}-{k}"
    if market=="3連複":return _sorted_key((i,j,k))
    if market=="2車単":return f"{i}-{j}"
    if market=="2車複":return _sorted_key((i,j))
    if market=="ワイド":return tuple(_sorted_key(a) for a in ((i,j),(i,k),(j,k)))
    if market=="2枠単":
        if frame_map is None:raise ValueError("2枠 market requires explicit frame map")
        return f"{frame_map[i]}-{frame_map[j]}"
    if market=="2枠複":
        if frame_map is None:raise ValueError("2枠 market requires explicit frame map")
        return _sorted_key((frame_map[i],frame_map[j]))
    raise ValueError("Unsupported market: "+str(market))

def frame_mapping(card,cars):
    if "frame_by_car" not in card:return None
    raw=card["frame_by_car"]
    if not isinstance(raw,dict) or {int(k) for k in raw}!=set(cars):
        raise ValueError("Frame map must specify exactly all cars")
    obj={int(k):int(v) for k,v in raw.items()}
    if any(v not in range(1,7) for v in obj.values()):
        raise ValueError("Frame numbers must be 1-6")
    return obj

def aggregate_probability(triples, market, frames=None):
    totals={}
    for t in triples:
        triple=(t["first_car_no"],t["second_car_no"],t["third_car_no"])
        p=t["joint_first_second_third_probability"]
        keys=outcome_selection(triple,market,frames)
        if market=="ワイド":
            for key in keys:totals[key]=totals.get(key,0.)+p
        else:
            totals[keys]=totals.get(keys,0.)+p
    mass=sum(totals.values())
    goal=3 if market=="ワイド" else 1
    if abs(mass-goal)>1e-8:
        raise RuntimeError(f"Invalid aggregated {market} mass: {mass}, expected {goal}")
    return totals

def validate_quotes(quotes,expected,market):
    if not isinstance(quotes,dict) or not quotes:
        raise ValueError("Odds book must be nonempty selection->decimal gross dividend mapping")
    if set(quotes)!=set(expected):
        miss=sorted(set(expected)-set(quotes))
        extra=sorted(set(quotes)-set(expected))
        raise ValueError(f"Incomplete/foreign {market} book: missing={miss[:6]} ({len(miss)}); extra={extra[:6]} ({len(extra)})")
    vals={}
    for name,value in quotes.items():
        if isinstance(value,bool):raise ValueError("Boolean quote forbidden")
        odd=float(value)
        if not math.isfinite(odd) or odd<=1.0:
            raise ValueError("Invalid gross payout multiplier: "+name)
        vals[name]=odd
    return vals

def inverse_price_public_probability(prices,market):
    if market=="ワイド":
        raise ValueError("Wide odds are NOT mutually exclusive and may not be normalized into a one-winner book")
    implied={key:1./odd for key,odd in prices.items()}
    total=sum(implied.values())
    if total<=0:raise ValueError("Invalid odds implied mass")
    return {key:v/total for key,v in implied.items()},total

def _price_phase_gate(meta, require_predated=True):
    if not isinstance(meta,dict):raise ValueError("Snapshot metadata required")
    kind=meta.get("price_phase")
    if kind not in ("final","predeadline"):
        raise ValueError("Known price_phase final or predeadline required")
    if kind=="final":
        if require_predated:raise ValueError("Final odds must never substitute for live predeadline odds")
        return
    try:
        capture=dt.datetime.fromisoformat(meta["captured_at"])
        deadline=dt.datetime.fromisoformat(meta["bet_close_at"])
    except (ValueError,TypeError,KeyError) as exc:
        raise ValueError("Timestamped capture and betting closure required") from exc
    if capture.tzinfo is None or deadline.tzinfo is None:
        raise ValueError("Both timestamps must be timezone-aware")
    mins=(deadline-capture).total_seconds()/60
    target=float(meta.get("target_minutes_before_close",15))
    tolerance=float(meta.get("tolerance_minutes",2))
    if not math.isfinite(mins) or not 0<target<240 or not 0<=tolerance<=10:
        raise ValueError("Invalid time information")
    if abs(mins-target)>tolerance:
        raise ValueError(f"Snapshot not within declared cutoff window: actual {mins:.3f} minutes before closure")

def load_coefficients(calibration,market,price_phase=None):
    if calibration is None:return None
    if not isinstance(calibration,dict) or calibration.get("status")!="CHRONOLOGICAL_MARKET_FIT_COMPLETED":
        raise ValueError("Market coefficients are not estimated and chronologically validated")
    if calibration.get("market")!=market:
        raise ValueError("Market-type specific coefficients required")
    if price_phase=="predeadline" and not calibration.get("approved_for_15min_runtime",False):
        raise ValueError("Final-odds trained coefficients cannot be used at 15 minutes before closure without direct snapshot-phase validation")
    if price_phase is not None and calibration.get("price_phase") not in (None,price_phase):
        raise ValueError("Model was calibrated to a different quote phase")
    par=calibration.get("coefficients",{})
    a=float(par["alpha_fundamental"]);b=float(par["beta_public"])
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Market fusion coefficient invalid")
    return a,b

def fuse(fundamental,public,alpha,beta):
    keys=sorted(fundamental)
    if set(public)!=set(keys):raise ValueError("Fundamental-public outcomes not aligned")
    logits=[alpha*math.log(max(fundamental[k],1e-300))+
            beta*math.log(max(public[k],1e-300)) for k in keys]
    anchor=max(logits)
    weights=[math.exp(v-anchor) for v in logits]
    total=sum(weights)
    return {k:w/total for k,w in zip(keys,weights)}

def analyze(card, second_model,third_model,quotes_by_market,metadata,calibration_by_market=None,allow_final_research=False):
    """Never returns claimed positive-EV picks for a market without fitted alpha/beta."""
    _price_phase_gate(metadata, require_predated=not allow_final_research)
    if not isinstance(quotes_by_market,dict) or not quotes_by_market:
        raise ValueError("At least one odds market required")
    if set(quotes_by_market)-set(SUPPORTED):
        raise ValueError("Unknown odds wager types")
    distribution=all_ordered_triples(card,second_model,third_model)
    triples=distribution["ordered_triples"]
    cars={x["car_no"] for x in distribution["first_marginal"]}
    frames=frame_mapping(card,cars)
    report={"status":"MARKET_COMPARISON_REQUIRES_FITTED_COEFFICIENTS",
            "scope":"G3開催 S級 only",
            "odds_snapshot":metadata,"price_phase":metadata["price_phase"],
            "calibration_missing_markets":[],
            "source":"G3 frozen first, second and third models",
            "markets":{}}
    proxy_cal=(calibration_by_market or {}).get("1着市場代理_3連単")
    if proxy_cal is not None:
        if "3連単" not in quotes_by_market:
            raise ValueError("First-win market proxy needs complete 3連単 odds")
        first_f={str(r["car_no"]):r["probability"] for r in distribution["first_marginal"]}
        trifecta_f=aggregate_probability(triples,"3連単")
        triodds=validate_quotes(quotes_by_market["3連単"],trifecta_f,"3連単")
        tri_public,_=inverse_price_public_probability(triodds,"3連単")
        public_first={car:0.0 for car in first_f}
        for outcome,p in tri_public.items():
            public_first[outcome.split("-")[0]]+=p
        alpha,beta=load_coefficients(proxy_cal,"1着市場代理_3連単",metadata["price_phase"])
        combined_first=fuse(first_f,public_first,alpha,beta)
        report["first_win_market_combination"]={
            "method":"Benter WIN-stage loglinear using 3連単 market-marginal as keirin proxy",
            "market_proxy_warning":"Deriving win probability from a 3連単 book is a keirin extension, not a method asserted to be in Benter 1994",
            "probabilities":[{"car_no":int(car),"fundamental":first_f[car],
                              "market_approx":public_first[car],"combined":combined_first[car]}
                             for car in sorted(first_f,key=int)]}
    for market,book in sorted(quotes_by_market.items()):
        if market in REQUIRES_FRAMES and frames is None:
            raise ValueError("Frame market requested without explicit frame mapping")
        f=aggregate_probability(triples,market,frames)
        prices=validate_quotes(book,f,market)
        public,inv_mass=(inverse_price_public_probability(prices,market)
                         if market in EXCLUSIVE else (None,None))
        coef=load_coefficients((calibration_by_market or {}).get(market),market,metadata["price_phase"])
        if coef and market=="ワイド":
            raise ValueError("Wide is nonexclusive; needs independently calibrated binary model, not multinomial logit")
        if coef is None:
            report["calibration_missing_markets"].append(market)
        fitted=fuse(f,public,*coef) if coef else None
        rows=[]
        for key in sorted(f):
            row={"selection":key,"fundamental_probability":f[key],
                 "gross_dividend":prices[key]}
            if public is not None:row["market_probability_approx"]=public[key]
            if fitted is not None:
                cp=fitted[key]
                row.update(combined_probability=cp,expected_return_multiple=cp*prices[key],
                           edge=cp*prices[key]-1,positive_expected_value=cp*prices[key]>1)
            else:
                row["combined_probability"]=None
                row["expected_return_multiple"]=None
                row["positive_expected_value"]=None
            rows.append(row)
        report["markets"][market]={
            "exhaustive_one_winner":market in EXCLUSIVE,
            "sum_fundamental_probabilities":sum(f.values()),
            "market_inverse_odds_mass":inv_mass,
            "normalization_is_assumption":"market inverse-odds shares normalized; marginal pricing may have selection-dependent distortions" if public is not None else "not normalized: three wide-winning pairs per podium",
            "calibration_status":"FITTED_CHRONOLOGICALLY" if coef else "NOT_FITTED",
            "rows":rows
        }
    if not report["calibration_missing_markets"]:
        report["status"]="RESEARCH_EXPECTED_RETURN_ESTIMATES_READY"
    return report

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

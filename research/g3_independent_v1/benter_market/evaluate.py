"""Fail-closed 7-market G3 ticket audit + Benter probability bridge.

IMPORTANT: This tool consumes odds externally supplied by a future conversation
(or a separately verified input adapter). It DOES NOT scrape or generate odds.
It refuses to label any ticket as EV+ without historically estimated
alpha/beta AND validated full market prices. It does not invent bet sizes.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from math import isfinite
from itertools import permutations
from pathlib import Path
import json,sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from predict_g3_ordered_triples import all_ordered_triples
from ticket_space import BET_TYPES,canonical,projection,market_key_to_string
from combination import public_first_from_complete_exacta,blended_first,require_probability_list

MODEL_SECOND=HERE.parent/"models/g3_second_conditional_v1.json"
MODEL_THIRD=HERE.parent/"models/g3_third_conditional_v1.json"

def timestamp(text):
    try:
        dt=datetime.fromisoformat(str(text).replace("Z","+00:00"))
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("Timestamp lacks timezone")
        return dt
    except (TypeError,ValueError) as exc:
        raise ValueError("Snapshot requires offset-aware ISO8601 times") from exc

def audit_snapshot(snapshot,race_id,mode="research"):
    if not isinstance(snapshot,dict):
        raise ValueError("No market snapshot: cannot price all ticket types")
    if snapshot.get("race_id")!=race_id:
        raise ValueError("Market snapshot belongs to a different race")
    if not snapshot.get("source_url") or not snapshot.get("capture_provenance"):
        raise ValueError("Require actual source URL and capture provenance; no invented odds")
    capture=timestamp(snapshot.get("captured_at_utc"))
    start=timestamp(snapshot.get("scheduled_start_at_utc"))
    remaining=(start-capture).total_seconds()/60
    if remaining<=0 and snapshot.get("odds_phase")!="final":
        raise ValueError("T-15 snapshot was not captured before race start")
    if mode=="live":
        if snapshot.get("odds_phase")!="T-15" or not 13<=remaining<=17:
            raise ValueError("Live-mode prices must be verified T-15 ±2 minutes")
    else:
        if snapshot.get("odds_phase") not in ("final","T-15"):
            raise ValueError("Unknown odds phase")
    if not isinstance(snapshot.get("sold_bet_types"),list) or len(set(snapshot["sold_bet_types"]))!=len(snapshot["sold_bet_types"]):
        raise ValueError("Sold ticket types must be a unique, explicit list")
    if not set(snapshot["sold_bet_types"]).issubset(BET_TYPES):
        raise ValueError("Unsupported bet type")
    quotes=snapshot.get("quotes")
    if not isinstance(quotes,list):raise ValueError("No odds quotes")
    catalog={}
    for row in quotes:
        typ=row.get("bet_type")
        if typ not in snapshot["sold_bet_types"]:
            raise ValueError("An unsold market contains quote rows")
        key=canonical(typ,row.get("selection"))
        token=(typ,key)
        if token in catalog:raise ValueError("Duplicate price for market ticket")
        if "decimal_odds" not in row:
            raise ValueError("Must supply exact decimal odds. Wide odds range is not an exact price")
        value=float(row["decimal_odds"])
        if not isfinite(value) or value<=1:
            raise ValueError("Nonpositive, unpriced or fake price not accepted")
        catalog[token]=value
    not_sold=set()
    for row in snapshot.get("not_offered_tickets",[]):
        typ=row.get("bet_type")
        if typ not in snapshot["sold_bet_types"]:
            raise ValueError("No 'not offered' for entirely unavailable market")
        key=canonical(typ,row.get("selection"))
        if not row.get("source_reason"):
            raise ValueError("Every unavailable ticket needs observed source reason")
        token=(typ,key)
        if token in catalog or token in not_sold:
            raise ValueError("Inconsistent unavailable ticket")
        not_sold.add(token)
    return catalog,not_sold,{
        "odds_phase":snapshot["odds_phase"],
        "minutes_before_scheduled_start":remaining,
        "captured_at_utc":snapshot["captured_at_utc"],
        "source_url":snapshot["source_url"],
        "capture_provenance":snapshot["capture_provenance"],
        "sold_bet_types":snapshot["sold_bet_types"]
    }

def evaluate_race(card,market_snapshot,calibration=None,mode="research"):
    if not card.get("race_id") or card.get("race_id")!=market_snapshot.get("race_id"):
        raise ValueError("Race ID missing/mismatched")
    q=require_probability_list(card.get("first_win"))
    quote_catalog,not_offered,provenance=audit_snapshot(market_snapshot,card["race_id"],mode)
    exacta=[{"selection":list(k),"decimal_odds":v}
            for (typ,k),v in quote_catalog.items() if typ=="2車単"]
    if "2車単" not in market_snapshot["sold_bet_types"]:
        return {"status":"BLOCKED_PUBLIC_WIN_PROXY_REQUIRES_2車単",
                "race_id":card["race_id"],"odds_audit":provenance}
    public,proxy=audit_market=public_first_from_complete_exacta(q,exacta)
    second=json.loads(MODEL_SECOND.read_text(encoding="utf-8"))
    third=json.loads(MODEL_THIRD.read_text(encoding="utf-8"))
    raw=all_ordered_triples(card,second,third)
    base=[{"first_car_no":x["first_car_no"],
           "second_car_no":x["second_car_no"],
           "third_car_no":x["third_car_no"],
           "probability":x["joint_first_second_third_probability"]}
          for x in raw["ordered_triples"]]
    bookprob=projection(base)
    checks={}
    for typ in market_snapshot["sold_bet_types"]:
        expected=set(bookprob[typ])
        with_price={key for (market,key),odds in quote_catalog.items() if market==typ}
        unpriced={key for (market,key) in not_offered if market==typ}
        if not (with_price|unpriced).issubset(expected):
            raise ValueError("A ticket market contains an impossible selection")
        missing=expected-with_price-unpriced
        if missing:
            raise ValueError(f"Incomplete sold {typ} book: {len(missing)} market outcomes missing")
        checks[typ]={"mathematically_possible":len(expected),
                     "actual_quoted":len(with_price),
                     "documented_not_offered":len(unpriced)}
    result={
        "scope":"G3 S級 only",
        "race_id":card["race_id"],
        "odds_audit":provenance,
        "market_proxy_method":proxy,
        "market_proxy_first":[{"car_no":i,"probability":v} for i,v in sorted(public.items())],
        "bet_type_book_audit":checks,
        "fundamental_first_from_G3_only":True,
        "frozen_second_and_third_used_unchanged":True,
        "number_of_3連単_joint_outcomes":len(base),
        "stake_sizing":"NOT_CALCULATED_REQUIRE_POOL_SIZE_AND_SELF_IMPACT",
        "no_deterministic_ticket_limit":True,
    }
    if calibration is None:
        return {
           **result,
           "status":"CALIBRATION_REQUIRED_NOT_EV_READY",
           "reason":"Benter alpha/beta not learned from matched G3 prior-year win probabilities + historical exacta market + realized winners",
           "public_market_model_not_yet_validated":True,
           "EV_positive_tickets":None,
           "fundamental_candidate_distribution":bookprob["3連単"] and {
               "-".join(map(str,k)):v for k,v in bookprob["3連単"].items()}
        }
    if calibration.get("status")!="G3_BENTER_PUBLIC_COMBINATION_CHRONOLOGICALLY_VALIDATED":
        raise ValueError("Refuse to treat unvalidated alpha/beta as true combined probabilities")
    if calibration.get("proxy_method")!="NORMALIZED_COMPLETE_2CAR_TAN_INVERSE_ODDS_AS_MARKET_PROXY":
        raise ValueError("Calibration not fitted with same G3 public odds proxy")
    if calibration.get("odds_phase")!=provenance["odds_phase"]:
        raise ValueError("Final-odds fit cannot be passed off as predeadline live calibration")
    combined=blended_first(q,public,float(calibration["alpha"]),float(calibration["beta"]))
    # Preserve G3 second/third CONDITIONAL distributions learned from
    # fundamental q; reweight only first marginal by combined Benter c.
    # This extension from Benter is explicit and needs separate trifecta validation.
    adjusted=[]
    for row in base:
        i=row["first_car_no"]
        adjusted.append({**row,"probability":row["probability"]*combined[i]/q[i]})
    adjusted_joint=projection(adjusted)
    ranked=[]
    for (typ,key),odds in quote_catalog.items():
        p=float(adjusted_joint[typ][key])
        er=p*odds
        if er>1.0:
            ranked.append({
                "bet_type":typ,"selection":list(key),
                "probability":p,"quoted_decimal_odds":odds,
                "expected_return_per_1_staked":er,
                "advantage":er-1,
                "stake_amount":None
            })
    ranked.sort(key=lambda v:(-v["advantage"],v["bet_type"],v["selection"]))
    return {
        **result,
        "status":"EXPLORATORY_MODEL_MARKET_COMPARISON_NOT_PROSPECTIVE_PROVEN",
        "combined_first":[{"car_no":i,"probability":v} for i,v in sorted(combined.items())],
        "all_positive_model_expected_return_tickets":ranked,
        "all_positive_tickets_count":len(ranked),
        "tickets_examined":len(quote_catalog),
        "prices_not_offered_or_cannot_bet":len(not_offered),
        "model_use_note":"Combined Benter correction applied to 1st marginal only; G3 2nd/3rd conditional distributions conditioned on original fundamental q. This extension is not asserted to be Benter's direct formula; joint calibration must be independently evaluated before betting.",
        "warning":"Betting all EV+ tickets simultaneously may incur pool-impact and joint Kelly interactions. Stake allocation absent by design."
    }

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--race",required=True,type=Path)
    ap.add_argument("--calibration",type=Path)
    ap.add_argument("--mode",choices=["research","live"],default="research")
    ap.add_argument("--output",required=True,type=Path)
    args=ap.parse_args()
    obj=json.loads(args.race.read_text(encoding="utf-8"))
    cal=json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    value=evaluate_race(obj["racecard"],obj["market_snapshot"],cal,mode=args.mode)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(value["status"],"race_id",value["race_id"],flush=True)

if __name__=="__main__":main()

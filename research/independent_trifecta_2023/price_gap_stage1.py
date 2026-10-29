#!/usr/bin/env python3
"""Stage 1: test 2023 PRICE GAPS using *raw* 3連単 final odds O and no-odds AI p.

PRICE GAP = AI probability * displayed market payout multiple O.
Example: 20x market, independent fair 10x (p=.10) => p*O=2.

We compare the AI's predicted events in each price-gap band to actual
realized events (one winner per race). No actual buying, no proposed bet
formation, no fake fill-in prices, no withholding rate adjustment, no model
refit, no model selection based on the odds.

Input: exact same-ID and same-trifecta historical price/probability pairs
preserved by GitHub Action 40820189970, including original and corrected
probabilities. Retrospective 2024-calibrated p in 2023 is a BACKCAST, not
a chronological 2023 prediction. Original 2023 AI used 2023 Jan-Oct for
training; Nov-Dec was not used for its training.

Run:
 python research/independent_trifecta_2023/price_gap_stage1.py
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"data/audits/trifecta_market_association_full_matches_2023.jsonl.gz"
AUDIT=ROOT/"data/audits/trifecta_market_association_2023.json"
OUT=ROOT/"data/audits/trifecta_price_gap_stage1_2023.json"
LEDGER=ROOT/"data/audits/trifecta_price_gap_stage1_candidates_2023.jsonl.gz"

MODELS={
  "original_2023_odds_free":"p_ai_original_2023",
  "corrected_2024_fit_backcast_to_2023":"p_ai_corrected",
}
BANDS=[
  ("ratio_below_0.5",0,0.5),
  ("ratio_0.5_to_0.8",0.5,0.8),
  ("ratio_0.8_to_1.0",0.8,1.0),
  ("ratio_1.0_to_1.2",1.,1.2),
  ("ratio_1.2_to_1.5",1.2,1.5),
  ("ratio_1.5_to_2.0",1.5,2.),
  ("ratio_2.0_to_3.0",2.,3.),
  ("ratio_3.0_to_5.0",3.,5.),
  ("ratio_5.0_to_10.0",5.,10.),
  ("ratio_10.0_plus",10.,math.inf),
]
THRESHOLDS=[1.,1.2,1.5,2.,3.,5.,10.]
PRICE_BANDS=[
 ("odds_1_10",1,10),("odds_10_30",10,30),
 ("odds_30_100",30,100),("odds_100_300",100,300),
 ("odds_300_1000",300,1000),("odds_1000_plus",1000,math.inf)
]
SEGMENTS=(
 "all",
 "2023_JanOct_original_AI_training",
 "2023_NovDec_original_AI_unused",
 "field_7",
 "field_9",
 "line_3-2-2",
 "line_2-2-2-1",
)

class Bucket:
    __slots__=("tickets","p_expected","market_reciprocal","observed",
               "races_with_candidates","race_bern_variance","sum_price","sum_p_times_price",
               "sum_predicted_event_mass_squared","max_race_mass",
               "min_odds","max_odds")
    def __init__(self):
        self.tickets=0
        self.p_expected=0.
        self.market_reciprocal=0.
        self.observed=0
        self.races_with_candidates=0
        self.race_bern_variance=0.
        self.sum_price=0.
        self.sum_p_times_price=0.
        self.sum_predicted_event_mass_squared=0.
        self.max_race_mass=0.
        self.min_odds=math.inf
        self.max_odds=0.
    def update_ticket(self,p,odd,win):
        self.tickets+=1
        self.p_expected+=p
        self.market_reciprocal+=1./odd
        self.observed+=int(win)
        self.sum_price+=odd
        self.sum_p_times_price+=p*odd
        self.min_odds=min(self.min_odds,odd)
        self.max_odds=max(self.max_odds,odd)
    def complete_race(self,selected_probability):
        if selected_probability<=0:return
        if not 0<=selected_probability<=1.00000001:
            raise AssertionError("Price-gap selection probability mass exceeds one")
        self.races_with_candidates+=1
        self.max_race_mass=max(self.max_race_mass,selected_probability)
        self.sum_predicted_event_mass_squared+=selected_probability*selected_probability
        self.race_bern_variance+=selected_probability*(1-selected_probability)
    def summary(self,all_comparable_races):
        if self.tickets==0:return None
        error=self.observed-self.p_expected
        sd=math.sqrt(max(0,self.race_bern_variance))
        n=self.tickets
        return {
            "candidate_trifecta_tickets":n,
            "races_with_at_least_one_candidate":self.races_with_candidates,
            "races_with_candidates_pct":round(100*self.races_with_candidates/all_comparable_races,5),
            "mean_candidates_per_nonempty_race":round(n/self.races_with_candidates,6),
            "predicted_winners_sum_AI":round(self.p_expected,6),
            "realized_winners":self.observed,
            "mean_AI_probability_per_ticket_pct":round(100*self.p_expected/n,8),
            "actual_ticket_win_frequency_pct":round(100*self.observed/n,8),
            "realized_over_predicted_win_count":round(self.observed/self.p_expected,6) if self.p_expected else None,
            "predicted_events_per_race_pct":round(100*self.p_expected/all_comparable_races,6),
            "observed_events_per_race_pct":round(100*self.observed/all_comparable_races,6),
            "mean_unadjusted_inverse_odds_per_ticket_pct":round(100*self.market_reciprocal/n,8),
            "mean_raw_market_odds":round(self.sum_price/n,5),
            "mean_AI_probability_times_raw_odds":round(self.sum_p_times_price/n,6),
            "min_max_displayed_odds":[self.min_odds,self.max_odds],
            "observed_minus_expected_wins":round(error,6),
            "race_unit_z_vs_AI_probability":round(error/sd,5) if sd>0 else None,
            "race_unit_variance_under_AI":round(self.race_bern_variance,7),
        }

def labels_for_race(r):
    period=("2023_JanOct_original_AI_training" if r["subpopulation"]=="in_sample_2023_jan_oct"
            else "2023_NovDec_original_AI_unused")
    out=["all",period]
    if r["runners"]==7:out.append("field_7")
    if r["runners"]==9:out.append("field_9")
    if r["line_shape"]=="3-2-2":out.append("line_3-2-2")
    if r["line_shape"]=="2-2-2-1":out.append("line_2-2-2-1")
    return out

def choose_band(price_multiple):
    for label,lo,hi in BANDS:
        if lo<=price_multiple<hi:return label
    raise RuntimeError("no price band for "+repr(price_multiple))

def price_band(odds):
    for label,lo,hi in PRICE_BANDS:
        if lo<=odds<hi:return label
    raise RuntimeError("no odds band for "+repr(odds))

def unit_test():
    assert choose_band(2.)=="ratio_2.0_to_3.0"
    assert choose_band(1.2)=="ratio_1.2_to_1.5"
    assert choose_band(.9)=="ratio_0.8_to_1.0"
    assert choose_band(10.)=="ratio_10.0_plus"
    assert price_band(20)=="odds_10_30"
    b=Bucket()
    for p,odd,win in ((.10,20,False),(.03,70,True)):
        b.update_ticket(p,odd,win)
    b.complete_race(.13)
    assert b.tickets==2 and abs(b.p_expected-.13)<1e-9
    assert b.observed==1 and b.races_with_candidates==1
    assert abs(b.race_bern_variance-.13*.87)<1e-9
    assert abs(b.summary(1)["mean_AI_probability_times_raw_odds"]-2.05)<1e-9
    print("PRICE_GAP_BOUNDARY_AND_RACE_DEPENDENCY_SELFTEST_PASS",flush=True)

def run():
    unit_test()
    source=json.loads(AUDIT.read_text(encoding="utf-8"))
    assert source["counts"]["2023_price_complete_races"]==6301
    assert source["counts"]["2023_model_eligible_races"]==23440
    assert source["counts"]["matched_all_trifecta_and_model_races"]==5226
    assert source["market_odds_in_ai_training"] is False
    assert SOURCE.exists() and SOURCE.stat().st_size>0
    N=source["counts"]["matched_all_trifecta_and_model_races"]
    candidate_buckets=defaultdict(Bucket)
    odds_buckets=defaultdict(Bucket)
    example_corridor=defaultdict(Bucket)
    witnesses=[]
    matched=0
    ticket_pairs=0
    count_races_by_period=defaultdict(int)
    count_races_by_field=defaultdict(int)
    by_period={}
    # Keep a compact ledger of ALL price-gap candidate tickets for reproducibility;
    # it is an artifact only, not a file committed to source control.
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(SOURCE,"rt",encoding="utf-8") as data, \
         gzip.open(LEDGER,"wt",encoding="utf-8",compresslevel=5) as ledger:
        for line in data:
            race=json.loads(line)
            rid=race["race_id"]
            if not (race.get("market_final_price_only") is True):
                raise AssertionError("Market price type is not finalized")
            tickets=race["all_trifecta"]
            winner=race["actual_trifecta"]
            if winner not in tickets:
                raise AssertionError("Exact race does not include winner "+rid)
            if len(tickets)!=race["runners"]*(race["runners"]-1)*(race["runners"]-2):
                raise AssertionError("Not full trifecta ticket grid "+rid)
            segments=labels_for_race(race)
            count_races_by_period[race["subpopulation"]]+=1
            count_races_by_field[str(race["runners"])]+=1
            if rid in by_period:raise AssertionError("Duplicate race ID "+rid)
            by_period[rid]=True
            # A group can include many tickets within the SAME race.
            # For uncertainty, aggregate selected p first into event mass
            # and treat "winner belongs to group" as a single Bernoulli event.
            local_race_mass=defaultdict(float)
            local_race_odd_mass=defaultdict(float)
            local_corridor_mass=defaultdict(float)
            original_sum=0.
            corrected_sum=0.
            odds_inverse_sum=0.
            for name,cell in tickets.items():
                if not isinstance(name,str) or len(name.split("-"))!=3:
                    raise AssertionError("Malformed trifecta ticket name")
                odd=float(cell["final_odds"])
                inv=float(cell["market_raw_inverse_odds"])
                if not (math.isfinite(odd) and odd>=1):
                    raise AssertionError("Invalid displayed market odds "+rid)
                if abs(inv-1./odd)>max(1e-11,1e-8/odd):
                    raise AssertionError("Raw inverse odds provenance mismatch "+rid)
                obs=int(name==winner)
                original_sum+=float(cell["p_ai_original_2023"])
                corrected_sum+=float(cell["p_ai_corrected"])
                odds_inverse_sum+=inv
                ticket_pairs+=1
                for model,field in MODELS.items():
                    p=float(cell[field])
                    if not (0<p<1 and math.isfinite(p)):
                        raise AssertionError("Invalid independent p "+rid)
                    ratio=p*odd
                    if not math.isfinite(ratio):raise AssertionError("Invalid price ratio")
                    band=choose_band(ratio)
                    odd_band=price_band(odd)
                    corridor=(15<=odd<25 and .08<=p<.12) # market ~20, AI fair ~10
                    groups=[("band",band),("odds_"+odd_band,band)]
                    for threshold in THRESHOLDS:
                        if ratio>=threshold:
                            groups.append(("threshold",f"ratio_ge_{threshold:g}"))
                    # Main results always separated by AI version and time.
                    for segment in segments:
                        for typ,key in groups:
                            groupkey=(model,segment,typ,key)
                            bucket=candidate_buckets[groupkey]
                            bucket.update_ticket(p,odd,obs)
                            local_race_mass[groupkey]+=p
                        if corridor:
                            cor_key=(model,segment,"literal_example",
                                     "market_15to25x_AI_fair_8p33to12p5x")
                            example_corridor[cor_key].update_ticket(p,odd,obs)
                            local_corridor_mass[cor_key]+=p
                    # Output all candidates only (not pretending to purchase).
                    if ratio>=1.:
                        ledger.write(json.dumps({
                           "race_id":rid,"date":race["race_date"],
                           "period":race["subpopulation"],"field":race["runners"],
                           "line_shape":race["line_shape"],
                           "ticket":name,"AI_model":model,
                           "AI_probability":round(p,12),
                           "AI_fair_decimal_odds":round(1/p,7),
                           "actual_final_market_odds":odd,
                           "raw_price_ratio_p_times_odds":round(ratio,9),
                           "actual_win":bool(obs)
                        },ensure_ascii=False,separators=(",",":"))+"\n")
                        if len(witnesses)<12 and race["subpopulation"]=="forward_2023_nov_dec" and odd>=10 and p>=.02:
                            witnesses.append({
                              "race_id":rid,"date":race["race_date"],"model":model,
                              "ticket":name,"price":odd,"AI_fair_price":round(1/p,5),
                              "AI_probability_pct":round(100*p,6),
                              "gap_multiple":round(ratio,5),
                              "won":bool(obs)})
            if not (abs(original_sum-1)<1e-7 and abs(corrected_sum-1)<1e-7):
                raise AssertionError("All 3連単 AI probabilities do not add to 100% "+rid)
            if not 1.<odds_inverse_sum<1.6:
                raise AssertionError("Unexpected reciprocal sum of actual finalized odds "+rid)
            for groupkey,p_mass in local_race_mass.items():
                candidate_buckets[groupkey].complete_race(p_mass)
            for groupkey,p_mass in local_corridor_mass.items():
                example_corridor[groupkey].complete_race(p_mass)
            matched+=1
            if matched%1000==0:
                print("ALL_PRICE_GAP_TICKETS_MEASURED",matched,"RACES",ticket_pairs,"TICKETS",flush=True)
    assert matched==N,(matched,N)
    assert ticket_pairs==source["counts"]["matched_exact_trifecta_ticket_pairs"]
    expected_splits={"in_sample_2023_jan_oct":4835,"forward_2023_nov_dec":391}
    assert dict(count_races_by_period)==expected_splits

    # Segment denominators: do not pretend line-level subsets have all 5226 races.
    denom=dict.fromkeys(SEGMENTS,0)
    denom["all"]=N
    denom["2023_JanOct_original_AI_training"]=4835
    denom["2023_NovDec_original_AI_unused"]=391
    denom["field_7"]=count_races_by_field["7"]
    denom["field_9"]=count_races_by_field["9"]
    denom["line_3-2-2"]=source["by_group"]["shape_3-2-2"]["races"]
    denom["line_2-2-2-1"]=source["by_group"]["shape_2-2-2-1"]["races"]

    def summarize(model,segment,typ):
        labels=([x[0] for x in BANDS] if typ=="band" else
                [f"ratio_ge_{x:g}" for x in THRESHOLDS])
        return {label:candidate_buckets[(model,segment,typ,label)].summary(denom[segment])
                for label in labels}

    report={
      "state":"STAGE1_COMPLETED_RAW_FINAL_ODDS_PRICE_GAP_OBSERVATION_2023",
      "purpose":"For every matched 3連単 ticket compare independent fair-price estimate 1/p with raw observed final market odds O; group by p*O and compare AI predicted event counts to ACTUAL outcomes.",
      "market_vs_AI_ratio_formula":"raw_gap_multiple = independent_AI_probability * final_decimal_market_odds; market 20x and AI fair 10x means ratio 2",
      "no_market_used_in_AI_training":True,
      "no_market_takeout_adjustment_or_assumption":True,
      "not_actual_purchase":True,"no_bet_selection_or_purchase_cost":True,
      "no_2023_model_retraining":True,"all_ticket_odds_are_FINAL_not_Tminus15":True,
      "data_source":{"original_full_join_GitHub_Actions_run":40820189970,
        "original_full_join_artifact":"independent-2023-trifecta-market-association",
        "price_audit":"data/audits/trifecta_market_association_2023.json",
        "matched_full_ticket_gzip":"data/audits/trifecta_market_association_full_matches_2023.jsonl.gz"},
      "coverage":{
        "2023_complete_price_books":source["counts"]["2023_price_complete_races"],
        "2023_independent_AI_eligible_races":source["counts"]["2023_model_eligible_races"],
        "2023_races_full_exact_price_AI_join":N,
        "2023_independent_AI_eligible_but_no_complete_join":source["counts"]["2023_model_eligible_races"]-N,
        "2023_price_complete_but_not_AI_eligible":source["counts"]["market_complete_but_not_model_ready"],
        "all_matched_exact_trifecta_pairs":ticket_pairs,
        "split":dict(count_races_by_period),
        "field_size":dict(count_races_by_field),
        "not_full_2023_nationwide_population":True,
      },
      "models":{
        "original_2023_odds_free":{
          "interpretation":"Jan-Oct model training in-sample; 2023 Nov-Dec unused by original 2023 model",
          "all":{t:summarize("original_2023_odds_free","all",t) for t in ("band","threshold")},
          "JanOct_in_sample":{t:summarize("original_2023_odds_free","2023_JanOct_original_AI_training",t) for t in ("band","threshold")},
          "NovDec_forward":{t:summarize("original_2023_odds_free","2023_NovDec_original_AI_unused",t) for t in ("band","threshold")},
          "field_7":{t:summarize("original_2023_odds_free","field_7",t) for t in ("band","threshold")},
          "field_9":{t:summarize("original_2023_odds_free","field_9",t) for t in ("band","threshold")},
          "line_3-2-2":{t:summarize("original_2023_odds_free","line_3-2-2",t) for t in ("band","threshold")},
          "line_2-2-2-1":{t:summarize("original_2023_odds_free","line_2-2-2-1",t) for t in ("band","threshold")},
        },
        "corrected_2024_fit_backcast_to_2023":{
          "interpretation":"2024-H1 fitted calibration coefficients; cannot be treated as 2023 live predictions or independent 2023 backtest",
          "all":{t:summarize("corrected_2024_fit_backcast_to_2023","all",t) for t in ("band","threshold")},
          "JanOct_in_sample":{t:summarize("corrected_2024_fit_backcast_to_2023","2023_JanOct_original_AI_training",t) for t in ("band","threshold")},
          "NovDec_exploratory_backcast":{t:summarize("corrected_2024_fit_backcast_to_2023","2023_NovDec_original_AI_unused",t) for t in ("band","threshold")},
          "field_7":{t:summarize("corrected_2024_fit_backcast_to_2023","field_7",t) for t in ("band","threshold")},
          "field_9":{t:summarize("corrected_2024_fit_backcast_to_2023","field_9",t) for t in ("band","threshold")},
          "line_3-2-2":{t:summarize("corrected_2024_fit_backcast_to_2023","line_3-2-2",t) for t in ("band","threshold")},
          "line_2-2-2-1":{t:summarize("corrected_2024_fit_backcast_to_2023","line_2-2-2-1",t) for t in ("band","threshold")},
        }
      },
      "literal_market_20x_AI_fair_10x_examples":{
        "selection_formula":"15<=final_market_odds<25 AND .08<=p_ai<.12, independent of observed winners",
        "summary":{
           model:{
             segment:example_corridor[(model,segment,"literal_example",
                               "market_15to25x_AI_fair_8p33to12p5x")].summary(denom[segment])
             for segment in ("all","2023_NovDec_original_AI_unused")
           }
           for model in MODELS
        }
      },
      "nontuned_examples_from_2023_NovDec":witnesses,
      "candidate_ledger_artifact_only":str(LEDGER.relative_to(ROOT)),
      "warnings":[
        "2023 original AI fitted Jan-Oct 2023; 2023 Nov-Dec is only 391 matched race IDs.",
        "Corrected model uses 2024 H1 results and its scores in 2023 are retrospective only.",
        "The join of 5226 races is not the full 2023 race universe; no fabricated missing odds. There are 18214 otherwise AI-eligible races without a complete verified price join.",
        "Prices are finalized archived odds, NOT executable T-15-minute purchase snapshots.",
        "P*O>1 is an ex-ante price-gap OBSERVATION, not validation or strategy success. Test measured outcomes and confidence.",
        "No 3連単 outcomes are treated as independent within a race. Variance is calculated over mutually exclusive race events.",
        "No estimated deduction fraction is used: only published raw final price and independent AI probability.",
        "Market final odds cannot enter or train AI probabilities and no coefficients are reoptimized.",
        "Historical score/line availability as-of the race cutoff has not been separately audited.",
        "No ROI is reported: no purchases were made and no prices for missing races were invented."
      ]
    }
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RAW_PRICE_GAP_STAGE1_FINISHED",json.dumps({
      "races":N,"tickets":ticket_pairs,
      "original_2023_NovDec_above_1p5":report["models"]["original_2023_odds_free"]["NovDec_forward"]["threshold"]["ratio_ge_1.5"],
      "original_2023_NovDec_above_2":report["models"]["original_2023_odds_free"]["NovDec_forward"]["threshold"]["ratio_ge_2"],
      "corrected_2023_all_above_2":report["models"]["corrected_2024_fit_backcast_to_2023"]["all"]["threshold"]["ratio_ge_2"]
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    run()

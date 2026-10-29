#!/usr/bin/env python3
"""2023 reversed-price-gap paper-bet audit: market odds BELOW AI fair odds.

For every matched race and every possible trifecta, buy exactly 100 JPY when
  O_market < 1 / p_independent_AI  (equivalently p_AI * O_market < 1).
The market is then assigning a shorter price than the independently estimated
fair price. This is not necessarily proof of a special outside factor.

Calculate genuine payouts for winners by displayed finalized odds; all other
tickets lose. Buy ALL qualifying tickets; no selection using known outcomes.
Also report fixed smaller ratios, disjoint bands, price subgroups, actual ROI,
cost, hit count, chronological monthly net, and deterministic day bootstrap CI.

Important: 2023 odds are archived FINAL payout prices (not T-minus-15
purchasable prices), the 2023 original AI was trained Jan-Oct 2023, and the
corrected model was fitted on 2024 H1 (a retrospective 2023 backcast only).
These are paper simulations, not realized bets or verified executable prices.
"""
from __future__ import annotations
from collections import defaultdict
import gzip
import json
import math
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"data/audits/trifecta_market_association_full_matches_2023.jsonl.gz"
SOURCE_AUDIT=ROOT/"data/audits/trifecta_market_association_2023.json"
REPORT=ROOT/"data/audits/trifecta_market_shorter_odds_buyall_2023.json"
LEDGER=ROOT/"data/audits/trifecta_market_shorter_odds_buyall_tickets_2023.jsonl.gz"
MODELS={
    "original_2023_independent_AI":"p_ai_original_2023",
    "corrected_2024_fitted_AI_backcast":"p_ai_corrected",
}
LIMITS=(0.25,0.5,0.75,0.9,1.0)
EXCLUSIVE=((0,.25,"ratio_0_to_0p25"),
           (.25,.5,"ratio_0p25_to_0p5"),
           (.5,.75,"ratio_0p5_to_0p75"),
           (.75,.9,"ratio_0p75_to_0p9"),
           (.9,1.0,"ratio_0p9_to_1p0"),
           (1.,math.inf,"ratio_1p0_or_higher"))
ODD_BANDS=((1,10,"odds_1to10"),(10,30,"odds_10to30"),
           (30,100,"odds_30to100"),(100,300,"odds_100to300"),
           (300,1000,"odds_300to1000"),(1000,math.inf,"odds_1000plus"))
NOVDEC="forward_2023_nov_dec"

class Stat:
    __slots__=("tickets","hits","stake_yen","paid_yen","sum_p","sum_ratio",
               "days","max_single_payout_yen")
    def __init__(self):
        self.tickets=0
        self.hits=0
        self.stake_yen=0
        self.paid_yen=0
        self.sum_p=0.
        self.sum_ratio=0.
        self.days=0
        self.max_single_payout_yen=0
    def add(self,p,ratio,won,payout):
        self.tickets+=1
        self.hits+=int(won)
        self.stake_yen+=100
        self.paid_yen+=payout
        self.sum_p+=p
        self.sum_ratio+=ratio
        self.max_single_payout_yen=max(self.max_single_payout_yen,payout)
    def emit(self):
        n=self.tickets
        return {
          "purchased_trifecta_tickets":n,
          "100yen_stake_yen":self.stake_yen,
          "actual_winning_tickets":self.hits,
          "actual_hit_frequency_pct":round(100*self.hits/n,7) if n else None,
          "sum_independent_AI_predicted_winners":round(self.sum_p,6),
          "actual_over_AI_predicted_hit_count":round(self.hits/self.sum_p,6) if self.sum_p else None,
          "market_odds_payout_total_yen":self.paid_yen,
          "profit_loss_yen":self.paid_yen-self.stake_yen,
          "actual_recovery_pct":round(100*self.paid_yen/self.stake_yen,6) if self.stake_yen else None,
          "net_profit_loss_per_100yen_ticket_yen":round(100*(self.paid_yen/self.stake_yen-1),5) if self.stake_yen else None,
          "mean_AI_probability_pct":round(100*self.sum_p/n,7) if n else None,
          "mean_raw_price_ratio":round(self.sum_ratio/n,7) if n else None,
          "largest_single_winning_100yen_payout_yen":self.max_single_payout_yen,
          "largest_win_share_of_total_payout_pct":round(100*self.max_single_payout_yen/self.paid_yen,6) if self.paid_yen else None,
        }

def segment_names(r):
    period=("NovDec_original_AI_holdout" if r["subpopulation"]==NOVDEC
            else "JanOct_original_AI_training")
    out=["all",period,f"field_{r['runners']}"]
    if r["line_shape"] in ("3-2-2","2-2-2-1"):
        out.append("shape_"+r["line_shape"])
    return out

def band_of(x):
    for low,up,label in EXCLUSIVE:
        if low<=x<up:return label
    raise ValueError("Impossible ratio "+str(x))

def odds_band_of(x):
    for low,up,label in ODD_BANDS:
        if low<=x<up:return label
    raise ValueError("Invalid odds "+str(x))

def selectors(p,odds):
    ratio=p*odds
    ans=[("flat_all_tickets_baseline","ALL")]
    band=band_of(ratio)
    ans.append(("nonoverlapping_ratio_band",band))
    if ratio < 1.:
        ans.append(("all_market_shorter","ratio_below_1"))
        ans.append(("market_shorter_by_price_band",odds_band_of(odds)))
        for limit in LIMITS:
            if ratio<limit:
                ans.append(("cumulative_ratio_below",f"ratio_below_{limit:g}"))
    return ans,ratio

def audit_sanity():
    assert selectors(.05,10)[1]==.5
    assert selectors(.05,10)[0][-1]==("cumulative_ratio_below","ratio_below_1")
    assert not any(t=="all_market_shorter" for t,_ in selectors(.1,20)[0])
    assert band_of(.25)=="ratio_0p25_to_0p5"
    assert odds_band_of(20)=="odds_10to30"
    s=Stat()
    s.add(.02,.2,False,0)
    s.add(.03,.3,True,2500)
    x=s.emit()
    assert x["100yen_stake_yen"]==200 and x["market_odds_payout_total_yen"]==2500
    assert x["actual_recovery_pct"]==1250 and x["profit_loss_yen"]==2300
    print("REVERSE_GAP_NO_PURCHASE_LOOKAHEAD_SELFTEST_PASS",flush=True)

def bootstrap_day_cluster(daygroup,repeat=1200):
    """Simple observed calendar-day clustered nonparametric bootstrap (illustrative)."""
    vals=[(s.stake_yen,s.paid_yen) for _,s in sorted(daygroup.items()) if s.stake_yen]
    if not vals:return None
    total_stake=sum(s for s,p in vals)
    total_paid=sum(p for s,p in vals)
    seed=random.Random(20231029)
    results=[]
    for _ in range(repeat):
        stake=paid=0
        for _ in range(len(vals)):
            s,p=vals[seed.randrange(len(vals))]
            stake+=s
            paid+=p
        if stake:results.append(100*paid/stake)
    results.sort()
    if not results:return None
    return {"calendar_days_with_bets":len(vals),"bootstrap_samples":len(results),
        "observed_recovery_pct":round(100*total_paid/total_stake,6),
        "day_cluster_bootstrap_95pct_recovery_range":[
            round(results[int(.025*(len(results)-1))],4),
            round(results[int(.975*(len(results)-1))],4)],
        "warning":"Descriptive historical-day bootstrap only. Not independent future profitability CI; final odds are not executable snapshots."}

def main():
    audit_sanity()
    source=json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    c=source["counts"]
    assert c["2023_price_complete_races"]==6301
    assert c["2023_model_eligible_races"]==23440
    assert c["matched_all_trifecta_and_model_races"]==5226
    assert c["matched_exact_trifecta_ticket_pairs"]==1732548
    assert SRC.is_file() and SRC.stat().st_size>0

    # Key = (model, population, selection, key)
    totals=defaultdict(Stat)
    # Calendar day and month grouped on the all population only.
    date_stat=defaultdict(Stat)
    month_stat=defaultdict(Stat)
    counters=defaultdict(int)
    seen_races=set()
    examples=[]
    matched_tickets=0

    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(SRC,"rt",encoding="utf-8") as fd, \
         gzip.open(LEDGER,"wt",encoding="utf-8",compresslevel=5) as ledger:
        for line in fd:
            r=json.loads(line)
            rid=r["race_id"];date=r["race_date"];month=date[:7]
            if rid in seen_races:raise AssertionError("Duplicate race "+rid)
            seen_races.add(rid)
            if not r["market_final_price_only"]:raise AssertionError("Not raw final odds")
            labels=segment_names(r)
            counters["races_"+r["subpopulation"]]+=1
            counters["races_total"]+=1
            counters["field_"+str(r["runners"])]+=1
            truth=r["actual_trifecta"]
            tickets=r["all_trifecta"]
            if truth not in tickets:raise AssertionError("Winner missing")
            if len(tickets)!=r["runners"]*(r["runners"]-1)*(r["runners"]-2):
                raise AssertionError("Incomplete 3ren-tan price ticket book")
            sums={model:0.0 for model in MODELS}
            for name, values in tickets.items():
                market_odds=float(values["final_odds"])
                if not 1<=market_odds<=9999.9 or not math.isfinite(market_odds):
                    raise AssertionError("Invalid market odds "+rid)
                if abs(float(values["market_raw_inverse_odds"])-1/market_odds)>1e-8:
                    raise AssertionError("Price vs raw inverse mismatch")
                matched_tickets+=1
                won=(name==truth)
                payout=int(round(100*market_odds)) if won else 0
                for model,p_field in MODELS.items():
                    p=float(values[p_field])
                    if not 0<p<1 or not math.isfinite(p):
                        raise AssertionError("Invalid AI probability")
                    sums[model]+=p
                    rules,ratio=selectors(p,market_odds)
                    for selection,key in rules:
                        for population in labels:
                            totals[(model,population,selection,key)].add(p,ratio,won,payout)
                        if selection in ("flat_all_tickets_baseline",
                                         "all_market_shorter","cumulative_ratio_below"):
                            dkey=(model,r["subpopulation"],selection,key,date)
                            date_stat[dkey].add(p,ratio,won,payout)
                            mkey=(model,r["subpopulation"],selection,key,month)
                            month_stat[mkey].add(p,ratio,won,payout)
                    if ratio<1:
                        ledger.write(json.dumps({
                          "race_id":rid,"date":date,"2023_AI_training_split":r["subpopulation"],
                          "line_shape":r["line_shape"],"runners":r["runners"],
                          "trifecta":name,"model":model,
                          "independent_probability":round(p,12),
                          "AI_fair_odds":round(1/p,8),
                          "market_final_odds":market_odds,
                          "market_to_AI_fair_odds_ratio":round(ratio,9),
                          "stake_yen":100,"actual_win":bool(won),
                          "gross_payout_yen":payout
                        },ensure_ascii=False,separators=(",",":"))+"\n")
                        if won and r["subpopulation"]==NOVDEC and len(examples)<15:
                            examples.append({
                              "race_id":rid,"date":date,"trifecta":name,
                              "model":model,"AI_probability_pct":round(100*p,6),
                              "AI_fair_odds":round(1/p,4),
                              "market_final_odds":market_odds,
                              "ratio":round(ratio,6),
                              "paid_yen_for_100yen":payout})
            if any(abs(s-1)>1e-7 for s in sums.values()):
                raise AssertionError("Trifecta probabilities do not add to 1")
            if counters["races_total"]%1000==0:
                print("MARKET_SHORTER_THAN_AI_PAPER_BUYALL_ANALYZED",
                      counters["races_total"],"RACES",
                      matched_tickets,"TICKETS",flush=True)
    if counters["races_total"]!=5226 or matched_tickets!=1732548:
        raise AssertionError("Population changed")
    if counters["races_"+NOVDEC]!=391 or counters["races_in_sample_2023_jan_oct"]!=4835:
        raise AssertionError("Chronological split changed")

    report={"state":"COMPLETED_MARKET_SHORTER_ODDS_BUYALL_2023_PAPER_REPLAY",
      "hypothesis":"Market odds shorter than no-odds AI fair price may reflect omitted outside factors; hypothesis itself is NOT established by this test.",
      "rule":"For each full-book 3連単 ticket, independently buy 100 yen iff O_market < 1 / p_AI; equivalently p_AI*O_market<1. Do not require a ticket to have ever won.",
      "not_real_world_purchase":True,
      "historical_price_type":"final published archived odds; NOT pre-close executable price",
      "not_retrained_AI":True,
      "takeout_rate_not_assumed":True,
      "no_profitable_filters_chosen_after_results":True,
      "same_race_multiple_tickets_all_staked":True,
      "stake_per_selected_trifecta_yen":100,
      "actual_payout_convention":"Only actual 3連単 winner pays 100 yen * displayed final multiplier; otherwise zero.",
      "coverage":{"2023_AI_eligible_races":23440,
                  "2023_complete_verified_trifecta_and_trio_odds_books":6301,
                  "races_exact_ticket_join":counters["races_total"],
                  "3ren_tan_ticket_comparisons":matched_tickets,
                  "AI_eligible_no_full_verified_market_join":23440-5226,
                  "JanOct_in_sample_races":4835,
                  "NovDec_original_AI_holdout_races":391,
                  "races_by_field":{str(k)[6:]:v for k,v in counters.items() if k.startswith("field_")}},
      "models":{},
      "examples_NovDec_of_actually_winning_market_shorter_tickets":examples,
      "source_actions_run":40820189970,
      "purchased_ticket_ledger_artifact_path":str(LEDGER.relative_to(ROOT)),
      "limits":["The most important constraint: 2023 final odds are not a purchasable price snapshot, so these are simulated returns at final price, NOT verified executable historical trades.",
                "Original AI fitted 2023 Jan-Oct: results during that period are in-sample; only 391 Nov-Dec races are original AI's out-of-sample period.",
                "2024-H1 fitted calibrator is backcast into 2023: not valid as an out-of-sample chronological 2023 prediction.",
                "Only 5226 of 23440 AI-eligible races have fully verified 3連単/3連複 books and exact-ID joins.",
                "The market's shorter price might reflect more information but may also be ordinary chance, miscalibration, or normal pricing; no causal evidence of an outside factor.",
                "No cost cap. Each qualifying trifecta gets exactly 100 yen; many tickets per race may cause significant capital use.",
                "Conditions are fixed before consulting outcomes; no optimization or best-band cherry-picking.",
                "All-in-one fixed 100-yen bets are not equivalent to a recomposed formation. No false formations are claimed.",
                "Multiple trifecta tickets within one race are mutually exclusive; day-level bootstrap is only a descriptive sensitivity diagnostic.",
                "No assumed takeout fraction or price normalization is applied; actual finalized decimal payouts are used directly."]}
    period_map={
      "all":"all",
      "JanOct_training":"JanOct_original_AI_training",
      "NovDec_holdout_original_only":"NovDec_original_AI_holdout",
      "field_7":"field_7",
      "field_9":"field_9",
      "shape_3-2-2":"shape_3-2-2",
      "shape_2-2-2-1":"shape_2-2-2-1"
    }
    rule_list=[
      ("flat_all_tickets_baseline","ALL"),
      ("all_market_shorter","ratio_below_1"),
      *[("cumulative_ratio_below",f"ratio_below_{lim:g}") for lim in LIMITS],
    ]
    def monthly_stats(model,period,selection,key):
        return {month:stat.emit() for (mo,pe,se,ky,month),stat in sorted(month_stat.items())
                if mo==model and pe==period and se==selection and ky==key}
    def ci_for(model,period,selection,key):
        group={date:stat for (mo,pe,se,ky,date),stat in date_stat.items()
               if mo==model and pe==period and se==selection and ky==key}
        return bootstrap_day_cluster(group)
    for model in MODELS:
        entry={"caveat":("2023 original model, in-sample Jan-Oct; Nov-Dec unused for its training"
                   if model=="original_2023_independent_AI" else
                   "Model calibrated in 2024 H1; all 2023 results are descriptive future-fitted backcast"),
               "summary":{}}
        for section,population in period_map.items():
            s={}
            for selection,key in rule_list:
                stat=totals.get((model,population,selection,key))
                s[("baseline" if selection=="flat_all_tickets_baseline" else key)]=stat.emit() if stat else None
            s["disjoint_price_ratio_groups"]={
                label:(totals[(model,population,"nonoverlapping_ratio_band",label)].emit()
                     if (model,population,"nonoverlapping_ratio_band",label) in totals else None)
                for _,_,label in EXCLUSIVE}
            s["market_price_band_for_shorter_only"]={
                label:(totals[(model,population,"market_shorter_by_price_band",label)].emit()
                      if (model,population,"market_shorter_by_price_band",label) in totals else None)
                for _,_,label in ODD_BANDS}
            entry["summary"][section]=s
        for period in ("in_sample_2023_jan_oct",NOVDEC):
            keyname=("JanOct_training" if period!="forward_2023_nov_dec" else "NovDec_holdout_original_only")
            entry.setdefault("monthly",{})[keyname]={
                ("baseline" if selection=="flat_all_tickets_baseline" else key):
                    monthly_stats(model,period,selection,key)
                for selection,key in rule_list}
            entry.setdefault("day_cluster_bootstrap_95pct",{})[keyname]={
                ("baseline" if selection=="flat_all_tickets_baseline" else key):
                    ci_for(model,period,selection,key)
                for selection,key in rule_list}
        report["models"][model]=entry

    for model in MODELS:
        for section in ("all","JanOct_training","NovDec_holdout_original_only"):
            baseline=report["models"][model]["summary"][section]["baseline"]
            expected={"all":5226,"JanOct_training":4835,"NovDec_holdout_original_only":391}[section]
            assert baseline["actual_winning_tickets"]==expected
            assert abs(baseline["sum_independent_AI_predicted_winners"]-expected)<.001
        all_short=report["models"][model]["summary"]["all"]["ratio_below_1"]
        disjoint=report["models"][model]["summary"]["all"]["disjoint_price_ratio_groups"]
        assert sum(s["purchased_trifecta_tickets"] for k,s in disjoint.items()
                   if k!="ratio_1p0_or_higher")==all_short["purchased_trifecta_tickets"]
        assert sum(s["market_odds_payout_total_yen"] for k,s in disjoint.items()
                   if k!="ratio_1p0_or_higher")==all_short["market_odds_payout_total_yen"]
        for sub in ("JanOct_training","NovDec_holdout_original_only"):
            report_section=report["models"][model]["summary"][sub]
            assert sum(s["purchased_trifecta_tickets"] for s in
                       [report_section["ratio_below_1"]])>0

    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    compact={}
    for m in MODELS:
        compact[m]={}
        for k in ("all","NovDec_holdout_original_only"):
            d=report["models"][m]["summary"][k]
            compact[m][k]={threshold:{z:d[threshold][z] for z in
                ("purchased_trifecta_tickets","100yen_stake_yen",
                 "actual_winning_tickets","market_odds_payout_total_yen",
                 "profit_loss_yen","actual_recovery_pct")}
                for threshold in ("ratio_below_0.25","ratio_below_0.5",
                                  "ratio_below_0.75","ratio_below_0.9",
                                  "ratio_below_1")}
    print("MARKET_SHORTER_THAN_AI_PRICE_BUYALL_RESULT",
          json.dumps(compact,ensure_ascii=False),flush=True)

if __name__=="__main__":main()

#!/usr/bin/env python3
"""V6 full-trifecta probability audit against 75%-ROI hypothesis.

ALL ordered 3ren-tan tickets of EVERY eligible race receive V6 p, no singled
out '10%' betting strategy and no selecting winning tickets retroactively.
For each exact probability p and pooled predeclared p interval:
   - one hypothetical 100 JPY stake per ticket (including all losers),
   - sum(p), actual winners, settled payout from ONLY winning ticket,
   - realized ROI and SIGNED & ABSOLUTE deviation from exact 75% benchmark.
75% is the user's benchmark. Neither 80% nor 70% is marked a success.
Two complementary diagnostics: sum observed vs expected winners to test
probability calibration, and observed ROI vs 75% to test market-price/payback
compatibility. ROIs do NOT by themselves establish probability calibration.

Frozen baseline from V10 successful study: V2/V3(.15)/V5/V6 structure,
clean score>0 2023 Jan-Oct re-fit; nothing in model is modified.
2023 fit is in sample, Nov-Dec development, 2024-26 previously inspected.
No market odds or payout enters predictions. Missing/conflicting payout cases
are excluded and counted. Genuinely unknown holdout is not claimed.
"""
from __future__ import annotations
import json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import v6_settled_payout_only_2023_2026 as prior
from v8_targeted_weak_podium_calibration import row_probs

OUT=ROOT/"data/audits/v6_all_trifecta_p_vs_settled_75pct_2023_2026.json"
YEARS={"2024":25348,"2025":24635,"2026":11797}
# All probabilities are raw, unchanged probabilities. Intervals are display
# groups only and are never applied as a betting selection filter.
COARSE=[
 ("p_lt_0p01pct",0,.0001),
 ("p_0p01_to_0p05pct",.0001,.0005),
 ("p_0p05_to_0p1pct",.0005,.001),
 ("p_0p1_to_0p5pct",.001,.005),
 ("p_0p5_to_1pct",.005,.01),
 ("p_1_to_2pct",.01,.02),
 ("p_2_to_5pct",.02,.05),
 ("p_5_to_10pct",.05,.1),
 ("p_10_to_20pct",.1,.2),
 ("p_20pct_plus",.2,1.00000001)]
FINE_CUTS=[0,.00001,.00005,.0001,.0002,.0003,.0005,.00075,
           .001,.0015,.002,.003,.004,.005,.0075,.01,
           .0125,.015,.0175,.02,.025,.03,.04,.05,.06,.075,
           .09,.10,.11,.125,.15,.175,.2,.25,.3,.4,.5,.75,1.00000001]
FINE=[(f"p_{100*lo:.5f}_to_{100*hi:.5f}pct",lo,hi)
      for lo,hi in zip(FINE_CUTS,FINE_CUTS[1:])]
PERIODS=("2023_train","2023_dev","2024","2025","2026")
TARGET=75.0

class Stat:
    def __init__(self):
        self.tickets=0;self.races=0
        self.p_mass=0.;self.hit_count=0;self.paid=0.
        self.day_set=set();self.day_mass=defaultdict(float)
        self.day_hit=defaultdict(int);self.day_stake=defaultdict(int)
        self.day_return=defaultdict(float)
    def ticket(self,day,p,hit,payout):
        self.tickets+=1;self.p_mass+=p;self.hit_count+=int(hit)
        self.day_set.add(day);self.day_mass[day]+=p
        self.day_hit[day]+=int(hit);self.day_stake[day]+=100
        if hit:
            self.paid+=payout;self.day_return[day]+=payout
    def race(self,prob_mass):
        if prob_mass>1e-16:
            assert -.00000001<=prob_mass<=1.00000001
            self.races+=1
    def emit(self):
        if self.tickets==0:return None
        invested=100*self.tickets
        roi=100*self.paid/invested
        expected_payout=invested*TARGET/100
        # Day residual for probability calibration only: not a multiple-
        # comparison corrected significance test.
        dates=list(self.day_mass)
        residual=sum(self.day_hit[x]-self.day_mass[x] for x in dates)
        dayvar=sum((self.day_hit[x]-self.day_mass[x])**2 for x in dates)
        z=residual/math.sqrt(dayvar) if dayvar>0 else None
        return {
          "all_tickets_in_probability_group":self.tickets,
          "races_with_probability_mass":self.races,
          "calendar_days":len(self.day_set),
          "mean_v6_probability_pct":round(100*self.p_mass/self.tickets,8),
          "expected_winning_tickets":round(self.p_mass,5),
          "actual_winning_tickets":self.hit_count,
          "observed_hit_pct":round(100*self.hit_count/self.tickets,8),
          "actual_minus_v6_expected_wins":round(self.hit_count-self.p_mass,5),
          "probability_calibration_residual_day_diagnostic_z":round(z,5) if z is not None else None,
          "hypothetical_stake_yen":invested,
          "actual_settled_payout_yen":round(self.paid,3),
          "expected_payout_at_75pct_yen":round(expected_payout,3),
          "realized_roi_pct":round(roi,6),
          "signed_roi_error_vs_75_pct_points":round(roi-TARGET,6),
          "ABS_roi_error_vs_75_pct_points":round(abs(roi-TARGET),6),
          "signed_payout_error_vs_75_yen":round(self.paid-expected_payout,3),
          "exact_75pct_success":abs(roi-TARGET)<1e-9
        }

class Audit:
    def __init__(self):
        self.s=defaultdict(Stat);self.n=0
        self.winner_price_products=defaultdict(list)
        self.full_mass_failures=0;self.tickets_evaluated=0
    def race(self,e,period,dist,winning_payout):
        y=e["order"][:3];day=e["date"]
        groups=("TOTAL",period,"2024_to_2026") if period in ("2024","2025","2026") else ("TOTAL",period)
        sums=defaultdict(float)
        winners=0;p_sum=0.
        if len(dist)!=len(e["scores"])*(len(e["scores"])-1)*(len(e["scores"])-2):
            raise AssertionError("Incomplete all-ticket V6 enumeration")
        for t,p in dist.items():
            if not (0<p<1 and math.isfinite(p)):
                raise AssertionError("Probability invalid, cannot audit every ticket")
            hit=t==y
            winners+=int(hit);p_sum+=p
            b=next((name for name,lo,hi in COARSE if lo<=p<hi),None)
            f=next((name for name,lo,hi in FINE if lo<=p<hi),None)
            if b is None or f is None:raise AssertionError(("Unbinned full-ticket p",p))
            payout=winning_payout if hit else 0
            for group in groups:
                for key in ("ALL",b,f):
                    st=self.s[(group,key)]
                    st.ticket(day,p,hit,payout)
                    sums[(group,key)]+=p
            if hit:
                # If market fair probability equals p and 25% takeout, this
                # equals 75 JPY for the 100-JPY ticket; no losing odds needed.
                value=p*winning_payout
                for group in groups:self.winner_price_products[group].append(value)
        if winners!=1 or abs(p_sum-1)>1e-8:
            self.full_mass_failures+=1
            raise AssertionError(("V6 probability or result conservation",winners,p_sum))
        for key,p_mass in sums.items():
            self.s[key].race(p_mass)
        self.tickets_evaluated+=len(dist);self.n+=1

def main():
    # The illustrative 10% example is ONLY a unit test, not ticket selection.
    unit=prior.Stat()
    for i in range(100):
        unit.ticket(.1,i<10,750,"toy")
        unit.race(.1)
    assert unit.as_dict()["actual_settled_payout_yen"]==7500
    print("V6_10PCT_IS_EXAMPLE_ONLY_75PCT_UNITTEST_OK",flush=True)

    all23,src=prior.obs.ingest()
    train={rid:e for rid,e in all23.items()
           if e["date"]<="2023-10-31" and prior.weak.score_gate(e)=="ELIGIBLE"}
    assert len(train)==19017,("V10 clean training sample differs",len(train))
    wg,wr=prior.v2.fit_structural(train)
    pw,_=prior.v3.train_pattern_gate(train)
    d5=prior.v5.fit_residual(train,wg,wr)
    wl,wc=prior.v6.fit_opponent(train,wg,wr,d5)
    args=(wg,wr,d5,wl,wc,pw)
    del train
    print("V6_FULL_TICKETS_MODEL_FIXED_REFIT_COMPLETE",flush=True)
    sections={"2023":all23}
    for year,load in (("2024",prior.r24.ingest_2024),
                      ("2025",prior.r25.read_2025),
                      ("2026",prior.r26.ingest_2026)):
        events,audit=load()
        assert len(events)==YEARS[year],("Original V6 population changed",year)
        sections[year]=events
    payout,conflicts,pay_source_audit=prior.load_settled_payments(
        set().union(*(set(z) for z in sections.values())))
    audit=Audit();coverage={}
    for year,events in sections.items():
        counts=Counter()
        for rid,e in events.items():
            counts["source_races"]+=1
            gate=prior.weak.score_gate(e)
            if gate!="ELIGIBLE":
                counts["excluded_"+gate]+=1;continue
            counts["score_valid"]+=1
            if rid in conflicts:
                counts["conflicting_payout_excluded"]+=1;continue
            pay=payout.get(rid)
            if pay is None:
                counts["missing_payout_excluded"]+=1;continue
            if pay[0]!=tuple(e["order"][:3]):
                counts["payout_result_mismatch_excluded"]+=1;continue
            dist=row_probs(e,*args)
            period=("2023_train" if e["date"]<="2023-10-31" else "2023_dev") if year=="2023" else year
            audit.race(e,period,dist,pay[1])
            counts["matched_races_evaluated"]+=1
            if counts["matched_races_evaluated"]%5000==0:
                print("V6_ALL_TICKETS_PROGRESS",year,counts["matched_races_evaluated"],flush=True)
        assert counts["score_valid"]==sum(counts[k] for k in (
            "conflicting_payout_excluded","missing_payout_excluded",
            "payout_result_mismatch_excluded","matched_races_evaluated"))
        coverage[year]=dict(counts)
        print("V6_ALL_TICKETS_YEAR_DONE",year,json.dumps({
            "races":counts["matched_races_evaluated"],
            "all_ticket_roi_75_check":audit.s[(year if year!="2023" else "2023_dev","ALL")].emit(),
            "coverage":dict(counts)
        },ensure_ascii=False),flush=True)

    groups=("TOTAL",*PERIODS,"2024_to_2026")
    def winner_stats(group):
        vals=audit.winner_price_products[group]
        if not vals:return None
        vals=sorted(vals)
        n=len(vals)
        mean=sum(vals)/n
        return {"n_winners":n,
                "mean_p_winner_times_100yen_payout_yen":round(mean,5),
                "median_p_winner_times_100yen_payout_yen":round(vals[n//2],5),
                "mean_minus_75_yen":round(mean-75,5),
                "this_is_price_agreement_NOT_probability_calibration":True}
    result={
        "state":"V6_ALL_TRIFECTA_ROI_75PCT_VS_ACTUAL_COMPLETE",
        "model":"Original V2/V3/V5/V6 algorithms, score-valid V10 2023 Jan-Oct model training",
        "zero_score_races_excluded":True,
        "all_tickets_no_probability_selection":True,
        "model_unchanged":True,
        "no_odds_file_read":True,
        "using_only_actual_winning_settled_payout":True,
        "ideal_75pct_roi_benchmark":75,
        "both_above_and_below_75_are_errors":True,
        "no_unseen_period_claim":True,
        "period_notes":{"2023_train":"training data, not OOS",
                        "2023_dev":"development 2023 Nov-Dec",
                        "2024":"previously inspected retrospective",
                        "2025":"previously inspected retrospective",
                        "2026":"retrospective archived through 2026-06"},
        "covered_races":audit.n,
        "covered_all_distinct_ticket_probabilities":audit.tickets_evaluated,
        "probability_normalization_failures":audit.full_mass_failures,
        "coverage":coverage,
        "settled_payout_sources":pay_source_audit,
        "coarse_probability_bands":COARSE,
        "fine_probability_bands":FINE,
        "full_population":{group:audit.s[group,"ALL"].emit() for group in groups},
        "per_probability_band":{
            group:{b:audit.s[group,b].emit() for b,_,_ in COARSE}
            for group in groups
        },
        "fine_probability_band":{
            group:{b:audit.s[group,b].emit() for b,_,_ in FINE}
            for group in groups
        },
        "winner_only_price_factor":{group:winner_stats(group) for group in groups},
        "critical_interpretation":[
            "Every single ordered ticket receives a probability and is hypothetically bought exactly once for 100 JPY; no 10% subset.",
            "Observed ROI is compared symmetrically with 75%: 80% means +5pt error and 70% means -5pt error.",
            "ROI for buying ALL tickets does not depend on model ranking, so it alone cannot validate V6; the by-probability-band calibration is essential.",
            "A model may be calibrated without yielding a 75% ROI in a probability cell if market prices differ from independent model probabilities.",
            "Low-probability bins and rare huge-payout races can have large sampling error, and matching 75% does not establish profit or true holdout correctness.",
            "Only settled winning payout rows are required; never treat missing payout as zero.",
            "Predictor uses pre-race score and predicted lines, never odds or result."
        ]
    }
    assert result["full_population"]["TOTAL"]["actual_winning_tickets"]==audit.n
    assert abs(result["full_population"]["TOTAL"]["expected_winning_tickets"]-audit.n)<.02
    assert sum(result["per_probability_band"]["TOTAL"][b]["all_tickets_in_probability_group"] for b,_,_ in COARSE)==audit.tickets_evaluated
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V6_FULL_ALL_TICKET_FINAL_RESULT",json.dumps({
        "count_races":audit.n,"count_all_tickets":audit.tickets_evaluated,
        "coverage":coverage,
        "full_population":result["full_population"],
        "all_coarse_2024_to_2026":result["per_probability_band"]["2024_to_2026"],
        "winner_price_2024_to_2026":result["winner_only_price_factor"]["2024_to_2026"],
    },ensure_ascii=False),flush=True)
if __name__=="__main__":main()

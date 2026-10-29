#!/usr/bin/env python3
"""FROZEN V6 (V10 score-valid training protocol) versus 2023-2026 3rentan prices.

Training algorithm and training period are identical to the successful V10
rank/role/slot audit. All probabilities are computed WITHOUT odds or results.
Compare:
 (1) Expected vs observed wins in predeclared AI-probability bands.
 (2) Real final-odds payouts vs V6-expected payouts and a normalized market
     probability benchmark (theoretical 75% is only an illustration).
 (3) One pre-race AI-only ticket per race nearest 10% (9.5%-10.5% and 8%-12%).
In-sample 2023 Jan-Oct, development 2023 Nov-Dec, previously researched
2024-2026 are reported SEPARATELY. This is NOT a new independent holdout and
final odds are NOT confirmed purchasable pre-race prices.
No market odds used to train, adjust, select, or filter ticket candidates.
Missing whole-book odds are excluded WITH explicit coverage counts.
Zero/missing-score races are excluded wholesale. No model code is changed.
"""
from __future__ import annotations
import gzip, itertools, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as obs
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import v5_rider_score_role_size_interactions as v5
import v6_opponent_competition_structure as v6
import calibration_audit as r24
import validate_2025 as r25
import v4_pattern_floor_2026 as r26
import weak_rider_podium_zero_excluded as weak
from v8_targeted_weak_podium_calibration import row_probs

OUT=ROOT/"data/audits/v6_market_price_recovery_calibration_2023_2026.json"
YEARS={"2024":25348,"2025":24635,"2026":11797}
BANDS=(
 ("lt_0p01pct",0,.0001),
 ("0p01_to_0p05pct",.0001,.0005),
 ("0p05_to_0p1pct",.0005,.001),
 ("0p1_to_0p5pct",.001,.005),
 ("0p5_to_1pct",.005,.01),
 ("1_to_2pct",.01,.02),
 ("2_to_5pct",.02,.05),
 ("5_to_10pct",.05,.1),
 ("10_to_20pct",.1,.2),
 ("20pct_plus",.2,1.0000001),
)
TARGETS=(("nearest_10pct_9p5_to_10p5",.095,.105),
         ("nearest_10pct_8_to_12",.08,.12))
PAYOUT_100=100.
THEORETICAL_NET_MULTIPLIER=.75

class Stat:
    __slots__=("bets","hits","sum_p","return_yen","expected_v6_return_yen",
               "expected_market_return_yen","stake_yen","variance","days",
               "races_with_bets","max_win_payout","virtual_75_return_yen")
    def __init__(self):
        self.bets=0;self.hits=0;self.sum_p=0.
        self.return_yen=0.;self.expected_v6_return_yen=0.
        self.expected_market_return_yen=0.;self.stake_yen=0.
        self.variance=0.;self.days=set();self.races_with_bets=0
        self.max_win_payout=0.;self.virtual_75_return_yen=0.
    def add(self,day,p,odd,hit,inv_book_total):
        self.bets+=1;self.hits+=int(hit);self.sum_p+=p
        self.stake_yen+=100;self.return_yen+=100*odd*int(hit)
        self.expected_v6_return_yen+=100*p*odd
        self.expected_market_return_yen+=100/inv_book_total
        if hit and p>0:
            self.virtual_75_return_yen+=100*.75/p
            self.max_win_payout=max(self.max_win_payout,100*odd)
        self.days.add(day)
    def finish_race(self,race_prob):
        if not -1e-8<=race_prob<=1+1e-8:
            raise AssertionError(("Invalid within-race band probability",race_prob))
        if race_prob>0:
            self.races_with_bets+=1
            self.variance+=race_prob*(1-race_prob)
    def pack(self):
        if not self.bets:return None
        err=self.hits-self.sum_p
        se=math.sqrt(self.variance)
        b=self.stake_yen
        return {
            "tickets":self.bets,"races_with_candidates":self.races_with_bets,
            "days":len(self.days),"stake_100_yen_each":int(b),
            "predicted_wins":round(self.sum_p,5),"observed_wins":self.hits,
            "predicted_hit_pct":round(self.sum_p/self.bets*100,6),
            "observed_hit_pct":round(self.hits/self.bets*100,6),
            "observed_minus_predicted_hits":round(err,5),
            "observed_over_expected":round(self.hits/self.sum_p,6) if self.sum_p else None,
            "calibration_z_race_unit":round(err/se,5) if se>0 else None,
            "actual_payout_yen":round(self.return_yen,3),
            "realized_roi_pct":round(100*self.return_yen/b,5),
            "V6_expected_payout_at_final_market_odds_yen":round(self.expected_v6_return_yen,3),
            "V6_expected_roi_at_final_odds_pct":round(self.expected_v6_return_yen/b*100,5),
            "normalized_market_expected_roi_pct":round(self.expected_market_return_yen/b*100,5),
            "virtual_75pct_of_AI_fair_odds_realized_roi_pct":round(
                self.virtual_75_return_yen/b*100,5),
            "virtual_75pct_AI_fair_odds_model_expectation_pct":75.,
            "largest_winning_payout_share_pct":round(
                self.max_win_payout/self.return_yen*100,5) if self.return_yen else None
        }

class Table:
    def __init__(self):
        self.stats=defaultdict(Stat)
        self.races=0
    def add_race(self,year,day,dist,odds,winning):
        labels=["all",year]
        bucket_mass=defaultdict(float)
        # Collect winners and expected mass inside the same race; tickets are
        # mutually exclusive, so variance must be on race event probability.
        inv=sum(1./o for o in odds.values())
        assert inv>0 and math.isfinite(inv)
        for trip,p in dist.items():
            odd=odds[trip]
            hit=(trip==winning)
            b=next(name for name,lo,hi in BANDS if lo<=p<hi)
            for segment in labels:
                self.stats[(segment,"all")].add(day,p,odd,hit,inv)
                self.stats[(segment,b)].add(day,p,odd,hit,inv)
            bucket_mass[b]+=p
        for segment in labels:
            self.stats[(segment,"all")].finish_race(1.)
            for name,_,_ in BANDS:
                self.stats[(segment,name)].finish_race(bucket_mass[name])
        # Exactly one candidate per race and target when the interval exists;
        # selection sees ONLY V6 probabilities and canonical ticket numbers.
        for name,lo,hi in TARGETS:
            cs=[(abs(p-.1),trip,p) for trip,p in dist.items() if lo<=p<hi]
            if not cs:continue
            _,ticket,p=min(cs)
            for segment in labels:
                self.stats[(segment,name)].add(day,p,odds[ticket],ticket==winning,inv)
                self.stats[(segment,name)].finish_race(p)
        self.races+=1
    def pack(self):
        return {s:{label:self.stats[(s,label)].pack() for label in
                ["all",*[b for b,_,_ in BANDS],*[b for b,_,_ in TARGETS]]}
                for s in ["all","2023_train","2023_dev","2024","2025","2026"]}

def parse_market(path,event_ids):
    if not path.is_file():
        raise FileNotFoundError(f"Verified complete-odds archive missing: {path}")
    kept={};counts=Counter();duplicates=set()
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            d=json.loads(line)
            counts["archive_records"]+=1
            if set(d)!={"race_id","race_type","odds"}:
                raise AssertionError("Unexpected odds-only source schema")
            rid=str(d["race_id"])
            if rid not in event_ids:continue
            counts["matched_id"]+=1
            if rid in kept:duplicates.add(rid)
            kept[rid]=d["odds"]
    if duplicates:
        raise AssertionError(("Duplicate odds book for exact race IDs",sorted(duplicates)[:5]))
    return kept,dict(counts)

def validated_prices(e,book):
    if not isinstance(book,dict):return None
    cs=tuple(sorted(e["scores"],key=int))
    expect={"-".join(t) for t in itertools.permutations(cs,3)}
    three=book.get("3連単")
    trio=book.get("3連複")
    if not isinstance(three,dict) or not isinstance(trio,dict):
        return None
    if set(three)!=expect or len(trio)!=math.comb(len(cs),3):
        return None
    if set(trio)!={"=".join(t) for t in itertools.combinations(cs,3)}:
        return None
    prices={}
    for k,v in three.items():
        try:odd=float(v)
        except (TypeError,ValueError):return None
        if not math.isfinite(odd) or odd<1:return None
        prices[tuple(k.split("-"))]=odd
    # The output archive was already certified whole-book, but recheck it.
    for v in trio.values():
        try:p=float(v)
        except (TypeError,ValueError):return None
        if not math.isfinite(p) or p<1:return None
    return prices

def base_tests():
    # Ideal 10%-chance tickets each have a 7.5x 100-yen expected payout
    # when the market prices that same event at 7.5x after 25% deduction.
    s=Stat()
    for k in range(100):
        s.add("day",.10,7.5,k<10,1/.75)
        s.finish_race(.10)
    ans=s.pack()
    assert ans["stake_100_yen_each"]==10000
    assert ans["predicted_wins"]==10 and ans["observed_wins"]==10
    assert ans["actual_payout_yen"]==7500
    assert ans["realized_roi_pct"]==75.
    assert ans["V6_expected_roi_at_final_odds_pct"]==75.
    assert ans["virtual_75pct_of_AI_fair_odds_realized_roi_pct"]==75.
    print("V6_10PCT_100_RACES_7500YEN_SELFTEST_PASS",flush=True)

def evaluate_group(events,period,train_parts,table,coverage):
    eligible={rid:e for rid,e in events.items() if weak.score_gate(e)=="ELIGIBLE"}
    coverage[period]={"source_races":len(events),
                      "score_valid":len(eligible),
                      "missing_or_zero_score":len(events)-len(eligible)}
    oddsfile=ROOT/f"data/odds_only/odds_{period[:4]}.jsonl.gz"
    books,scan=parse_market(oddsfile,eligible)
    coverage[period]["archive"]=scan
    counts=Counter()
    for i,(rid,e) in enumerate(eligible.items(),1):
        book=books.get(rid)
        if book is None:
            counts["missing_full_market"]+=1;continue
        prices=validated_prices(e,book)
        if prices is None:
            counts["invalid_or_roster_mismatch"]+=1;continue
        dist=row_probs(e,*train_parts)
        if len(dist)!=len(prices) or set(dist)!=set(prices):
            raise AssertionError(("Ticket grid mismatch",rid))
        if abs(sum(dist.values())-1)>1e-8:
            raise AssertionError(("V6 full trifecta probability not 1",rid))
        if not all(0<p<1 and math.isfinite(p) for p in dist.values()):
            raise AssertionError(("V6 nonfinite or zero ticket",rid))
        year=("2023_train" if e["date"]<="2023-10-31" else "2023_dev") if period=="2023" else period
        table.add_race(year,e["date"],dist,prices,e["order"])
        counts["evaluated_complete_market"]+=1
        if counts["evaluated_complete_market"]%2500==0:
            print("V6_PRICE_CENSUS_PROGRESS",period,counts["evaluated_complete_market"],flush=True)
    coverage[period]["reasons"]=dict(counts)
    if sum(counts.values())-counts.get("evaluated_complete_market",0) != len(eligible):
        # All score-valid events must fall into exactly one outcome.
        raise AssertionError(("Coverage accounting mismatch",period,coverage[period]))
    print("V6_PRICE_YEAR_COMPLETED",json.dumps({
        "year":period,"coverage":coverage[period],
        "near10":table.stats[(period if period!="2023" else "2023_dev",
                              "nearest_10pct_8_to_12")].pack() if period!="2023" else
                 table.stats[("2023_dev","nearest_10pct_8_to_12")].pack()
        },ensure_ascii=False),flush=True)

def main():
    base_tests()
    all23,source23=obs.ingest()
    train={rid:e for rid,e in all23.items()
           if e["date"]<="2023-10-31" and weak.score_gate(e)=="ELIGIBLE"}
    if len(train)!=19017:raise AssertionError(("V10 baseline training changed",len(train)))
    print("V6_PRICE_FROZEN_TRAIN_BEGIN",len(train),flush=True)
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    args=(wg,wr,d5,wl,wc,pw)
    print("V6_PRICE_FROZEN_V10_MODEL_FITTED_NO_ODDS",flush=True)
    del train
    table=Table();coverage={}
    evaluate_group(all23,"2023",args,table,coverage)
    del all23
    for yr,reader in (("2024",r24.ingest_2024),
                      ("2025",r25.read_2025),
                      ("2026",r26.ingest_2026)):
        events,source=reader()
        if len(events)!=YEARS[yr]:raise AssertionError(("Input drift",yr,len(events)))
        evaluate_group(events,yr,args,table,coverage)
        del events
    stats=table.pack()
    check=stats["all"]["all"]
    assert check["observed_wins"]==table.races
    assert abs(check["predicted_wins"]-table.races)<.01
    assert check["tickets"]>table.races
    output={
        "state":"V6_FROZEN_ODDS_PRICE_CALIBRATION_2023_2026_COMPLETE",
        "training":"V10 score-nonzero V6 model code unchanged; trained 2023 Jan-Oct 19017 races",
        "eval_windows":{"2023_train":"in-sample","2023_dev":"dev previously inspected",
                        "2024":"retrospective inspected",
                        "2025":"retrospective inspected","2026":"retrospective inspected"},
        "validation_notes":[
            "p and ticket selection computed from pre-race card, BEFORE odds and result comparison",
            "No model update, odds training, probability floor, post-hoc threshold selection",
            "Full, valid 3rentan and 3renpuku final odds required; no fabricated or filled prices",
            "Final odds cannot establish pre-race bet executable prices",
            "75% is illustrative, not universal; normalized market baseline is measured from whole-book sum of 1/odds",
            "Correct ROI alone does not prove calibrated probabilities; compare actual vs predicted hits",
            "2023 Jan-Oct was used to fit model; 2024-2026 previously explored and not fresh unknown-period proof",
            "One ticket selected per race per 10% interval, no outcome or price filter",
            "All tickets in a race are mutually exclusive; z uses race-level selection probability mass"
        ],
        "model_model_version":"V6 full trifecta post V3=.15 pattern gate; score-zero races excluded as V10",
        "market_source":"data/odds_only/odds_YEAR.jsonl.gz, archived FINAL price",
        "unit_stake_yen":100,"illustrative_theoretical_roi_pct":75,
        "race_counts":table.races,
        "coverage":coverage,
        "statistics":stats,
        "source_2023":source23
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V6_PRICE_FINAL_RESULT",json.dumps({
        "race_counts":table.races,
        "coverage":{yr:val["reasons"] for yr,val in coverage.items()},
        "near_10pct":{yr:stats[yr]["nearest_10pct_9p5_to_10p5"]
                      for yr in ("2023_train","2023_dev","2024","2025","2026")},
        "by_year_all_tickets":{yr:stats[yr]["all"] for yr in
                              ("2023_train","2023_dev","2024","2025","2026")}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

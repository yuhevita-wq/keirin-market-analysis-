#!/usr/bin/env python3
"""FROZEN V6 audit using ONLY settled winning payouts; no full odds.

The model probabilities are identical to the successfully run V10 protocol:
algorithm V2/V3(.15)/V5/V6, score-zero excluded, fit on 2023 Jan-Oct.
The chosen tickets are based only on V6 probability and vehicle numbers.
Three independent quantities are kept distinct:
 - Expected vs observed winners (calibration).
 - Actual settled payouts and genuine ROI (requires ONLY winning payout).
 - Hypothetical 75% of 1/p fair price (diagnostic, NOT a market promise).
No full-odds market file is loaded. No payouts used to fit, choose or alter V6.
No model version files are changed and no profitable subset is tuned.
All 2024-2026 results are previously inspected, not untouched OOS.
"""
from __future__ import annotations
import csv,gzip,io,json,math,sys,zipfile
from collections import Counter,defaultdict
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

OUT=ROOT/"data/audits/v6_settled_payout_only_calibration_2023_2026.json"
YEARS={"2024":25348,"2025":24635,"2026":11797}
BANDS=(("lt_0p01pct",0,.0001),("0p01_to_0p05pct",.0001,.0005),
       ("0p05_to_0p1pct",.0005,.001),("0p1_to_0p5pct",.001,.005),
       ("0p5_to_1pct",.005,.01),("1_to_2pct",.01,.02),
       ("2_to_5pct",.02,.05),("5_to_10pct",.05,.1),
       ("10_to_20pct",.1,.2),("20pct_plus",.2,1.0000001))
TARGETS=(("nearest_10pct_9p5_to_10p5",.095,.105),
         ("nearest_10pct_8_to_12",.08,.12))
PERIODS=("2023_train","2023_dev","2024","2025","2026")
EXPECTED_PAYOUT_RATIO=.75

class Stat:
    def __init__(self):
        self.bets=0;self.hits=0;self.sum_p=0.;self.paid_yen=0.
        self.virtual_paid_yen=0.;self.max_payout=0.;self.races=0
        self.sum_race_variance=0.
        self.days=set()
    def ticket(self,p,hit,settled_payout,day):
        self.bets+=1
        self.hits+=int(hit)
        self.sum_p+=p
        self.days.add(day)
        if hit:
            self.paid_yen+=settled_payout
            self.virtual_paid_yen+=100*.75/p
            self.max_payout=max(self.max_payout,settled_payout)
    def race(self,p_mass):
        if not -1e-8<=p_mass<=1+1e-8:raise AssertionError("Invalid race mass")
        if p_mass>0:
            self.races+=1
            self.sum_race_variance+=p_mass*(1-p_mass)
    def as_dict(self):
        if not self.bets:return None
        stake=100*self.bets
        z=(self.hits-self.sum_p)/math.sqrt(self.sum_race_variance) if self.sum_race_variance>0 else None
        return {
            "tickets":self.bets,"races_with_tickets":self.races,"calendar_days":len(self.days),
            "stake_yen":stake,
            "predicted_hit_count":round(self.sum_p,5),
            "actual_hit_count":self.hits,
            "predicted_hit_pct":round(100*self.sum_p/self.bets,6),
            "observed_hit_pct":round(100*self.hits/self.bets,6),
            "actual_minus_predicted_count":round(self.hits-self.sum_p,5),
            "actual_over_predicted_hits":round(self.hits/self.sum_p,6) if self.sum_p else None,
            "race_cluster_diagnostic_z":round(z,5) if z is not None else None,
            "actual_settled_payout_yen":round(self.paid_yen,3),
            "actual_roi_pct":round(100*self.paid_yen/stake,5),
            "theoretical_75pct_benchmark_payout_yen":round(.75*stake,3),
            "actual_minus_75pct_benchmark_yen":round(self.paid_yen-.75*stake,3),
            "hypothetical_75pct_of_1_over_p_virtual_payout_yen":round(self.virtual_paid_yen,3),
            "hypothetical_75pct_fair_price_virtual_roi_pct":round(100*self.virtual_paid_yen/stake,5),
            "max_one_hit_share_of_return_pct":round(100*self.max_payout/self.paid_yen,5) if self.paid_yen else None,
        }

def payout_sources():
    # Canonical source paths only; aliases under all_classes deliberately omitted.
    for root in ("data/a_class_12","data/a3_challenge",
                 "data/2023","data/2024","data/2025",
                 "data/2026_h1","data/2026_future_block1"):
        folder=ROOT/root
        if folder.exists():
            for p in sorted(folder.rglob("payouts.csv")):yield ("csv",p)
    folder=ROOT/"data/grade_races"
    if folder.exists():
        for p in sorted(folder.rglob("*.zip")):
            if any(str(y) in str(p) for y in (2023,2024,2025,2026)):
                yield ("zip",p)

def rows_csv(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)

def rows_zip(p):
    with zipfile.ZipFile(p) as z:
        names=[n for n in z.namelist() if n.rsplit("/",1)[-1]=="payouts.csv"]
        for name in names:
            with z.open(name) as raw:
                with io.TextIOWrapper(raw,encoding="utf-8-sig",newline="") as f:
                    yield from csv.DictReader(f)

def load_settled_payments(wanted):
    # Return a single paid 3ren-tan combination/payout per race.
    # Conflicting recorded outcomes are NOT silently resolved.
    found=defaultdict(set);audit=Counter()
    for typ,path in payout_sources():
        audit["source_files_checked"]+=1
        try:
            iterator=rows_zip(path) if typ=="zip" else rows_csv(path)
            for row in iterator:
                rid=str(row.get("race_id","")).strip()
                if rid not in wanted:continue
                if str(row.get("ticket_type","")).strip() not in ("3連単","trifecta"):continue
                audit["three_ticket_source_rows"]+=1
                if str(row.get("status","")).strip().lower() not in ("paid","confirmed"):
                    audit["not_paid"]+=1;continue
                combo=str(row.get("combination","")).strip()
                bits=combo.split("-")
                if len(bits)!=3 or len(set(bits))!=3 or not all(c.isdigit() and len(c)==1 for c in bits):
                    audit["invalid_winner_combination"]+=1;continue
                try:v=int(str(row.get("payout_yen","")).replace(",",""))
                except (ValueError,TypeError):
                    audit["invalid_payout"]+=1;continue
                if v<=0:
                    audit["nonpositive_payout"]+=1;continue
                found[rid].add((tuple(bits),v))
        except (OSError,zipfile.BadZipFile,UnicodeError) as ex:
            audit["unreadable_source"]+=1
            print("PAYOUT_SOURCE_UNREADABLE",str(path),type(ex).__name__,flush=True)
    values={};conflicted=set()
    for rid,variants in found.items():
        if len(variants)==1:values[rid]=next(iter(variants))
        else:conflicted.add(rid)
    audit["races_valid_payout_row"]=len(values)
    audit["races_conflicting_paid_rows"]=len(conflicted)
    print("V6_SETTLED_PAYOUT_SOURCE_AUDIT",json.dumps(dict(audit)),flush=True)
    return values,conflicted,dict(audit)

class Audit:
    def __init__(self):
        self.stats=defaultdict(Stat)
        self.races=0;self.coverage=defaultdict(Counter)
    def race(self,period,e,dist,pay):
        y=tuple(e["order"][:3])
        win_combo,won_yen=pay
        if y!=win_combo:raise AssertionError(("PAYOUT_MISMATCH",e["date"],y,win_combo))
        labels=("all",period)
        mass=defaultdict(float)
        for ticket,p in dist.items():
            if not 0<p<1 or not math.isfinite(p):
                raise AssertionError(("Invalid V6 p",p))
            hit=(ticket==y)
            b=next(name for name,lo,hi in BANDS if lo<=p<hi)
            for seg in labels:
                self.stats[seg,"all"].ticket(p,hit,won_yen,e["date"])
                self.stats[seg,b].ticket(p,hit,won_yen,e["date"])
            mass[b]+=p
        if abs(sum(mass.values())-1)>1e-7:raise AssertionError("V6 total mass not 1")
        for seg in labels:
            self.stats[seg,"all"].race(1.)
            for b,lo,hi in BANDS:
                self.stats[seg,b].race(mass[b])
        for name,lo,hi in TARGETS:
            cs=[(abs(p-.1),t,p) for t,p in dist.items() if lo<=p<hi]
            if not cs:continue
            _,ticket,p=min(cs)
            for seg in labels:
                self.stats[seg,name].ticket(p,ticket==y,won_yen,e["date"])
                self.stats[seg,name].race(p)
        self.races+=1

def check_7500():
    s=Stat()
    for k in range(100):
        s.ticket(.1,k<10,750,"day")
        s.race(.1)
    x=s.as_dict()
    assert x["stake_yen"]==10000
    assert x["predicted_hit_count"]==10 and x["actual_hit_count"]==10
    assert x["actual_settled_payout_yen"]==7500
    assert x["actual_roi_pct"]==75
    assert x["hypothetical_75pct_fair_price_virtual_roi_pct"]==75
    print("V6_10PCT_100_RACES_SETTLED_7500YEN_SELFTEST_PASS",flush=True)

def main():
    check_7500()
    all23,src=obs.ingest()
    train={rid:e for rid,e in all23.items()
           if e["date"]<="2023-10-31" and weak.score_gate(e)=="ELIGIBLE"}
    if len(train)!=19017:raise AssertionError(("V10 baseline drift",len(train)))
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    args=(wg,wr,d5,wl,wc,pw)
    del train
    print("V6_PAYOUT_FROZEN_SCORE_VALID_FIT_COMPLETE",flush=True)

    sections={"2023":all23}
    for year,reader in (("2024",r24.ingest_2024),
                        ("2025",r25.read_2025),
                        ("2026",r26.ingest_2026)):
        es,source=reader()
        if len(es)!=YEARS[year]:raise AssertionError(("Source population drift",year,len(es)))
        sections[year]=es
    wanted=set().union(*(set(d) for d in sections.values()))
    payouts,conflicted,pay_audit=load_settled_payments(wanted)
    audit=Audit()
    for year,events in sections.items():
        counts=Counter()
        for rid,e in events.items():
            counts["source_races"]+=1
            gate=weak.score_gate(e)
            if gate!="ELIGIBLE":
                counts["score_or_card_excluded_"+gate]+=1;continue
            counts["score_valid"]+=1
            if rid in conflicted:
                counts["conflicting_paid_payout_excluded"]+=1;continue
            p=payouts.get(rid)
            if p is None:
                counts["missing_paid_payout_excluded"]+=1;continue
            if p[0]!=tuple(e["order"][:3]):
                counts["payout_vs_result_disagreement_excluded"]+=1;continue
            dist=row_probs(e,*args)
            if abs(sum(dist.values())-1)>1e-8:
                raise AssertionError(("Not complete 3rentan probability",rid))
            expected=len(e["scores"])*(len(e["scores"])-1)*(len(e["scores"])-2)
            if len(dist)!=expected:raise AssertionError(("Incomplete ticket set",rid))
            period=("2023_train" if e["date"]<="2023-10-31" else "2023_dev") if year=="2023" else year
            audit.race(period,e,dist,p)
            counts["matched_verified_evaluated"]+=1
            if counts["matched_verified_evaluated"]%2500==0:
                print("V6_SETTLED_YEAR_PROGRESS",year,counts["matched_verified_evaluated"],flush=True)
        if counts["score_valid"]!=sum(counts[k] for k in (
            "matched_verified_evaluated","conflicting_paid_payout_excluded",
            "missing_paid_payout_excluded","payout_vs_result_disagreement_excluded")):
            raise AssertionError(("Coverage accounting mismatch",year,dict(counts)))
        audit.coverage[year]=counts
        print("V6_SETTLED_YEAR_DONE",year,json.dumps(dict(counts)),flush=True)
    labels=["all",*PERIODS]
    fields=["all",*(x for x,_,_ in BANDS),*(x for x,_,_ in TARGETS)]
    stats={tag:{name:audit.stats[tag,name].as_dict() for name in fields} for tag in labels}
    total=stats["all"]["all"]
    assert total["actual_hit_count"]==audit.races
    assert abs(total["predicted_hit_count"]-audit.races)<.01
    result={
      "state":"FROZEN_V6_SETTLED_PAYOUT_ONLY_2023_2026_COMPLETE",
      "model":"V10 zero-score-excluded V6, same 2023 Jan-Oct fitting protocol",
      "market_odds_required":False,
      "inputs":"race card, top-three result, settled winning 3rentan payout only",
      "bet_unit_yen":100,
      "benchmark":"75% is illustrative 25% deduction with market fair price equal to V6; NOT a guaranteed payout or statistical calibration criterion",
      "period_meanings":{"2023_train":"2023 Jan-Oct in sample",
                         "2023_dev":"2023 Nov-Dec development, previously inspected",
                         "2024":"previously inspected retrospective",
                         "2025":"previously inspected retrospective",
                         "2026":"previously inspected retrospective"},
      "limitations":[
        "Unrecorded winning payout excluded, not silently treated as zero",
        "Different true market probability from V6 implies expected ROI need not be 75%",
        "Observed ROI can be volatile even for perfectly calibrated probabilities",
        "No odds teachers, no selection based on payout, no V6 code change",
        "All ticket bands refer to theoretical one-unit purchases of every combination",
        "10% nearest ticket selected strictly before race via V6 probability",
        "Only valid race cards and settled payout matches included; incomplete coverage reported",
        "2024-2026 already inspected and not fresh truly out-of-sample"
      ],
      "races_evaluated":audit.races,
      "payout_source_audit":pay_audit,
      "coverage":{y:dict(v) for y,v in audit.coverage.items()},
      "results":stats,
      "source_2023":src
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("V6_SETTLED_FINAL_RESULT",json.dumps({
      "races_evaluated":audit.races,
      "coverage":result["coverage"],
      "near_10pct":{x:stats[x]["nearest_10pct_9p5_to_10p5"] for x in PERIODS},
      "near_10pct_wide":{x:stats[x]["nearest_10pct_8_to_12"] for x in PERIODS},
      "full_bands":{x:{b:stats[x][b] for b,_,_ in BANDS} for x in ("2024","2025","2026")}
    },ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()

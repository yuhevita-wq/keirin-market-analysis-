#!/usr/bin/env python3
"""Frozen V6: WHERE exactly did full trifecta probability mass go wrong?

This is observation, NOT a fitted repair. Enumerate every 3ren-tan ticket
in all 2023-Nov/Dec development + matched 2024-2026 races; preserve original
V6 p and sum(p)=1 within each race. No market odds, payouts as teacher,
rules conditioned on the realized winner, nor selected tickets. Results
are used ONLY as labels after full ticket assignment.

For each disjoint partition (p-band, finish line pattern, winner role,
score rank, and their combinations):
 - expected winning count: sum V6 p over ALL tickets in the cell
 - observed winning count: count of exactly winning ordered ticket in cell
 - signed error = observed-expected (positive=receives more p; negative=donor)
 - per-race Bernoulli variance: sum over races q_cell*(1-q_cell).
2024/25/26 directions checked independently; earlier years have already
been inspected in other research, so this is not a truly untouched holdout.
No uncertain high-dimensional pattern automatically becomes a correction.
"""
from __future__ import annotations
import json,math,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import v6_settled_payout_only_2023_2026 as prior
from v8_targeted_weak_podium_calibration import row_probs

OUT=ROOT/"data/audits/v6_probability_donor_receiver_structural_2023dev_2026.json"
YEARS={"2024":25348,"2025":24635,"2026":11797}
BANDS=[
 ("lt_0.01pct",0,.0001),
 ("0.01_to_0.05pct",.0001,.0005),
 ("0.05_to_0.1pct",.0005,.001),
 ("0.1_to_0.5pct",.001,.005),
 ("0.5_to_1pct",.005,.01),
 ("1_to_2pct",.01,.02),
 ("2_to_5pct",.02,.05),
 ("5_to_10pct",.05,.1),
 ("10_to_20pct",.1,.2),
 ("20pct_plus",.2,1.00000001)]
AXES=("P","PAT","R1","RK1","P_PAT","P_R1","P_RK1","PAT_R1",
      "PAT_RK1","P_PAT_R1","P_PAT_RK1","P_PAT_R1_RK1","P_PAT_R1_R2")
PERIODS=("2023_dev","2024","2025","2026")
class Cell:
    __slots__=("tickets","expected","actual","variance","races_with_mass")
    def __init__(self):
        self.tickets=0
        self.expected=0.
        self.actual=0
        self.variance=0.
        self.races_with_mass=0
    def add(self,n,q,y):
        self.tickets+=n
        self.expected+=q
        self.actual+=y
        self.variance+=q*(1-q)
        self.races_with_mass+=1
    def output(self):
        n=self.tickets;ex=self.expected;obs=self.actual;gap=obs-ex
        return {"tickets":n,"expected":round(ex,6),"actual":obs,
                "observed_minus_expected":round(gap,6),
                "expected_minus_observed":round(-gap,6),
                "prediction_relative_error_pct":round(100*gap/ex,5) if ex else None,
                "predicted_hit_pct":round(100*ex/n,7),
                "observed_hit_pct":round(100*obs/n,7),
                "rate_error_pct_points":round(100*gap/n,7),
                "mass_error_pp_per_race":None,
                "race_binomial_z_diagnostic":round(gap/math.sqrt(self.variance),4)
                        if self.variance>0 else None,
                "races_with_cell_mass":self.races_with_mass}

def pband(p):
    for name,l,h in BANDS:
        if l<=p<h:return name
    raise AssertionError(("Unbinned positive V6 p",p))

def structure(t,e,roles):
    a,b,c=t; pos=e["positions"];g1=pos[a][0];g2=pos[b][0];g3=pos[c][0]
    if g1==g2==g3:pat="AAA"
    elif g1==g2:pat="AAB"
    elif g1==g3:pat="ABA"
    elif g2==g3:pat="ABB"
    else:pat="ABC"
    return pat,roles[a],roles[b],"r"+str(e["ranks"][a])

def all_keys(p,pat,role1,role2,rank1):
    bb=pband(p)
    return (("P",bb),
            ("PAT",pat),
            ("R1",role1),
            ("RK1",rank1),
            ("P_PAT",bb+"|"+pat),
            ("P_R1",bb+"|"+role1),
            ("P_RK1",bb+"|"+rank1),
            ("PAT_R1",pat+"|"+role1),
            ("PAT_RK1",pat+"|"+rank1),
            ("P_PAT_R1",bb+"|"+pat+"|"+role1),
            ("P_PAT_RK1",bb+"|"+pat+"|"+rank1),
            ("P_PAT_R1_RK1",bb+"|"+pat+"|"+role1+"|"+rank1),
            ("P_PAT_R1_R2",bb+"|"+pat+"|"+role1+"|"+role2))

def race(cells,e,dist):
    count=len(e["scores"]);target=tuple(e["order"][:3])
    if len(dist)!=count*(count-1)*(count-2):
        raise AssertionError("All ordered triple probabilities required")
    roles={c:prior.v2.rlabel(e,c) for c in e["scores"]}
    local=defaultdict(lambda:[0,0.])
    m=0.
    true_keys=None
    for t,p in dist.items():
        if not math.isfinite(p) or not 0<p<1:raise AssertionError(("Invalid V6 probability",t,p))
        f=structure(t,e,roles)
        keys=all_keys(p,*f)
        if t==target:true_keys=set(keys)
        for k in keys:
            z=local[k];z[0]+=1;z[1]+=p
        m+=p
    if true_keys is None or abs(m-1)>1e-8:
        raise AssertionError(("Missing winning ticket or probability mass",target,m))
    for k,(n,q) in local.items():
        if q<=0 or q>1+1e-8:raise AssertionError(("Invalid cell q",k,q))
        cells[k].add(n,q,int(k in true_keys))
    return len(dist)

def pack(cells,races):
    groups={axis:{} for axis in AXES}
    for (axis,k),s in sorted(cells.items()):
        v=s.output()
        v["mass_error_pp_per_race"]=round(100*(s.actual-s.expected)/races,7)
        groups[axis][k]=v
    for axis,sub in groups.items():
        n=sum(c["tickets"] for c in sub.values())
        observed=sum(c["actual"] for c in sub.values())
        expected=sum(c["expected"] for c in sub.values())
        assert n==cells[("P","lt_0.01pct")].tickets+sum(
            c["tickets"] for k,c in groups["P"].items() if k!="lt_0.01pct"),("p refs",axis)
        assert observed==races,(axis,"Observed winner mass wrong",observed,races)
        assert abs(expected-races)<.15,(axis,"Predicted mass wrong",expected,races)
    return groups

def stable_cells(outputs):
    d=outputs["2024_to_2026"]
    reports={}
    for axis in AXES:
        cand=[]
        for key,s in d["cells"][axis].items():
            ex=s["expected"];diff=s["observed_minus_expected"]
            if ex<100 or abs(diff)<25:continue
            yearly=[outputs[year]["cells"][axis].get(key) for year in ("2024","2025","2026")]
            if not all(v is not None and v["expected"]>=15 for v in yearly):continue
            yearly_d=[v["observed_minus_expected"] for v in yearly]
            same_sign=all((x>0)==(diff>0) and x!=0 for x in yearly_d)
            if not same_sign:continue
            detail={"key":key,"classification":"RECEIVER_V6_UNDERALLOCATED" if diff>0 else "DONOR_V6_OVERALLOCATED",
                 "expected":ex,"observed":s["actual"],"gap":diff,
                 "rate_expected_pct":s["predicted_hit_pct"],
                 "rate_observed_pct":s["observed_hit_pct"],
                 "error_pp_per_race":s["mass_error_pp_per_race"],
                 "z_diagnostic":s["race_binomial_z_diagnostic"],
                 "yearly_diffs":{yr:round(v["observed_minus_expected"],5)
                                 for yr,v in zip(("2024","2025","2026"),yearly)},
                 "yearly_expected":{yr:round(v["expected"],5)
                                 for yr,v in zip(("2024","2025","2026"),yearly)},
                 "yearly_races":{yr:outputs[yr]["races"] for yr in ("2024","2025","2026")}}
            cand.append(detail)
        cand.sort(key=lambda x:abs(x["gap"]),reverse=True)
        reports[axis]={"stable_sign_count":len(cand),
                       "largest_signed_mass_errors":cand[:35],
                       "largest_receivers":sorted((x for x in cand if x["gap"]>0),
                                             key=lambda x:-x["gap"])[:20],
                       "largest_donors":sorted((x for x in cand if x["gap"]<0),
                                             key=lambda x:x["gap"])[:20]}
    return reports

def main():
    print("V6_STRUCTURAL_AUDIT_STARTED",flush=True)
    all23,_=prior.obs.ingest()
    train={rid:e for rid,e in all23.items()
           if e["date"]<="2023-10-31" and prior.weak.score_gate(e)=="ELIGIBLE"}
    assert len(train)==19017
    wg,wr=prior.v2.fit_structural(train)
    pw,_=prior.v3.train_pattern_gate(train)
    d5=prior.v5.fit_residual(train,wg,wr)
    wl,wc=prior.v6.fit_opponent(train,wg,wr,d5)
    args=(wg,wr,d5,wl,wc,pw)
    del train
    print("V6_SAME_MODEL_FIT_19017_2023_JAN_OCT",flush=True)
    sections={"2023_dev":{rid:e for rid,e in all23.items()
                           if e["date"]>"2023-10-31"}}
    del all23
    for year,reader in (("2024",prior.r24.ingest_2024),
                        ("2025",prior.r25.read_2025),
                        ("2026",prior.r26.ingest_2026)):
        entries,_=reader()
        assert len(entries)==YEARS[year]
        sections[year]=entries
    wanted=set().union(*(set(x) for x in sections.values()))
    payout,conflicts,payout_src=prior.load_settled_payments(wanted)
    outputs={};allcells=defaultdict(Cell);coverage={}
    for year,events in sections.items():
        year_cells=defaultdict(Cell);counts=Counter();nb=0
        for rid,e in events.items():
            counts["source"]+=1
            if prior.weak.score_gate(e)!="ELIGIBLE":
                counts["invalid_pre_race_card"]+=1;continue
            if rid in conflicts:
                counts["payout_conflict"]+=1;continue
            pay=payout.get(rid)
            if pay is None:
                counts["missing_verifiable_winner_payout"]+=1;continue
            if pay[0]!=tuple(e["order"][:3]):
                counts["payout_result_disagreement"]+=1;continue
            d=row_probs(e,*args)
            nb+=race(year_cells,e,d)
            counts["evaluated"]+=1
            if counts["evaluated"]%5000==0:
                print("V6_STRUCTURAL_PROGRESS",year,counts["evaluated"],"tickets",nb,flush=True)
        assert counts["source"]==sum(v for k,v in counts.items() if k!="source")
        coverage[year]=dict(counts)
        outputs[year]={"races":counts["evaluated"],
                       "tickets":nb,
                       "cells":pack(year_cells,counts["evaluated"])}
        if year in ("2024","2025","2026"):
            for k,c in year_cells.items():
                a=allcells[k]
                a.tickets+=c.tickets;a.expected+=c.expected
                a.actual+=c.actual;a.variance+=c.variance
                a.races_with_mass+=c.races_with_mass
        print("V6_STRUCTURAL_YEAR_DONE",year,json.dumps({
            "coverage":dict(counts),
            "probability_bands":outputs[year]["cells"]["P"],
            "patterns":outputs[year]["cells"]["PAT"]},
            ensure_ascii=False),flush=True)
        del year_cells,events
    cumulative_races=sum(outputs[yr]["races"] for yr in ("2024","2025","2026"))
    cumulative_tickets=sum(outputs[yr]["tickets"] for yr in ("2024","2025","2026"))
    outputs["2024_to_2026"]={"races":cumulative_races,
                            "tickets":cumulative_tickets,
                            "cells":pack(allcells,cumulative_races)}
    assert cumulative_races==60776,("not exact same settled-payout matched population",cumulative_races)
    assert cumulative_tickets==13994280,("not exact previous full all ticket total",cumulative_tickets)
    stable=stable_cells(outputs)
    result={
        "state":"FROZEN_V6_ALL_TICKET_DONOR_RECIPIENT_STRUCTURAL_AUDIT_COMPLETE",
        "model_frozen":True,
        "model_training":"2023 Jan-Oct 19017 score-valid; exact prior payout-only V6 refit",
        "no_odds_or_payout_uses_for_prediction":True,
        "payout_only_used_to_match_previous_2024_26_audit_cohort":True,
        "prediction_target":"Full ordered 3ren-tan ticket distribution sum=1 each race",
        "definition":"OBSERVED_MINUS_V6_EXPECTED wins; positive RECEIVER (underallocated), negative DONOR (overallocated)",
        "per_race_mass_conservation":"Each disjoint structural axis sums to 1 predicted and 1 observed per race; deficits and surpluses cancel within each axis",
        "caveats":[
          "No transfer weights fitted; no V6 modification.",
          "A coarser axis and its intersections overlap; never add donor errors from different axes together.",
          "Per-race Bernoulli z is descriptive (no cluster or multiple testing correction), not proof of predictability.",
          "Sign stability across 2024,2025,2026 is RETROSPECTIVE: all years inspected before in other studies.",
          "With rare events, realized error can be substantial even if probabilities are correctly calibrated.",
          "Even a stable cell need not identify WHICH individual tickets within the cell should receive mass; finer structural cells must also validate.",
          "Excluding zero-score and unavailable winning-payout races produces a conditional sample.",
          "Payoff and odds have NOT entered labels or p; this is pure probability accuracy, NOT betting-profit proof.",
          "2023 development period is NOT newly unknown.",
        ],
        "coverage":coverage,
        "results":outputs,
        "stable_sign_both_years":stable,
        "settled_payout_provenance":payout_src,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    overview={}
    for axis in ("P","PAT","R1","RK1","P_PAT","P_PAT_R1","P_PAT_R1_RK1"):
        overview[axis]={"top_receivers":stable[axis]["largest_receivers"][:12],
                        "top_donors":stable[axis]["largest_donors"][:12]}
    print("V6_DONOR_RECEIVER_FINAL_RESULT",json.dumps({
       "cohorts":{y:{"races":v["races"],"tickets":v["tickets"]}
                  for y,v in outputs.items()},
       "probability_bands":outputs["2024_to_2026"]["cells"]["P"],
       "line_patterns":outputs["2024_to_2026"]["cells"]["PAT"],
       "winner_roles":outputs["2024_to_2026"]["cells"]["R1"],
       "winner_rank":outputs["2024_to_2026"]["cells"]["RK1"],
       "stable_cells":overview},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

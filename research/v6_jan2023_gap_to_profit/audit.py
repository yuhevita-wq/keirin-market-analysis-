#!/usr/bin/env python3
"""JAN 2023 fixed V6: price disagreement size -> actual winning frequency and ROI.

No V6 retraining changes, no odds teachers, no after-result picking. Compare
(1) every ordered triple in full valid market books and (2) original frozen
V6 formation components on the exact same 2023 January races.
This is descriptive IN-SAMPLE, archived FINAL-odds research, NOT live returns.
"""
from __future__ import annotations
import gzip, itertools, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
sys.path.insert(0,str(ROOT/"research/v6_january_2023_formation_ev"))
import observe as obs
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import v5_rider_score_role_size_interactions as v5
import v6_opponent_competition_structure as v6
import weak_rider_podium_zero_excluded as weak
from v8_targeted_weak_podium_calibration import row_probs
from v6_settled_payout_only_2023_2026 import load_settled_payments
from prepare import parse_market,select

OUT=ROOT/"data/audits/v6_jan2023_gap_to_actual_profit"
GAP_BOUND=(.5,.75,1,1.1,1.2,4/3,1.5,1.75,2,3,5,10)
EV_BOUND=(.5,.75,.9,1,1.1,1.2,1.5,2,3,5,10)
GAP_THRESH=(1,1.1,1.2,4/3,1.5,1.75,2,3,5,10)
EV_THRESH=(1,.9,1.1,1.2,1.5,2,3,5,10)

def bin_name(x,bounds):
    for i,z in enumerate(bounds):
        if x<z:return ("below_"+str(z)) if i==0 else (str(bounds[i-1])+"_to_"+str(z))
    return str(bounds[-1])+"_plus"

def stat():
    return dict(tickets=0,hits=0,predicted=0.,market_predicted=0.,
                stake=0,payout=0,v6_payout=0.,market_payout=0.,
                max_win=0,days=set(),race_mass=defaultdict(float))

def update(s,rid,day,p,q,odds,hit,official_paid):
    s["tickets"]+=1
    s["predicted"]+=p
    s["market_predicted"]+=q
    s["hits"]+=int(hit)
    s["stake"]+=100
    s["v6_payout"]+=100*p*odds
    s["market_payout"]+=100*q*odds
    actual=(official_paid if hit else 0)
    s["payout"]+=actual
    s["max_win"]=max(s["max_win"],actual)
    s["days"].add(day)
    s["race_mass"][rid]+=p

def pack(s):
    n=s["tickets"]
    if not n:return None
    var=sum(min(1,max(0,p))*(1-min(1,max(0,p))) for p in s["race_mass"].values())
    return {
      "tickets":n,"races_with_tickets":len(s["race_mass"]),"days":len(s["days"]),
      "predicted_hits":round(s["predicted"],5),
      "market_normalized_predicted_hits":round(s["market_predicted"],5),
      "observed_hits":s["hits"],
      "predicted_hit_rate_pct":round(s["predicted"]/n*100,6),
      "actual_hit_rate_pct":round(s["hits"]/n*100,6),
      "observed_over_v6_predicted":round(s["hits"]/s["predicted"],6) if s["predicted"] else None,
      "observed_over_market_predicted":round(s["hits"]/s["market_predicted"],6) if s["market_predicted"] else None,
      "calibration_z_descriptive":round((s["hits"]-s["predicted"])/math.sqrt(var),5) if var>0 else None,
      "invested_yen":s["stake"],"official_payout_yen":s["payout"],
      "profit_yen":s["payout"]-s["stake"],
      "actual_roi_pct":round(s["payout"]/s["stake"]*100,5),
      "v6_expected_roi_pct":round(s["v6_payout"]/s["stake"]*100,5),
      "market_normalized_expected_roi_pct":round(s["market_payout"]/s["stake"]*100,5),
      "largest_payout_share_pct":round(s["max_win"]/s["payout"]*100,5) if s["payout"] else None,
    }

def collect(rid,day,dist,prices,winner,paid,tickets,stats):
    inv=sum(1/x for x in prices.values())
    if not 1.05<inv<1.8:raise AssertionError(("Unexpected inverse price sum",rid,inv))
    for t,p in dist.items():
        odds=prices[t]
        q=(1/odds)/inv
        rel=p/q
        ev=p*odds
        hit=(t==winner)
        for cohort in ("all_valid_triples", "frozen_formation_components"):
            if cohort=="frozen_formation_components" and (tickets is None or t not in tickets):
                continue
            roots=stats[cohort]
            labels=[
                ("overall","all"),
                ("relative_gap_bins",bin_name(rel,GAP_BOUND)),
                ("net_ev_bins",bin_name(ev,EV_BOUND)),
            ]
            if rel>=1:
                labels += [("relative_gap_cumulative",">="+str(t0)) for t0 in GAP_THRESH if rel>=t0]
            labels += [("net_ev_cumulative",">="+str(t0)) for t0 in EV_THRESH if ev>=t0]
            for typ,lab in labels:
                update(roots[typ][lab],rid,day,p,q,odds,hit,paid)

def main():
    all23,audit=obs.ingest()
    train={rid:e for rid,e in all23.items()
           if e["date"]<="2023-10-31" and weak.score_gate(e)=="ELIGIBLE"}
    assert len(train)==19017,("V6 training reference drift",len(train))
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    model=(wg,wr,d5,wl,wc,pw)
    del train
    jan={rid:{k:v for k,v in e.items() if k!="order"}
         for rid,e in all23.items() if "2023-01-01"<=e["date"]<="2023-01-31"}
    assert len(jan)==1666,("January race count drift",len(jan))
    market={}
    with gzip.open(ROOT/"data/odds_only/odds_2023.jsonl.gz","rt",encoding="utf8") as f:
        for line in f:
            row=json.loads(line)
            rid=str(row.get("race_id",""))
            if rid not in jan:continue
            if rid in market:raise AssertionError(("duplicate market",rid))
            market[rid]=row
    # Stage 1: generate every model distribution and original frozen formation
    # without accessing outcome fields; seal as in-memory immutable records.
    sealed={}
    counts=Counter()
    for rid,e in sorted(jan.items(),key=lambda x:(x[1]["date"],x[0])):
        counts["jan_valid_card"]+=1
        if rid not in market:
            counts["missing_full_odds"]+=1;continue
        prices,why=parse_market(market[rid],sorted(e["scores"],key=int))
        if prices is None:
            counts[why]+=1;continue
        dist=row_probs(e,*model)
        assert set(dist)==set(prices) and abs(sum(dist.values())-1)<1e-8
        chosen,examined=select(e,dist,prices)
        tickets=set(tuple(x.split("-")) for x in chosen["tickets"]) if chosen else None
        sealed[rid]=(e["date"],dist,prices,tickets)
        counts["complete_valid_races"]+=1
        counts["formation_races"]+=int(chosen is not None)
        counts["formation_tickets"]+=len(tickets) if tickets else 0
        if counts["complete_valid_races"]%250==0:
            print("V6_GAP_STAGE1",counts["complete_valid_races"],flush=True)
    assert counts["complete_valid_races"]==1205,dict(counts)
    assert counts["formation_races"]==1202 and counts["formation_tickets"]==9260,dict(counts)
    print("V6_GAP_NO_RESULTS_SELECTION_SEALED",json.dumps(dict(counts)),flush=True)
    # Stage 2: outcomes and official payouts. Exclude any non-verifiable race
    # from BOTH cohorts, rather than substituting the final odds for payout.
    official,conflicts,payout_audit=load_settled_payments(set(sealed))
    stats={cohort:{
        typ:defaultdict(stat) for typ in
        ("overall","relative_gap_bins","net_ev_bins",
         "relative_gap_cumulative","net_ev_cumulative")}
        for cohort in ("all_valid_triples","frozen_formation_components")}
    unsettled=Counter()
    for rid,(day,dist,prices,tickets) in sealed.items():
        order=tuple(all23[rid]["order"][:3])
        if rid in conflicts:
            unsettled["official_payout_conflict"]+=1;continue
        if rid not in official:
            unsettled["official_payout_missing"]+=1;continue
        if tuple(official[rid][0])!=order:
            unsettled["official_winner_disagrees"]+=1;continue
        collect(rid,day,dist,prices,order,official[rid][1],tickets,stats)
        unsettled["verified_races"]+=1
    result={
      "state":"V6_JAN2023_EXACT_PRICE_GAP_TO_OBSERVED_HITS_AND_PROFIT",
      "period":"2023-01-01..2023-01-31",
      "population":"ALL 3ren-tan tickets in fully available archived final 3ren-tan odds books; also compare selected frozen formation components on the same races",
      "frozen_V6_training":"2023 Jan-Oct, 19017 score-valid races; Jan is IN-SAMPLE",
      "market_probability":"q=(1/final_odds)/sum_all_3ren_tan(1/final_odds)",
      "relative_gap_ratio":"p_V6/q (1.0=market price-normalized consensus)",
      "relative_gap_pct":"100*(p_V6/q-1); prices include ~25% deduction",
      "net_single_ticket_ev_ratio":"p_V6*posted_final_odds (1.0=100% predicted net ROI)",
      "no_automatic_double_75pct_deduction":True,
      "flat_stake_each_ticket_yen":100,
      "settlement":"actual official 3ren-tan 100-yen payout for verified winners ONLY",
      "no_ticket_reselection_after_results":True,
      "model_or_existing_formation_rules_changed":False,
      "counts":dict(counts),"settlement":dict(unsettled),"official_payout_source":payout_audit,
      "statistics":{
       cohort:{typ:{lab:pack(s) for lab,s in sorted(k.items())}
               for typ,k in kinds.items()}
       for cohort,kinds in stats.items()},
      "warnings":[
       "IN SAMPLE: Jan 2023 races were used to fit V6; not forward deployable evidence.",
       "Historical FINAL odds collected later, not verified purchasable before bet deadline.",
       "Relative price disagreements and individual EV predictions are NOT guaranteed true profit.",
       "Full-market all-ticket universe buys every combination and is a diagnostic, not a feasible recommended bet.",
       "Report number of trials, predicted vs observed wins and payout concentration. Sparse high odds are volatile.",
       "No inference of an optimal cutoff from this January data; any cutoff needs distinct future evaluation."
      ]
    }
    if not unsettled["verified_races"]:raise AssertionError("no official settled races")
    top=result["statistics"]["frozen_formation_components"]["overall"]["all"]
    if not (top["tickets"]==9260 and top["observed_hits"]==63
            and top["official_payout_yen"]==551660):
        raise AssertionError(("Historical frozen formation reproduction differs",top))
    full=result["statistics"]["all_valid_triples"]["overall"]["all"]
    assert full["observed_hits"]==unsettled["verified_races"]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"price_gap_profit_full.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    digest={"coverage":dict(counts),"settlement":dict(unsettled),
            "overall":{c:result["statistics"][c]["overall"]["all"] for c in result["statistics"]},
            "all_ticket_relative_gap_bins":result["statistics"]["all_valid_triples"]["relative_gap_bins"],
            "all_ticket_predicted_ev_bins":result["statistics"]["all_valid_triples"]["net_ev_bins"],
            "formation_relative_gap_bins":result["statistics"]["frozen_formation_components"]["relative_gap_bins"],
            "formation_net_ev_bins":result["statistics"]["frozen_formation_components"]["net_ev_bins"],
            "formation_gap_cumulative":result["statistics"]["frozen_formation_components"]["relative_gap_cumulative"],
            "all_ticket_gap_cumulative":result["statistics"]["all_valid_triples"]["relative_gap_cumulative"],
            "warnings":result["warnings"]}
    (OUT/"gap_profit_summary.json").write_text(json.dumps(digest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("V6_GAP_TO_PROFIT_COMPLETED",json.dumps(digest,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

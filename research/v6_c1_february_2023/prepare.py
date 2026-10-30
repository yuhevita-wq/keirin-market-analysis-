#!/usr/bin/env python3
"""February 2023: freeze both V6 and V6-C1 full-formation decisions BEFORE settlement.

Original Jan selection thresholds/formation candidates are reused unchanged for
direct comparison, not endorsed as optimal. V6-C1 is the SAVED all-years fit,
which includes February outcomes. All final odds are retrospective, not a
verified pre-close snapshot. Neither model is fit or adjusted here.
"""
from __future__ import annotations
import gzip,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
sys.path.insert(0,str(ROOT/"research/v6_january_2023_formation_ev"))
sys.path.insert(0,str(ROOT/"research/v6_c1_full_data"))
import observe as obs
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import v5_rider_score_role_size_interactions as v5
import v6_opponent_competition_structure as v6
import weak_rider_podium_zero_excluded as weak
from v8_targeted_weak_podium_calibration import row_probs
from prepare import parse_market, select, MIN_POINTS,MAX_POINTS,MIN_FLAT_EV,MIN_COMPOSITE,MIN_PSET,MIN_VALUE_TICKETS,MIN_VALUE_SHARE,MAX_SEEDS
from apply import read_model,apply_calibration

START="2023-02-01"
END="2023-02-28"
ODDS=ROOT/"data/odds_only/odds_2023.jsonl.gz"
DIR=ROOT/"data/audits/v6_c1_formation_feb2023"
VERSIONS=("V6","V6-C1")

def main():
    assert (MIN_POINTS,MAX_POINTS,MIN_FLAT_EV,MIN_COMPOSITE,MIN_PSET,MIN_VALUE_TICKETS,MIN_VALUE_SHARE,MAX_SEEDS)==(4,12,1.2,3.5,.01,2,.35,15), "January selector changed"
    fitted_c1=read_model()
    assert fitted_c1["selected_stage"]=="pattern_pband_role_interactions"
    full,_=obs.ingest()
    train={rid:e for rid,e in full.items()
           if e["date"]<="2023-10-31" and weak.score_gate(e)=="ELIGIBLE"}
    assert len(train)==19017,"Frozen original V6 train population drift"
    wg,wr=v2.fit_structural(train)
    pw,_=v3.train_pattern_gate(train)
    d5=v5.fit_residual(train,wg,wr)
    wl,wc=v6.fit_opponent(train,wg,wr,d5)
    original_model=(wg,wr,d5,wl,wc,pw)
    del train
    # Remove actual result completely from each February inference card.
    feb={rid:{k:v for k,v in e.items() if k!="order"}
         for rid,e in full.items()
         if START<=e["date"]<=END and weak.score_gate(e)=="ELIGIBLE"}
    assert feb,"No eligible February race cards"
    market={}
    with gzip.open(ODDS,"rt",encoding="utf-8") as fh:
        for line in fh:
            row=json.loads(line)
            rid=str(row["race_id"])
            if rid in feb:
                if rid in market:raise AssertionError(("Duplicate archive market",rid))
                market[rid]=row
    DIR.mkdir(parents=True,exist_ok=True)
    counts={name:Counter() for name in VERSIONS}
    by_day={name:Counter() for name in VERSIONS}
    files={name:(DIR/("sealed_"+name.replace("-","_")+"_formations.jsonl")).open("w",encoding="utf-8")
           for name in VERSIONS}
    try:
        for i,(rid,e) in enumerate(sorted(feb.items(),key=lambda x:(x[1]["date"],x[0])),1):
            row=market.get(rid)
            for name in VERSIONS:counts[name]["feb_eligible_v6_card"]+=1
            if row is None:
                for name in VERSIONS:counts[name]["missing_market_book"]+=1
                continue
            prices,why=parse_market(row,sorted(e["scores"],key=int))
            if prices is None:
                for name in VERSIONS:counts[name][why]+=1
                continue
            base=row_probs(e,*original_model)
            assert abs(sum(base.values())-1)<1e-8 and set(base)==set(prices)
            corrected=apply_calibration(e,base,fitted_c1)
            assert abs(sum(corrected.values())-1)<1e-9 and set(corrected)==set(base)
            for name,dist in (("V6",base),("V6-C1",corrected)):
                z=counts[name];z["complete_final_odds_and_v6"]+=1
                bet,_=select(e,dist,prices)
                if bet is None:
                    z["no_formation_passed_frozen_rules"]+=1
                    continue
                assert 4<=bet["points"]<=12
                bet.update({
                   "model_name":name,"race_id":rid,"date":e["date"],
                   "race_type":row.get("race_type"),"odds_phase":"historical_final_NOT_preclose",
                   "training_overlap":True,
                   "no_result_used_in_selection":True,
                   "p_field_note":"v6_p and v6_flat_ev hold SELECTED MODEL values; for V6-C1 they are calibrated probabilities"
                })
                files[name].write(json.dumps(bet,ensure_ascii=False,separators=(",",":"))+"\n")
                z["purchase_races"]+=1
                z["purchase_points"]+=bet["points"]
                z["contains_below_100_ev_component"]+=int(bet["components_expected_below_100pct"]>0)
                by_day[name][e["date"]]+=1
            if i%250==0:print("FEB2023_PRE_RESULT_PROGRESS",i,{k:dict(v) for k,v in counts.items()},flush=True)
    finally:
        for fp in files.values():fp.close()
    metadata={
       "state":"V6_C1_FEB2023_BOTH_MODELS_PRE_RESULT_SEALED",
       "period":[START,END],
       "model_version_used":{"V6":"unchanged 2023 Jan-Oct original refit",
                             "V6-C1":"saved 84065-race full-2023-2026 fitted model"},
       "historical_training_overlap":True,
       "archived_final_odds_not_preclose":True,
       "no_result_join_before_seal":True,
       "original_jan_selector_unmodified":True,
       "original_jan_4_to_12_points_arbitrary_not_optimal":True,
       "flat_stake_per_ticket_yen":100,
       "settings":{"min_points":MIN_POINTS,"max_points":MAX_POINTS,
                   "min_flat_ev":MIN_FLAT_EV,"min_synthetic":MIN_COMPOSITE,
                   "min_formation_probability":MIN_PSET,
                   "min_positive_tickets":MIN_VALUE_TICKETS,
                   "min_positive_fraction":MIN_VALUE_SHARE,"max_seeds":MAX_SEEDS},
       "coverage":{name:dict(z) for name,z in counts.items()},
       "purchase_races_by_day":{name:dict(z) for name,z in by_day.items()},
       "sha256":{name:hashlib.sha256((DIR/("sealed_"+name.replace("-","_")+"_formations.jsonl")).read_bytes()).hexdigest()
                 for name in VERSIONS}
    }
    (DIR/"sealed_decision_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FEB2023_BOTH_MODELS_PRE_RESULT_SEALED",json.dumps({"coverage":metadata["coverage"],"sha256":metadata["sha256"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()

#!/usr/bin/env python3
"""Diagnostics on sealed V6 January 2023 formations, no reselection."""
from __future__ import annotations
import json,math,statistics
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"data/audits/v6_formation_ev_2023_01"
def accum():
    return {"races":0,"hits":0,"sum_p_set":0.,"stakes":0.,"actual_payout":0.,
            "v6_expected_payout":0.,"points":0,"positive":0,"negative":0}
def add(s,x):
    s["races"]+=1;s["hits"]+=int(x["hit"])
    s["sum_p_set"]+=x["v6_formation_probability_pct"]/100
    s["stakes"]+=x["stake_yen"]
    s["actual_payout"]+=x["verified_payout_yen"] or 0
    s["v6_expected_payout"]+=x["stake_yen"]*x["v6_flat_stake_expected_roi_pct"]/100
    s["points"]+=x["points"]
    s["positive"]+=x["components_expected_over_100pct"]
    s["negative"]+=x["components_expected_below_100pct"]
def emit(s):
    o=dict(s)
    o["predicted_hits"]=round(o.pop("sum_p_set"),5)
    o["observed_over_predicted_hits"]=round(o["hits"]/o["predicted_hits"],5) if o["predicted_hits"] else None
    o["mean_predicted_hit_pct"]=round(100*o["predicted_hits"]/o["races"],5) if o["races"] else None
    o["observed_hit_pct"]=round(100*o["hits"]/o["races"],5) if o["races"] else None
    o["realized_roi_pct"]=round(100*o["actual_payout"]/o["stakes"],5) if o["stakes"] else None
    o["v6_expected_roi_pct"]=round(100*o["v6_expected_payout"]/o["stakes"],5) if o["stakes"] else None
    o["mean_points"]=round(o["points"]/o["races"],4) if o["races"] else None
    o["v6_expected_payout"]=round(o["v6_expected_payout"],2)
    return o
def binp(p):
    if p<5:return "<5pct"
    if p<10:return "5to10pct"
    if p<20:return "10to20pct"
    if p<40:return "20to40pct"
    return "40pct+"
def binev(p):
    if p<150:return "120to150pct"
    if p<200:return "150to200pct"
    if p<300:return "200to300pct"
    if p<500:return "300to500pct"
    return "500pct+"
def main():
    d=json.loads((BASE/"jan2023_formation_roi_report.json").read_text())
    rows=[json.loads(z) for z in (BASE/"settled_formation_purchases.jsonl").read_text().splitlines()]
    assert all(z["settlement_status"].startswith("verified_") for z in rows), "Cannot compute official payout ROI with unresolved settlement"
    bucket=defaultdict(accum); ev=defaultdict(accum); shape=defaultdict(accum)
    days=defaultdict(accum);allstat=accum();paid=[]
    for z in rows:
        add(allstat,z);add(bucket[binp(z["v6_formation_probability_pct"])],z)
        add(ev[binev(z["v6_flat_stake_expected_roi_pct"])],z)
        add(shape[z["shape"]],z);add(days[z["date"]],z)
        if z["hit"]:
            paid.append({"date":z["date"],"race_id":z["race_id"],"formation":z["formation"],
                         "payout":z["verified_payout_yen"],"stake":z["stake_yen"],
                         "v6_expected_roi_pct":z["v6_flat_stake_expected_roi_pct"],
                         "v6_expected_win_pct":z["v6_formation_probability_pct"]})
    assert allstat["races"]==d["actual_verified_official_payouts"]["races"]
    assert allstat["actual_payout"]==d["actual_verified_official_payouts"]["payout_yen"]
    obs=allstat["hits"];ex=allstat["sum_p_set"]
    var=sum(p*(1-p) for p in (z["v6_formation_probability_pct"]/100 for z in rows))
    diagnosed={
      "state":"JAN2023_SEALED_V6_FORMATIONS_POSTHOC_CALIBRATION_DIAGNOSED",
      "previous_decision_sha256":d["decision_hash_sha256"],
      "no_change_to_earlier_selections":True,
      "total":emit(allstat),
      "overall_hit_calibration_z_descriptive":round((obs-ex)/math.sqrt(var),5) if var>0 else None,
      "by_v6_predicted_formation_hit_probability":{k:emit(x) for k,x in sorted(bucket.items())},
      "by_v6_expected_flat_roi_band":{k:emit(x) for k,x in sorted(ev.items())},
      "by_formation_shape":{k:emit(x) for k,x in sorted(shape.items())},
      "by_day":{k:emit(x) for k,x in sorted(days.items())},
      "winning_formations_sorted_by_paid":sorted(paid,key=lambda x:-x["payout"]),
      "winning_payout_top_5_share":round(sum(z["payout"] for z in sorted(paid,key=lambda x:-x["payout"])[:5])/max(1,allstat["actual_payout"]),6),
      "historical_final_odds_vs_official":d["official_payout_to_final_odds_ratio"],
      "coverage":d["coverage"],
      "warnings":["Jan 2023 fitted into V6, thus 2023 Jan is in-sample.",
                  "All odds are final archival, not pre-deadline tradable snapshots.",
                  "Many candidates per race inspected; selected-formations have selection-induced calibration distortions.",
                  "The selection was frozen before the outcome reveal and has NOT been tuned after observing its performance."],
    }
    output=BASE/"jan2023_why_formation_ev_failed.json"
    output.write_text(json.dumps(diagnosed,ensure_ascii=False,indent=2)+"\n")
    print("JAN2023_FORMATION_EV_DIAGNOSIS",json.dumps({
      "overall":diagnosed["total"],
      "overall_calibration_z":diagnosed["overall_hit_calibration_z_descriptive"],
      "by_probability":diagnosed["by_v6_predicted_formation_hit_probability"],
      "by_expected_ev":diagnosed["by_v6_expected_flat_roi_band"],
      "by_shape":diagnosed["by_formation_shape"],
      "top_winners":diagnosed["winning_formations_sorted_by_paid"][:8],
      "win_payout_top5_share":diagnosed["winning_payout_top_5_share"],
      "price_to_official_ratio":diagnosed["historical_final_odds_vs_official"]
    },ensure_ascii=False),flush=True)
if __name__=="__main__":main()

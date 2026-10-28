#!/usr/bin/env python3
"""Audit counterfactual actual-car Cartesian formations in the *same* three random
races, without rewriting the original 0-yen SKIP decisions.

Select highest estimated full-basket ROI 2-,3-,4-,6-point REAL rectangles from
all actual rider slots. Calibrate on 2021-24 only; use neither 2025-26 result
for candidate selection nor for ticket/price projections. Label all proposed
amounts as non-purchased ZERO under the validated conservative gate.
"""
from __future__ import annotations
import itertools,json
from datetime import datetime,timezone
from pathlib import Path
from research.three_random_joint_formation_allocation_20261029 import evaluate as prev
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research/three_random_joint_formation_allocation_20261029"
COUNTS=(2,3,4,6)
def discovery(m,probs):
    n=m["n"];cars=m["cars"]
    edge={t:probs[t]*m["prices"][t] for t in m["prices"]}
    best={}
    # Build real 3-car distinct tickets. No market rank subset templates,
    # no fixed line patterns, no cherry-picked results.
    masks=[list(itertools.combinations(cars,k)) for k in range(1,min(4,n)+1)]
    seen=set()
    for ia in range(1,5):
        for ib in range(1,5):
            for ic in range(1,5):
                if ia*ib*ic>12:continue
                for slots in itertools.product(masks[ia-1],masks[ib-1],masks[ic-1]):
                    tickets=tuple(sorted(t for t in itertools.product(*slots) if len(set(t))==3))
                    N=len(tickets)
                    if N not in COUNTS or tickets in seen:continue
                    seen.add(tickets)
                    # For a fixed wager of 100 yen on each ticket, expected
                    # payout/stake = mean(p_i O_i). This is NOT the optimized
                    # live decision; actual optimum with all pO<1 is zero.
                    avg=sum(edge[t] for t in tickets)/N
                    key=(avg,-len(slots[0])-len(slots[1])-len(slots[2]),slots)
                    if N not in best or key>best[N][0]:
                        best[N]=(key,slots,tickets)
    answer=[]
    for N in COUNTS:
        if N not in best:continue
        (_,_,_),slots,tickets=best[N]
        answer.append({"points":N,"formation":"-".join("".join(map(str,p)) for p in slots),
                       "1着":slots[0],"2着":slots[1],"3着":slots[2],
                       "equal_stake_estimated_roi_pct":round(100*sum(edge[t] for t in tickets)/N,3),
                       "Kelly_optimal_actual_stake_yen":0,
                       "tickets":[{"combination":"-".join(map(str,t)),
                                   "odds":m["prices"][t],
                                   "calibrated_probability":round(probs[t],8),
                                   "estimated_per_ticket_roi_pct":round(100*edge[t],3),
                                   "actual_purchase_yen":0}
                                  for t in tickets]})
    return answer,len(seen),max(edge.values())

def main():
    labels=prev.load_labels()
    a,e,f,three,tr_n,test_n=prev.train_and_select(labels)
    previous=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    if [rid for y,rid,w,m in three]!=[x["race_id"] for x in previous["races"]]:
        raise AssertionError("Different sampled races; unsafe to compare")
    samples=[]
    for yr,rid,finish,m in three:
        probs=prev.calibrated_probs(m,f)
        candidates,count,maxedge=discovery(m,probs)
        samples.append({"race_id":rid,"year":yr,"n":m["n"],
          "favorite":"-".join(map(str,m["favorite"])),
          "settlement_actual_result":"-".join(map(str,finish)),
          "real_formations_considered":count,
          "max_estimated_gross_roi_single_combination_pct":round(100*maxedge,4),
          "decision_unchanged":"SKIP",
          "stake_yen":0,"formation_examples_counterfactual_only":candidates})
    report={"status":"POSTHOC_GEOMETRY_REPORT_NO_DECISION_CHANGES",
      "generated_utc":datetime.now(timezone.utc).isoformat(),
      "original_random_pool":test_n,"train_races":tr_n,
      "origin":"research/three_random_joint_formation_allocation_20261029/result.json",
      "caution":"All figures are candidate inspections only. 0 yen is optimal under strict nonpositive model EV; no results used to select formations.",
      "races":samples}
    (OUT/"geometry_diagnostic.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("THREE_SAME_RACES_GEOMETRY",json.dumps({
        "races":[{"rid":s["race_id"],"top_single_ev":s["max_estimated_gross_roi_single_combination_pct"],
                  "forms":[{"points":z["points"],"formation":z["formation"],
                  "basket_roi":z["equal_stake_estimated_roi_pct"],
                  "prices":[(t["combination"],t["odds"],t["estimated_per_ticket_roi_pct"])
                  for t in z["tickets"]]} for z in s["formation_examples_counterfactual_only"]]}
                  for s in samples]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

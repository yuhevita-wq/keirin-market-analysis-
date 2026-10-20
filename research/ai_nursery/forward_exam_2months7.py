"""Frozen JULY out-of-sample comparison: GEN-007 before vs after 61 training days.

This is a separate exam AFTER all continuation training dates. Neither
checkpoint nor behavior policy is updated; identical race IDs and RNG seeds.
Never use July holdout for adjusting the learning rule or model selection.
"""
from __future__ import annotations
import hashlib,json
from collections import Counter
from datetime import date
import numpy as np

from train_week import ROOT,WEB,D,load_cards,load_results,load_odds,eligible
from growth_audit import score_one,aggregate,paired_ci
from behavior_policy import BehaviorPolicy
START="2023-07-01"
END="2023-07-07"
OUT=ROOT/"research/ai_nursery/outputs/GEN-007/continuation-2m"


def main():
    parent_dir=ROOT/"research/ai_nursery/outputs/GEN-007"
    parents=json.loads((parent_dir/"checkpoint.json").read_text(encoding="utf-8"))
    stage=json.loads((OUT/"checkpoint.json").read_text(encoding="utf-8"))
    prior_policy=BehaviorPolicy.load(parent_dir/"behavior_policy.json")
    final_policy=BehaviorPolicy.load(OUT/"behavior_policy.json")
    if stage["parent_weights_sha256"]!=parents["weights_sha256"]:
        raise RuntimeError("Incorrect prediction lineage")
    if stage["parent_behavior_policy_sha256"]!=prior_policy.digest():
        raise RuntimeError("Incorrect policy lineage")
    if stage["training_span"][-1]>=START:
        raise RuntimeError("Future evaluation not fully out of training window")
    weights=[]
    for cp in (parents,stage):
        w=np.asarray(cp["weights"],dtype=np.float64)
        if w.shape!=(D,) or hashlib.sha256(w.tobytes()).hexdigest()!=cp["weights_sha256"]:
            raise RuntimeError("Corrupt frozen weights")
        weights.append(w)

    cards,_=load_cards()
    calendar={rid:o for rid,o in cards.items()
              if START<=o["race"].get("race_date","")<=END}
    results=load_results()
    odds=load_odds(set(calendar).intersection(results))
    missing=Counter()
    races=[]
    for rid,obj in calendar.items():
        if rid not in results:missing["missing_result"]+=1;continue
        if rid not in odds:missing["missing_complete_odds"]+=1;continue
        item={**obj,"odds":odds[rid],"outcome":results[rid],"race_id":rid}
        if not eligible(item):missing["incomplete_or_inconsistent"]+=1;continue
        races.append(item)
    races.sort(key=lambda a:(
        a["race"]["race_date"],a["race"].get("start_time",""),
        a["race"].get("track",""),int(a["race"].get("race_no","0") or 0),a["race_id"]))
    if not races:
        raise RuntimeError("No July 2023 complete odds, results, racecards; cannot claim future test")
    groups={}
    result_rows=[]
    for label,w,policy in (("GEN-007-before",weights[0],prior_policy),
                          ("GEN-007-after-2months",weights[1],final_policy)):
        bank=10000
        rows=[]
        for r in races:
            row=score_one(r,"GEN-007",w,bank,policy)
            rows.append(row)
            bank=row["wallet_after_yen"]
        groups[label]=rows
        result_rows.append({"label":label,"metrics":aggregate(rows),
                            "final_paper_wallet_yen":bank})
    old=groups["GEN-007-before"]
    new=groups["GEN-007-after-2months"]
    report={
        "id":"GEN-007-C2-JULY-FROZEN-EXAM",
        "status":"COMPLETE",
        "holdout_period":[START,END],
        "training_data_ends":"2023-06-11",
        "pretrained_parent_sha256":parents["weights_sha256"],
        "post_continuation_sha256":stage["weights_sha256"],
        "same_races_no_learning":True,
        "only_final_odds_public_market":True,
        "total_racecards":len(calendar),
        "tested_races":len(races),
        "excluded":dict(missing),
        "race_id_sha256":hashlib.sha256("\n".join(r["race_id"] for r in races).encode()).hexdigest(),
        "results":result_rows,
        "paired_differences_after_minus_before":{
            k:paired_ci(new,old,k)
            for k in ("logloss_tan","logloss_fuku","net_proxy_yen","marks","sheets","skipped")
        },
        "note":"The July holdout is later than both models training data; this is paper-only with final odds, not a live-betting guarantee. No hyperparameter changes should be made using July results."
    }
    (OUT/"forward_july_exam.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (WEB/"nursery-gen007-phase2-forward-exam.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUT/"forward_july_exam_per_race.jsonl").open("w",encoding="utf-8") as f:
        for name,rows in groups.items():
            for r in rows:f.write(json.dumps({"model":name,**r},ensure_ascii=False,separators=(",",":"))+"\n")
    print("GEN007_FORWARD_JULY_EXAM_COMPLETE",len(races),report["results"])


if __name__=="__main__":
    main()

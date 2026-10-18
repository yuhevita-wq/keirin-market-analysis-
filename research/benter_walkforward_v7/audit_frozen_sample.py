#!/usr/bin/env python3
"""Read results ONLY AFTER independent forecast CSVs have been frozen in Git."""
from __future__ import annotations
import csv,json,hashlib,sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from freeze_prior_win import PERIODS,sources,rows

MIN_RACES=3198
MIN_SECOND=29679
MIN_THIRD=26481

def main():
    frozen=HERE/"frozen"
    forecast={}
    manifests={}
    for period in PERIODS:
        pred=frozen/f"independent_win_{period}.csv"
        manifest=frozen/f"manifest_{period}.json"
        if not pred.exists() or not manifest.exists():
            raise RuntimeError(f"Missing not-yet-frozen period: {period}")
        meta=json.loads(manifest.read_text(encoding="utf-8"))
        assert meta["prediction_sha256"]==hashlib.sha256(pred.read_bytes()).hexdigest()
        manifests[period]=meta
        for r in rows(pred):
            rid=r["race_id"]
            d=forecast.setdefault(rid,{"period":period,"date":r["race_date"],"cars":set(),"sum_prob":0.0})
            if d["period"]!=period or d["date"]!=r["race_date"]:
                raise RuntimeError(f"Overlapping frozen targets: {rid}")
            car=int(r["car_no"])
            if car in d["cars"]:raise RuntimeError(f"Repeated car in forecast {rid}")
            d["cars"].add(car)
            d["sum_prob"]+=float(r["independent_win_probability"])
    for rid,d in forecast.items():
        if abs(d["sum_prob"]-1.0)>1e-8:
            raise RuntimeError(f"Probabilities not normalized: {rid}")
    # Results are post-forecast labels, NOT predictors. Keep unknown outcomes missing.
    outcomes=defaultdict(lambda:defaultdict(list))
    for folder in sources():
        p=folder/"results.csv"
        if not p.exists():continue
        for r in rows(p):
            rid=r.get("race_id","")
            if rid in forecast:
                place=str(r.get("finish_position","")).strip()
                if place in ("1","2","3"):
                    outcomes[rid][place].append(r.get("car_no"))
    groups={"fit":{"races":0,"second_candidates":0,"third_candidates":0,"field_sizes":Counter(),"ids":[]},
            "chronological_check":{"races":0,"second_candidates":0,"third_candidates":0,"field_sizes":Counter(),"ids":[]}}
    skipped=Counter()
    for rid,info in forecast.items():
        rec=outcomes.get(rid,{})
        try:
            podium=[int(rec[str(k)][0]) for k in (1,2,3) if len(rec.get(str(k),[]))==1]
        except ValueError:
            podium=[]
        if len(podium)!=3 or len(set(podium))!=3 or not set(podium).issubset(info["cars"]):
            skipped["missing_or_ambiguous_podium"]+=1
            continue
        # PRE-DECLARED split boundary: exact chronological split, not chosen by
        # inspecting the winners or performance.
        split="fit" if info["date"]<"2025-04-01" else "chronological_check"
        group=groups[split]
        n=len(info["cars"])
        group["races"]+=1
        group["second_candidates"]+=n-1
        group["third_candidates"]+=n-2
        group["field_sizes"][n]+=1
        group["ids"].append(rid)
    report={
        "scope":"F1 S級 only (as enforced in forecast build)",
        "status":"FROZEN_FORECAST_PODIUM_AUDIT",
        "date_split":"fit: through 2025-03-31; chronological_check: from 2025-04-01",
        "disclaimer":"Chronological evaluation is not globally virgin: earlier studies inspected parts of these dates",
        "data_limitation":"Retroactively collected date-specific racecards, not fully certified original real-time snapshots",
        "V4_calibration_applied":False,
        "V7_coefficients_unchanged":True,
        "minima":{"races":MIN_RACES,"second_candidate_rows":MIN_SECOND,"third_candidate_rows":MIN_THIRD},
        "forecast_races":len(forecast),
        "unusable_podium_count":dict(skipped),
        "periods":{k:{"forecast_races":m["target_races"],"training_races":m["training_races"],
                      "max_training_date":m["train_max_date"],"forecast_sha256":m["prediction_sha256"]}
                   for k,m in manifests.items()},
        "partition":{},
    }
    for split,item in groups.items():
        size=dict(item["field_sizes"])
        audit={
            "eligible_races":item["races"],
            "second_candidate_rows":item["second_candidates"],
            "third_candidate_rows":item["third_candidates"],
            "field_size_distribution":size,
            "race_ids_sha256":hashlib.sha256(("\n".join(sorted(item["ids"]))+"\n").encode()).hexdigest(),
        }
        audit["passes_benter_sample_gate"]=(audit["eligible_races"]>=MIN_RACES and
                                          audit["second_candidate_rows"]>=MIN_SECOND and
                                          audit["third_candidate_rows"]>=MIN_THIRD)
        report["partition"][split]=audit
    report["both_partitions_pass"]=all(x["passes_benter_sample_gate"] for x in report["partition"].values())
    output=HERE/"frozen_sample_audit.json"
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="periods"},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()

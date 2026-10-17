#!/usr/bin/env python3
"""Out-of-time Sept 2026 score: frozen July-August calibrated V2 vs raw V2.

Never fits/calibrates or modifies predictions here. This script may ONLY run
after forecast commit has been pushed to GitHub.
"""
from __future__ import annotations
import csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).parent
PRED=HERE/"sep_predictions_frozen.csv"
TRUTH=HERE/"sep_holdout/results.csv"
CONTROL=HERE/"calibrator_frozen.json"
OUT=HERE/"holdout_report.json"
RACE=HERE/"holdout_by_race.csv"
BIN_EDGES=[0,.025,.05,.1,.15,.2,.25,.3,.4,.5,.6,1.000001]

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def metric(rows, field):
    n=len(rows)
    vals=[r[field] for r in rows]
    null_loss=sum(math.log(r["n"]) for r in rows)/n
    loss=sum(-math.log(max(v,1e-300)) for v in vals)/n
    return {"races":n,"log_loss":loss,"pseudo_R2_benter_1994":1-loss/null_loss,
            "brier":sum(r[field+"_brier"] for r in rows)/n}
def bucket(rows,field,selector):
    selected=[]
    for r in rows:
        for p,w in r["riders"]:
            if selector(p):
                selected.append((p[field],w))
    if not selected:return {"riders":0}
    return {
        "riders":len(selected),"predicted":float(np.mean([p for p,y in selected])),
        "actual":float(np.mean([y for p,y in selected])),
        "prediction_minus_actual":float(np.mean([p-y for p,y in selected])),
        "predicted_winners":float(sum(p for p,y in selected)),
        "actual_winners":int(sum(y for p,y in selected))}
def main():
    manifest=json.loads(CONTROL.read_text(encoding="utf-8"))
    assert manifest["frozen_independent_eval_window"]=="2026-09-01 to 2026-09-30"
    assert manifest["target_period_results_opened_before_forecasting"] is False
    predictions=defaultdict(list)
    for p in read(PRED):
        assert "2026-09-01"<=p["race_date"]<="2026-09-30"
        predictions[p["race_id"]].append(p)
    actual=defaultdict(list)
    for r in read(TRUTH):
        if "2026-09-01"<=r.get("race_date","")<="2026-09-30":
            actual[r["race_id"]].append(r)
    records=[]
    excluded=[]
    for rid,rs in predictions.items():
        winners=[r["car_no"] for r in actual.get(rid,[]) if r.get("finish_position","").strip()=="1"]
        cars={r["car_no"] for r in rs}
        if len(winners)!=1 or winners[0] not in cars:
            excluded.append({"race_id":rid,"reason":"Missing or ambiguous winner","winner_count":len(winners)})
            continue
        w=winners[0]
        pa=np.array([float(r["original_probability"]) for r in rs])
        pb=np.array([float(r["calibrated_probability"]) for r in rs])
        ys=np.array([int(r["car_no"]==w) for r in rs])
        assert abs(pa.sum()-1)<1e-8 and abs(pb.sum()-1)<1e-8
        assert np.argmax(pa)==np.argmax(pb)
        record={
            "race_id":rid,"date":rs[0]["race_date"],"track":rs[0]["track"],
            "race_no":int(rs[0]["race_no"]),"actual_winner":w,
            "n":len(rs),"original_probability":float(pa[ys==1][0]),
            "calibrated_probability":float(pb[ys==1][0]),
            "original_probability_brier":float(np.sum((pa-ys)**2)),
            "calibrated_probability_brier":float(np.sum((pb-ys)**2)),
            "top1_car":rs[int(np.argmax(pa))]["car_no"],
            "top1_hit":int(rs[int(np.argmax(pa))]["car_no"]==w),
            "riders":[({"original_probability":float(a),"calibrated_probability":float(b)},int(y))
                      for a,b,y in zip(pa,pb,ys)],
        }
        records.append(record)
    if len(records)<260:
        raise RuntimeError(f"Not enough scored Sept races: {len(records)}")
    orig=metric(records,"original_probability")
    corrected=metric(records,"calibrated_probability")
    high_selector=lambda p:p["original_probability"]>=.4
    original_high=bucket(records,"original_probability",high_selector)
    calibrated_high=bucket(records,"calibrated_probability",high_selector)
    binned=[]
    for lo,hi in zip(BIN_EDGES[:-1],BIN_EDGES[1:]):
        fun=lambda p,lo=lo,hi=hi:lo<=p["original_probability"]<hi
        o=bucket(records,"original_probability",fun)
        c=bucket(records,"calibrated_probability",fun)
        if o["riders"]==0:continue
        binned.append({"original_probability_bin":[lo,hi],"n_riders":o["riders"],
                       "before":o["predicted"],"after":c["predicted"],"actual":o["actual"]})
    # Each replicate samples entire race dates (not individual riders).
    groups=defaultdict(list)
    for r in records:groups[r["date"]].append(r)
    days=list(groups.values())
    rng=np.random.default_rng(20261018)
    boot=[]
    for _ in range(3000):
        rows=[r for j in rng.integers(len(days),size=len(days)) for r in days[j]]
        ll=float(np.mean([math.log(max(r["calibrated_probability"],1e-300))-
                          math.log(max(r["original_probability"],1e-300)) for r in rows]))
        s_high=[(p,y) for r in rows for p,y in r["riders"]
                if p["original_probability"]>=.4]
        high_bias_orig=np.mean([p["original_probability"]-y for p,y in s_high]) if s_high else np.nan
        high_bias_cal=np.mean([p["calibrated_probability"]-y for p,y in s_high]) if s_high else np.nan
        boot.append([ll,high_bias_orig,high_bias_cal])
    boot=np.array(boot)
    report={
        "status":"SEPARATE_OUT_OF_TIME_CALIBRATION_TEST",
        "window":"2026-09-01 through 2026-09-30",
        "calibration_training_window":"2026-07-02 to 2026-08-30 only",
        "calibrator":manifest["calibrator_selection"],
        "forecast_was_committed_before_target_outcomes_loaded":True,
        "predicted_races":len(predictions),"scored_races":len(records),
        "unscorable":excluded,
        "race_dates":len(groups),
        "top1_hits_same_for_both":sum(r["top1_hit"] for r in records),
        "top1_accuracy_same_for_both":sum(r["top1_hit"] for r in records)/len(records),
        "before":orig,"after":corrected,
        "comparison":{"logloss_improvement_positive_is_good":orig["log_loss"]-corrected["log_loss"],
                      "brier_improvement_positive_is_good":orig["brier"]-corrected["brier"],
                      "bootstrap_by_date_95pct_logloss_improvement":np.quantile(boot[:,0],[.025,.975]).tolist()},
        "high_40pct_bucket_frozen_from_original_predictions":{
            "before":original_high,"after":calibrated_high,
            "before_abs_calibration_error":abs(original_high["prediction_minus_actual"]),
            "after_abs_calibration_error":abs(calibrated_high["prediction_minus_actual"]),
            "bootstrap_95pct_original_predicted_minus_actual":np.quantile(boot[:,1],[.025,.975]).tolist(),
            "bootstrap_95pct_corrected_predicted_minus_actual":np.quantile(boot[:,2],[.025,.975]).tolist(),
        },
        "bins":binned,
        "caveat":"Historical per-race rider cards are retrospectively archived, not contemporaneous live forecasts.",
        "other_caveat":"Strong Sept performance validates temporal transfer of a calibration fixed on Jul-Aug; no betting profitability or full-race generality claim.",
    }
    with RACE.open("w",encoding="utf-8",newline="") as f:
        fields=["race_id","date","track","race_no","actual_winner","n","original_probability",
                "calibrated_probability","original_probability_brier","calibrated_probability_brier",
                "top1_car","top1_hit"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for rec in records:w.writerow({k:rec[k] for k in fields})
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SEPT_OOT",json.dumps({k:report[k] for k in ("scored_races","top1_accuracy_same_for_both",
           "before","after","comparison","high_40pct_bucket_frozen_from_original_predictions")},
           ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()

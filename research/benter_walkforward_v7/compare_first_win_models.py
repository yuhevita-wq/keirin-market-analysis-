#!/usr/bin/env python3
"""Same-race F1 S-class comparison: original V2 vs term999-categorical V2.
Term999 candidate proposed after old evaluation analysis -> no blind test claim.
"""
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE/"first_win_research"
NEW=HERE/"first_win_research_term_category"
OUT=HERE/"first_win_category_comparison.json"

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return {r["race_id"]:r for r in csv.DictReader(f)}

def bootstrap(group,seed=20261018,reps=2500):
    dates=sorted(group)
    s=np.array([sum(group[d]) for d in dates])
    n=np.array([len(group[d]) for d in dates])
    gen=np.random.default_rng(seed)
    inds=gen.integers(len(s),size=(reps,len(s)))
    z=s[inds].sum(axis=1)/n[inds].sum(axis=1)
    return {"gain_nll_positive_is_new_better":float(s.sum()/n.sum()),
            "day_cluster_95":[float(x) for x in np.quantile(z,[.025,.975])],
            "n_days":len(dates)}

def main():
    a=rows(OLD/"heldout_first_by_race.csv")
    b=rows(NEW/"heldout_first_by_race.csv")
    if len(a)!=7688 or set(a)!=set(b):
        raise RuntimeError("New model must be evaluated on identical F1 S-class race IDs")
    grouped=defaultdict(list)
    for rid in sorted(a):
        x,y=a[rid],b[rid]
        for name in ("date","period","winner","n"):
            if x[name]!=y[name]:raise RuntimeError(f"Mismatch {rid}: {name}")
        grouped[y["period"]].append((x,y))
    result={}
    for stage in ("raw","selected"):
        s=defaultdict(list)
        perperiod={}
        for period,obs in grouped.items():
            improvements=[]
            for x,y in obs:
                old=float(x[f"{stage}_log_loss"])
                new=float(y[f"{stage}_log_loss"])
                improvements.append(old-new)
                s[x["date"]].append(old-new)
            perperiod[period]={"races":len(obs),"nll_gain_positive_is_new_better":float(np.mean(improvements))}
        result[stage]={"paired":bootstrap(s),"per_period":dict(sorted(perperiod.items()))}
    orig=json.loads((OLD/"first_win_report.json").read_text(encoding="utf-8"))
    cand=json.loads((NEW/"first_win_report.json").read_text(encoding="utf-8"))
    manifest={
        "scope":"F1 S級 only",
        "status":"PAIRWISE_IDENTICAL_RACE_V2_VS_FOREIGN_TERM_CATEGORICAL_CANDIDATE",
        "same_scored_races":len(a),
        "race_ids_sha256":hashlib.sha256(("\n".join(sorted(a))+"\n").encode()).hexdigest(),
        "all_generated_race_count":14081,
        "old_model":"V2 term entered continuously including special 999",
        "new_model":"V2 term 999 treated as special foreign category indicator; normal term numeric only",
        "selection_history_warning":"New model family motivated by old evaluation errors, so the same evaluation range is no longer globally untouched",
        "no_odds_used":True,
        "V7_frozen_parameters_modified":False,
        "raw_vs_raw":result["raw"],
        "development_selected_calibration_vs_development_selected_calibration":result["selected"],
        "old_calibration_selected":orig["selected_method"],
        "new_calibration_selected":cand["selected_method"],
        "old_first_evaluation":{k:orig["selected_evaluation"][k] for k in ("races","mean_log_loss","mean_brier","top1_accuracy","benter_pseudo_r2_vs_uniform")},
        "new_first_evaluation":{k:cand["selected_evaluation"][k] for k in ("races","mean_log_loss","mean_brier","top1_accuracy","benter_pseudo_r2_vs_uniform")},
        "deferred":"No 2nd / 3rd model changes or market odds integration."
    }
    OUT.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FIRST_MODEL_COMPARISON",json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=="__main__":main()

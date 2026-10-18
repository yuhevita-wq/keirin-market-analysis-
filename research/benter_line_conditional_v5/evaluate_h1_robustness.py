#!/usr/bin/env python3
"""Additional chronological robustness check: H1 2026 F1 S class.

Uses FROZEN V2 win probabilities from 2026-H1 (prior V2 research).
V5 conditional line coefficients were fitted only on Jul-Aug 2026 and
selected using July->August. They are not adjusted using H1 outcomes.
Thus this is an external-order-condition coefficient transfer check
but NOT an untouched holdout for the underlying fundamental win model.
"""
from __future__ import annotations
import csv,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from model_line_conditional import cprob,parse_line,calibrate
RACES=ROOT/"data/2026_h1/s_class_f1_all/races.csv"
TRUTH=ROOT/"data/2026_h1/s_class_f1_all/results.csv"
FROZEN=ROOT/"research/benter_structural_discovery_v2/output/test_full_plus_line_win_probability.csv"
PARAM=HERE/"parameters_pretest.json"
OUT=HERE/"h1_order_robustness.json"

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def main():
    config=json.loads(PARAM.read_text(encoding="utf-8"))
    line2=np.asarray(config["parameters"]["second"]["theta"],dtype=float)
    line3=np.asarray(config["parameters"]["third"]["theta"],dtype=float)
    base2=np.asarray([config["baseline_without_line_frozen"]["gamma"]]+[0.]*6)
    base3=np.asarray([config["baseline_without_line_frozen"]["delta"]]+[0.]*9)
    card={r["race_id"]:r for r in read(RACES)}
    outcomes=defaultdict(lambda:defaultdict(list))
    for r in read(TRUTH):
        try:i=int(r["finish_position"]);c=int(r["car_no"])
        except(ValueError,KeyError):continue
        if 1<=i<=3:outcomes[r["race_id"]][i].append(c)
    groups=defaultdict(list)
    for row in read(FROZEN):groups[row["race_id"]].append(row)
    records=[];skips=Counter()
    for rid,rs in groups.items():
        if rid not in card:skips["no_card"]+=1;continue
        outcome=outcomes.get(rid,{})
        if any(len(outcome.get(j,[]))!=1 for j in (1,2,3)):
            skips["ambiguous_or_missing_1_2_3"]+=1;continue
        rs.sort(key=lambda r:int(r["car_no"]))
        cars=[int(v["car_no"]) for v in rs]
        lines=parse_line(card[rid].get("predicted_line_formation",""),cars)
        if lines is None:skips["bad_lines"]+=1;continue
        try:idx=[cars.index(outcome[i][0]) for i in (1,2,3)]
        except ValueError:skips["winner_not_in_entries"]+=1;continue
        if len(set(idx))!=3:skips["duplicate_finish"]+=1;continue
        orig=np.array([float(v["estimated_win_probability"]) for v in rs])
        if abs(orig.sum()-1)>1e-6 or min(orig)<=0:skips["bad_win_probability"]+=1;continue
        q=calibrate(orig)
        pos={car:(lid,position) for lid,line in enumerate(lines) for position,car in enumerate(line)}
        rec={"date":rs[0]["date"],"race_id":rid,"cars":cars,"win_idx":idx,"prob":q,
            "line_idx":np.array([pos[c][0] for c in cars]),
            "pos_idx":np.array([pos[c][1] for c in cars]),
            "size":np.array([len(lines[pos[c][0]]) for c in cars])}
        recs={}
        for kind,p2,p3 in (("baseline",base2,base3),("line_both",line2,line3),
                           ("line_second_only",line2,base3),("line_third_only",base2,line3)):
            i,j,k=idx
            sec=cprob(rec,2,i,None,p2)[j]
            third=cprob(rec,3,i,j,p3)[k]
            loss2=-math.log(sec)
            loss3=-math.log(third)
            recs[kind]={"second":loss2,"third":loss3,"ordered":-math.log(q[i])+loss2+loss3}
        records.append({"date":rs[0]["date"],"r":recs,"same_1_2":int(rec["line_idx"][idx[0]]==rec["line_idx"][idx[1]])})
    n=len(records)
    assert n>2400,(n,dict(skips))
    models={key:{f:float(np.mean([r["r"][key][f] for r in records]))
                  for f in ("second","third","ordered")}
            for key in ("baseline","line_second_only","line_third_only","line_both")}
    days=defaultdict(list)
    for i,r in enumerate(records):days[r["date"]].append(i)
    rng=np.random.default_rng(20261018);clusters=list(days.values())
    diffs={}
    for key in ("line_second_only","line_third_only","line_both"):
        res={}
        for f in ("second","third","ordered"):
            diff=np.array([r["r"]["baseline"][f]-r["r"][key][f] for r in records])
            draws=[]
            for _ in range(2000):
                indices=np.concatenate([np.asarray(clusters[i]) for i in rng.integers(len(clusters),size=len(clusters))])
                draws.append(float(diff[indices].mean()))
            res[f]={"mean_logloss_improvement":float(diff.mean()),
                    "bootstrap_day_95pct":[float(z) for z in np.quantile(draws,[.025,.975])]}
        diffs[key]=res
    report={
        "model":"V5 line-conditioned ranking coefficients frozen on July-Aug, applied to earlier H1 F1 S-class outcomes",
        "scope":"2026-01-01 through 2026-06-30; 1/2/3 uniquely ranked",
        "forecasted_races":len(groups),"eligible":n,"skipped":dict(skips),
        "same_line_first_second_count":sum(r["same_1_2"] for r in records),
        "n_days":len(days),"metrics":models,"comparisons":diffs,
        "limitation":"This H1 has been used as V2 original win-model development holdout; supplementary robustness only",
        "parameter_change":False
    }
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()

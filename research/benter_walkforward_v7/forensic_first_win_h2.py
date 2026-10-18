#!/usr/bin/env python3
"""Describe extreme July-Aug 2026 F1 S-class V2 first-probability inputs.
This is explanatory audit only: does not retrain models or alter probabilities.
"""
import csv,json,math,statistics,hashlib
from collections import defaultdict,Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
H2=ROOT/"data/2026_future_block1/s_class_f1_20260701_20260830"
H1=ROOT/"data/2026_h1/s_class_f1_all"
FEATURES=("score","age","term","s_count","b_count","nige_count","makuri_count",
"sashi_count","mark_count","first_count","second_count","third_count","outside_count",
"win_rate","top2_rate","top3_rate","gear")

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        yield from csv.DictReader(f)

def value(v):
    try:
        f=float(str(v or "").replace(",","").replace("%","").strip())
        if math.isfinite(f):return f
    except ValueError: pass
    return None

def describe(seq):
    x=sorted(v for v in seq if v is not None)
    if not x:return {"n":0}
    return {"n":len(x),"p01":round(x[int((len(x)-1)*.01)],6),
            "median":round(x[len(x)//2],6),
            "p99":round(x[int((len(x)-1)*.99)],6),
            "min":x[0],"max":x[-1],"missing":len(seq)-len(x)}

def main():
    by_h2=defaultdict(list);specs=defaultdict(dict)
    for p,target in ((H2,by_h2),(H1,None)):
        # Keep raw values for per-feature comparison, but do not use outcomes in features.
        pass
    h2list=list(rows(H2/"entries.csv"))
    h1list=list(rows(H1/"entries.csv"))
    for row in h2list:
        by_h2[row["race_id"]].append(row)
    for row in rows(H2/"races.csv"):
        specs[row["race_id"]]=row
    dist={}
    for key in FEATURES:
        dist[key]={"2026H1":describe([value(r.get(key)) for r in h1list]),
                   "2026JulAug":describe([value(r.get(key)) for r in h2list])}
    target=list(rows(HERE/"frozen/independent_win_2026julaug.csv"))
    forecast=defaultdict(dict)
    for row in target:
        forecast[row["race_id"]][int(row["car_no"])]=float(row["independent_win_probability"])
    tails=json.loads((HERE/"first_win_research/first_win_tail_audit.json").read_text(encoding="utf-8"))
    failure_ids={(x["race_id"],x["car"]) for x in tails["counterexamples"] if x["period"]=="2026julaug"}
    extremes=[]
    for rid,prob in forecast.items():
        for car,p in prob.items():
            if p<.8 and (rid,car) not in failure_ids:continue
            entr=by_h2.get(rid,[])
            e=next((z for z in entr if int(z["car_no"])==car),None)
            if e is None:continue
            fields={key:value(e.get(key)) for key in FEATURES}
            cards=[]
            for z in entr:
                no=int(z["car_no"])
                cards.append({"car":no,"prob":round(prob.get(no,0),9),
                              "score":value(z.get("score")),
                              "win_rate":value(z.get("win_rate")),
                              "first_count":value(z.get("first_count")),
                              "top2_rate":value(z.get("top2_rate")),
                              "gear":value(z.get("gear"))})
            extremes.append({"race_id":rid,"date":e.get("race_date"),"car":car,"p":p,
                             "actual_winner":int((rid,car) not in failure_ids) if p>=.8 else 0,
                             "race_type":specs.get(rid,{}).get("race_type"),
                             "lineup":specs.get(rid,{}).get("predicted_line_formation"),
                             "features":fields,
                             "race_candidates":sorted(cards,key=lambda x:-x["prob"])})
    out={
        "description":"Source-input forensics for unusual July-Aug 2026 first place predictions",
        "training_cutoff":"2026-06-30",
        "status":"EXPLORATORY_DIAGNOSTIC_ONLY",
        "model_modified":False,
        "by_feature_distribution":dist,
        "extreme_candidates":extremes,
        "high_probability_80pct_candidates":sum(x["p"]>=.8 for x in extremes),
        "candidate_extreme_failure_examples":sum((x["race_id"],x["car"]) in failure_ids for x in extremes),
        "caveat":"Figures describe race-specific retrospectively collected cards; not a causal attribution nor a certified original live snapshot."
    }
    outpath=HERE/"first_win_research/first_win_input_forensics_2026julaug.json"
    outpath.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FORENSIC",json.dumps({
       "features":{k:v for k,v in dist.items() if k in ("score","win_rate","first_count","top2_rate","gear")},
       "n":len(extremes),
       "top_highprob":sorted([x for x in extremes if x["p"]>=.8],key=lambda x:-x["p"])[:5]
    },ensure_ascii=False),flush=True)
if __name__=="__main__":main()

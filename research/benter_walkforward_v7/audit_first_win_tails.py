#!/usr/bin/env python3
"""Diagnostic-only first-win extreme probability residuals on frozen evaluation races.
Do NOT select/tune models from this retrospective evaluation inspection.
"""
from __future__ import annotations
import json,math,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from research_first_win import load_races,FIT_LIMIT,OUT

def agg(values):
    n=len(values)
    e=sum(x["p"] for x in values)
    a=sum(x["won"] for x in values)
    var=sum(x["p"]*(1-x["p"]) for x in values)
    return {"candidate_rows":n,"predicted_winners":round(e,3),
            "actual_winners":a,"predicted_win_rate":round(e/n,5) if n else None,
            "actual_win_rate":round(a/n,5) if n else None,
            "normal_reference_z":round((a-e)/math.sqrt(var),3) if var else None}

def main():
    races,_=load_races()
    evaluation=[r for r in races if r["date"]>=FIT_LIMIT]
    buckets={"below_1pct":[],"80pct_and_up":[],"50pct_and_up":[]}
    examples=[]
    byperiod=defaultdict(lambda:defaultdict(list))
    bysize=defaultdict(lambda:defaultdict(list))
    for r in evaluation:
        for index,prob in enumerate(r["p"]):
            p=float(prob)
            won=int(index==r["winner"])
            if p<.01:key="below_1pct"
            elif p>=.8:key="80pct_and_up"
            else:key=None
            if key:
                row={"race_id":r["race_id"],"date":r["date"],"period":r["period"],
                     "car":r["cars"][index],"n":len(r["cars"]),"p":p,"won":won}
                buckets[key].append(row)
                byperiod[key][r["period"]].append(row)
                bysize[key][str(len(r["cars"]))].append(row)
                if key=="80pct_and_up" and not won:
                    examples.append(row)
                if key=="below_1pct" and won:
                    examples.append(row)
            if p>=.5:
                buckets["50pct_and_up"].append({"p":p,"won":won})
    result={
      "scope":"F1 S級 only",
      "model":"frozen walk-forward uncalibrated V2 independent win probabilities",
      "evaluation_races":len(evaluation),
      "period":"2025-04 through 2026-09",
      "warning":"Exploratory evaluation diagnostics. Do not modify model or select new calibration based on them",
      "tails":{key:agg(v) for key,v in buckets.items()},
      "by_period":{key:{period:agg(v) for period,v in groups.items()} for key,groups in byperiod.items()},
      "by_field_size":{key:{size:agg(v) for size,v in groups.items()} for key,groups in bysize.items()},
      "counterexamples":sorted(examples,key=lambda x:(x["date"],x["race_id"])),
    }
    OUT.mkdir(exist_ok=True)
    (OUT/"first_win_tail_audit.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FIRST_TAILS",json.dumps({k:v for k,v in result.items() if k!="counterexamples"},
                                    ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

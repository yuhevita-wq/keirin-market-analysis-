#!/usr/bin/env python3
from __future__ import annotations
import json, importlib.util
from pathlib import Path
from statistics import mean

spec=importlib.util.spec_from_file_location("v27","scripts/keirin_shogi_v27_second_inside_outside_mixture.py")
v27=importlib.util.module_from_spec(spec); spec.loader.exec_module(v27)

OUT=Path("results/keirin_shogi/v28_second_slot_allocation")
OUT.mkdir(parents=True,exist_ok=True)

# Structural 2-slot second row:
# high P(inside)  -> both v21 first candidates
# low P(inside)   -> best two outsiders
# middle          -> best one inside + best one outsider
# Exactly 2 second candidates (except pathological one-candidate edge cases).

def group_probs(base,vr,mods):
    fc=list(map(v27.v25.ino,vr["candidates"]))
    inside=[n for n in fc if n in base]
    outside=[n for n in base if n not in set(fc)]
    q=float(mods["gate"].predict_proba(v27.gate_features(base,vr).reshape(1,-1))[0,1])
    pi=v27.normalized_group(mods["inside"],base,inside,fc,vr,"inside")
    po=v27.normalized_group(mods["outside"],base,outside,fc,vr,"outside")
    si=sorted(pi.items(),key=lambda kv:(-kv[1],kv[0]))
    so=sorted(po.items(),key=lambda kv:(-kv[1],kv[0]))
    return fc,q,si,so

def choose_slots(base,vr,mods,low,high):
    fc,q,si,so=group_probs(base,vr,mods)
    if q>=high:
        c=[n for n in fc if n in base][:2]
        mode="both_inside"
    elif q<=low:
        c=[n for n,p in so[:2]]
        mode="two_outside"
    else:
        c=[]
        if si:c.append(si[0][0])
        if so and so[0][0] not in c:c.append(so[0][0])
        mode="split"
    # fill to 2 from remaining highest structural probabilities, but never exceed 2
    if len(c)<2:
        rem=[]
        for n,p in si: rem.append((n,q*p))
        for n,p in so: rem.append((n,(1-q)*p))
        rem.sort(key=lambda kv:(-kv[1],kv[0]))
        for n,p in rem:
            if n not in c:c.append(n)
            if len(c)>=2:break
    return c[:2],q,mode

def evaluate(ds,mods,low,high):
    logs=[]
    for rid,dt,base,order,vr in ds:
        if not vr.get("v21_participate"):continue
        c,q,mode=choose_slots(base,vr,mods,low,high)
        f,s=order[:2];fc=list(map(v27.v25.ino,vr["candidates"]))
        rel="same_line" if base[f]["line_id"]==base[s]["line_id"] else "different_line"
        logs.append({
          "race_id":rid,"race_date":dt,"actual_first":f,"actual_second":s,
          "first_candidates":fc,"first_hit":int(f in fc),
          "second_in_first_candidates":int(s in fc),"relation":rel,
          "q_inside":q,"mode":mode,"second_candidates":c,
          "second_hit":int(s in c),"second_candidate_count":len(c)
        })
    return logs

def metrics(logs):
    n=len(logs);first=[x for x in logs if x["first_hit"]]
    both=[x for x in first if x["second_in_first_candidates"]]
    outside=[x for x in first if not x["second_in_first_candidates"]]
    hard=[x for x in outside if x["relation"]=="different_line"]
    modes={}
    for mode in ("both_inside","split","two_outside"):
        rr=[x for x in logs if x["mode"]==mode]
        modes[mode]={"races":len(rr),"capture":sum(x["second_hit"] for x in rr)/len(rr) if rr else 0}
    return {
      "races":n,
      "second_capture":sum(x["second_hit"] for x in logs)/n,
      "avg_second_candidates":mean(x["second_candidate_count"] for x in logs),
      "second_given_first_capture":sum(x["second_hit"] for x in first)/len(first),
      "when_both_first_second_in_first_candidates":sum(x["second_hit"] for x in both)/len(both),
      "outside_first_candidates_capture_given_first":sum(x["second_hit"] for x in outside)/len(outside),
      "hard_diff_line_outside_capture_given_first":sum(x["second_hit"] for x in hard)/len(hard),
      "hard_races":len(hard),
      "mode_stats":modes
    }

def main():
    vrows,eb,rb=v27.prepared()
    train=v27.datasets(vrows,eb,rb,v27.v25.TRAIN_START,v27.v25.TRAIN_END)
    cal=v27.datasets(vrows,eb,rb,v27.v25.CAL_START,v27.v25.CAL_END)
    test=v27.datasets(vrows,eb,rb,v27.v25.TEST_START,v27.v25.TEST_END)

    variants=[
      {"name":"all_d2","participant":False,"depth":2,"C":.5},
      {"name":"all_d3","participant":False,"depth":3,"C":.5},
      {"name":"participant_d2","participant":True,"depth":2,"C":.5},
      {"name":"participant_d3","participant":True,"depth":3,"C":.5},
    ]
    grid=[];models={}
    for v in variants:
        mods=v27.fit_models(train,v["participant"],v["C"],v["depth"]);models[v["name"]]=mods
        for low in (.20,.30,.40,.50):
            for high in (.50,.60,.70,.80):
                if low>=high:continue
                logs=evaluate(cal,mods,low,high);m=metrics(logs)
                obj=(m["second_given_first_capture"]
                     +0.65*m["hard_diff_line_outside_capture_given_first"]
                     +0.30*m["outside_first_candidates_capture_given_first"]
                     +0.15*m["when_both_first_second_in_first_candidates"])
                grid.append({"variant":v["name"],"low":low,"high":high,"objective":obj,**m})
    grid.sort(key=lambda x:(x["objective"],x["second_given_first_capture"],x["hard_diff_line_outside_capture_given_first"]),reverse=True)
    best=grid[0];mods=models[best["variant"]]
    logs=evaluate(test,mods,best["low"],best["high"]);tm=metrics(logs)

    baseline={"second_capture":0.5342465753424658,"second_given_first_capture":0.5934065934065934,
              "avg_second_candidates":2.0332681017612524,"hard_diff_line_outside_capture_given_first":0.1259259259259259,
              "when_both_first_second_in_first_candidates":0.851063829787234}
    summary={
      "algorithm":"keirin_shogi_v28_second_slot_allocation",
      "concept":"Allocate the 2nd-row slots by regime instead of blending everything into one ranking. High inside probability uses both v21 first candidates; low uses two outsiders; middle uses one+one.",
      "first_core":"v21 frozen",
      "selected_calibration":best,
      "test_metrics":tm,
      "v23_baseline":baseline,
      "delta_vs_v23":{k:tm[k]-baseline[k] for k in baseline if k in tm},
      "calibration_top10":grid[:10],
      "leakage_guard":"Regime thresholds selected only on 2025-10-27..12-28 calibration. 2026 H1 untouched until final test. Odds unused."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"race_log.json").write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"calibration.json").write_text(json.dumps(grid,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()

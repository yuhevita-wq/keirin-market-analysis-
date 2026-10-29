#!/usr/bin/env python3
"""Truly forward-year validation: frozen 2023 model + fixed 2024 H1 calibrator.

2025 outcomes are read ONLY for scoring predictions. Fixed calibration coefficients
are loaded unchanged from the already-committed 2024 H1-fit/H2-approved audit.
No 2025 fit, hyperparameter tuning, model selection or market odds.
"""
from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import hashlib
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
import observe as fixed

CAL=ROOT/"data/audits/independent_trifecta_recalibration_2024.json"
LABELS=ROOT/"data/result_universe_v2/combined/results_2025.jsonl.gz"
OUT=ROOT/"data/audits/independent_trifecta_final_validation_2025.json"
EXAMPLE=ROOT/"data/audits/independent_trifecta_final_validation_sample_2025.json"
ARCHIVE=ROOT/"data/audits/independent_trifecta_all_probabilities_2025.jsonl.gz"
BINS=[
 ("0-0.1%",0.,.001),("0.1-0.5%",.001,.005),
 ("0.5-1%",.005,.01),("1-2%",.01,.02),
 ("2-5%",.02,.05),("5-10%",.05,.1),
 ("10-20%",.1,.2),("20-100%",.2,1.000001)
]
EVENTS=("same_line_12","same_line_13","same_line_23","same_line_123")
MIN_P=1e-15

def readers():
    for base in (ROOT/"data/a_class_12",ROOT/"data/a3_challenge",ROOT/"data/2025"):
        paths=(base.glob("2025_*/races.csv") if base.name in {"a_class_12","a3_challenge"}
               else base.glob("**/races.csv"))
        for racepath in sorted(paths):
            entrypath=racepath.with_name("entries.csv")
            if not entrypath.is_file():continue
            def factory(a=racepath,b=entrypath):
                with a.open("r",encoding="utf-8-sig",newline="") as f:
                    rr=list(csv.DictReader(f))
                with b.open("r",encoding="utf-8-sig",newline="") as f:
                    ee=list(csv.DictReader(f))
                return rr,ee
            yield str(racepath.relative_to(ROOT)),factory
    for z in sorted((ROOT/"data/grade_races").rglob("*2025*.zip")):
        with zipfile.ZipFile(z) as a:names=set(a.namelist())
        for name in sorted(names):
            if not name.endswith("races.csv"):continue
            ep=name[:-9]+"entries.csv"
            if ep not in names:continue
            def factory(z=z,n=name,ep=ep):
                with zipfile.ZipFile(z) as archive:
                    rr=list(csv.DictReader(io.StringIO(archive.read(n).decode("utf-8-sig"))))
                    ee=list(csv.DictReader(io.StringIO(archive.read(ep).decode("utf-8-sig"))))
                return rr,ee
            yield str(z.relative_to(ROOT))+":"+name,factory

def class_of(path):
    if "/a_class_12/" in path:return "A1_A2"
    if "/a3_challenge/" in path:return "A3"
    if "/s_class_" in path:return "S_F1"
    if "/grade_races/" in path:return "S_grade"
    return "other"

def read_2025():
    results={}
    with gzip.open(LABELS,"rt",encoding="utf-8") as f:
        for line in f:
            item=json.loads(line)
            top=item.get("finish_order")
            if isinstance(top,list) and len(top)>=3:
                cars=tuple(map(str,top[:3]))
                if len(set(cars))==3 and all(c.isdigit() for c in cars):
                    results[str(item["race_id"])]=cars
    manifest={};entries=defaultdict(dict);audit=Counter();used_source=Counter()
    for path,factory in readers():
        records,rows=factory()
        ids=set()
        for r in records:
            rid=str(r.get("race_id","")).strip()
            date=str(r.get("race_date","")).strip()
            if rid not in results or not ("2025-01-01"<=date<="2025-12-31"):continue
            info={"date":date,"line":str(r.get("predicted_line_formation") or ""),
                  "line_status":str(r.get("line_status") or ""),
                  "class":class_of(path),"source":path}
            if rid in manifest:
                audit["duplicate_manifests"]+=1
                if manifest[rid]["line_status"]!="published" and info["line_status"]=="published":
                    manifest[rid]=info
            else:manifest[rid]=info
            ids.add(rid)
        n=0
        for r in rows:
            rid=str(r.get("race_id","")).strip()
            if rid not in ids:continue
            car=str(r.get("car_no","")).strip()
            if not car.isdigit():continue
            try:
                score=float(r.get("score") or "")
                if not math.isfinite(score):raise ValueError
            except (ValueError,TypeError):
                audit["invalid_score_rows"]+=1;continue
            if car in entries[rid]:
                audit["duplicate_entry_rows"]+=1;continue
            entries[rid][car]=score;n+=1
        used_source[path]+=n
        print("SOURCE",path,n,flush=True)
    eligible={};by_class=Counter();by_size=Counter()
    for rid,m in manifest.items():
        scores=entries.get(rid,{})
        if len(scores)<4 or not set(results[rid]).issubset(scores):
            audit["missing_card_or_podium"]+=1;continue
        if m["line_status"]!="published":
            audit["missing_published_line"]+=1;continue
        groups=fixed.parse_line(m["line"],scores)
        if groups is None:
            audit["invalid_line_coverage"]+=1;continue
        positions={car:(g+1,pos+1,len(grp)) for g,grp in enumerate(groups) for pos,car in enumerate(grp)}
        ranks={car:min(i+1,4) for i,car in enumerate(sorted(scores,key=lambda car:(-scores[car],int(car))))}
        eligible[rid]={
          "date":m["date"],"order":results[rid],"scores":scores,
          "positions":positions,"ranks":ranks,"groups":groups,"race_class":m["class"]}
        by_class[m["class"]]+=1;by_size[len(scores)]+=1
    eligible=dict(sorted(eligible.items(),key=lambda x:(x[1]["date"],x[0])))
    return eligible,{
      "verified_2025_top3_results":len(results),"result_races_with_manifest":len(manifest),
      "eligible_2025":len(eligible),"matched_percent_of_results":round(len(eligible)*100/len(results),4),
      "class_counts":dict(by_class),"field_sizes":dict(by_size),
      "exclusions":dict(audit),"individual_files_source_count":dict(used_source)
    }

def same_line_flags(e,ticket):
    a,b,c=ticket;la=e["positions"][a][0];lb=e["positions"][b][0];lc=e["positions"][c][0]
    return (int(la==lb),int(la==lc),int(lb==lc),int(la==lb==lc))

def recalibrate(e,original,params):
    alpha=float(params["alpha"]); betas=tuple(float(v) for v in params["line_coefs"])
    scores={}
    for ticket,p in original.items():
        if not p>0:raise ValueError("Model has zero probability")
        flags=same_line_flags(e,ticket)
        scores[ticket]=alpha*math.log(p)+sum(b*x for b,x in zip(betas,flags))
    max_score=max(scores.values())
    val={k:math.exp(s-max_score) for k,s in scores.items()}
    total=sum(val.values())
    out={k:v/total for k,v in val.items()}
    if abs(sum(out.values())-1)>1e-9:raise ValueError("Calibration failed normalization")
    return out

def empty_metric():
    return {"n":0,"nll":0.,"brier":0.,"ptrue":0.,"top1":0,"top3":0}
def sum_metric(m,dist,truth):
    p=dist[truth];v=-math.log(max(p,MIN_P))
    m["n"]+=1;m["nll"]+=v;m["brier"]+=1.-2*p+sum(x*x for x in dist.values());m["ptrue"]+=p
    top=sorted(dist,key=lambda t:(-dist[t],t))[:3]
    m["top1"]+=int(top[0]==truth);m["top3"]+=int(truth in top)
    return v
def finish_metric(m):
    n=m["n"]
    return {"races":n,"nll":round(m["nll"]/n,8),
            "brier":round(m["brier"]/n,8),"mean_probability_of_actual_pct":round(100*m["ptrue"]/n,6),
            "top1_hit_pct":round(100*m["top1"]/n,5),"top3_hit_pct":round(100*m["top3"]/n,5)}

def main():
    calibration=json.loads(CAL.read_text("utf-8"))
    assert calibration["state"]=="COMPLETE_FROZEN_2023_CALIBRATED_2024_H1_TEST_2024_H2"
    assert calibration["2025_or_later_accessed"] is False
    adopted=calibration["model_adopted"]
    assert adopted=="joint" and calibration["accepted"] is True
    alpha=calibration["fitted_coefficients"][adopted]
    orig,oldcov=fixed.ingest()
    training={rid:e for rid,e in orig.items() if e["date"]<=fixed.TRAIN_END}
    if len(training)!=19248:raise ValueError("2023 training changed")
    model=fixed.train_model(training,True)
    del orig,training
    print("2023_FROZEN_MODEL_FITTED_ONLY_PRIOR_YEARS",flush=True)
    events,coverage=read_2025()
    if len(events)<20000:raise ValueError("2025 audit missing too many races")
    print("2025_ELIGIBLE_UNSEEN",len(events),flush=True)
    allm={name:empty_metric() for name in ("unchanged","adopted")}
    byclass=defaultdict(lambda:{name:empty_metric() for name in allm})
    bymonth=defaultdict(lambda:{name:empty_metric() for name in allm})
    byshape=defaultdict(lambda:{name:empty_metric() for name in allm})
    bands={key:{lab:{"tickets":0,"expected":0.,"actual":0,"var":0.} for lab,_,_ in BINS} for key in allm}
    structures={key:{ev:{"obs":0,"exp":0.,"var":0.} for ev in EVENTS} for key in allm}
    day_delta=defaultdict(lambda:[0.,0])
    n_tickets=0;sample=None
    with gzip.open(ARCHIVE,"wt",encoding="utf-8",compresslevel=6) as out:
        for i,(rid,e) in enumerate(events.items()):
            original=fixed.distribution(e,model,True)
            calibrated=recalibrate(e,original,alpha)
            truth=e["order"]
            shape="-".join(map(str,sorted(map(len,e["groups"]),reverse=True)))
            n_tickets+=len(original)
            loss={}
            for label,dist in (("unchanged",original),("adopted",calibrated)):
                loss[label]=sum_metric(allm[label],dist,truth)
                sum_metric(byclass[e["race_class"]][label],dist,truth)
                sum_metric(bymonth[e["date"][:7]][label],dist,truth)
                sum_metric(byshape[shape][label],dist,truth)
                # Probabilities within a single race are NOT independent.
                predmass=defaultdict(float)
                agg=defaultdict(float)
                for t,p in dist.items():
                    selected=next(x for x,lo,hi in BINS if lo<=p<hi)
                    z=bands[label][selected]
                    z["tickets"]+=1;z["expected"]+=p;z["actual"]+=int(t==truth)
                    predmass[selected]+=p
                    fl=same_line_flags(e,t)
                    for j,ev in enumerate(EVENTS):
                        if fl[j]:agg[ev]+=p
                for key,prob in predmass.items():
                    bands[label][key]["var"]+=max(0.,prob*(1-prob))
                target_flag=same_line_flags(e,truth)
                for j,ev in enumerate(EVENTS):
                    p=agg[ev];s=structures[label][ev]
                    s["exp"]+=p;s["obs"]+=target_flag[j];s["var"]+=max(0.,p*(1-p))
            day_delta[e["date"]][0]+=loss["unchanged"]-loss["adopted"]
            day_delta[e["date"]][1]+=1
            # Original and calibrated complete 2025 distributions are archived,
            # so a later market comparison never needs to recreate the model.
            data={"race_id":rid,"date":e["date"],"true_result":"-".join(truth),
                  "class":e["race_class"],"calibration":"frozen-2024-H1-joint",
                  "all_trifecta_probability":{
                    "-".join(t):round(float(v),12) for t,v in sorted(calibrated.items())}}
            out.write(json.dumps(data,ensure_ascii=False,separators=(",",":"))+"\n")
            if sample is None and len(e["scores"])==7:
                sample={"race_id":rid,"date":e["date"],"true_result":"-".join(truth),
                        "initial_probability_of_actual_pct":100*original[truth],
                        "calibrated_probability_of_actual_pct":100*calibrated[truth],
                        "all_210":[{"ticket":"-".join(t),"before_pct":round(100*original[t],8),
                                     "after_pct":round(100*calibrated[t],8)}
                                     for t in sorted(original)]}
            if (i+1)%2500==0:print("2025_BLIND_RACES_SCORED",i+1,len(events),flush=True)
    n=len(events)
    average=sum(x[0] for x in day_delta.values())/n
    robust_se=math.sqrt(sum((x[0]-average*x[1])**2 for x in day_delta.values()))/n
    outbands={}
    for label in allm:
        outbands[label]={}
        for b,s in bands[label].items():
            if not s["tickets"]:continue
            pred=s["expected"];obs=s["actual"];nn=s["tickets"]
            sd=math.sqrt(s["var"])
            outbands[label][b]={
               "tickets":nn,"mean_prediction_pct":round(100*pred/nn,6),
               "actual_frequency_pct":round(100*obs/nn,6),
               "expected_hits":round(pred,3),"observed_hits":obs,
               "observed_over_expected":round(obs/pred,5) if pred else None,
               "z_score":round((obs-pred)/sd,4) if sd else None}
    outevents={}
    for label in allm:
        outevents[label]={}
        for ev,s in structures[label].items():
            outevents[label][ev]={
               "expected_pct":round(100*s["exp"]/n,5),
               "actual_pct":round(100*s["obs"]/n,5),
               "difference_pp":round(100*(s["obs"]-s["exp"])/n,5),
               "null_z":round((s["obs"]-s["exp"])/math.sqrt(s["var"]),4)}
    ece={label:round(100*sum(abs(v["actual"]-v["expected"]) for v in bands[label].values())/
                       n_tickets,8) for label in allm}
    metrics={k:finish_metric(m) for k,m in allm.items()}
    ci=[average-1.96*robust_se,average+1.96*robust_se]
    conclusion=(average>0 and ci[0]>0 and ece["adopted"]<ece["unchanged"])
    report={
      "state":"COMPLETE_2025_FINAL_OUT_OF_TIME_VALIDATION",
      "2025_no_training_or_parameter_selection":True,"no_odds_or_prices_used":True,
      "2023_original_model_training":["2023-01-01","2023-10-31"],
      "2024_calibration_fit":["2024-01-01","2024-06-30"],
      "2024_acceptance_test":["2024-07-01","2024-12-31"],
      "2025_final_test":["2025-01-01","2025-12-31"],
      "frozen_calibrator":calibration["model_adopted"],
      "frozen_coefficients":alpha,
      "2025_result_coverage":coverage,
      "prior_2023_train_count":19248,
      "total_2025_3trifecta_ticket_probabilities":n_tickets,
      "metrics":metrics,
      "calibration_bands":outbands,
      "structure_calibration":outevents,
      "ece_ticket_weighted_pp":ece,
      "nll_improvement":{
        "unchanged_minus_calibrated":round(average,8),
        "day_cluster_robust_se":round(robust_se,8),
        "95pct_reference_interval":[round(x,8) for x in ci],
        "number_of_days":len(day_delta)},
      "by_class":{k:{name:finish_metric(v) for name,v in group.items()} for k,group in byclass.items()},
      "by_month":{k:{name:finish_metric(v) for name,v in group.items()} for k,group in sorted(bymonth.items())},
      "by_line_shape_top_12":{k:{name:finish_metric(v) for name,v in byshape[k].items()}
                   for k in sorted(byshape,key=lambda s:-byshape[s]["adopted"]["n"])[:12]},
      "strict_final_test_passed":bool(conclusion),
      "strict_success_criterion":"NLL improvement day-cluster 95% lower bound > 0 AND fixed-bin ECE reduction on 2025",
      "full_2025_probability_file":str(ARCHIVE.relative_to(ROOT)),
      "sample_210_combinations":str(EXAMPLE.relative_to(ROOT)),
      "limitations":[
        "Score and predicted-line historical snapshots cannot guarantee original at-cutoff availability.",
        "Only exact-ID results with usable riders and published full line enter; exclusion counts disclosed.",
        "Bins contain dependent mutually exclusive 3連単 tickets; null variances aggregate race-level event masses.",
        "2025 remains an independent validation relative to the 2023 model and 2024 calibrator; future re-tuning on 2025 invalidates that status.",
        "No profit/ROI or market comparison conducted."],
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    EXAMPLE.write_text(json.dumps(sample,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FINAL_2025_OUT_OF_TIME_COMPLETE",json.dumps({
      "eligible":n,"trifecta_probabilities":n_tickets,
      "before":metrics["unchanged"],"after":metrics["adopted"],
      "nll_ci":report["nll_improvement"],"ece":ece,
      "passed":bool(conclusion),
      "ten_twenty":{k:v.get("10-20%") for k,v in outbands.items()},
      "same_line_all":{k:v.get("same_line_123") for k,v in outevents.items()}
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()

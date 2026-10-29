#!/usr/bin/env python3
"""Frozen 2023 odds-free trifecta formula -> 2024 untouched calibration audit.

Uses the unchanged research/independent_trifecta_2023/observe.py function:
 - 2023-01..10 for historical fit, 2023-11..12 never fitted
 - 2024 ALL months for independent assessment only (not for fitting/tuning)
 - score rank / published line and result labels only, NO odds or payouts
Exports all 2024 3連単 probabilities to an Actions artifact, and aggregate JSON
with race-cluster-corrected calibration diagnostics.
"""
from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/independent_trifecta_2023"))
import observe as fixed  # noqa: E402

AUDIT = ROOT / "data/audits/independent_trifecta_calibration_2024.json"
SAMPLE = ROOT / "data/audits/independent_trifecta_sample_2024.json"
ALL_TICKETS = ROOT / "data/audits/independent_trifecta_all_tickets_2024.jsonl.gz"
LABELS_2024 = ROOT / "data/result_universe_v2/combined/results_2024.jsonl.gz"
MODELS = ("uniform", "line_role_only", "line_role_score_rank")
BUCKETS = ("0–0.1%", "0.1–0.5%", "0.5–1%", "1–2%", "2–5%", "5–10%", "10–20%", "20–100%")
EVENTS = ("same_line_12","same_line_13","same_line_23","same_line_123","three_distinct_lines",
          "winner_leader","winner_bante","score_rank_1_wins")
PRIMARY = "line_role_score_rank"

def bucket(p):
    if p < .001: return BUCKETS[0]
    if p < .005: return BUCKETS[1]
    if p < .01: return BUCKETS[2]
    if p < .02: return BUCKETS[3]
    if p < .05: return BUCKETS[4]
    if p < .10: return BUCKETS[5]
    if p < .20: return BUCKETS[6]
    return BUCKETS[7]

def source_readers_2024():
    for directory in (ROOT/"data/a_class_12", ROOT/"data/a3_challenge",ROOT/"data/2024"):
        paths = (directory.glob("2024_*/races.csv") if directory.name in ("a_class_12","a3_challenge")
                 else directory.glob("**/races.csv"))
        for rp in sorted(paths):
            ep=rp.with_name("entries.csv")
            if not ep.is_file():continue
            def opener(a=rp,b=ep):
                with a.open(encoding="utf-8-sig",newline="") as f:
                    rac=list(csv.DictReader(f))
                with b.open(encoding="utf-8-sig",newline="") as f:
                    ent=list(csv.DictReader(f))
                return rac,ent
            yield str(rp.relative_to(ROOT)),opener
    for z in sorted((ROOT/"data/grade_races").rglob("*2024*.zip")):
        if not z.is_file(): continue
        with zipfile.ZipFile(z) as f:
            filenames=set(f.namelist())
        for n in sorted(filenames):
            if not n.endswith("races.csv"):continue
            ep=n[:-9]+"entries.csv"
            if ep not in filenames:continue
            def opener(path=z,a=n,b=ep):
                with zipfile.ZipFile(path) as zf:
                    race=list(csv.DictReader(io.StringIO(zf.read(a).decode("utf-8-sig"))))
                    entries=list(csv.DictReader(io.StringIO(zf.read(b).decode("utf-8-sig"))))
                return race,entries
            yield str(z.relative_to(ROOT))+":"+n,opener

def classify(path,row):
    if "/a_class_12/" in path: return "A1_A2"
    if "/a3_challenge/" in path: return "A3"
    if "/s_class_" in path: return "S_F1"
    if "/grade_races/" in path: return "S_grade"
    return "other"

def ingest_2024():
    truth={}
    with gzip.open(LABELS_2024,"rt",encoding="utf-8") as handle:
        for line in handle:
            v=json.loads(line);top=v.get("finish_order")
            if not isinstance(top,list) or len(top)<3:continue
            top=tuple(map(str,top[:3]))
            if len(set(top))!=3 or not all(x.isdigit() for x in top):continue
            truth[str(v["race_id"])]=top
    metadata={};cards=defaultdict(dict);provenance=Counter();conflicts=0
    for name,read in source_readers_2024():
        races,entries=read();selected=set()
        for r in races:
            rid=str(r.get("race_id","")).strip()
            day=str(r.get("race_date","")).strip()
            if rid not in truth or not ("2024-01-01"<=day<="2024-12-31"):continue
            info={"date":day,
                  "line":str(r.get("predicted_line_formation") or ""),
                  "line_status":str(r.get("line_status") or ""),
                  "class":classify(name,r),"source":name}
            if rid in metadata:
                provenance["duplicate_manifests"]+=1
                if info["line"] != metadata[rid]["line"]:
                    conflicts+=1
                # Prefer published line if a duplicate archive has more detail.
                if metadata[rid]["line_status"]!="published" and info["line_status"]=="published":
                    metadata[rid]=info
            else:metadata[rid]=info
            selected.add(rid)
        count=0
        for r in entries:
            rid=str(r.get("race_id","")).strip()
            if rid not in selected:continue
            car=str(r.get("car_no","")).strip()
            if not car.isdigit(): continue
            try:
                value=float(r.get("score") or "")
                if not math.isfinite(value):raise ValueError
            except (ValueError,TypeError):
                provenance["invalid_score_rows"]+=1
                continue
            if car in cards[rid]:
                provenance["duplicated_entry_rows"]+=1
                continue
            cards[rid][car]=value;count+=1
        print("SOURCE",name,"usable_entry_rows",count,flush=True)
    eligible={};by_class=Counter();by_field=Counter()
    for rid,info in metadata.items():
        scores=cards.get(rid,{})
        if len(scores)<4 or not all(car in scores for car in truth[rid]):
            provenance["missing_entries_or_podium"]+=1
            continue
        if info["line_status"]!="published":
            provenance["no_published_line"]+=1
            continue
        groups=fixed.parse_line(info["line"],scores)
        if groups is None:
            provenance["line_coverage_invalid"]+=1
            continue
        positions={car:(g+1,i+1,len(grp)) for g,grp in enumerate(groups) for i,car in enumerate(grp)}
        sorted_score=sorted(scores,key=lambda c:(-scores[c],int(c)))
        ranks={car:min(i+1,4) for i,car in enumerate(sorted_score)}
        eligible[rid]={"date":info["date"],"order":truth[rid],
                       "scores":scores,"groups":groups,"ranks":ranks,
                       "positions":positions,"race_class":info["class"]}
        by_class[info["class"]]+=1;by_field[len(scores)]+=1
    eligible=dict(sorted(eligible.items(),key=lambda item:(item[1]["date"],item[0])))
    return eligible,{"verified_2024_result_labels":len(truth),
                     "races_with_2024_manifest":len(metadata),
                     "usable_matching_cards_and_line":len(eligible),
                     "coverage_pct_of_verified_labels":round(100*len(eligible)/len(truth),4),
                     "class_counts":dict(by_class),"field_size_counts":dict(by_field),
                     "duplicate_line_conflicts_observed":conflicts,"exclusions":dict(provenance)}

def flags(e,triple):
    a,b,c=triple; pos=e["positions"]
    line=(pos[a][0],pos[b][0],pos[c][0])
    return {
      "same_line_12":line[0]==line[1],
      "same_line_13":line[0]==line[2],
      "same_line_23":line[1]==line[2],
      "same_line_123":line[0]==line[1]==line[2],
      "three_distinct_lines":len(set(line))==3,
      "winner_leader":pos[a][1]==1,
      "winner_bante":pos[a][1]==2,
      "score_rank_1_wins":e["ranks"][a]==1,
    }

def blank_metric():
    return {"races":0,"nll_sum":0.0,"brier_sum":0.0,
            "ptrue_sum":0.0,"top1_hits":0,"top3_hits":0,
            "days":defaultdict(lambda:[0,0.])}

def aggregate_add(stat,day,dist,truth,prange):
    n=len(dist)
    ptrue=dist[truth]
    nll=-math.log(max(ptrue,1e-15))
    stat["races"]+=1
    stat["nll_sum"]+=nll
    stat["brier_sum"]+=1.0-2.0*ptrue+sum(x*x for x in dist.values())
    stat["ptrue_sum"]+=ptrue
    top=sorted(dist,key=lambda x:(-dist[x],x))[:3]
    stat["top1_hits"]+=int(top[0]==truth)
    stat["top3_hits"]+=int(truth in top)
    ds=stat["days"][day]
    ds[0]+=1;ds[1]+=nll
    return ptrue,top,nll

def pack_metric(m):
    n=m["races"]
    if not n:return None
    return {"races":n,"mean_nll":round(m["nll_sum"]/n,8),
            "mean_multiclass_brier":round(m["brier_sum"]/n,8),
            "p_true_mean_pct":round(m["ptrue_sum"]*100/n,6),
            "top1_hit_pct":round(100*m["top1_hits"]/n,5),
            "top3_hit_pct":round(100*m["top3_hits"]/n,5)}

def bucket_result(s,byday):
    # Each race contributes a single Bernoulli success for this bucket.
    # The per-ticket 3連単 indicators within a race are NOT independent.
    tickets,exp_hits,actual,var=s
    sd=math.sqrt(var)
    day_se=math.sqrt(sum(v*v for v in byday.values()))
    return {"ticket_count":tickets,"predicted_hits":round(exp_hits,5),
            "actual_hits":actual,"mean_predicted_pct":round(100*exp_hits/tickets,7) if tickets else None,
            "empirical_hit_pct":round(100*actual/tickets,7) if tickets else None,
            "observed_over_expected":round(actual/exp_hits,5) if exp_hits else None,
            "actual_minus_expected_hits":round(actual-exp_hits,5),
            "z_score_under_calibrated_null":round((actual-exp_hits)/sd,4) if sd else None,
            "null_reference_95pct_hit_interval": [round(exp_hits-1.96*sd,2),round(exp_hits+1.96*sd,2)],
            "date_cluster_residual_se_hits":round(day_se,4)}

def structure_result(v,byday):
    obs,expected,var,n=v;sd=math.sqrt(var)
    return {"races":n,"actual_count":obs,"expected_count":round(expected,4),
            "actual_rate_pct":round(100*obs/n,5),
            "expected_rate_pct":round(100*expected/n,5),
            "observed_minus_expected_pp":round(100*(obs-expected)/n,5),
            "null_z_score":round((obs-expected)/sd,4) if sd else None,
            "reference_95pct_pp":[round(100*(expected-1.96*sd)/n,4),
                                  round(100*(expected+1.96*sd)/n,4)],
            "date_cluster_residual_se_pp":round(100*math.sqrt(sum(x*x for x in byday.values()))/n,5)}

def main():
    # No 2024 targets can affect the fixed 2023 training model.
    old,old_audit=fixed.ingest()
    tr={k:v for k,v in old.items() if v["date"]<=fixed.TRAIN_END}
    if len(tr)!=19248:
        raise RuntimeError(f"Frozen 2023 training universe changed: {len(tr)} != 19248")
    models={"line_role_only":fixed.train_model(tr,False),
            "line_role_score_rank":fixed.train_model(tr,True)}
    del old, tr
    print("FROZEN_2023_TRAIN_19248",flush=True)
    events,coverage=ingest_2024()
    if len(events)<15000:
        raise RuntimeError(f"2024 coverage unexpectedly low: {len(events)}")
    print("VALID_2024",len(events),flush=True)
    overall={name:blank_metric() for name in MODELS}
    by_class=defaultdict(lambda:{name:blank_metric() for name in MODELS})
    by_shape=defaultdict(lambda:{name:blank_metric() for name in MODELS})
    by_field=defaultdict(lambda:{name:blank_metric() for name in MODELS})
    by_month=defaultdict(lambda:{name:blank_metric() for name in MODELS})
    raw_buckets={name:{b:[0,0.,0,0.] for b in BUCKETS} for name in MODELS}
    bucket_day={name:defaultdict(float) for name in MODELS}
    raw_events={name:{ev:[0,0.,0.,0] for ev in EVENTS} for name in MODELS}
    event_day={name:defaultdict(float) for name in MODELS}
    per_day_loss_delta={name:defaultdict(lambda:[0.,0]) for name in MODELS[1:]}
    sample=None
    AUDIT.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(ALL_TICKETS,"wt",encoding="utf-8",compresslevel=6) as export:
        for idx,(rid,e) in enumerate(events.items()):
            cars=sorted(e["scores"],key=int);n=len(cars)
            truth=e["order"];day=e["date"];month=day[:7]
            size_shape="-".join(map(str,sorted([len(g) for g in e["groups"]],reverse=True)))
            cls=e["race_class"]
            win_flags=flags(e,truth)
            uniform=1/(n*(n-1)*(n-2))
            models_dist={
              "uniform":{t:uniform for t in itertools.permutations(cars,3)},
              "line_role_only":fixed.distribution(e,models["line_role_only"],False),
              "line_role_score_rank":fixed.distribution(e,models["line_role_score_rank"],True)
            }
            # No unrounded input is silently renormalized after scoring.
            for name,dist in models_dist.items():
                ptrue,top,nll=aggregate_add(overall[name],day,dist,truth,None)
                for container,key in ((by_class,cls),(by_shape,size_shape),(by_field,str(n)),(by_month,month)):
                    aggregate_add(container[key][name],day,dist,truth,None)
                if name!="uniform":
                    d=per_day_loss_delta[name][day]
                    d[0]+=nll-(-math.log(uniform));d[1]+=1
                masses=defaultdict(float)
                ep=defaultdict(float)
                for comb,p in dist.items():
                    group=bucket(p)
                    z=raw_buckets[name][group]
                    z[0]+=1;z[1]+=p;z[2]+=int(comb==truth)
                    masses[group]+=p
                    if name==PRIMARY or name=="line_role_only":
                        fl=flags(e,comb)
                        for ev in EVENTS:
                            if fl[ev]:ep[ev]+=p
                for group,m in masses.items():
                    raw_buckets[name][group][3]+=m*(1-m)
                    bucket_day[name][day,group]+=int(group==bucket(ptrue))-m
                if name!= "uniform":
                    for ev in EVENTS:
                        pred=ep[ev]
                        obs=int(win_flags[ev])
                        z=raw_events[name][ev]
                        z[0]+=obs;z[1]+=pred;z[2]+=pred*(1-pred);z[3]+=1
                        event_day[name][day,ev]+=obs-pred
            primary=models_dist[PRIMARY]
            all_ticket_record={
                "race_id":rid,"date":day,"riders":n,"true_result":"-".join(truth),
                "probabilities":{"-".join(t):round(p,12) for t,p in sorted(primary.items())}
            }
            export.write(json.dumps(all_ticket_record,ensure_ascii=False,separators=(",",":"))+"\n")
            if sample is None and n==7:
                sorted_items=sorted(primary.items(),key=lambda item:(-item[1],item[0]))
                rank=next(i+1 for i,(k,p) in enumerate(sorted_items) if k==truth)
                sample={
                  "race_id":rid,"race_date":day,"participants":n,
                  "true_result":"-".join(truth),"true_ticket_probability_pct":round(100*primary[truth],8),
                  "true_ticket_rank":rank,"probability_sum":sum(primary.values()),
                  "top_10":[{"ticket":"-".join(k),"probability_pct":round(100*p,7)}
                            for k,p in sorted_items[:10]],
                  "all_210_tickets":[{"ticket":"-".join(k),"probability_pct":round(100*p,9)}
                                     for k,p in sorted(primary.items())]
                }
            if (idx+1)%2000==0:
                print("EVALUATED_2024",idx+1,"/",len(events),flush=True)
    metric={k:pack_metric(v) for k,v in overall.items()}
    bands={}
    for model in MODELS:
        bands[model]={group:bucket_result(raw_buckets[model][group],
                  {day:val for (day,g),val in bucket_day[model].items() if g==group})
                  for group in BUCKETS if raw_buckets[model][group][0]}
    structures={}
    for name in MODELS[1:]:
        structures[name]={ev:structure_result(raw_events[name][ev],
                    {day:val for (day,k),val in event_day[name].items() if k==ev})
                    for ev in EVENTS}
    improvement={}
    n=len(events)
    for name in MODELS[1:]:
        days=per_day_loss_delta[name]
        delta=sum(v[0] for v in days.values())/n  # negative means beats uniform
        diff_std=math.sqrt(sum((v[0]-delta*v[1])**2 for v in days.values()))/n
        improvement[name]={
          "nll_uniform_minus_model":round(-delta,8),
          "delta_daily_cluster_se":round(diff_std,8),
          "delta_95pct_ci": [round(-delta-1.96*diff_std,8),round(-delta+1.96*diff_std,8)],
          "days":len(days)}
    ece={}
    for name in MODELS:
        bins=bands[name]
        total_tickets=sum(x["ticket_count"] for x in bins.values())
        ece[name]=round(sum(x["ticket_count"]*abs(x["empirical_hit_pct"]-x["mean_predicted_pct"])
                            for x in bins.values())/total_tickets,7)
    report={
      "state":"COMPLETED_2024_NO_ODDS_FROZEN_2023",
      "purpose":"All-trifecta independent model probability calibration, not profitability",
      "train_period":["2023-01-01","2023-10-31"],"train_races":19248,
      "test_period":["2024-01-01","2024-12-31"],"test_races":len(events),
      "frozen_formula_source":"research/independent_trifecta_2023/observe.py",
      "frozen_formula_initial_commit":"88a57ccf04e08fce8dc5a1cf11722b2b6e387439",
      "model":"Hierarchically-smoothed conditional race-specific trifecta probabilities; score rank and published line positions; fixed smoothing 40/80",
      "market_odds_features":False,"market_odds_evaluation":False,"prices_read":False,
      "outcomes_used_in_2024_training":False,"2025_and_2026_read":False,
      "number_of_3trifecta_ticket_predictions":sum(s[0] for s in raw_buckets[PRIMARY].values()),
      "coverage":coverage,
      "overall":metric,
      "nll_improvement_vs_uniform":improvement,
      "calibration_bins":bands,
      "ticket_weighted_ece_percentage_points":ece,
      "structural_calibration":structures,
      "by_class":{k:{name:pack_metric(m) for name,m in d.items()} for k,d in sorted(by_class.items())},
      "by_field_size":{k:{name:pack_metric(m) for name,m in d.items()} for k,d in sorted(by_field.items())},
      "by_line_shape_top_12":{k:{name:pack_metric(m) for name,m in by_shape[k].items()}
                          for k in sorted(by_shape,key=lambda k:-by_shape[k][PRIMARY]["races"])[:12]},
      "by_month":{k:{name:pack_metric(m) for name,m in d.items()} for k,d in sorted(by_month.items())},
      "all_tickets_archive":str(ALL_TICKETS.relative_to(ROOT)),
      "all_tickets_archive_note":"All probabilities exported for every eligible 2024 race; archived in Actions artifact, not committed to Git",
      "example_sample":str(SAMPLE.relative_to(ROOT)),
      "limitations":[
        "Published line formation and rider score archives were captured retrospectively; original pre-race timestamp of all inputs not verified.",
        "Use race-ID deduplication and exact results join; missing/malformed cards and unavailable lines excluded. Never extrapolate to excluded population.",
        "Predicted ticket outcomes in the same race are mutually exclusive, not independent. Expected-vs-observed null variance aggregates probabilities at race level.",
        "95% model-reference bands assume between-race independence; additional date-cluster residual SE also reported (approximately robust to within-day dependence).",
        "Low ticket-level ECE does not imply strong high-probability calibration. Inspect the fixed 5%+ bins and structural events separately.",
        "Held-out 2024 used only for evaluation, not fitting. Later changes require a fresh 2025+ holdout.",
        "No profitability/ROI, betting odds, or stakes involved."]}
    AUDIT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    SAMPLE.write_text(json.dumps(sample,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("CALIBRATION_2024_COMPLETE",json.dumps({
       "test_races":len(events),
       "ticket_predictions":report["number_of_3trifecta_ticket_predictions"],
       "nll":{k:v["mean_nll"] for k,v in metric.items()},
       "top1":{k:v["top1_hit_pct"] for k,v in metric.items()},
       "calibration_5_10":bands[PRIMARY].get("5–10%"),
       "calibration_10_20":bands[PRIMARY].get("10–20%"),
       "calibration_20_100":bands[PRIMARY].get("20–100%")
    },ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()

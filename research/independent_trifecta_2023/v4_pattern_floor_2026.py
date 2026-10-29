#!/usr/bin/env python3
"""Research V4: small probability floor WITHIN each frozen V3 line-pattern.

Research hypothesis: V3 concentrates too much probability on favored ordered
triples within each AAA/AAB/ABA/ABB/ABC structural event. Pool extremely small
fractions of V3's probability among SAME-pattern valid triples only.

q(t) = (1-epsilon)*p_V3(t) + epsilon*P_V3(pattern(t))/count(pattern(t)).
For each race, this preserves P(AAA), ..., P(ABC) EXACTLY; all q sum to one.
For all triples q(t)>= (1-epsilon)*p_V3(t); epsilon<=.02, so any ticket
loses at most 2% of its OWN assigned probability. No arbitrary division by 5.
Does NOT use market odds, payout, popularity, or post-race information.

Chronological design:
- V2, V3 original frozen parameters trained ONLY 2023 Jan-Oct.
- Candidate epsilons and nonregression gates specified here BEFORE 2026 test.
- Fit SELECT epsilon on previously inspected 2024 outcomes; report 2025,
  also previously inspected, as a replication check.
- Inspect 2026 ONLY AFTER epsilon has been chosen from 2024; report as first
  out-of-year forward test for this V4 hypothesis, without reoptimizing.
- 2026 test includes available published-line and known-results races only,
  up to Oct 29, 2026. No model promotion if statistical guards fail.
- Exact same V3 (.15 mix) is baseline throughout, never modified.
"""
from __future__ import annotations

import csv, gzip, io, json, math, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2023"))
sys.path.insert(0,str(ROOT/"research/independent_trifecta_2024"))
import observe as original
import line_dependence_v2 as v2
import aab_guarded_v3 as v3
import calibration_audit as year24
import validate_2025 as year25

OUT=ROOT/"data/audits/independent_trifecta_v4_pattern_floor_2026.json"
EPSILONS=(0.,0.0025,0.005,0.01,0.02)
PATS=("AAA","AAB","ABA","ABB","ABC")
TAIL_BINS=("<0.01%","0.01-0.05%","0.05-0.1%")
ALL_BINS=(
    (0.,0.0001,"<0.01%"),
    (0.0001,0.0005,"0.01-0.05%"),
    (0.0005,0.001,"0.05-0.1%"),
    (0.001,0.005,"0.1-0.5%"),
    (0.005,0.01,"0.5-1%"),
    (0.01,0.02,"1-2%"),
    (0.02,0.05,"2-5%"),
    (0.05,0.1,"5-10%"),
    (0.1,0.2,"10-20%"),
    (0.2,1.000001,"20%+"),
)
CUTOFF_2026="2026-10-29"

def bin_of(p):
    for a,b,name in ALL_BINS:
        if a<=p<b:return name
    raise AssertionError(("Invalid ticket p",p))

def files_2026():
    roots=[
       ROOT/"data/a_class_12",ROOT/"data/a3_challenge",
       ROOT/"data/2026_h1",ROOT/"data/2026_future_block1",
    ]
    seen=set()
    for base in roots:
        if base.name in ("a_class_12","a3_challenge"):
            paths=sorted(base.glob("2026_*/races.csv"))
            paths+=sorted(base.glob("pilot_2026*/races.csv"))
        else:
            paths=sorted(base.glob("**/races.csv"))
        for path in paths:
            ep=path.with_name("entries.csv")
            if ep.is_file() and path not in seen:
                seen.add(path)
                with path.open(encoding="utf-8-sig",newline="") as f:
                    races=list(csv.DictReader(f))
                with ep.open(encoding="utf-8-sig",newline="") as f:
                    entries=list(csv.DictReader(f))
                yield str(path.relative_to(ROOT)),races,entries
    for path in sorted((ROOT/"data/grade_races").rglob("*2026*.zip")):
        with zipfile.ZipFile(path) as z:
            names=set(z.namelist())
            for name in sorted(names):
                if not name.endswith("races.csv"):continue
                ep=name[:-9]+"entries.csv"
                if ep not in names:continue
                rr=list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig"))))
                ee=list(csv.DictReader(io.StringIO(z.read(ep).decode("utf-8-sig"))))
                yield str(path.relative_to(ROOT))+":"+name,rr,ee

def ingest_2026():
    truth={}
    with gzip.open(ROOT/"data/result_universe_v2/combined/results_2026.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            record=json.loads(line)
            top=record.get("finish_order")
            if not isinstance(top,list) or len(top)<3:continue
            top=tuple(map(str,top[:3]))
            if len(set(top))==3 and all(x.isdigit() for x in top):
                truth[str(record["race_id"])]=top

    meta={};scores=defaultdict(dict);audit=Counter();sources=Counter()
    for path,races,entries in files_2026():
        selected=set()
        for r in races:
            rid=str(r.get("race_id","")).strip()
            day=str(r.get("race_date","")).strip()
            if rid not in truth or not ("2026-01-01"<=day<=CUTOFF_2026):
                continue
            info={"date":day,"line":str(r.get("predicted_line_formation") or ""),
                  "status":str(r.get("line_status") or ""), "source":path}
            if rid in meta:
                audit["duplicate_manifest"]+=1
                if meta[rid]["status"]!="published" and info["status"]=="published":
                    meta[rid]=info
            else:meta[rid]=info
            selected.add(rid)
        for ent in entries:
            rid=str(ent.get("race_id","")).strip()
            if rid not in selected:continue
            car=str(ent.get("car_no","")).strip()
            if not car.isdigit():continue
            try:
                score=float(ent.get("score") or "")
                if not math.isfinite(score):raise ValueError
            except (ValueError,TypeError):
                audit["missing_or_invalid_score"]+=1
                continue
            if car not in scores[rid]:
                scores[rid][car]=score
                sources[path]+=1
            else:
                audit["duplicate_entry"]+=1

    valid={};by_shape=Counter()
    for rid,m in meta.items():
        e=scores.get(rid,{})
        if len(e)<4 or not set(truth[rid]).issubset(e):
            audit["incomplete_podium_or_cards"]+=1;continue
        if m["status"]!="published":
            audit["unpublished_line"]+=1;continue
        groups=original.parse_line(m["line"],e)
        if groups is None:
            audit["invalid_line_coverage"]+=1;continue
        positions={car:(g+1,pos+1,len(gg))
                   for g,gg in enumerate(groups) for pos,car in enumerate(gg)}
        ranks={car:min(i+1,4) for i,car in enumerate(sorted(e,key=lambda c:(-e[c],int(c))))}
        item={"date":m["date"],"order":truth[rid],"scores":e,
              "groups":groups,"positions":positions,"ranks":ranks}
        valid[rid]=item
        by_shape[v2.line_shape(item)]+=1
    valid=dict(sorted(valid.items(),key=lambda kv:(kv[1]["date"],kv[0])))
    return valid,{
       "verified_2026_result_labels_with_top3":len(truth),
       "matched_metadata_ids_up_to_cutoff":len(meta),
       "usable_2026_races":len(valid),
       "cutoff":CUTOFF_2026,
       "line_shapes":dict(by_shape.most_common()),
       "source_entries":dict(sources),
       "exclusions":dict(audit),
       "data_source_disclaimer":"2026 source coverage incomplete; deduplicated by race ID, published line required"
    }

def candidate_distribution(base, pats, epsilon):
    """Preserve each of five pattern masses while sharing <=2% within pattern."""
    m=defaultdict(float);n=Counter()
    for t,p in base.items():
        m[pats[t]]+=p;n[pats[t]]+=1
    if abs(sum(m.values())-1)>1e-9:
        raise AssertionError("V3 is not normalized")
    uniform={k:m[k]/n[k] for k in m}
    new={t:(1-epsilon)*p+epsilon*uniform[pats[t]] for t,p in base.items()}
    mass=defaultdict(float)
    for t,p in new.items():
        if p<(1-epsilon)*base[t]-1e-15:raise AssertionError("Mass floor violated")
        mass[pats[t]]+=p
    for k in PATS:
        if abs(m[k]-mass[k])>1e-9:raise AssertionError(("Pattern mass changed",k))
    if abs(sum(new.values())-1)>1e-9:raise AssertionError("Total not one")
    return new

def bzero():
    return [0.,0,0,0.,defaultdict(float)]

class Eval:
    def __init__(self):
        self.n=0;self.ticket_count=0
        self.nll=0.;self.brier=0.;self.ptrue=0.
        self.day=defaultdict(lambda:[0.,0.,0])
        self.tail=defaultdict(bzero)
        self.own=defaultdict(bzero)
        self.by_pattern=defaultdict(lambda:[0,0.,0.])
        self.shape_tail=defaultdict(bzero)
    def add(self,e,base,candidate,pats):
        truth=e["order"];n=len(candidate);day=e["date"]
        if truth not in candidate:raise AssertionError("True trifecta absent")
        self.n+=1;self.ticket_count+=n
        ptrue=candidate[truth]
        loss=-math.log(ptrue)
        brier=1-2*ptrue+sum(z*z for z in candidate.values())
        self.nll+=loss;self.brier+=brier;self.ptrue+=ptrue
        z=self.day[day];z[0]+=loss;z[1]+=brier;z[2]+=1
        pz=self.by_pattern[pats[truth]]
        pz[0]+=1;pz[1]+=loss;pz[2]+=brier
        local_fixed=defaultdict(lambda:[0.,0,0])
        local_own=defaultdict(lambda:[0.,0,0])
        locshape=defaultdict(lambda:[0.,0,0])
        shape=v2.line_shape(e)
        for t,q in candidate.items():
            lowname=bin_of(base[t]);newname=bin_of(q)
            win=int(t==truth)
            lf=local_fixed[lowname];lf[0]+=q;lf[1]+=win;lf[2]+=1
            lo=local_own[newname];lo[0]+=q;lo[1]+=win;lo[2]+=1
            if lowname in TAIL_BINS:
                zs=locshape[shape+"|"+lowname]
                zs[0]+=q;zs[1]+=win;zs[2]+=1
        for d,store in ((local_fixed,self.tail),(local_own,self.own),(locshape,self.shape_tail)):
            for key,(expected,actual,tickets) in d.items():
                z=store[key]
                z[0]+=expected;z[1]+=actual;z[2]+=tickets
                z[3]+=max(0.,expected*(1-expected))
                z[4][day]+=actual-expected
        if abs(sum(x[0] for x in local_fixed.values())-1)>1e-8 or sum(x[1] for x in local_fixed.values())!=1:
            raise AssertionError("Ticket probability buckets not exhaustive")

    @staticmethod
    def bin_summary(z):
        expected,actual,tickets,var,byday=z
        days=list(byday.values())
        n=len(days)
        mean=sum(days)/n if n else 0.
        se=math.sqrt(n/(n-1)*sum((x-mean)**2 for x in days)) if n>1 else 0.
        delta=actual-expected
        return {"tickets":tickets,"expected":round(expected,5),"actual":actual,
                "obs_over_exp":round(actual/expected,5) if expected>0 else None,
                "actual_minus_predicted":round(delta,5),
                "mean_predicted_pct":round(100*expected/tickets,7) if tickets else None,
                "observed_pct":round(100*actual/tickets,7) if tickets else None,
                "day_cluster_z":round(delta/se,4) if se else None,
                "expected_below_30":expected<30,
                "days":n}
    def pack(self):
        return {
          "races":self.n,
          "trifecta_combinations":self.ticket_count,
          "NLL":round(self.nll/self.n,9),
          "Brier":round(self.brier/self.n,9),
          "mean_prob_of_true_pct":round(100*self.ptrue/self.n,7),
          "by_realized_pattern_NLL":{
             k:{"realized_events":z[0],"mean_NLL":round(z[1]/z[0],8),
                "mean_Brier":round(z[2]/z[0],8)}
             for k,z in self.by_pattern.items() if z[0]
          },
          "fixed_V3_ticket_cohort_probability_bands":{
            k:self.bin_summary(z) for k,z in sorted(self.tail.items())},
          "candidate_own_probability_bands":{
            k:self.bin_summary(z) for k,z in sorted(self.own.items())},
          "low_tail_by_line_shape_on_fixed_V3_cohort":{
            k:self.bin_summary(z) for k,z in sorted(self.shape_tail.items())}
        }

def paired_ci(base,candidate,idx):
    days=sorted(set(base.day)|set(candidate.day))
    delta=sum(base.day[d][idx]-candidate.day[d][idx] for d in days)/base.n
    se=math.sqrt(sum((base.day[d][idx]-candidate.day[d][idx]-delta*base.day[d][2])**2 for d in days))/base.n
    return {"V3_minus_V4_positive_improvement":round(delta,9),
            "day_cluster_standard_error":round(se,9),
            "approx_95pct_CI":[round(delta-1.96*se,9),round(delta+1.96*se,9)],
            "days":len(days)}

def lowtail_abs_error(e):
    return sum(abs(e.tail[k][1]-e.tail[k][0]) for k in TAIL_BINS)
def lowtail_aggregate_abs_error(e):
    return abs(sum(e.tail[k][1]-e.tail[k][0] for k in TAIL_BINS))
def strict_eligibility(base,cand):
    return {
        "NLL_gain":base.nll-cand.nll,
        "Brier_gain":base.brier-cand.brier,
        "ultra_rare_absolute_error_gain":abs(base.tail["<0.01%"][1]-base.tail["<0.01%"][0])-abs(cand.tail["<0.01%"][1]-cand.tail["<0.01%"][0]),
        "all_rare_sum_absolute_error_gain":lowtail_abs_error(base)-lowtail_abs_error(cand),
        "all_rare_aggregate_absolute_error_gain":lowtail_aggregate_abs_error(base)-lowtail_aggregate_abs_error(cand),
        "passed":bool(base.nll-cand.nll>1e-8
                    and base.brier-cand.brier>=-1e-8
                    and abs(base.tail["<0.01%"][1]-base.tail["<0.01%"][0])-
                        abs(cand.tail["<0.01%"][1]-cand.tail["<0.01%"][0])>0
                    and lowtail_abs_error(base)-lowtail_abs_error(cand)>0
                    and lowtail_aggregate_abs_error(base)-lowtail_aggregate_abs_error(cand)>0)
    }

def evaluate(events,wg,wr,pattern_weights,epsilons,tag):
    scores={x:Eval() for x in epsilons}
    for i,(rid,e) in enumerate(events.items(),1):
        p2=v3.base_distribution(e,wg,wr)
        p3map,pats=v3.variants(e,p2,v3.predict_non_aaa(e,pattern_weights))
        base=p3map[0.15][0]
        for eps in epsilons:
            new=base if eps==0 else candidate_distribution(base,pats,eps)
            scores[eps].add(e,base,new,pats)
        if i%2000==0:
            print("V4_SCORING",tag,i,"complete races",flush=True)
    return scores

def main():
    assert EPSILONS==(0.,0.0025,0.005,0.01,0.02)
    assert original.TRAIN_END=="2023-10-31"
    v3.selftest()
    historical,_=original.ingest()
    training={k:v for k,v in historical.items() if v["date"]<=original.TRAIN_END}
    assert len(training)==19248
    wg,wr=v2.fit_structural(training)
    pw,n_trained=v3.train_pattern_gate(training)
    del training,historical
    print("OLD_V2_V3_MODELS_FROZEN_JANOCT_2023",flush=True)

    # The 2024 observations already informed V3 selection, so this is fit
    # rather than a genuinely blind holdout for the new V4 hypothesis.
    events24,audit24=year24.ingest_2024()
    assert len(events24)==25348
    fit=evaluate(events24,wg,wr,pw,EPSILONS,"2024_FIT")
    passed={eps:strict_eligibility(fit[0.],fit[eps]) for eps in EPSILONS if eps}
    viable=[eps for eps in EPSILONS if eps and passed[eps]["passed"]]
    eps_selected=min(viable,key=lambda v:(fit[v].nll,v)) if viable else 0.
    print("V4_EPS_SELECTED_FROM_2024_ONLY",eps_selected,
          "with",len(viable),"eligible candidates",flush=True)
    del events24

    # 2025 was inspected in prior research and is therefore only confirmation.
    events25,audit25=year25.read_2025()
    assert len(events25)==24635
    confirm=evaluate(events25,wg,wr,pw,(0.,eps_selected) if eps_selected else (0.,),"2025_CONFIRM")
    assert abs(confirm[0.].nll/len(events25)-4.36613863)<1e-5, "2025 frozen V3 reproduction failed"
    confirm_guard=strict_eligibility(confirm[0.],confirm[eps_selected])
    del events25

    # All pre-2026 choices now fixed. No retraining or alternate epsilon
    # selection from the 2026 results is permitted.
    events26,audit26=ingest_2026()
    if len(events26)<3000:
        raise RuntimeError(("2026 current-year result/card coverage insufficient; no selection",len(events26)))
    print("V4_2026_NEVER_FITTED_FORWARD_RACES",len(events26),flush=True)
    test=evaluate(events26,wg,wr,pw,(0.,eps_selected) if eps_selected else (0.,),"2026_UNTOUCHED")
    score_ci={t:{
        "NLL":paired_ci(before[0.],before[eps_selected],0),
        "Brier":paired_ci(before[0.],before[eps_selected],1)}
        for t,before in (("2025",confirm),("2026",test))}
    test_guard=strict_eligibility(test[0.],test[eps_selected])
    accepted=(eps_selected>0 and confirm_guard["passed"] and test_guard["passed"]
              and score_ci["2026"]["NLL"]["approx_95pct_CI"][0]>0)
    report={
      "state":"V4_CONDITIONAL_PATTERN_FLOOR_2026_FORWARD_EVALUATION_COMPLETE",
      "hypothesis":"A tiny per-race within-pattern uniform mixture rescues V3 ultra-rare trifecta probability estimates with at most 2% reduction of any existing ticket's own probability.",
      "formula":"q(t)=(1-epsilon)*p_V3(t)+epsilon*mass_V3(pattern(t))/count_valid_triples(pattern(t))",
      "V3_frozen_source_hash":"04bb3a7956a5a5d43a987194df557838b6ac81e5",
      "2023_train_races":19248,
      "2024_fit_races":25348,
      "2025_confirm_races":24635,
      "2026_first_forward_test_races":len(events26),
      "2026_cutoff":CUTOFF_2026,
      "candidate_epsilon_values":EPSILONS,
      "2024_fit_predeclared_guards":"Improve NLL, avoid worsened Brier, improve same-V3-ticket ultra-rare (<0.01%) absolute error, per-low-bin total absolute error and cumulative <0.1% residual.",
      "2024_fit_candidate_guards":{str(k):v for k,v in passed.items()},
      "2024_epsilon_selected_once":eps_selected,
      "2024_fit_results":{str(k):v.pack() for k,v in fit.items()},
      "2025_reproduction_results":{str(k):v.pack() for k,v in confirm.items()},
      "2025_replication_guard":confirm_guard,
      "2026_forward_results":{str(k):v.pack() for k,v in test.items()},
      "2026_forward_guard":test_guard,
      "independent_score_interval":score_ci,
      "strict_adoption_allowed":bool(accepted),
      "if_not_pass":"KEEP_FROZEN_V3_AS_BEFORE",
      "all_five_pattern_masses_preserved_race_by_race":True,
      "maximum_percentage_reduction_per_ticket":round(100*max(EPSILONS),3),
      "all_trifecta_probs_sum_exactly_one":True,
      "odds_or_market_prices_used":False,
      "2025_2026_used_to_fit_model_or_select_epsilon":False,
      "note_on_future_validation":"2025 previously inspected and not treated as new blind data. 2026 results first evaluated after 2024 selection; further adaptation invalidates 2026 as future untouched evaluation.",
      "limitations":[
        "2024 model selection was based on previously inspected V3 work, so 2024 counts are not an independent discovery validation.",
        "The conditional-uniform component protects pattern totals but changes individual AAA tickets, unlike V3's AAA-ticket conservation relative to V2.",
        "Correcting a tail of probabilities does not demonstrate positive expected ROI versus market odds.",
        "Some conditional score and published line snapshots may have been collected after event; cutoff-time provenance not independently audited.",
        "2026 only includes matched published-line and result snapshots in available sources, and may under-cover S/G graded events.",
        "No zero-2024-2025 results can be inferred as real-world impossibility; test strictly uses observed race outcomes.",
        "Day-cluster intervals are approximate, and multi-bin comparisons are exploratory."
      ],
      "coverage_2024":audit24,"coverage_2025":audit25,"coverage_2026":audit26,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    def tiny(s):
        return {str(k):{"NLL":round(v.nll/v.n,8),"Brier":round(v.brier/v.n,8),
                        "low_tail":{
                          t:{"expected":round(v.tail[t][0],3),"actual":v.tail[t][1]}
                          for t in TAIL_BINS},
                        "pattern_nll":{g:round(z[1]/z[0],6)
                                       for g,z in v.by_pattern.items() if z[0]}}
                for k,v in s.items()}
    print("V4_CONDITIONAL_FLOOR_EXPERIMENT_COMPLETE",json.dumps({
       "epsilon":eps_selected,
       "fit_2024":tiny(fit),
       "confirm_2025":tiny(confirm),
       "test_2026":tiny(test),
       "2025_confirm_guard":confirm_guard,
       "2026_forward_guard":test_guard,
       "forward_CIs":score_ci,
       "strict_accept":accepted,"2026_eligible_races":len(events26)},
       ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

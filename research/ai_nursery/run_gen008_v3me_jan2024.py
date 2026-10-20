"""Chronologically train empirical market error and freeze a NEW December exam.

The examined October period is now *development data*, clearly recorded as
such. Its earlier exam reports remain unchanged. December has never been
used by this candidate to fit, calibrate or choose settings.
"""
from __future__ import annotations

import hashlib,json,math
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_week import ROOT
from source_bridge import joined_races
from gen008_recovered_exam import expanded_joined_races
from gen008_frozen_exam import load_frozen
from parallel_gen008 import pure_pre_result,seal
from gen008_experiment import score_papers,model_digest
from run_parallel_week import forward_compatible_probabilities,revealed_metric
from gen008_v3 import matrix,blend,joint_portfolio,settle_v3
from gen008_v3_market_error import (
 FIRST_FIT,LAST_FIT,FIRST_DIAG,LAST_DIAG,FIRST_TEST,LAST_TEST,
 FEATURE_NAMES,RIDGE,logloss_report,market_features,
 corrected_probability,best_expected_value_stats,portfolio,settle
)

BASE=ROOT/"research/ai_nursery/outputs/GEN-008-V3-MARKET-ERROR"
OUT=BASE/"frozen-jan-01-to-14-2024"
WEB=ROOT/"docs/ai-lab/gen008-v3-market-error-jan-2024.json"
STAGE="GEN008-V3-MARKET-ERROR-FROZEN-JAN-2024"
V3_PARENT=ROOT/"research/ai_nursery/outputs/GEN-008-V3/first-independent-november-2023/v3_checkpoint.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_rows():
    """Keep fit & diagnostic history unchanged; 2024 is only a new forward exam."""
    import gzip
    from source_bridge import load_cards_year,stream_podium
    from train_week import eligible
    from recover_gen008_v3me_jan2024_odds import (
       PRICES,REPORT,validate_books,fingerprint,FIRST,LAST)
    original=joined_races(2023,"2023-09-15","2023-09-30")
    sep,sep_source=original
    octob,oct_source=expanded_joined_races(2023,"2023-10-01","2023-10-14")
    assert len(octob)==917
    early=sep+octob
    early.sort(key=lambda r:(r["race"]["race_date"],r["race"].get("start_time",""),
        r["race"].get("track",""),int(r["race"].get("race_no") or 0),r["race_id"]))
    train=[r for r in early if FIRST_FIT<=r["race"]["race_date"]<=LAST_FIT]
    diagnostic=[r for r in early if FIRST_DIAG<=r["race"]["race_date"]<=LAST_DIAG]
    assert len(train)+len(diagnostic)==len(early)
    assert len(train)==761 and len(diagnostic)==473
    testing,test_source=joined_races(2024,FIRST,LAST)
    unfilled=json.loads(REPORT.read_text(encoding="utf-8"))
    assert unfilled["target_dates"]==[FIRST,LAST]
    assert unfilled["no_result_labels_read"] and unfilled["no_imputed_odds"]
    assert unfilled["books_sha256"]==fingerprint(PRICES)
    cards,_,_=load_cards_year(2024,FIRST,LAST)
    outcomes,_=stream_podium(2024,set(cards))
    old_ids={r["race_id"] for r in testing}
    recovered={}
    with gzip.open(PRICES,"rt",encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():continue
            obj=json.loads(line)
            rid=obj["race_id"]
            assert rid in cards and rid not in old_ids and rid not in recovered
            assert obj["phase"]=="FINAL"
            validate_books(rid,sorted(cards[rid]["entries"]),obj["odds"])
            recovered[rid]=obj
    assert len(recovered)==unfilled["recovered_total"]
    added=0;no_label=0;invalid=0
    for rid,obj in sorted(recovered.items()):
        if rid not in outcomes:
            no_label+=1
            continue
        row={"race_id":rid,**cards[rid],"odds":obj["odds"],"outcome":outcomes[rid]}
        if not eligible(row):
            invalid+=1
            continue
        testing.append(row);added+=1
    testing.sort(key=lambda r:(r["race"]["race_date"],r["race"].get("start_time",""),
          r["race"].get("track",""),int(r["race"].get("race_no") or 0),r["race_id"]))
    assert len(testing)==len(old_ids)+added
    assert len({r["race_id"] for r in testing})==len(testing)
    assert {r["race_id"] for r in early}.isdisjoint(r["race_id"] for r in testing)
    test_source={**test_source,
      "complete_races":len(testing),
      "joined_race_ids_sha256":hashlib.sha256("\\n".join(r["race_id"] for r in testing).encode()).hexdigest(),
      "join_status":{**test_source["join_status"],"joined_and_eligible":len(testing)},
      "expanded_original_odds":{
        "original_available_races":len(old_ids),"newly_recovered_books":len(recovered),
        "additional_eligible":added,"missing_labels":no_label,"invalid_race":invalid,
        "market_parse_failures":unfilled["remaining_unrecovered"],
        "original_sha256":unfilled["books_sha256"],"no_imputation":True}}
    print("GEN008_V3ME_JAN_SOURCE_READY",json.dumps({
       "races":len(testing),"added":added,"market_missing":unfilled["remaining_unrecovered"]}),flush=True)
    return train,diagnostic,testing,{"september":sep_source,"october":oct_source,"january_2024":test_source}


def frozen_fit(rows, weights, temperature):
    """Immutable frozen September/October fitted market-error coefficients."""
    path=ROOT/"research/ai_nursery/outputs/GEN-008-V3-MARKET-ERROR/dec-2023-expanded-authentic-prices/market_error_fit.json"
    frozen=read(path)
    assert frozen["status"]=="FIT_COMPLETE" and frozen["train_races"]==761
    assert frozen["train_dates"]==[FIRST_FIT,LAST_FIT]
    assert frozen["future_december_labels_used_in_fit"] is False
    assert not frozen.get("no_october_holdout_claim") is False
    assert frozen["input_NEW_weights_sha256"]==model_digest(weights)
    assert float(frozen["input_NEW_temperature"])==temperature
    return frozen


def summarize(records):
    comparison={}
    for branch in ("old","new","v3","v3_market_error"):
        x=[r for r in records if r["branch"]==branch]
        n=len(x)
        spent=sum(r["actual"]["stake_yen"] for r in x)
        pay=sum(r["actual"]["payout_yen"] for r in x)
        comparison[branch]={
            "races":n,"entered":sum(r["entered"] for r in x),
            "skipped":sum(not r["entered"] for r in x),
            "hit_races":sum(r["actual"]["hit"] for r in x),
            "marks":sum(r["actual"]["marks"] for r in x),
            "stake_yen":spent,"payout_yen":pay,"net_yen":pay-spent,
            "roi_pct":100*pay/spent if spent else None,
            "mean_trifecta_logloss":sum(r["trifecta_logloss"] for r in x)/n,
            "mean_trio_logloss":sum(r["trio_logloss"] for r in x)/n}
    return comparison


def run():
    if OUT.exists() or WEB.exists():
        raise RuntimeError("Immutable market-error result files already exist")
    prior,snap=load_frozen()
    v3_checkpoint=read(V3_PARENT)
    assert v3_checkpoint["model"]=="GEN-008-V3" if "model" in v3_checkpoint else v3_checkpoint["id"]=="GEN-008-V3"
    assert v3_checkpoint["parent_new_prediction"]==snap["new"]["prediction"]
    assert v3_checkpoint["parent_new_policy"]==snap["new"]["behavior"]
    alpha=float(v3_checkpoint["alpha"])
    train,diagnostics,testing,reports=source_rows()
    learned=frozen_fit(train,prior["new"].weights,prior["new"].temperature)
    learned["input_NEW_weights_sha256"]=snap["new"]["prediction"]
    learned["input_NEW_policy_sha256"]=snap["new"]["behavior"]
    learned["input_NEW_temperature"]=snap["new"]["temperature"]
    learned["frozen_v3_alpha"]=alpha
    learned["used_previously_examined_october_as_development"]=True
    learned["september_training_and_october_training_are_only_development"]=True
    learned["january_2024_untouched_forward_period"]=["2024-01-01","2024-01-14"]
    learned["diagnostic_races"]=len(diagnostics)
    learned["diagnostic_dates"]=[FIRST_DIAG,LAST_DIAG]
    learned["diagnostic_no_refit"]=logloss_report(
        diagnostics,learned["coefficients"],prior["new"].weights,prior["new"].temperature,alpha)
    learned["train_logloss"]=logloss_report(
        train,learned["coefficients"],prior["new"].weights,prior["new"].temperature,alpha)
    learned["algorithm_sha256"]=hashlib.sha256(
        json.dumps({
          "features":learned["feature_names"],"params":learned["coefficients"],
          "ridge":RIDGE,"parent":snap["new"]["prediction"]
        },sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert learned["algorithm_sha256"]==read(ROOT/"research/ai_nursery/outputs/GEN-008-V3-MARKET-ERROR/dec-2023-expanded-authentic-prices/market_error_fit.json")["algorithm_sha256"]
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"market_error_fit.json").write_text(json.dumps(learned,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    allresults=[]
    byday=defaultdict(list)
    selected_counts={"race_with_positive_ev":0,"positive_trifecta_tickets":0,
                     "positive_trio_tickets":0,"max_ev":None}
    with (OUT/"decisions_sealed_before_result.jsonl").open("x",encoding="utf-8") as decisions, \
         (OUT/"settlements_after_result.jsonl").open("x",encoding="utf-8") as revealed:
        for i,row in enumerate(testing,1):
            pre=pure_pre_result(row)
            controls,controlpayload={},{}
            for name in ("old","new"):
                controls[name],controlpayload[name]=prior[name].precommit(pre)
                assert controls[name]["prediction_before_sha256"]==snap[name]["prediction"]
                assert controls[name]["policy_before_sha256"]==snap[name]["behavior"]
            legacy_v3=joint_portfolio(pre,prior["new"].weights,prior["new"].temperature,alpha)
            features=market_features(pre,prior["new"].weights,prior["new"].temperature)
            q=corrected_probability(features,learned["coefficients"])
            choice=portfolio(pre,features,q)
            ev=best_expected_value_stats(pre,features,q)
            if ev["positive_3rentan_tickets"] or ev["positive_3renpuku_tickets"]:
                selected_counts["race_with_positive_ev"]+=1
            selected_counts["positive_trifecta_tickets"]+=ev["positive_3rentan_tickets"]
            selected_counts["positive_trio_tickets"]+=ev["positive_3renpuku_tickets"]
            selected_counts["max_ev"]=max(selected_counts["max_ev"] or -1.,ev["max_3rentan_ev_per_yen"],ev["max_3renpuku_ev_per_yen"])
            seal(decisions,{
                "stage":STAGE,"sequence":i,"date":pre["race"]["race_date"],
                "race_id":pre["race_id"],"market_model_sha256":learned["algorithm_sha256"],
                "input_model_sha256":snap["new"]["prediction"],
                "pre_result":True,"old":controls["old"],"new":controls["new"],
                "v3":legacy_v3,"v3_market_error":choice,
                "estimated_ev_summary":ev
            })
            # Result is not accessed before all four branches have been sealed.
            finish=row["outcome"]
            for branch in ("old","new","v3","v3_market_error"):
                if branch in ("old","new"):
                    paid=score_papers(controls[branch]["pages"],pre["odds"],finish)
                    probs=forward_compatible_probabilities(controlpayload[branch][0],controlpayload[branch])
                    ll=revealed_metric(finish,probs)
                elif branch=="v3":
                    paid=settle_v3(legacy_v3,finish)
                    keys,pmod,m=matrix(pre,prior["new"].weights,prior["new"].temperature)
                    prob=blend(pmod,m,alpha)
                    trioq=sum(float(p) for key,p in zip(keys,prob) if
                        set(key.split("-"))==set(map(str,finish)))
                    ll={"trifecta_logloss":-math.log(max(float(prob[keys.index("-".join(map(str,finish)))]),1e-50)),
                        "trio_logloss":-math.log(max(trioq,1e-50))}
                else:
                    paid=settle(choice,finish)
                    idx=features["keys"].index("-".join(map(str,finish)))
                    trioix=features["trio_group"][idx]
                    trioq=np.bincount(features["trio_group"],weights=q,minlength=len(features["trio_names"]))
                    ll={"trifecta_logloss":-math.log(max(float(q[idx]),1e-50)),
                        "trio_logloss":-math.log(max(float(trioq[trioix]),1e-50))}
                recorded={
                    "stage":STAGE,"sequence":i,"date":pre["race"]["race_date"],
                    "race_id":pre["race_id"],"branch":branch,"entered":paid["stake_yen"]>0,
                    "finish_only_after_seal":list(finish),
                    "actual":paid,**ll,"real_money_bet":False}
                seal(revealed,recorded)
                allresults.append(recorded)
                byday[recorded["date"]].append(recorded)
            if i%100==0:
                print("MARKET_ERROR_FROZEN_JAN24_PROGRESS",i,len(testing),flush=True)
    ending={name:{"prediction":model_digest(br.weights),
                   "behavior":br.policy.digest(),"temperature":br.temperature}
            for name,br in prior.items()}
    assert ending==snap,"Read-only frozen exam changed control weights or policy"
    comparison=summarize(allresults)
    daily=[{"date":date,"races":len(z)//4,
            **{name+"_net":sum(v["actual"]["net_yen"] for v in z if v["branch"]==name)
               for name in comparison}}
            for date,z in sorted(byday.items())]
    model_ll=learned["diagnostic_no_refit"]["mean_logloss"]
    summary={
        "stage":STAGE,"status":"COMPLETE",
        "model_objective":"Infer systematic past-result/market-pricing residuals without treating odds outliers as proven edge",
        "train_period":[FIRST_FIT,LAST_FIT],"train_races":len(train),
        "diagnostic_period":[FIRST_DIAG,LAST_DIAG],"diagnostic_races":len(diagnostics),
        "test_period":["2024-01-01","2024-01-14"],"test_races":len(testing),
        "former_october_holdout_now_dev":True,
        "december_outcomes_never_used_to_fit":True,
        "january_2024_outcomes_never_used_to_fit":True,
        "frozen_prior_market_error_parameters":True,
        "pre_result_decisions_sealed_before_outcomes":True,
        "old_new_v3_frozen_unchanged":True,
        "same_races_for_all_four":True,
        "historical_final_odds_only":True,"actual_real_bets":False,
        "market_baseline_and_no_refit_diagnostic":model_ll,
        "fitted_model_sha256":learned["algorithm_sha256"],
        "market_mispricing_detection":selected_counts,
        "frozen_parent_start":snap,"frozen_parent_end":ending,
        "source_reports":reports,"comparison":comparison,"daily":daily,
        "limitation":"A statistically learned residual is not a demonstrated profitable exploitable market error. This is paper settlement on historical FINAL prices only."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    WEB.parent.mkdir(parents=True,exist_ok=True)
    WEB.write_text(json.dumps({
      "stage":STAGE,"status":"COMPLETE","test_races":len(testing),
      "comparison":comparison,"market_mispricing_detection":selected_counts,
      "source":str((OUT/"summary.json").relative_to(ROOT)),
      "historical_final_odds_only":True},ensure_ascii=False,indent=2)+"\n")
    print("GEN008_V3_MARKET_ERROR_FROZEN_EXAM_DONE",
          json.dumps({"fit":len(train),"diagnostic":len(diagnostics),
          "test":len(testing),"coefs":learned["coefficients"],
          "diagnostic":model_ll,"detection":selected_counts,
          "comparison":comparison},ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()

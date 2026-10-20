"""GEN-008 parallel laboratory: train OLD and NEW side by side, when requested.

CREATION ONLY: importing or invoking with no --run-training does not train.
Both branches inherit identical frozen GEN-007-C2 prediction and action weights.
Control = unchanged GEN-007 learning rules.
Candidate = shadow opportunity, coherent markets and past-only calibration.
Pre-result decisions and shadow tickets are sealed to disk BEFORE results.
Chronological training: 2023-08-01 .. 2023-09-30.
Frozen later examination: 2023-10-01 .. 2023-10-14.
This script intentionally never overwrites any old model or its archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from train_week import ROOT,D,LR,feature_matrix,softmax
from behavior_policy import make_learned_sheets,learn_from_race,BehaviorPolicy
from profit_objective import ProfitPolicy,signed_ticket_updates
from source_bridge import joined_races_for_training,joined_races
from gen008_experiment import (
    ShadowProfitPolicy,inherited_shadow_policy,coherent_probabilities,
    calibration_metrics,update_temperature,make_shadow_sheet,score_papers,
    race_seed,model_digest,
)

PARENT=ROOT/"research/ai_nursery/outputs/GEN-007/continuation-2m"
OUT=ROOT/"research/ai_nursery/outputs/GEN-008-PARALLEL"
TRAIN_START="2023-08-01"
TRAIN_END="2023-09-30"
TEST_START="2023-10-01"
TEST_END="2023-10-14"
COST=100
VERSION="GEN-008-PARALLEL-CREATED"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seal(f,obj):
    f.write(json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    f.flush()
    os.fsync(f.fileno())


def pure_pre_result(row):
    """No label/payout can reach either branch's prediction code."""
    return {"race_id":row["race_id"],"entries":row["entries"],
            "race":row["race"],"odds":row["odds"]}


def load_both_parents():
    saved=read_json(PARENT/"checkpoint.json")
    if saved.get("stage_id")!="GEN-007-C2" or saved.get("after_step")!=5328:
        raise RuntimeError("Frozen GEN-007-C2 parent not verified")
    original=BehaviorPolicy.load(PARENT/"behavior_policy.json")
    if original.digest()!=saved["behavior_policy_sha256"]:
        raise RuntimeError("Inherited action policy digest incorrect")
    w=np.asarray(saved["weights"],dtype=np.float64)
    if w.shape!=(D,) or model_digest(w)!=saved["weights_sha256"]:
        raise RuntimeError("Inherited predictor digest incorrect")
    baseline=ProfitPolicy.inherit(PARENT/"behavior_policy.json")
    candidate=inherited_shadow_policy(PARENT/"behavior_policy.json")
    if baseline.digest()!=candidate.digest():
        raise RuntimeError("Two children do not have identical initial action weights")
    return saved,w,baseline,candidate


@dataclass
class Branch:
    name:str
    weights:np.ndarray
    policy:ProfitPolicy
    temperature:float=1.0

    def precommit(self,pre):
        arms=feature_matrix(pre["entries"],pre["odds"])
        X=np.stack([a["x"] for a in arms])
        prior=np.array([a["prior"] for a in arms],dtype=np.float64)
        if self.name=="old":
            probabilities=softmax(X@self.weights+prior)
            odds_probabilities=None
        else:
            probabilities,odds_probabilities=coherent_probabilities(
                arms,self.weights,self.temperature)
        rng=np.random.default_rng(race_seed(pre["race_id"],"actual-decision"))
        pages,decisions=make_learned_sheets(
            arms,probabilities,pre["odds"],pre["entries"],10000,rng,self.policy)
        shadow=(make_shadow_sheet(arms,probabilities,pre["odds"],pre["entries"],
                                  pre["race_id"]) if self.name=="new" else None)
        chosen={
            "id":pre["race_id"],"branch":self.name,
            "parent":VERSION,"prediction_before_sha256":model_digest(self.weights),
            "policy_before_sha256":self.policy.digest(),
            "temperature_before":float(self.temperature),
            "pages":pages,"decisions":[{"kind":d.kind,"action":d.action,
                "probability":d.probability,"context":d.context,
                "sheet_index":d.sheet_index} for d in decisions],
            "shadow_page":shadow,
            "shadow_is_not_spent":True,
            "actual_paper_stake_yen":sum(page["total_yen"] for page in pages),
            "actual_paper_marks":sum(len(page["picks"]) for page in pages),
        }
        return chosen,(arms,probabilities,decisions,odds_probabilities,pages,shadow)

    def settle_and_learn(self,pre,finish,payload,decision_record):
        arms,mix,decisions,odds_probs,pages,shadow=payload
        actual=score_papers(pages,pre["odds"],finish)
        correct={("3連単","-".join(map(str,finish))),
                 ("3連複","=".join(map(str,sorted(finish))))}
        X=np.stack([a["x"] for a in arms])
        if self.name=="old":
            winning_idx=[i for i,a in enumerate(arms)
                         if (a["market"],a["selection"]) in correct]
            if len(winning_idx)!=2:raise AssertionError("missing winning market")
            loss=float(-sum(math.log(max(float(mix[j]),1e-30))
                            for j in winning_idx)/2)
            cls_delta=LR*(X[winning_idx].mean(axis=0)-mix@X)
            ticket_delta,credits=signed_ticket_updates(arms,mix,pages,correct)
            actions=learn_from_race(self.policy,decisions,
                                    page_increment_nets(pages,pre["odds"],finish),
                                    actual["net_yen"])
            metrics={"legacy_two_market_joint_logloss":loss}
            skipped_shadow=None
        else:
            assert odds_probs is not None and shadow is not None
            metrics=calibration_metrics(odds_probs,finish)
            win="-".join(map(str,finish))
            idx=next(i for i,a in enumerate(arms)
                     if a["market"]=="3連単" and a["selection"]==win)
            triple_idx=[i for i,a in enumerate(arms) if a["market"]=="3連単"]
            q=np.asarray([odds_probs["3連単"][arms[i]["selection"]]
                          for i in triple_idx],dtype=float)
            cls_delta=LR*(X[idx]-q@X[triple_idx])
            # Shadow is an independent precommitted one-sheet opportunity,
            # evaluated on EVERY race regardless of whether actual entered.
            shadow_score=score_papers([shadow],pre["odds"],finish)
            skipped_shadow=(shadow_score["net_yen"] if not pages else None)
            feedback_pages=pages if pages else [shadow]
            ticket_delta,credits=signed_ticket_updates(
                arms,mix,feedback_pages,correct)
            increments=page_increment_nets(pages,pre["odds"],finish)
            actions=[]
            for d in decisions:
                if d.kind=="enter":
                    actions.append(self.policy.update_enter_shadow(
                        d,shadow_score["net_yen"]))
                else:
                    realized=increments[d.sheet_index] if d.action else 0
                    actions.append(self.policy.update(d,realized))
            # This update knows only the finish of the race just settled.
            self.temperature=update_temperature(
                self.temperature,arms,self.weights.copy(),finish)
            metrics.update({
                "shadow_net_yen":shadow_score["net_yen"],
                "shadow_stake_yen":shadow_score["stake_yen"],
                "shadow_missed_profit_if_skipped_yen":skipped_shadow,
                "shadow_used_as_real_stake":False,
                "temperature_after":self.temperature,
            })
        self.weights+=cls_delta+ticket_delta
        if not np.isfinite(self.weights).all():
            raise RuntimeError("non-finite updated model")
        return {
            "race_id":pre["race_id"],"branch":self.name,
            "entered":bool(pages),"actual":actual,
            "reported_net_yen":actual["net_yen"],
            "updates":actions,"ticket_credit_count":len(credits),
            "prediction_after_sha256":model_digest(self.weights),
            "policy_after_sha256":self.policy.digest(),
            **metrics
        }


def page_increment_nets(pages,odds,finish):
    return [score_papers([p],odds,finish)["net_yen"] for p in pages]


def frozen_score(br,pre,finish,payload):
    """Frozen forward evaluation; no weights, temperature or policy updates."""
    _,_,_,q,pages,shadow=payload
    actual=score_papers(pages,pre["odds"],finish)
    stat={"race_id":pre["race_id"],"branch":br.name,
          "entered":bool(pages),"actual":actual,
          "model_hash":model_digest(br.weights),
          "policy_hash":br.policy.digest()}
    if br.name=="new":
        stat.update(calibration_metrics(q,finish))
        shadow_score=score_papers([shadow],pre["odds"],finish)
        stat["shadow_opportunity_net_yen"]=shadow_score["net_yen"]
        stat["shadow_missed_profitable_if_skipped"]=bool(
            not pages and shadow_score["net_yen"]>0)
    else:
        arms,mix,_,_,_,_=payload
        pi={a["selection"]:mix[i] for i,a in enumerate(arms)
            if a["market"]=="3連単"}
        pt=pi["-".join(map(str,finish))]/max(1e-30,sum(pi.values()))
        stat["trifecta_logloss"]=-math.log(max(pt,1e-30))
    return stat


def aggregate(rows):
    out={}
    for name in ("old","new"):
        x=[r for r in rows if r["branch"]==name]
        stake=sum(r["actual"]["stake_yen"] for r in x)
        paid=sum(r["actual"]["payout_yen"] for r in x)
        assert len(x)*2==len(rows)
        out[name]={
            "races":len(x),"entered":sum(r["entered"] for r in x),
            "skipped":sum(not r["entered"] for r in x),
            "actual_staked_yen":stake,"actual_payout_yen":paid,
            "actual_net_yen":paid-stake,
            "paper_roi_percent":round(100*paid/stake,3) if stake else None,
            "hit_races":sum(r["actual"]["hit"] for r in x),
            "trifecta_logloss":round(sum(r["trifecta_logloss"] for r in x)/len(x),6),
        }
        if name=="new":
            out[name]["shadow_profitable_skips"]=sum(
                r["shadow_missed_profitable_if_skipped"] for r in x)
            out[name]["shadow_opportunity_net_yen"]=sum(
                r["shadow_opportunity_net_yen"] for r in x)
    return out


def run():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("A parallel experiment already exists; never overwrite evidence")
    assert date.fromisoformat(TRAIN_START)>date(2023,6,11)
    assert date.fromisoformat(TRAIN_END)<date.fromisoformat(TEST_START)
    assert date.fromisoformat(TEST_START)<date.fromisoformat(TEST_END)
    training,train_sources=joined_races_for_training(TRAIN_START,TRAIN_END)
    holdout,holdout_sources=joined_races(2023,TEST_START,TEST_END)
    if not training or not holdout:
        raise RuntimeError("Empty exact-ID training or independent test source")
    if {r["race_id"] for r in training}&{r["race_id"] for r in holdout}:
        raise RuntimeError("Training/test race ID leakage")
    parent,w,old_policy,new_policy=load_both_parents()
    current=Branch("old",w.copy(),old_policy)
    improved=Branch("new",w.copy(),new_policy)
    assert model_digest(current.weights)==model_digest(improved.weights)
    OUT.mkdir(parents=True,exist_ok=False)
    train_stats=[]
    with (OUT/"sealed_train_decisions.jsonl").open("x",encoding="utf-8") as paper,\
         (OUT/"revealed_train_results.jsonl").open("x",encoding="utf-8") as ledger:
        for row in training:
            pre=pure_pre_result(row)
            old_p,old_payload=current.precommit(pre)
            new_p,new_payload=improved.precommit(pre)
            # IMPORTANT: must seal both records BEFORE accessing row[outcome].
            seal(paper,{"race_id":pre["race_id"],"date":pre["race"]["race_date"],
                        "old":old_p,"new":new_p})
            finish=row["outcome"]
            for model,payload,decision in (
                (current,old_payload,old_p),(improved,new_payload,new_p)):
                settled=model.settle_and_learn(pre,finish,payload,decision)
                train_stats.append(settled)
                seal(ledger,settled)
    saved={}
    for model in (current,improved):
        sub=OUT/model.name
        sub.mkdir()
        weights_obj={
            "model":model.name,"parent_checkpoint_sha256":parent["weights_sha256"],
            "parent_policy_sha256":parent["behavior_policy_sha256"],
            "after_step":parent["after_step"]+len(training),
            "trained_dates":[TRAIN_START,TRAIN_END],
            "weights":model.weights.tolist(),
            "weights_sha256":model_digest(model.weights),
            "temperature":model.temperature,
            "source":"independent frozen GEN-007-C2 cloned weights"
        }
        (sub/"checkpoint.json").write_text(json.dumps(weights_obj,ensure_ascii=False)+"\n")
        model.policy.save(sub/"behavior_policy.json")
        saved[model.name]={
            "weights_sha256":weights_obj["weights_sha256"],
            "policy_sha256":model.policy.digest(),
            "temperature":model.temperature,
        }
    frozen_pre={x.name:(model_digest(x.weights),x.policy.digest(),x.temperature)
                for x in (current,improved)}
    test_stats=[]
    with (OUT/"sealed_forward_decisions.jsonl").open("x",encoding="utf-8") as paper,\
         (OUT/"revealed_forward_results.jsonl").open("x",encoding="utf-8") as ledger:
        for row in holdout:
            pre=pure_pre_result(row)
            a,pa=current.precommit(pre)
            b,pb=improved.precommit(pre)
            seal(paper,{"race_id":pre["race_id"],"date":pre["race"]["race_date"],
                        "old":a,"new":b})
            finish=row["outcome"]
            for model,payload in ((current,pa),(improved,pb)):
                result=frozen_score(model,pre,finish,payload)
                test_stats.append(result)
                seal(ledger,result)
    assert frozen_pre=={x.name:(model_digest(x.weights),x.policy.digest(),x.temperature)
                        for x in (current,improved)},"Frozen examination modified a model"
    report={
        "experiment":"GEN-008-PARALLEL",
        "status":"TRAIN_AND_FROZEN_EXAM_COMPLETED",
        "control":"identical inherited GEN-007 algorithm with continued training",
        "candidate":"shadow-on-skip + coherent market probabilities + online past-only temperature",
        "both_started_with_same_weights":True,
        "parent_weights_sha256":parent["weights_sha256"],
        "parent_policy_sha256":parent["behavior_policy_sha256"],
        "training_dates":[TRAIN_START,TRAIN_END],
        "forward_exam_dates":[TEST_START,TEST_END],
        "train_races":len(training),"forward_exam_races":len(holdout),
        "actual_training_paper_profit":{
            name:sum(t["actual"]["net_yen"] for t in train_stats if t["branch"]==name)
            for name in ("old","new")
        },
        "training_archive_reports":train_sources,
        "forward_archive_report":holdout_sources,
        "checkpoint_digests":saved,
        "frozen_forward_comparison":aggregate(test_stats),
        "no_real_bets":True,
        "historical_final_odds_only":True,
        "shadow_not_counted_as_real_cashflow":True,
        "old_GEN007_records_modified":False,
        "warning":"Single pre-sealed shadow page does not reveal the profit of ALL possible skipped tickets. Forward experiment cannot guarantee future profit."
    }
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("GEN008_PARALLEL_FROZEN_EXAM_COMPLETE",json.dumps({
        "train_races":len(training),"holdout_races":len(holdout),
        "comparison":report["frozen_forward_comparison"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-training",action="store_true",
                        help="Explicitly launch both children; DOES NOT modify GEN-007")
    args=parser.parse_args()
    if not args.run_training:
        print("GEN-008 created; no training launched. Pass --run-training explicitly.")
    else:
        run()

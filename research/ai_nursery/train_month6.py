"""GEN-006: one calendar month of sequential paper-only keirin learning.

2023-02-05 through 2023-03-04, inclusive (28 days). GEN-005 weights,
virtual wallet, accumulated rewards and stress accounting are inherited.
New: a genuinely updated ENTER/SKIP and CONTINUE/STOP policy, with per-mark
signed gain/loss credit. No racing tactics, fixed formation, paper limits or
optimized rules are taught. The June 1-7 frozen holdout is never used here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import date,timedelta

import numpy as np

from train_week import ROOT,WEB,D,LR,load_cards,load_odds,load_results,eligible,feature_matrix,softmax
from train_week3 import spend_marks
from train_week5 import award_excess
from behavior_policy import (
    BehaviorPolicy,make_learned_sheets,mark_credit_weights,learn_from_race
)

GEN="GEN-006"
PARENT="GEN-005"
PARENT_DIR=ROOT/"research/ai_nursery/outputs/GEN-005"
OUT=ROOT/"research/ai_nursery/outputs/GEN-006"
FIRST=date(2023,2,5)
LAST=date(2023,3,4)
EXAM_START=date(2023,6,1)
SEED=20261026
COST=100
REFILL=10000


def digest(w):
    return hashlib.sha256(w.tobytes()).hexdigest()


def write_sealed(f,obj):
    f.write(json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n")
    f.flush()
    os.fsync(f.fileno())


def full_calendar():
    cards,_=load_cards()
    raw={rid:obj for rid,obj in cards.items()
         if FIRST.isoformat()<=obj["race"].get("race_date","")<=LAST.isoformat()}
    results=load_results()
    odds=load_odds(set(raw).intersection(results))
    excluded=Counter()
    races=[]
    for rid,obj in raw.items():
        if rid not in results:
            excluded["missing_result"]+=1
            continue
        if rid not in odds:
            excluded["missing_complete_odds"]+=1
            continue
        row={**obj,"odds":odds[rid],"outcome":results[rid],"race_id":rid}
        if not eligible(row):
            excluded["incomplete_or_inconsistent"]+=1
            continue
        races.append(row)
    races.sort(key=lambda row:(
        row["race"]["race_date"],row["race"].get("start_time",""),
        row["race"].get("track",""),int(row["race"].get("race_no","0") or 0),
        row["race_id"]
    ))
    assert LAST<EXAM_START, "Training overlaps reserved common exam"
    if not races:raise RuntimeError("No full-odds, full-card, actual-finish joined races")
    return raw,races,excluded


def choose_wallet_reward(bank,spent,refills,paid):
    """Audit-friendly, exact inherited 10k replenishment and reward separation."""
    post_marks=bank+len(refills)*REFILL-spent
    assert post_marks>=0
    after_payout=post_marks+paid
    new_bank,reward=award_excess(after_payout)
    assert new_bank+reward==bank+len(refills)*REFILL-spent+paid
    return new_bank,reward


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("GEN-006 already exists: refusing to overwrite learned history")
    parent=json.loads((PARENT_DIR/"checkpoint.json").read_text(encoding="utf-8"))
    if parent.get("id")!=PARENT or parent.get("after_step")!=2309 or parent.get("training_span",[])[-1]!="2023-02-04":
        raise RuntimeError("GEN-005 parent mismatch")
    weights=np.asarray(parent["weights"],dtype=np.float64)
    if weights.shape!=(D,) or digest(weights)!=parent["weights_sha256"]:
        raise RuntimeError("GEN-005 weights not intact")
    parent_hash=digest(weights)
    bank=int(parent["bankroll_balance_yen"])
    inherited_bank=bank
    old_reward_fund=int(parent.get("cumulative_reward_wallet_yen",0))
    bank,opening_extra=award_excess(bank)
    earned=0
    reward_account=old_reward_fund+opening_extra
    funds_added=0
    stress_events=0
    policy=BehaviorPolicy()
    initial_policy_hash=policy.digest()
    rng=np.random.default_rng(SEED)
    raw,races,excluded=full_calendar()
    OUT.mkdir(parents=True,exist_ok=True)
    WEB.mkdir(parents=True,exist_ok=True)

    journal=[]
    by_kind=Counter()
    paper_counts=Counter()
    with (OUT/"action_decisions.jsonl").open("x",encoding="utf-8") as decision_file, \
         (OUT/"formation_mark_sheets.jsonl").open("x",encoding="utf-8") as sheets_file, \
         (OUT/"learning_history.jsonl").open("x",encoding="utf-8") as learning_file:
        for i,row in enumerate(races,1):
            race=row["race"]
            w_before=digest(weights)
            p_before=policy.digest()
            pre_wallet=bank
            arms=feature_matrix(row["entries"],row["odds"])
            X=np.stack([a["x"] for a in arms])
            probability=softmax(X@weights+np.asarray([a["prior"] for a in arms]))
            paper,decisions=make_learned_sheets(arms,probability,row["odds"],row["entries"],bank,rng,policy)
            exposure=Counter((s["type"],pick) for s in paper for pick in s["picks"])
            post_paper,spent,refills=spend_marks(paper,bank)
            assert spent==100*sum(exposure.values())
            assert len(decisions)>=1 and decisions[0].kind=="enter"
            assert (not paper)==(not decisions[0].action)
            if paper:
                assert len(paper)>0
            else:
                assert not refills and spent==0
            statement={
                "generation":GEN,"parent":PARENT,"generation_step":i,
                "step":parent["after_step"]+i,
                "race_id":row["race_id"],"date":race["race_date"],
                "track":race.get("track"),"race_no":race.get("race_no"),
                "race_type":race.get("race_type"),
                "weights_before_sha256":w_before,
                "policy_hash_before":p_before,
                "wallet_before_yen":pre_wallet,
                "decisions":[d.json() for d in decisions],
                "skip":not paper,
                "marks":sum(exposure.values()),
                "sheet_count":len(paper),
                "paper_stake_yen":spent,
                "refills_before_result":refills
            }
            write_sealed(decision_file,statement)
            form_statement={
                "step":statement["step"],"generation_step":i,"race_id":row["race_id"],
                "date":race["race_date"],"track":race.get("track"),
                "sheets":paper,"sheet_count":len(paper),
                "marks":statement["marks"],
                "stake_yen":spent,"wallet_before_yen":pre_wallet,
                "wallet_after_marks_yen":post_paper,
                "refills_before_result":refills,
                "weights_before_sha256":w_before,"policy_hash_before":p_before
            }
            write_sealed(sheets_file,form_statement)

            # Results are only read AFTER prediction and decision files were fsynced.
            finish=row["outcome"]
            correct={
                ("3連単","-".join(map(str,finish))),
                ("3連複","=".join(map(str,sorted(finish))))
            }
            winning_idx=[j for j,a in enumerate(arms)
                         if (a["market"],a["selection"]) in correct]
            if len(winning_idx)!=2:raise RuntimeError("Two known complete-market outcomes required")
            logloss=-sum(math.log(max(float(probability[j]),1e-30))
                         for j in winning_idx)/2
            # Each purchased 100-yen mark earns its own payout; preserve overlaps.
            payout=int(round(sum(
                exposure[t]*100*row["odds"][t[0]][t[1]]
                for t in correct if t in exposure)))
            net=payout-spent
            nextbank,reward=choose_wallet_reward(bank,spent,refills,payout)
            assert nextbank==award_excess(post_paper+payout)[0]
            bank=nextbank
            reward_account+=reward
            earned+=reward
            funds_added+=len(refills)*REFILL
            stress_events+=len(refills)

            # Result supervision remains independent from learned behavior.
            classification_delta=LR*(X[winning_idx].mean(axis=0)-probability@X)
            # Ticket-level signed credit replaces all-mark reward/stress reinforcement.
            ticket_delta,credits=mark_credit_weights(arms,probability,paper,correct)
            weights+=classification_delta+ticket_delta
            if not np.isfinite(weights).all():raise RuntimeError("Invalid updated weights")

            page_net=[]
            for page in paper:
                gain=sum(100*row["odds"][page["type"]][t]
                         for t in page["picks"] if (page["type"],t) in correct)
                page_net.append(int(round(gain-100*len(page["picks"]))))
            assert sum(page_net)==net
            # Future page starts at index N on "continue" decision taken after N sheets.
            actions=learn_from_race(policy,decisions,page_net,net)
            assert len(actions)==len(decisions)
            for d in decisions:
                by_kind[(d.kind,d.action)]+=1
            paper_counts["races"]+=1
            paper_counts["entered"]+=int(bool(paper))
            paper_counts["skipped"]+=int(not paper)
            paper_counts["sheets"]+=len(paper)
            paper_counts["marks"]+=sum(exposure.values())
            paper_counts["hits"]+=int(any(exposure[t] for t in correct))
            entry={
                "step":statement["step"],"generation_step":i,"race_id":row["race_id"],
                "date":race["race_date"],"race_type":race.get("race_type"),
                "finish_order":"-".join(map(str,finish)),
                "prediction_before_sha256":w_before,
                "prediction_after_sha256":digest(weights),
                "policy_before_sha256":p_before,
                "policy_after_sha256":policy.digest(),
                "wallet_before_yen":pre_wallet,"wallet_after_yen":bank,
                "refill_events":len(refills),"refilled_yen":len(refills)*REFILL,
                "stress_events":len(refills),"stress_units_yen":len(refills)*REFILL,
                "sheets":len(paper),"marks":sum(exposure.values()),
                "skipped":not paper,"spent_yen":spent,"payout_proxy_yen":payout,
                "net_proxy_yen":net,
                "reward_yen":reward,
                "hit":any(exposure[t] for t in correct),
                "logloss_before_update":float(logloss),
                "ticket_credit_update_l2":float(np.linalg.norm(ticket_delta)),
                "ticket_positive_feedbacks":sum(x["signed_feedback"]>0 for x in credits),
                "ticket_negative_feedbacks":sum(x["signed_feedback"]<0 for x in credits),
                "classification_update_l2":float(np.linalg.norm(classification_delta)),
                "policy_updates":actions
            }
            write_sealed(learning_file,entry)
            journal.append(entry)
            if i%100==0:
                print("GEN006_PROGRESS",i,len(races),"wallet",bank,
                      "skips",paper_counts["skipped"],
                      "policy_updates",policy.update_count,flush=True)

    if not journal:raise RuntimeError("No learned race")
    stakes=sum(x["spent_yen"] for x in journal)
    pays=sum(x["payout_proxy_yen"] for x in journal)
    assert bank+reward_account==inherited_bank+old_reward_fund+funds_added+pays-stakes

    days=[]
    for delta in range((LAST-FIRST).days+1):
        day=(FIRST+timedelta(days=delta)).isoformat()
        rr=[r for r in journal if r["date"]==day]
        days.append({
            "date":day,"races":len(rr),
            "entered":sum(not r["skipped"] for r in rr),
            "skipped":sum(r["skipped"] for r in rr),
            "sheets":sum(r["sheets"] for r in rr),
            "marks":sum(r["marks"] for r in rr),
            "hit_races":sum(r["hit"] for r in rr),
            "spent_yen":sum(r["spent_yen"] for r in rr),
            "payout_proxy_yen":sum(r["payout_proxy_yen"] for r in rr),
            "net_proxy_yen":sum(r["net_proxy_yen"] for r in rr),
            "reward_yen":sum(r["reward_yen"] for r in rr),
            "stress_events":sum(r["stress_events"] for r in rr),
            "closing_wallet_yen":rr[-1]["wallet_after_yen"] if rr else None,
            "mean_logloss":float(np.mean([r["logloss_before_update"] for r in rr])) if rr else None
        })
    model={
        "id":GEN,"parent_id":PARENT,
        "after_step":parent["after_step"]+len(journal),"dimension":D,
        "weights":[float(x) for x in weights],
        "weights_sha256":digest(weights),
        "parent_weights_sha256":parent_hash,
        "random_seed":SEED,"training_span":[FIRST.isoformat(),LAST.isoformat()],
        "bankroll_balance_yen":bank,
        "cumulative_reward_wallet_yen":reward_account,
        "reward_earned_this_month_yen":earned,
        "inherited_reward_wallet_yen":old_reward_fund,
        "refill_events":stress_events,
        "behavior_policy_sha256":policy.digest()
    }
    policy_data=policy.save(OUT/"behavior_policy.json")
    assert policy.count["enter"]==len(journal)
    assert policy.count["continue"]>0, "No continue decisions were trained"
    assert model["behavior_policy_sha256"]==policy_data["weights_sha256"]
    summary={
        "id":GEN,"parent_id":PARENT,"status":"COMPLETE",
        "training_span":model["training_span"],
        "duration_calendar_days":(LAST-FIRST).days+1,
        "parent_step":parent["after_step"],"total_step":model["after_step"],
        "parent_weights_sha256":parent_hash,
        "behavior_policy_weights_sha256":policy.digest(),
        "behavior_policy_initial_sha256":initial_policy_hash,
        "training_racecards":len(raw),"races":len(journal),
        "excluded":dict(excluded),
        "race_types":dict(Counter(r["race_type"] for r in journal)),
        "action_count_enter":policy.count["enter"],
        "action_count_continue":policy.count["continue"],
        "action_prob_final_enter_baseline":float(1/(1+math.exp(-policy.weights["enter"][0]))),
        "behavior_weights_changed_enter":bool(np.linalg.norm(policy.weights["enter"])>0),
        "behavior_weights_changed_continue":bool(np.linalg.norm(policy.weights["continue"])>0),
        "action_decisions":{str(k):v for k,v in by_kind.items()},
        "races_entered":paper_counts["entered"],"races_skipped":paper_counts["skipped"],
        "sheets":paper_counts["sheets"],"marks":paper_counts["marks"],
        "hit_races":paper_counts["hits"],"hit_rate_per_all_races":paper_counts["hits"]/len(journal),
        "starting_wallet_yen":inherited_bank,
        "starting_reward_fund_yen":old_reward_fund,
        "opening_excess_reward_yen":opening_extra,
        "new_reward_yen":earned,
        "cumulative_reward_fund_yen":reward_account,
        "stress_events":stress_events,"refill_events":stress_events,
        "refill_yen":funds_added,"spent_yen":stakes,"payout_proxy_yen":pays,
        "net_proxy_yen":pays-stakes,"ending_wallet_yen":bank,
        "roi_proxy_percent":100*pays/stakes if stakes else None,
        "mean_train_logloss":float(np.mean([r["logloss_before_update"] for r in journal])),
        "first_30_train_logloss":float(np.mean([r["logloss_before_update"] for r in journal[:30]])),
        "last_30_train_logloss":float(np.mean([r["logloss_before_update"] for r in journal[-30:]])),
        "daily":days,
        "out_of_sample_exam":"2023-06-01..2023-06-07; separate workflow; not touched during learning",
        "note":"Both behavior action policies trained sequentially after sealed decisions. All skipped races still teach the separate classification model after the result is revealed.",
        "limitations":[
            "Historical paper-only evaluation, no real bets or deposits.",
            "Uses public FINAL rather than verified real-time pre-close odds.",
            "Only complete 3連単 and 3連複 markets are available.",
            "Actions are learned from sampled decisions; one-month in-sample return does not prove improvement.",
            "Replenishment stress and accumulated rewards are virtual educational feedback, not physical feelings.",
            "Only a common untouched later exam can compare frozen GEN-006 to GEN-005 fairly."
        ]
    }
    (OUT/"checkpoint.json").write_text(json.dumps(model,ensure_ascii=False)+"\n",encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for path,obj in (
        (WEB/"nursery-gen006-checkpoint.json",model),
        (WEB/"nursery-gen006-summary.json",summary),
    ):
        path.write_text(json.dumps(obj,ensure_ascii=False)+"\n",encoding="utf-8")
    (WEB/"nursery-gen006-history.json").write_text(json.dumps(journal,ensure_ascii=False)+"\n",encoding="utf-8")
    # Mark generation as COMPLETE in the same atomic commit as trained artifacts.
    # The follow-on June holdout audit will then discover this checkpoint.
    registry_path=WEB/"generations.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("current_generation")!=GEN:
        raise RuntimeError("Wrong current generation in registry")
    entry=registry["generations"][-1]
    if entry.get("id")!=GEN or entry.get("parent_id")!=PARENT:
        raise RuntimeError("Expected final registry entry to be GEN-006")
    entry.update({
        "status":"COMPLETE","races":len(journal),
        "races_entered":paper_counts["entered"],"races_skipped":paper_counts["skipped"],
        "sheets":paper_counts["sheets"],"marks":paper_counts["marks"],
        "hit_races":paper_counts["hits"],
        "policy_enter_updates":policy.count["enter"],
        "policy_continue_updates":policy.count["continue"],
        "policy_sha256":policy.digest(),
        "weights_sha256":model["weights_sha256"],
        "net_proxy_yen":pays-stakes,"ending_wallet_yen":bank,
        "refill_events":stress_events,
        "new_reward_yen":earned,
        "trained_behavior_policy":True
    })
    registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN006_MONTH_COMPLETE",json.dumps({
        "races":len(journal),"entered":paper_counts["entered"],
        "skips":paper_counts["skipped"],
        "policy_updates":policy.update_count,
        "net_yen":pays-stakes,"ending_wallet":bank,
        "checkpoint_sha256":model["weights_sha256"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

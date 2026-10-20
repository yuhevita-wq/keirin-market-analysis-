"""GEN-007 continuation: sixty-one active training days over two months.

Training begins 2023-04-05 following GEN-007's previous 2023-04-04
checkpoint. The locked 2023-06-01..07 exam is NEVER trained on; to finish
61 training days, resume 2023-06-08 through 11. This continuation is
still GEN-007 (same learned skill), not a new generation. Both predictor
and learned enter/continue action-policy weights are inherited.
The sole success objective is positive TOTAL NET PROFIT in yen:
sum(payouts - stakes) > 0 with nonzero betting exposure.
Virtual wallet, refill and bonus ledgers are preserved for historical
accounting only: they neither enter the action policy nor count as income.
No racing tactics or preselected betting templates are added.
The immutable June 1-7 holdout is strictly outside this training.
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
from behavior_policy import make_learned_sheets,learn_from_race
from profit_objective import ProfitPolicy,net_profit_yen,successful,OBJECTIVE,signed_ticket_updates

GEN="GEN-007"
PARENT="GEN-007"
PARENT_DIR=ROOT/"research/ai_nursery/outputs/GEN-007"
OUT=ROOT/"research/ai_nursery/outputs/GEN-007/continuation-2m"
FIRST=date(2023,4,5)
LAST=date(2023,6,11)
EXAM_START=date(2023,6,1)
EXAM_END=date(2023,6,7)
TRAINING_DAYS=61
STAGE_ID="GEN-007-C2"
FUTURE_HOLDOUT_START=date(2023,7,1)
SEED=20261028
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
         if FIRST.isoformat()<=obj["race"].get("race_date","")<=LAST.isoformat()
         and not EXAM_START.isoformat()<=obj["race"].get("race_date","")<=EXAM_END.isoformat()}
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
    assert FIRST<EXAM_START<=EXAM_END<LAST<FUTURE_HOLDOUT_START
    training_dates=[FIRST+timedelta(days=i) for i in range((LAST-FIRST).days+1)
                    if not EXAM_START<=FIRST+timedelta(days=i)<=EXAM_END]
    assert len(training_dates)==TRAINING_DAYS
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
        raise RuntimeError("GEN-007 continuation already exists: never overwrite frozen checkpoints")
    parent=json.loads((PARENT_DIR/"checkpoint.json").read_text(encoding="utf-8"))
    if parent.get("id")!=PARENT or parent.get("after_step")!=3853 or parent.get("training_span",[])[-1]!="2023-04-04":
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
    policy=ProfitPolicy.inherit(ROOT/"research/ai_nursery/outputs/GEN-007/behavior_policy.json")
    parent_policy_hash=policy.digest()
    if parent_policy_hash!=parent["behavior_policy_sha256"]:
        raise RuntimeError("GEN-007 behavior policy hash is not inherited intact")
    initial_policy_hash=policy.digest()
    previous_enter_updates=policy.count["enter"]
    previous_continue_updates=policy.count["continue"]
    rng=np.random.default_rng(SEED)
    raw,races,excluded=full_calendar()
    OUT.mkdir(parents=True,exist_ok=True)
    WEB.mkdir(parents=True,exist_ok=True)

    journal=[]
    running_net_yen=0
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
            # Capital is bookkeeping ONLY. It has no input to the action policy.
            paper,decisions=make_learned_sheets(arms,probability,row["odds"],row["entries"],10000,rng,policy)
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
                "generation":GEN,"stage_id":STAGE_ID,"parent":PARENT,"generation_step":i,
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
            net=int(net_profit_yen(payout,spent))
            running_net_yen+=net
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
            ticket_delta,credits=signed_ticket_updates(arms,probability,paper,correct)
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
                "net_profit_goal_running_yen":running_net_yen,
                "goal_positive_so_far":successful(running_net_yen,sum(z["spent_yen"] for z in journal)+spent),
                "objective":OBJECTIVE,
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
                print("GEN007_PROGRESS",i,len(races),"wallet",bank,
                      "skips",paper_counts["skipped"],
                      "policy_updates",policy.update_count,flush=True)

    if not journal:raise RuntimeError("No learned race")
    stakes=sum(x["spent_yen"] for x in journal)
    pays=sum(x["payout_proxy_yen"] for x in journal)
    assert bank+reward_account==inherited_bank+old_reward_fund+funds_added+pays-stakes

    days=[]
    for delta in range((LAST-FIRST).days+1):
        current=FIRST+timedelta(days=delta)
        if EXAM_START<=current<=EXAM_END:
            continue
        day=current.isoformat()
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
        "id":GEN,"stage_id":STAGE_ID,"parent_id":PARENT,
        "optimization_objective":OBJECTIVE,
        "parent_behavior_policy_sha256":parent_policy_hash,
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
    assert policy.count["enter"]-previous_enter_updates==len(journal)
    assert policy.count["continue"]-previous_continue_updates>0, "No continue decisions were trained"
    assert model["behavior_policy_sha256"]==policy_data["weights_sha256"]
    summary={
        "id":GEN,"stage_id":STAGE_ID,"parent_id":PARENT,"status":"COMPLETE",
        "optimization_objective":OBJECTIVE,
        "goal_formula":"sum_all_races(payout_proxy_yen - stake_yen) > 0",
        "goal_excludes":["total_capital","wallet_balance","refill","reward_vault","stress","hit_rate","roi"],
        "goal_strictly_positive_and_nonzero_exposure":True,
        "goal_achieved_training":successful(pays-stakes,stakes),
        "net_profit_total_yen":pays-stakes,
        "zero_stake_is_success":False,
        "capital_used_as_policy_feature":False,
        "parent_behavior_policy_sha256":parent_policy_hash,
        "policy_updates_this_generation_enter":policy.count["enter"]-previous_enter_updates,
        "policy_updates_this_generation_continue":policy.count["continue"]-previous_continue_updates,
        "training_span":model["training_span"],
        "duration_calendar_days":(LAST-FIRST).days+1,
        "active_training_days":TRAINING_DAYS,
        "excluded_holdout_dates":[EXAM_START.isoformat(),EXAM_END.isoformat()],
        "training_active_dates_start":FIRST.isoformat(),
        "training_active_dates_end":LAST.isoformat(),
        "parent_stage_checkpoint":"research/ai_nursery/outputs/GEN-007/checkpoint.json",
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
        "out_of_sample_exam":"2023-07-01..2023-07-07; later than all training; old June 1-7 is excluded",
        "note":"Both behavior action policies trained sequentially after sealed decisions. All skipped races still teach the separate classification model after the result is revealed.",
        "limitations":[
            "Historical paper-only evaluation, no real bets or deposits.",
            "Uses public FINAL rather than verified real-time pre-close odds.",
            "Only complete 3連単 and 3連複 markets are available.",
            "Actions are learned from sampled decisions; two-month in-sample return does not prove improvement.",
            "Replenishment stress and accumulated rewards are virtual educational feedback, not physical feelings.",
            "Continuation must be tested only on a later (July or later) untouched holdout.",
            "A month reaching a positive training net is not proof that future trades will profit."
        ]
    }
    (OUT/"checkpoint.json").write_text(json.dumps(model,ensure_ascii=False)+"\n",encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for path,obj in (
        (WEB/"nursery-gen007-phase2-checkpoint.json",model),
        (WEB/"nursery-gen007-phase2-summary.json",summary),
    ):
        path.write_text(json.dumps(obj,ensure_ascii=False)+"\n",encoding="utf-8")
    (WEB/"nursery-gen007-phase2-history.json").write_text(json.dumps(journal,ensure_ascii=False)+"\n",encoding="utf-8")
    # Record a separate continuation stage. Never replace the original GEN-007 checkpoint.
    # The original June holdout is earlier than final training dates and cannot
    # serve as a valid forward test for this continuation.
    registry_path=WEB/"generations.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("current_generation")!=GEN:
        raise RuntimeError("Wrong current generation in registry")
    entry=registry["generations"][-1]
    if entry.get("id")!=GEN or entry.get("parent_id")!="GEN-006":
        raise RuntimeError("Expected final registry entry to be GEN-007")
    if entry.get("id")!=GEN:
        raise RuntimeError("Wrong same-generation continuation registry target")
    stages=entry.setdefault("continuations",[])
    if not stages or stages[-1].get("id")!=STAGE_ID:
        raise RuntimeError("Missing pre-registered continuation")
    stages[-1].update({
        "status":"COMPLETE","races":len(journal),
        "active_training_days":TRAINING_DAYS,
        "training_span":[FIRST.isoformat(),LAST.isoformat()],
        "excluded_holdout":["2023-06-01","2023-06-07"],
        "races_entered":paper_counts["entered"],"races_skipped":paper_counts["skipped"],
        "sheets":paper_counts["sheets"],"marks":paper_counts["marks"],
        "hit_races":paper_counts["hits"],
        "policy_enter_updates":policy.count["enter"],
        "policy_continue_updates":policy.count["continue"],
        "net_profit_goal_yen":pays-stakes,
        "net_profit_goal_met":successful(pays-stakes,stakes),
        "optimization_objective":OBJECTIVE,
        "policy_sha256":policy.digest(),
        "weights_sha256":model["weights_sha256"],
        "net_proxy_yen":pays-stakes,"ending_wallet_yen":bank,
        "refill_events":stress_events,
        "new_reward_yen":earned,
        "trained_behavior_policy":True
    })
    registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("GEN007_CONTINUATION_2M_COMPLETE",json.dumps({
        "races":len(journal),"entered":paper_counts["entered"],
        "skips":paper_counts["skipped"],
        "policy_updates":policy.update_count,
        "net_yen":pays-stakes,"ending_wallet":bank,
        "checkpoint_sha256":model["weights_sha256"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

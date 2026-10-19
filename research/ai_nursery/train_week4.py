"""GEN-004, 2023-01-22..28: reward exactly the portion of wallet above 10,000 yen.

Only NEW feedback to the infant: exceeding the 10,000-yen wallet target
earns a reward exactly equal to the excess. The excess is moved from wallet
to the separate reward account; the next race starts with at most 10,000.
GEN-003 wallet, model weights, historical learning and formations continue.
All currency is PAPER ONLY. Outcomes remain sealed until paper predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import date, timedelta

import numpy as np

from train_week2 import (
    ROOT, WEB, D, LR, COST_PER_MARK, load_cards, load_results,
    load_odds, eligible, feature_matrix, softmax, make_sheets,
)
from train_week3 import spend_marks

GEN="GEN-004"
PARENT="GEN-003"
PARENT_DIR=ROOT/"research/ai_nursery/outputs/GEN-003"
OUT=ROOT/"research/ai_nursery/outputs/GEN-004"
FIRST=date(2023,1,22)
LAST=FIRST+timedelta(days=6)
SEED=20261022
WALLET_TARGET_YEN=10000
REPLENISH_YEN=10000


def award_excess(bank):
    """The monetary reward is EXACTLY the positive excess above 10k."""
    if bank<0:
        raise ValueError("negative wallet")
    awarded=max(0,bank-WALLET_TARGET_YEN)
    return bank-awarded,awarded


def reward_learning_delta(X, prob, arms, marked, rewarded_yen):
    """Learn from the reward itself; no new handicapping or selection rules.

    Scores are the same GEN-003 outcome-conditioned softmax features.
    If positive reward, reinforce features of selections actually submitted
    before the race, centered against the full pre-race probability baseline.
    Reward enters monotonically, bounded by LR for numerical stability.
    No reward => exactly zero extra learning update.
    """
    if rewarded_yen<=0:
        return np.zeros(D,dtype=np.float64)
    index={(a["market"],a["selection"]):i for i,a in enumerate(arms)}
    selection_ids=[index[(sheet["type"],ticket)]
                   for sheet in marked
                   for ticket in sheet["picks"]]
    if not selection_ids:
        raise ValueError("no submitted marks to reinforce")
    # Amount paid as reward remains exact; only the gradient step is stabilized.
    intensity=rewarded_yen/(WALLET_TARGET_YEN+rewarded_yen)
    return LR*intensity*(X[selection_ids].mean(axis=0)-prob@X)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("GEN-004 results already exist: never overwrite or retrain them")
    prior=json.loads((PARENT_DIR/"checkpoint.json").read_text(encoding="utf-8"))
    if (prior.get("id")!=PARENT or prior.get("after_step")!=1518
            or prior.get("training_span",[])[-1]!="2023-01-21"):
        raise RuntimeError("Incorrect GEN-003 ancestor")
    weights=np.asarray(prior["weights"],dtype=np.float64)
    parent_hash=hashlib.sha256(weights.tobytes()).hexdigest()
    if weights.shape!=(D,) or parent_hash!=prior["weights_sha256"]:
        raise RuntimeError("Ancestor learning checkpoint corruption")
    rng=np.random.default_rng(SEED)

    # Inherit the physical wallet; do NOT hand out a new initial 10k.
    inherited_balance=int(prior["bankroll_balance_yen"])
    if inherited_balance<0:
        raise RuntimeError("Invalid negative inherited wallet")
    bank,opening_reward=award_excess(inherited_balance)
    earned_rewards=0
    rewards_total=opening_reward
    total_new_deposits=0
    total_refill_events=0

    cards,_=load_cards()
    week={rid:item for rid,item in cards.items()
          if FIRST.isoformat()<=item["race"].get("race_date","")<=LAST.isoformat()}
    outcomes=load_results()
    odds=load_odds(set(week).intersection(outcomes))
    excluded=Counter()
    usable=[]
    for rid,item in week.items():
        if rid not in outcomes:
            excluded["missing_result"]+=1
            continue
        if rid not in odds:
            excluded["missing_complete_odds"]+=1
            continue
        row={**item,"odds":odds[rid],"outcome":outcomes[rid],"race_id":rid}
        if not eligible(row):
            excluded["incomplete_or_inconsistent"]+=1
            continue
        usable.append(row)
    usable.sort(key=lambda r:(
        r["race"]["race_date"],r["race"].get("start_time",""),
        r["race"].get("track",""),int(r["race"].get("race_no","0") or 0),r["race_id"]
    ))
    if not usable:
        raise RuntimeError("No valid racecards + full odds + finishing order in Jan 22-28")

    OUT.mkdir(parents=True,exist_ok=True)
    WEB.mkdir(parents=True,exist_ok=True)
    journal=[]
    with (OUT/"formation_mark_sheets.jsonl").open("x",encoding="utf-8") as marks:
        for step,r in enumerate(usable,1):
            checksum_before=hashlib.sha256(weights.tobytes()).hexdigest()
            balance_before=bank
            arms=feature_matrix(r["entries"],r["odds"])
            X=np.stack([a["x"] for a in arms])
            probabilities=softmax(X@weights+np.array([a["prior"] for a in arms]))
            pages=make_sheets(arms,probabilities,r["odds"],r["entries"],rng)
            exposures=Counter()
            for page in pages:
                for ticket in page["picks"]:
                    exposures[(page["type"],ticket)]+=1
            balance_after_marks,spent,refills=spend_marks(pages,bank)
            assert spent==COST_PER_MARK*sum(exposures.values())
            total_new_deposits+=len(refills)*REPLENISH_YEN
            total_refill_events+=len(refills)
            bank=balance_after_marks

            # The first log intentionally NEVER contains the result.
            sealed={
                "step":prior["after_step"]+step,"generation_step":step,
                "generation":GEN,"parent":PARENT,
                "race_id":r["race_id"],"date":r["race"]["race_date"],
                "track":r["race"].get("track"),"race_no":r["race"].get("race_no"),
                "race_type":r["race"].get("race_type"),
                "checkpoint_before_sha256":checksum_before,
                "wallet_before_yen":balance_before,
                "refills_before_result":refills,
                "refilled_yen":len(refills)*REPLENISH_YEN,
                "sheets":pages,"sheet_count":len(pages),
                "total_marks":sum(exposures.values()),
                "unique_marks":len(exposures),
                "duplicate_marks":sum(exposures.values())-len(exposures),
                "spent_yen":spent,
                "wallet_after_marks_yen":bank
            }
            marks.write(json.dumps(sealed,ensure_ascii=False,separators=(",",":"))+"\n")
            marks.flush()
            os.fsync(marks.fileno())

            # Only after sheet seals, reveal true finishing order and winnings.
            finish=r["outcome"]
            winning={
                ("3連単","-".join(map(str,finish))),
                ("3連複","=".join(map(str,sorted(finish))))
            }
            true_ids=[i for i,a in enumerate(arms) if (a["market"],a["selection"]) in winning]
            if len(true_ids)!=2:
                raise RuntimeError("two true combinations required for learning")
            loss=-sum(math.log(max(float(probabilities[i]),1e-30)) for i in true_ids)/2
            pay=int(round(sum(
                COST_PER_MARK*exposures[x]*r["odds"][x[0]][x[1]]
                for x in winning if x in exposures
            )))
            bank_before_reward=bank+pay
            bank,reward=award_excess(bank_before_reward)
            earned_rewards+=reward
            rewards_total+=reward

            # Retain the old true-outcome learning and ADD only new reward signal.
            outcome_gradient=LR*(X[true_ids].mean(axis=0)-probabilities@X)
            reinforcement=reward_learning_delta(X,probabilities,arms,pages,reward)
            weights+=outcome_gradient+reinforcement
            journal.append({
                "step":sealed["step"],"generation_step":step,
                "race_id":r["race_id"],"date":sealed["date"],"track":sealed["track"],
                "race_no":sealed["race_no"],"race_type":sealed["race_type"],
                "wallet_before_yen":balance_before,"wallet_after_marks_yen":balance_after_marks,
                "wallet_before_reward_yen":bank_before_reward,
                "wallet_after_reward_yen":bank,
                "refill_events":len(refills),"refilled_yen":len(refills)*REPLENISH_YEN,
                "sheet_count":len(pages),"marks":sealed["total_marks"],
                "spent_yen":spent,"payout_proxy_yen":pay,
                "reward_yen":reward,
                "reward_learning_intensity":reward/(WALLET_TARGET_YEN+reward) if reward else 0,
                "reward_gradient_norm":float(np.linalg.norm(reinforcement)),
                "outcome_gradient_norm":float(np.linalg.norm(outcome_gradient)),
                "finish_order":"-".join(map(str,finish)),
                "winning_ticket_exposures":sum(exposures[x] for x in winning),
                "hit":any(exposures[x] for x in winning),
                "loss_before_update":float(loss),
                "checkpoint_before_sha256":checksum_before,
                "checkpoint_after_sha256":hashlib.sha256(weights.tobytes()).hexdigest()
            })
            if step%50==0:
                print("GEN004_PROGRESS",step,len(usable),"balance",bank,
                      "rewards",rewards_total,"refills",total_refill_events,flush=True)

    spent_total=sum(x["spent_yen"] for x in journal)
    paid_total=sum(x["payout_proxy_yen"] for x in journal)
    assert bank+rewards_total==inherited_balance+total_new_deposits+paid_total-spent_total

    days=[]
    for d in (FIRST+timedelta(days=i) for i in range(7)):
        rr=[x for x in journal if x["date"]==d.isoformat()]
        days.append({
            "date":d.isoformat(),"races":len(rr),
            "sheets":sum(x["sheet_count"] for x in rr),
            "marks":sum(x["marks"] for x in rr),
            "hit_races":sum(x["hit"] for x in rr),
            "spent_yen":sum(x["spent_yen"] for x in rr),
            "payout_proxy_yen":sum(x["payout_proxy_yen"] for x in rr),
            "reward_events":sum(x["reward_yen"]>0 for x in rr),
            "earned_rewards_yen":sum(x["reward_yen"] for x in rr),
            "refill_events":sum(x["refill_events"] for x in rr),
            "closing_wallet_yen":rr[-1]["wallet_after_reward_yen"] if rr else None,
            "mean_loss":float(np.mean([x["loss_before_update"] for x in rr])) if rr else None
        })
    summary={
        "id":GEN,"parent_id":PARENT,
        "status":"GEN004_WEEK_COMPLETE",
        "dates":[FIRST.isoformat(),LAST.isoformat()],
        "instruction_added":"財布が10000円を超えたら超えた分をそのままご褒美として渡す",
        "parent_checkpoint_sha256":parent_hash,
        "parent_step":prior["after_step"],
        "total_step":prior["after_step"]+len(journal),
        "parent_final_wallet_yen":inherited_balance,
        "opening_inherited_excess_reward_yen":opening_reward,
        "reward_target_yen":WALLET_TARGET_YEN,
        "reward_separation":"exact positive excess transferred out of wallet to rewards",
        "learning_added":"positive reward reinforces only marks chosen before that race; existing outcome loss training unchanged",
        "reward_gradient_scale":"reward/(10000+reward), bounded; actual reward balance is not scaled",
        "races":len(journal),"raw_racecards_in_week":len(week),
        "missing_races":dict(excluded),
        "race_types":dict(Counter(x["race_type"] for x in journal)),
        "sheets":sum(x["sheet_count"] for x in journal),
        "marks":sum(x["marks"] for x in journal),
        "max_sheets_in_one_race":max(x["sheet_count"] for x in journal),
        "max_marks_in_one_race":max(x["marks"] for x in journal),
        "multi_sheet_races":sum(x["sheet_count"]>1 for x in journal),
        "hit_races":sum(x["hit"] for x in journal),
        "hit_rate_percent":100*sum(x["hit"] for x in journal)/len(journal),
        "reward_events":sum(x["reward_yen"]>0 for x in journal),
        "earned_rewards_yen":earned_rewards,
        "total_rewards_including_opening_yen":rewards_total,
        "refill_events":total_refill_events,"refill_deposits_yen":total_new_deposits,
        "new_contributions_yen":total_new_deposits,
        "total_spent_yen":spent_total,"payout_proxy_yen":paid_total,
        "net_betting_proxy_yen":paid_total-spent_total,
        "wallet_final_yen":bank,
        "cashflow_invariant":"wallet_final + opening_and_earned_rewards = inherited_wallet + new_refills + payout - stakes",
        "roi_proxy_percent":100*paid_total/spent_total if spent_total else None,
        "mean_loss":float(np.mean([x["loss_before_update"] for x in journal])),
        "first_30_loss":float(np.mean([x["loss_before_update"] for x in journal[:30]])),
        "last_30_loss":float(np.mean([x["loss_before_update"] for x in journal[-30:]])),
        "daily":days,
        "limitations":[
            "Historical paper-only experiment; no actual cash transfer or wagering.",
            "Reward fund is a virtual ledger and is not net betting profit.",
            "The opening excess is inherited from GEN-003, not earned in this training week.",
            "Replenishments remain excluded from returns or reward profit.",
            "Only 3連単 and 3連複 complete markets are present in public joined odds.",
            "These are FINAL prices and retrospectively collected racecards, not certified pre-close odds.",
            "This reward rule is a new feedback signal, but it does not by itself prove positive expected value.",
            "Outcome supervision and reward reinforcement use historical labels after saving paper predictions.",
            "The previous neutral stochastic formation stopping rule remains unchanged."
        ]
    }
    checkpoint={
        "id":GEN,"parent_id":PARENT,"after_step":summary["total_step"],"dimension":D,
        "weights":[float(x) for x in weights],
        "weights_sha256":hashlib.sha256(weights.tobytes()).hexdigest(),
        "parent_weights_sha256":parent_hash,"random_seed":SEED,
        "training_span":[FIRST.isoformat(),LAST.isoformat()],
        "bankroll_balance_yen":bank,
        "cumulative_reward_wallet_yen":rewards_total,
        "reward_earned_this_week_yen":earned_rewards,
        "reward_opening_inheritance_yen":opening_reward,
        "refill_events":total_refill_events
    }
    outputs=(
        (OUT/"summary.json",summary),
        (OUT/"checkpoint.json",checkpoint),
        (WEB/"nursery-gen004-summary.json",summary),
        (WEB/"nursery-gen004-checkpoint.json",checkpoint),
        (WEB/"nursery-gen004-history.json",journal)
    )
    for path,obj in outputs:
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2 if path.name=="summary.json" else None)+"\n",encoding="utf-8")
    with (OUT/"learning_history.jsonl").open("x",encoding="utf-8") as f:
        for row in journal:
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    print("GEN004_COMPLETE",json.dumps({
        "races":len(journal),"earned_rewards_yen":earned_rewards,
        "reward_events":summary["reward_events"],"refill_events":total_refill_events,
        "wallet_final_yen":bank,"marks":summary["marks"],
        "net_betting_proxy_yen":summary["net_betting_proxy_yen"],
        "model_sha":checkpoint["weights_sha256"][:16]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

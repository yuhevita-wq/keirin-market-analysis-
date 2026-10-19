"""Audit GEN-006: a full 28-day run, ancestry, sealed actions and actual policy updates."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

from train_month6 import ROOT,OUT,WEB,FIRST,LAST,GEN,PARENT,REFILL
from train_week3 import spend_marks
from train_week5 import award_excess
from behavior_policy import BehaviorPolicy


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check():
    cp=read(OUT/"checkpoint.json")
    summary=read(OUT/"summary.json")
    parent=read(ROOT/"research/ai_nursery/outputs/GEN-005/checkpoint.json")
    forms=[json.loads(s) for s in (OUT/"formation_mark_sheets.jsonl").read_text(encoding="utf-8").splitlines()]
    actions=[json.loads(s) for s in (OUT/"action_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    history=[json.loads(s) for s in (OUT/"learning_history.jsonl").read_text(encoding="utf-8").splitlines()]
    policy=BehaviorPolicy.load(OUT/"behavior_policy.json")
    assert summary["id"]==cp["id"]==GEN and summary["parent_id"]==cp["parent_id"]==PARENT
    assert summary["duration_calendar_days"]==28
    assert cp["training_span"]==[FIRST.isoformat(),LAST.isoformat()]
    assert LAST.isoformat()=="2023-03-04" and LAST.isoformat()<"2023-06-01"
    assert cp["after_step"]==parent["after_step"]+len(history)
    assert parent["after_step"]==2309
    assert len(forms)==len(actions)==len(history)==summary["races"]>0
    assert summary["races"]+sum(summary["excluded"].values())==summary["training_racecards"]
    assert cp["parent_weights_sha256"]==parent["weights_sha256"]
    assert summary["parent_weights_sha256"]==parent["weights_sha256"]
    weights=np.asarray(cp["weights"],dtype=np.float64)
    assert weights.shape==(256,)
    assert hashlib.sha256(weights.tobytes()).hexdigest()==cp["weights_sha256"]
    assert cp["weights_sha256"]!=parent["weights_sha256"]
    assert policy.digest()==cp["behavior_policy_sha256"]==summary["behavior_policy_weights_sha256"]
    assert summary["behavior_weights_changed_enter"] and summary["behavior_weights_changed_continue"]
    assert policy.update_count==policy.count["enter"]+policy.count["continue"]
    assert policy.count["enter"]==len(actions) and policy.count["continue"]>0
    assert len({x["race_id"] for x in actions})==len(actions)
    assert all(FIRST.isoformat()<=x["date"]<=LAST.isoformat() for x in actions)
    wallet=int(parent["bankroll_balance_yen"])
    old_rewards=int(parent["cumulative_reward_wallet_yen"])
    wallet,opening=award_excess(wallet)
    spent=paid=refills=earned=sheets_count=mark_count=hits=skips=0
    decision_count=Counter()
    total_positive=total_negative=0
    for i,(a,f,h) in enumerate(zip(actions,forms,history),1):
        assert a["generation"]==GEN and a["parent"]==PARENT
        assert a["step"]==f["step"]==h["step"]==parent["after_step"]+i
        assert a["generation_step"]==i and a["race_id"]==f["race_id"]==h["race_id"]
        assert "finish_order" not in a and "payout_proxy_yen" not in a
        assert "finish_order" not in f and "payout_proxy_yen" not in f
        assert a["policy_hash_before"]==h["policy_before_sha256"]==f["policy_hash_before"]
        assert a["weights_before_sha256"]==h["prediction_before_sha256"]==f["weights_before_sha256"]
        assert len(a["decisions"])==len(h["policy_updates"])>=1
        assert a["skip"]==h["skipped"]==(len(f["sheets"])==0)
        assert a["paper_stake_yen"]==h["spent_yen"]==f["stake_yen"]
        assert a["marks"]==h["marks"]==f["marks"]
        assert a["wallet_before_yen"]==h["wallet_before_yen"]==f["wallet_before_yen"]==wallet
        after,cost,additions=spend_marks(f["sheets"],wallet)
        assert after==f["wallet_after_marks_yen"]
        assert additions==f["refills_before_result"]==a["refills_before_result"]
        assert cost==a["paper_stake_yen"]
        assert a["skip"]==(a["decisions"][0]["action"]=="stop")
        for decision,change in zip(a["decisions"],h["policy_updates"]):
            kind=decision["kind"]
            decision_count[kind]+=1
            assert kind==change["kind"]
            assert decision["probability_of_continue"]>=0 and decision["probability_of_continue"]<=1
            assert len(decision["context"])==7
        if a["skip"]:
            assert cost==h["payout_proxy_yen"]==0 and not additions
            assert not h["hit"]
        returned=h["payout_proxy_yen"]
        after,winner_reward=award_excess(after+returned)
        assert h["reward_yen"]==winner_reward and h["wallet_after_yen"]==after
        assert h["net_proxy_yen"]==returned-cost
        assert h["stress_events"]==h["refill_events"]==len(additions)
        assert h["stress_units_yen"]==h["refilled_yen"]==len(additions)*REFILL
        assert h["ticket_positive_feedbacks"]+h["ticket_negative_feedbacks"]<=h["marks"]
        total_positive+=h["ticket_positive_feedbacks"]
        total_negative+=h["ticket_negative_feedbacks"]
        spent+=cost;paid+=returned;earned+=winner_reward;refills+=len(additions)
        mark_count+=h["marks"];sheets_count+=h["sheets"]
        hits+=int(h["hit"]);skips+=int(h["skipped"])
        wallet=after
        if i>1:
            assert a["weights_before_sha256"]==history[i-2]["prediction_after_sha256"]
            assert a["policy_hash_before"]==history[i-2]["policy_after_sha256"]
    assert summary["starting_wallet_yen"]==parent["bankroll_balance_yen"]
    assert summary["starting_reward_fund_yen"]==old_rewards
    assert summary["opening_excess_reward_yen"]==opening
    assert summary["new_reward_yen"]==earned
    assert summary["ending_wallet_yen"]==cp["bankroll_balance_yen"]==wallet
    assert summary["cumulative_reward_fund_yen"]==cp["cumulative_reward_wallet_yen"]==old_rewards+opening+earned
    assert wallet+old_rewards+opening+earned==parent["bankroll_balance_yen"]+old_rewards+refills*REFILL+paid-spent
    assert summary["stress_events"]==summary["refill_events"]==refills
    assert summary["refill_yen"]==refills*REFILL
    assert summary["marks"]==mark_count and summary["sheets"]==sheets_count
    assert summary["races_skipped"]==skips and summary["races_entered"]==len(history)-skips
    assert summary["hit_races"]==hits
    assert summary["spent_yen"]==spent and summary["payout_proxy_yen"]==paid
    assert summary["net_proxy_yen"]==paid-spent
    assert len(summary["daily"])==28 and sum(x["races"] for x in summary["daily"])==len(history)
    assert sum(x["skipped"] for x in summary["daily"])==skips
    assert sum(x["stress_events"] for x in summary["daily"])==refills
    assert decision_count["enter"]==policy.count["enter"]
    assert decision_count["continue"]==policy.count["continue"]
    assert policy.update_count==sum(decision_count.values())
    assert history[-1]["prediction_after_sha256"]==cp["weights_sha256"]
    assert history[-1]["policy_after_sha256"]==policy.digest()
    assert total_positive+total_negative<=mark_count
    assert read(WEB/"nursery-gen006-summary.json")["races"]==len(history)
    registry=read(WEB/"generations.json")
    gen=registry["generations"][-1]
    assert registry["current_generation"]==GEN and gen["id"]==GEN and gen["status"]=="COMPLETE"
    assert gen["parent_id"]==PARENT and gen["races"]==len(history)
    print("GEN006_MONTH_VERIFIED",len(history),"28-day races",skips,"skips",refills,"refills",policy.update_count,"policy updates")


if __name__=="__main__":
    check()

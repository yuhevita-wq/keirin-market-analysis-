"""Audit parent inheritance, sealed marks, exact excess rewards and financial identity."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from train_week4 import ROOT, OUT, PARENT_DIR, D, REPLENISH_YEN, award_excess, reward_learning_delta
from train_week3 import spend_marks

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parent=load(PARENT_DIR/"checkpoint.json")
    summary=load(OUT/"summary.json")
    cp=load(OUT/"checkpoint.json")
    marks=[json.loads(x) for x in (OUT/"formation_mark_sheets.jsonl").read_text(encoding="utf-8").splitlines()]
    history=[json.loads(x) for x in (OUT/"learning_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["id"]==cp["id"]=="GEN-004"
    assert summary["parent_id"]==cp["parent_id"]==parent["id"]=="GEN-003"
    assert summary["dates"]==["2023-01-22","2023-01-28"]
    assert len(marks)==len(history)==summary["races"]>0
    assert summary["parent_step"]==parent["after_step"]==1518
    assert summary["parent_checkpoint_sha256"]==cp["parent_weights_sha256"]==parent["weights_sha256"]
    assert marks[0]["checkpoint_before_sha256"]==parent["weights_sha256"]
    w=np.asarray(cp["weights"],dtype=np.float64)
    assert w.shape==(D,) and hashlib.sha256(w.tobytes()).hexdigest()==cp["weights_sha256"]
    assert cp["after_step"]==1518+len(history)
    assert len({x["race_id"] for x in marks})==len(marks)
    bank,opening=award_excess(parent["bankroll_balance_yen"])
    assert summary["opening_inherited_excess_reward_yen"]==opening
    assert opening==cp["reward_opening_inheritance_yen"]
    deposits=spent_total=paid_total=earned=hits=marks_total=pages_total=refill_events=reward_events=0
    for i,(a,b) in enumerate(zip(marks,history),1):
        assert a["generation_step"]==b["generation_step"]==i
        assert a["step"]==b["step"]==1518+i
        assert a["race_id"]==b["race_id"] and a["checkpoint_before_sha256"]==b["checkpoint_before_sha256"]
        assert "finish_order" not in a and "reward_yen" not in a and "hit" not in a
        assert a["wallet_before_yen"]==bank==b["wallet_before_yen"]
        assert len(a["sheets"])==a["sheet_count"]==b["sheet_count"]
        assert sum(p["point_count"] for p in a["sheets"])==a["total_marks"]==b["marks"]
        post,cost,refills=spend_marks(a["sheets"],bank)
        assert refills==a["refills_before_result"] and len(refills)==b["refill_events"]
        assert all(0<=r["balance_before_yen"]<100 and r["amount_yen"]==10000 for r in refills)
        assert post==a["wallet_after_marks_yen"]==b["wallet_after_marks_yen"]
        assert cost==a["spent_yen"]==b["spent_yen"]
        before_reward=post+b["payout_proxy_yen"]
        nextbank,reward=award_excess(before_reward)
        assert b["wallet_before_reward_yen"]==before_reward
        assert b["wallet_after_reward_yen"]==nextbank
        assert b["reward_yen"]==reward and nextbank<=10000
        if reward:
            assert nextbank==10000 and before_reward>10000
            assert b["reward_gradient_norm"]>0
            reward_events+=1
        else:
            assert b["reward_gradient_norm"]==b["reward_learning_intensity"]==0
        deposits+=len(refills)*REPLENISH_YEN
        refill_events+=len(refills)
        spent_total+=cost
        paid_total+=b["payout_proxy_yen"]
        earned+=reward
        hits+=int(b["hit"])
        marks_total+=a["total_marks"]
        pages_total+=a["sheet_count"]
        bank=nextbank
        if i>1:assert a["checkpoint_before_sha256"]==history[i-2]["checkpoint_after_sha256"]
    assert bank==summary["wallet_final_yen"]==cp["bankroll_balance_yen"]
    assert opening+earned==summary["total_rewards_including_opening_yen"]==cp["cumulative_reward_wallet_yen"]
    assert earned==summary["earned_rewards_yen"]==cp["reward_earned_this_week_yen"]
    assert bank+opening+earned==parent["bankroll_balance_yen"]+deposits+paid_total-spent_total
    assert deposits==summary["refill_deposits_yen"] and refill_events==summary["refill_events"]
    assert spent_total==summary["total_spent_yen"] and paid_total==summary["payout_proxy_yen"]
    assert pages_total==summary["sheets"] and marks_total==summary["marks"]
    assert reward_events==summary["reward_events"] and hits==summary["hit_races"]
    assert sum(d["races"] for d in summary["daily"])==len(marks)
    assert sum(d["earned_rewards_yen"] for d in summary["daily"])==earned
    assert cp["weights_sha256"]==history[-1]["checkpoint_after_sha256"]
    assert award_excess(9999)==(9999,0)
    assert award_excess(10000)==(10000,0)
    assert award_excess(10001)==(10000,1)
    assert award_excess(15780)==(10000,5780)
    m=np.zeros((2,D));m[0,0]=1;m[1,1]=1
    p=np.array([0.25,0.75])
    arms=[{"market":"3連単","selection":"1-2-3"},{"market":"3連単","selection":"2-1-3"}]
    page=[{"type":"3連単","picks":["1-2-3"]}]
    assert not np.any(reward_learning_delta(m,p,arms,page,0))
    assert np.linalg.norm(reward_learning_delta(m,p,arms,page,100))>0
    dash=load(ROOT/"docs/ai-lab/nursery-gen004-summary.json")
    assert dash["races"]==len(marks)
    registry=load(ROOT/"docs/ai-lab/generations.json")
    assert registry["current_generation"]=="GEN-004"
    assert registry["generations"][-1]["parent_id"]=="GEN-003"
    print("GEN004_INTEGRITY_OK",len(marks),"races",reward_events,"rewards",earned,"yen; bank",bank)

if __name__=="__main__":
    main()

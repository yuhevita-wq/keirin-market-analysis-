"""Audit GEN-005 inheritance, paper-only chronology and negative stress ledger."""
from __future__ import annotations
import hashlib,json
import numpy as np
from train_week5 import ROOT,OUT,PARENT_DIR,D,REPLENISH_YEN,award_excess
from train_week3 import spend_marks
def j(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
    parent=j(PARENT_DIR/"checkpoint.json")
    cp=j(OUT/"checkpoint.json");s=j(OUT/"summary.json")
    forms=[json.loads(x) for x in (OUT/"formation_mark_sheets.jsonl").read_text(encoding="utf-8").splitlines()]
    rows=[json.loads(x) for x in (OUT/"learning_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert s["id"]==cp["id"]=="GEN-005" and s["parent_id"]==cp["parent_id"]==parent["id"]=="GEN-004"
    assert s["dates"]==["2023-01-29","2023-02-04"]
    assert len(forms)==len(rows)==s["races"]>0 and len({f["race_id"] for f in forms})==len(forms)
    assert s["parent_step"]==parent["after_step"]==1942
    assert cp["after_step"]==1942+len(forms)
    assert forms[0]["checkpoint_before_sha256"]==s["parent_checkpoint_sha256"]==cp["parent_weights_sha256"]==parent["weights_sha256"]
    w=np.asarray(cp["weights"],dtype=np.float64)
    assert w.shape==(D,) and hashlib.sha256(w.tobytes()).hexdigest()==cp["weights_sha256"]
    assert cp["weights_sha256"]==rows[-1]["checkpoint_after_sha256"]
    original_bank=int(parent["bankroll_balance_yen"])
    old_rewards=int(parent["cumulative_reward_wallet_yen"])
    bank,opening=award_excess(original_bank)
    assert s["inherited_rewards_yen"]==cp["inherited_reward_wallet_yen"]==old_rewards
    assert opening==s["opening_inherited_excess_reward_yen"]
    deposited=spent=paid=earned=refills=stresses=0
    paper=marks=hits=reward_events=0
    for i,(f,r) in enumerate(zip(forms,rows),1):
        assert f["generation"]=="GEN-005" and f["parent"]=="GEN-004"
        assert f["step"]==r["step"]==1942+i and f["race_id"]==r["race_id"]
        assert f["checkpoint_before_sha256"]==r["checkpoint_before_sha256"]
        assert "finish_order" not in f and "reward_yen" not in f and "payout_proxy_yen" not in f
        assert f["wallet_before_yen"]==r["wallet_before_yen"]==bank
        new_bank,cost,events=spend_marks(f["sheets"],bank)
        assert f["wallet_after_marks_yen"]==r["wallet_after_marks_yen"]==new_bank
        assert f["spent_yen"]==r["spent_yen"]==cost==100*f["total_marks"]
        assert events==f["refills_before_result"]
        assert r["refilled_yen"]==len(events)*REPLENISH_YEN
        assert r["refill_events"]==r["stress_events"]==len(events)
        assert r["stress_units_yen"]==len(events)*REPLENISH_YEN
        assert r["stress_gradient_norm"]>0 if events else r["stress_gradient_norm"]==0
        if i>1:assert f["checkpoint_before_sha256"]==rows[i-2]["checkpoint_after_sha256"]
        before_reward=new_bank+r["payout_proxy_yen"]
        bank,reward=award_excess(before_reward)
        assert before_reward==r["wallet_before_reward_yen"]
        assert bank==r["wallet_after_reward_yen"] and reward==r["reward_yen"] and 0<=bank<=10000
        deposited+=len(events)*REPLENISH_YEN;stresses+=len(events)*REPLENISH_YEN;refills+=len(events)
        spent+=cost;paid+=r["payout_proxy_yen"];earned+=reward
        reward_events+=int(reward>0);paper+=f["sheet_count"];marks+=f["total_marks"];hits+=int(r["hit"])
    assert s["stress_events"]==cp["stress_events"]==s["refill_events"]==refills
    assert s["stress_units_yen"]==cp["stress_units_yen"]==stresses==deposited
    assert s["wallet_final_yen"]==cp["bankroll_balance_yen"]==bank
    assert s["total_rewards_including_opening_yen"]==cp["cumulative_reward_wallet_yen"]==old_rewards+opening+earned
    assert earned==s["earned_rewards_yen"] and s["reward_events"]==reward_events
    assert bank+old_rewards+opening+earned==original_bank+old_rewards+deposited+paid-spent
    assert s["payout_proxy_yen"]==paid and s["total_spent_yen"]==spent and s["net_betting_proxy_yen"]==paid-spent
    assert s["sheets"]==paper and s["marks"]==marks and s["hit_races"]==hits
    assert len(s["daily"])==7 and sum(d["races"] for d in s["daily"])==len(forms)
    assert sum(d["stress_events"] for d in s["daily"])==refills
    assert sum(d["stress_units_yen"] for d in s["daily"])==deposited
    assert sum(d["earned_rewards_yen"] for d in s["daily"])==earned
    assert s["raw_racecards_in_week"]>=len(forms)
    assert j(ROOT/"docs/ai-lab/nursery-gen005-summary.json")["races"]==s["races"]
    registry=j(ROOT/"docs/ai-lab/generations.json")
    assert registry["current_generation"]=="GEN-005" and registry["generations"][-1]["parent_id"]=="GEN-004"
    print("GEN005 VERIFIED",len(rows),"races",refills,"stress events",earned,"earned rewards",bank,"wallet")
if __name__=="__main__":main()

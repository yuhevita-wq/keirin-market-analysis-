"""GEN-007 continuation: 61-day signed-net-only learning, immutable June exam."""
from __future__ import annotations
import hashlib,json
from collections import Counter
from datetime import date
import numpy as np
from train_2months7 import ROOT,OUT,WEB,FIRST,LAST,GEN,PARENT,REFILL,EXAM_START,EXAM_END,TRAINING_DAYS,STAGE_ID
from train_week3 import spend_marks
from train_week5 import award_excess
from profit_objective import ProfitPolicy,net_profit_yen,successful,OBJECTIVE
from behavior_policy import BehaviorPolicy

def j(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    old=j(ROOT/"research/ai_nursery/outputs/GEN-007/checkpoint.json")
    ancestor=BehaviorPolicy.load(ROOT/"research/ai_nursery/outputs/GEN-007/behavior_policy.json")
    cp=j(OUT/"checkpoint.json");s=j(OUT/"summary.json")
    policy=BehaviorPolicy.load(OUT/"behavior_policy.json")
    action=[json.loads(x) for x in (OUT/"action_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    pages=[json.loads(x) for x in (OUT/"formation_mark_sheets.jsonl").read_text(encoding="utf-8").splitlines()]
    hist=[json.loads(x) for x in (OUT/"learning_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert cp["id"]==s["id"]==GEN=="GEN-007"
    assert old["id"]==PARENT=="GEN-007"
    assert cp["parent_weights_sha256"]==old["weights_sha256"]
    assert cp["parent_behavior_policy_sha256"]==ancestor.digest()==old["behavior_policy_sha256"]
    assert s["parent_behavior_policy_sha256"]==ancestor.digest()
    assert s["stage_id"]==cp["stage_id"]==STAGE_ID
    assert s["behavior_policy_initial_sha256"]==ancestor.digest()
    assert s["optimization_objective"]==OBJECTIVE==cp["optimization_objective"]
    assert s["goal_formula"]=="sum_all_races(payout_proxy_yen - stake_yen) > 0"
    assert s["goal_strictly_positive_and_nonzero_exposure"] and s["zero_stake_is_success"] is False
    assert s["capital_used_as_policy_feature"] is False
    assert s["duration_calendar_days"]==68 and s["active_training_days"]==61
    assert s["training_span"]==["2023-04-05","2023-06-11"]
    assert EXAM_START==date(2023,6,1) and EXAM_END==date(2023,6,7)
    assert LAST<date(2023,7,1)
    assert len(action)==len(pages)==len(hist)==s["races"]>0
    assert all(not EXAM_START.isoformat()<=x["date"]<=EXAM_END.isoformat() for x in action)
    assert s["races"]+sum(s["excluded"].values())==s["training_racecards"]
    assert cp["after_step"]==old["after_step"]+len(hist)
    w=np.asarray(cp["weights"],dtype=float)
    assert hashlib.sha256(w.tobytes()).hexdigest()==cp["weights_sha256"]
    assert cp["weights_sha256"]==hist[-1]["prediction_after_sha256"]
    assert cp["behavior_policy_sha256"]==policy.digest()==hist[-1]["policy_after_sha256"]
    assert policy.update_count>ancestor.update_count
    assert policy.count["enter"]-ancestor.count["enter"]==s["races"]
    assert policy.count["continue"]-ancestor.count["continue"]>0
    assert s["policy_updates_this_generation_enter"]==len(hist)
    assert s["policy_updates_this_generation_continue"]==policy.count["continue"]-ancestor.count["continue"]
    assert len({r["race_id"] for r in action})==len(action)
    bank,opening=award_excess(old["bankroll_balance_yen"])
    inherited_reward=old["cumulative_reward_wallet_yen"]
    spent=paid=earned=refills=skips=marks=sheets=0
    decisions=Counter()
    cumulative_net=0
    for i,(a,f,h) in enumerate(zip(action,pages,hist),1):
        assert a["generation_step"]==i and a["step"]==old["after_step"]+i
        assert f["step"]==h["step"]==a["step"] and f["race_id"]==h["race_id"]==a["race_id"]
        assert a["parent"]==PARENT and a["generation"]==GEN
        assert "finish_order" not in a and "payout_proxy_yen" not in a
        assert "finish_order" not in f and "payout_proxy_yen" not in f
        assert a["weights_before_sha256"]==f["weights_before_sha256"]==h["prediction_before_sha256"]
        assert a["policy_hash_before"]==f["policy_hash_before"]==h["policy_before_sha256"]
        assert len(a["decisions"])==len(h["policy_updates"])
        assert a["wallet_before_yen"]==f["wallet_before_yen"]==h["wallet_before_yen"]==bank
        after_marks,cost,adds=spend_marks(f["sheets"],bank)
        assert adds==f["refills_before_result"]==a["refills_before_result"]
        assert f["wallet_after_marks_yen"]==after_marks
        assert cost==a["paper_stake_yen"]==f["stake_yen"]==h["spent_yen"]==a["marks"]*100
        assert a["skip"]==h["skipped"]==(len(f["sheets"])==0)
        assert a["skip"]==(a["decisions"][0]["action"]=="stop")
        # All GEN-007 decisions see the same neutral wallet feature (not live capital).
        for d,u in zip(a["decisions"],h["policy_updates"]):
            assert d["context"][1]==1
            assert u["objective"]==OBJECTIVE
            assert abs(u["scaled_reward"] - u["realized_incremental_net_yen"]/10000)<0.000001
            decisions[d["kind"]]+=1
        if a["skip"]:assert h["payout_proxy_yen"]==0 and cost==0
        nextbank,reward=award_excess(after_marks+h["payout_proxy_yen"])
        assert nextbank==h["wallet_after_yen"] and reward==h["reward_yen"]
        net=int(net_profit_yen(h["payout_proxy_yen"],cost))
        assert h["net_proxy_yen"]==net
        cumulative_net+=net
        assert h["net_profit_goal_running_yen"]==cumulative_net
        assert h["goal_positive_so_far"]==successful(cumulative_net,spent+cost)
        assert h["objective"]==OBJECTIVE
        paid+=h["payout_proxy_yen"];spent+=cost;earned+=reward;refills+=len(adds)
        marks+=h["marks"];sheets+=h["sheets"];skips+=int(a["skip"]);bank=nextbank
        if i>1:
            assert a["weights_before_sha256"]==hist[i-2]["prediction_after_sha256"]
            assert a["policy_hash_before"]==hist[i-2]["policy_after_sha256"]
    assert s["net_profit_total_yen"]==s["net_proxy_yen"]==paid-spent==cumulative_net
    assert s["goal_achieved_training"]==successful(paid-spent,spent)
    assert not successful(0,0) and not successful(-1,100) and successful(1,100)
    assert s["spent_yen"]==spent and s["payout_proxy_yen"]==paid
    assert s["races_skipped"]==skips and s["marks"]==marks and s["sheets"]==sheets
    assert bank==s["ending_wallet_yen"]==cp["bankroll_balance_yen"]
    assert bank+inherited_reward+opening+earned==old["bankroll_balance_yen"]+inherited_reward+refills*REFILL+paid-spent
    assert len(s["daily"])==TRAINING_DAYS and sum(d["races"] for d in s["daily"])==len(action)
    assert sum(d["net_proxy_yen"] for d in s["daily"])==cumulative_net
    assert decisions["enter"]==s["policy_updates_this_generation_enter"]
    assert decisions["continue"]==s["policy_updates_this_generation_continue"]
    assert j(WEB/"nursery-gen007-phase2-summary.json")["net_profit_total_yen"]==cumulative_net
    reg=j(WEB/"generations.json")
    assert reg["current_generation"]==GEN and reg["generations"][-1]["status"]=="COMPLETE"
    assert reg["generations"][-1]["continuations"][-1]["id"]==STAGE_ID
    assert reg["generations"][-1]["continuations"][-1]["status"]=="COMPLETE"
    assert reg["generations"][-1]["continuations"][-1]["net_profit_goal_met"]==successful(paid-spent,spent)
    print("GEN007_2MONTHS_PROFIT_TEST_PASS",len(action),"races",cumulative_net,"net-yen",skips,"skips")

if __name__=="__main__":
    main()

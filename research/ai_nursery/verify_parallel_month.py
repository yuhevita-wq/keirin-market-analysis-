"""Audit the GEN-008 parallel monthly experiment without learning."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from behavior_policy import BehaviorPolicy
from gen008_experiment import model_digest
from run_parallel_month import (
    BASE,PARENT,OUT,WEB,DAYS,FIRST,LAST,ID,BRANCHES,check_parent,
)
from run_parallel_week import load_inherited
from train_week import D


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check():
    summary=read_json(OUT/"summary.json")
    assert summary["id"]==ID and summary["status"]=="COMPLETE"
    assert summary["training_only"] and summary["days"]==DAYS==31
    assert summary["dates"]==[FIRST.isoformat(),LAST.isoformat()]
    assert summary["reserved_frozen_holdout"]==["2023-10-01","2023-10-14"]
    assert summary["holdout_results_read_for_training"]==0
    assert summary["all_shadow_returns_excluded_from_cash"]
    assert summary["existing_GEN007_and_week01_week02_unchanged"]
    parents,origin=load_inherited(3)
    check_parent(parents,origin)
    sealed=jsonl(OUT/"sealed_pre_result_choices.jsonl")
    learned=jsonl(OUT/"unsealed_post_result_learning.jsonl")
    assert len(sealed)==summary["races"]>0
    assert len(learned)==2*len(sealed)
    ids=[x["race_id"] for x in sealed]
    assert len(ids)==len(set(ids))
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest()==summary["race_ids_sha256"]
    tallies={k:{"entered":0,"marks":0,"stake":0,"payout":0,"hit":0,
                "sum_trifecta_loss":0.0,"sum_trio_loss":0.0} for k in BRANCHES}
    shadow_skips=0
    shadow_positive=0
    shadow_net=0
    for pre,old,new in zip(sealed,learned[::2],learned[1::2]):
        assert FIRST.isoformat()<=pre["date"]<=LAST.isoformat()
        assert pre["race_id"]==old["race_id"]==new["race_id"]
        assert pre["sealing_happened_before_result"]
        assert "outcome" not in pre and "finish_order" not in pre
        assert "payout_yen" not in pre and "net_yen" not in pre
        assert pre["old"]["id"]==pre["new"]["id"]==pre["race_id"]
        assert pre["new"]["shadow_is_not_spent"]
        assert pre["new"]["shadow_page"] is not None
        for name,record in (("old",old),("new",new)):
            act=record["actual"]
            chosen=pre[name]
            assert record["branch"]==name
            assert record["shadow_counted_as_actual_cash"] is False
            assert act["net_yen"]==act["payout_yen"]-act["stake_yen"]
            assert act["stake_yen"]==chosen["actual_paper_stake_yen"]
            assert act["marks"]==chosen["actual_paper_marks"]
            assert act["stake_yen"]==100*act["marks"]
            assert record["entered"]==bool(chosen["pages"])
            assert record["trifecta_logloss"]>=0 and math.isfinite(record["trifecta_logloss"])
            assert record["trio_logloss"]>=0 and math.isfinite(record["trio_logloss"])
            v=tallies[name]
            v["entered"]+=int(record["entered"])
            v["marks"]+=act["marks"]
            v["stake"]+=act["stake_yen"]
            v["payout"]+=act["payout_yen"]
            v["hit"]+=int(act["hit"])
            v["sum_trifecta_loss"]+=record["trifecta_logloss"]
            v["sum_trio_loss"]+=record["trio_logloss"]
        assert (new["skipped_shadow_net_yen"] is None)==bool(new["entered"])
        if new["skipped_shadow_net_yen"] is not None:
            shadow_skips+=1
            shadow_positive+=int(new["skipped_shadow_net_yen"]>0)
            shadow_net+=new["skipped_shadow_net_yen"]
    for name in BRANCHES:
        expected=summary["comparison"][name]
        v=tallies[name]
        assert expected["races"]==len(sealed)
        assert expected["entered"]==v["entered"]
        assert expected["skipped"]==len(sealed)-v["entered"]
        assert expected["actual_marks"]==v["marks"]
        assert expected["hit_races"]==v["hit"]
        assert expected["actual_staked_yen"]==v["stake"]
        assert expected["actual_payout_proxy_yen"]==v["payout"]
        assert expected["actual_net_yen"]==v["payout"]-v["stake"]
        assert math.isclose(expected["mean_trifecta_logloss"],
                            v["sum_trifecta_loss"]/len(sealed))
        assert math.isclose(expected["mean_trio_logloss"],
                            v["sum_trio_loss"]/len(sealed))
        cp=read_json(OUT/name/"checkpoint.json")
        weights=np.asarray(cp["weights"],dtype=np.float64)
        assert cp["branch"]==name and cp["stage"]==ID
        assert cp["dates"]==summary["dates"]
        assert cp["cumulative_races_from_GEN007"]==5616+len(sealed)
        assert weights.shape==(D,) and model_digest(weights)==cp["weights_sha256"]
        assert BehaviorPolicy.load(OUT/name/"behavior_policy.json").digest()==cp["policy_sha256"]
        assert cp["parent_weights_sha256"]==summary["inherited_checkpoints"][name]["prediction"]
        assert cp["parent_policy_sha256"]==summary["inherited_checkpoints"][name]["behavior"]
        assert cp["weights_sha256"]==summary["closing_checkpoints"][name]["prediction"]
    new=summary["comparison"]["new"]
    assert new["skipped_with_presealed_shadow"]==shadow_skips
    assert new["profitable_skipped_shadows"]==shadow_positive
    assert new["shadow_net_on_skips_yen"]==shadow_net
    assert sum(x["races"] for x in summary["daily"])==len(sealed)
    assert len(summary["daily"])==31
    public=read_json(WEB)
    assert public["races"]==len(sealed)
    assert public["comparison"]==summary["comparison"]
    print("GEN008_MONTH_31D_AUDIT_PASS",
          json.dumps({"races":len(sealed),
          "old_net":summary["comparison"]["old"]["actual_net_yen"],
          "new_net":summary["comparison"]["new"]["actual_net_yen"]}),flush=True)


if __name__=="__main__":
    check()

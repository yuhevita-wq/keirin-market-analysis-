"""Audits GEN-008 week01 after both parallel learners finish."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from datetime import date,timedelta

import numpy as np

from train_gen008_week01 import ROOT,OUT,MANIFEST,FIRST,LAST,BOTH,read_json
from parallel_gen008 import PARENT,model_digest
from behavior_policy import BehaviorPolicy


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def audit():
    s=read_json(OUT/"summary.json")
    m=read_json(MANIFEST)
    assert s["status"]=="COMPLETE_TRAINING_ONLY"
    assert s["races"]>0 and s["days"]==7
    assert s["training_dates"]==[FIRST.isoformat(),LAST.isoformat()]
    assert m["races"]==s["races"]
    assert s["same_parent_predictor_sha256"]==read_json(PARENT/"checkpoint.json")["weights_sha256"]
    assert s["same_parent_action_policy_sha256"]==read_json(PARENT/"checkpoint.json")["behavior_policy_sha256"]
    assert s["later_frozen_exam_races_read_for_training"]==0
    assert s["net_counts_only_actual_purchased_paper"]
    assert s["shadow_paper_real_spend_yen"]==0
    sealed=list(lines(OUT/"sealed_before_results.jsonl"))
    results=list(lines(OUT/"after_results_learning.jsonl"))
    assert len(sealed)==s["races"]
    assert len(results)==2*s["races"]
    assert len({x["race_id"] for x in sealed})==len(sealed)
    for i,(before,a,b) in enumerate(zip(sealed,results[::2],results[1::2]),1):
        assert before["sequence"]==i
        assert a["sequence"]==b["sequence"]==i
        assert a["race_id"]==b["race_id"]==before["race_id"]
        assert a["branch"]=="old" and b["branch"]=="new"
        assert not before["results_visible_before_seal"]
        assert before["actual_stakes_exclude_shadow"]
        assert set(before["old"]).issuperset({"pages","decisions","prediction_before_sha256"})
        assert set(before["new"]).issuperset({"pages","decisions","shadow_page","temperature_before"})
        assert before["new"]["shadow_page"]["total_yen"]>0
        assert before["new"]["shadow_is_not_spent"]
        assert before["new"]["actual_paper_stake_yen"]==b["actual"]["stake_yen"]
        assert before["old"]["actual_paper_stake_yen"]==a["actual"]["stake_yen"]
        assert a["shadow_skipped_net_yen"] is None
        assert b["shadow_counted_as_actual"] is False
        for x in (a,b):
            assert x["actual"]["net_yen"]==x["actual"]["payout_yen"]-x["actual"]["stake_yen"]
            assert x["actual"]["stake_yen"]==100*x["actual"]["marks"]
            assert len(x["post_update_predictor_sha256"])==64
            assert x["pre_update"]["trifecta_mass"]>0
            assert x["pre_update"]["trio_mass"]>0
            assert np.isfinite(x["pre_update"]["trifecta_logloss"])
            assert np.isfinite(x["pre_update"]["trio_logloss"])
            assert FIRST.isoformat()<=x["date"]<=LAST.isoformat()
    for name in BOTH:
        cp=read_json(OUT/name/"checkpoint.json")
        pol=BehaviorPolicy.load(OUT/name/"behavior_policy.json")
        assert cp["trained_race_count"]==s["races"]
        assert cp["parent_weights_sha256"]==s["same_parent_predictor_sha256"]
        assert cp["parent_policy_sha256"]==s["same_parent_action_policy_sha256"]
        assert cp["weights_sha256"]==model_digest(cp["weights"])
        assert cp["after_step"]==5328+s["races"]
        assert pol.digest()==s["checkpoints"][name]["policy_sha256"]
        assert pol.count["enter"]==3019+s["races"]
        assert s["old_vs_new_training_scores"][name]["races"]==s["races"]
        assert s["checkpoints"][name]["weights_sha256"]==cp["weights_sha256"]
        rr=[r for r in results if r["branch"]==name]
        sc=s["old_vs_new_training_scores"][name]
        assert sum(r["actual"]["stake_yen"] for r in rr)==sc["staked_yen"]
        assert sum(r["actual"]["payout_yen"] for r in rr)==sc["payout_proxy_yen"]
        assert sc["pure_net_yen"]==sc["payout_proxy_yen"]-sc["staked_yen"]
    assert not (OUT/"frozen_forward_results.jsonl").exists()
    print("GEN008_WEEK01_FROZEN_CHECKPOINTS_AUDIT_PASS",s["races"])


if __name__=="__main__":
    audit()

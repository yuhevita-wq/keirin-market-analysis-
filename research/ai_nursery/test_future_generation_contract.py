"""Quality contract enforced for each new completed baby generation.

Older GEN-001..005 remain immutable historical controls. GEN-006+ must carry
a trained action policy and separate per-race action-decision/outcome logs,
then appear on the exact same frozen holdout before claims of growth.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from behavior_policy import BehaviorPolicy
from train_week import ROOT,WEB


def check():
    registry=json.loads((WEB/"generations.json").read_text(encoding="utf-8"))
    metrics=json.loads((WEB/"nursery-growth-audit.json").read_text(encoding="utf-8"))
    tested={x["generation"] for x in metrics["metrics"]}
    previous=None
    for entry in registry["generations"]:
        gen=entry["id"]
        if entry.get("status") not in ("COMPLETE","COMPLETE_BABY_BANKROLL_EXPERIMENT",
                                       "GEN004_WEEK_COMPLETE","GEN005_STRESS_WEEK_COMPLETE"):
            previous=entry
            continue
        if not Path(ROOT/entry["checkpoint"]).exists():
            raise AssertionError("No trained checkpoint for "+gen)
        cp=json.loads((ROOT/entry["checkpoint"]).read_text(encoding="utf-8"))
        assert cp["id"]==gen
        if int(gen.split("-")[1])<=5:
            previous=entry
            continue
        if cp["training_span"][1]>=metrics["holdout_start"]:
            raise AssertionError("Training overlaps locked holdout in "+gen)
        if previous is None or entry.get("parent_id")!=previous["id"]:
            raise AssertionError("Invalid parent lineage for "+gen)
        parent=json.loads((ROOT/previous["checkpoint"]).read_text(encoding="utf-8"))
        assert cp.get("parent_weights_sha256")==parent["weights_sha256"]
        if gen not in tested:
            raise AssertionError("Missing a new-generation frozen shared-race test for "+gen)
        d=ROOT/entry["training_records"]
        policy_path=d/"behavior_policy.json"
        decisions_path=d/"action_decisions.jsonl"
        if not policy_path.is_file() or not decisions_path.is_file():
            raise AssertionError("No learned action checkpoint or decisions for "+gen)
        policy=BehaviorPolicy.load(policy_path)
        if policy.update_count==0 or not all(v>0 for v in policy.count.values()):
            raise AssertionError("One or more action categories not learned in "+gen)
        lines=decisions_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise AssertionError("No action history for "+gen)
        actual_actions=0
        for line in lines:
            event=json.loads(line)
            if "outcome" in event or "finish_order" in event:
                raise AssertionError("Result leaked into pre-race action record")
            if not set(("race_id","policy_hash_before","decisions","marks","skip"))<=set(event):
                raise AssertionError("Incomplete pre-decision log")
            actual_actions+=len(event["decisions"])
        if actual_actions<policy.update_count:
            raise AssertionError("Policy update count is larger than logged decisions")
        previous=entry
    print("FUTURE_GENERATION_QUALITY_CONTRACT_PASS",len(tested),"generations checked")


if __name__=="__main__":
    check()

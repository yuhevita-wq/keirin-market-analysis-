"""Audit completed GEN-008 parallel week without changing weights."""
from __future__ import annotations
import argparse,hashlib,json,math
import numpy as np
from train_week import D
from behavior_policy import BehaviorPolicy
from gen008_experiment import model_digest
from run_parallel_week import paths,interval,load_inherited

def iter_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def check(week):
    start,end=interval(week)
    d=paths(week)
    summary=json.loads((d/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="COMPLETE" and summary["dates"]==[start,end]
    assert not summary["independent_held_out_exam_completed"]
    parents,_=load_inherited(week)
    if week==1:assert parents[0].policy.digest()==parents[1].policy.digest()
    before=list(iter_jsonl(d/"sealed_pre_result_choices.jsonl"))
    after=list(iter_jsonl(d/"unsealed_post_result_learning.jsonl"))
    assert len(before)==summary["races"]>0 and len(after)==2*len(before)
    ids=[r["race_id"] for r in before]
    assert len(ids)==len(set(ids))
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest()==summary["race_ids_sha256"]
    counts={name:{"stake":0,"payout":0,"marks":0,"entered":0} for name in ("old","new")}
    for pre,old,new in zip(before,after[::2],after[1::2]):
        assert pre["race_id"]==old["race_id"]==new["race_id"]
        assert old["branch"]=="old" and new["branch"]=="new"
        assert pre["old"]["id"]==pre["new"]["id"]==pre["race_id"]
        assert "outcome" not in pre and "finish_order" not in pre
        assert "payout_yen" not in pre and "net_yen" not in pre
        assert pre["new"]["shadow_page"] is not None and pre["new"]["shadow_is_not_spent"]
        for name,post,decision in (("old",old,pre["old"]),("new",new,pre["new"])):
            acc=post["actual"]
            assert acc["net_yen"]==acc["payout_yen"]-acc["stake_yen"]
            assert acc["stake_yen"]==decision["actual_paper_stake_yen"]
            assert acc["marks"]==decision["actual_paper_marks"]
            assert post["entered"]==bool(decision["pages"])
            assert math.isfinite(post["trifecta_logloss"])
            assert math.isfinite(post["trio_logloss"])
            counts[name]["stake"]+=acc["stake_yen"]
            counts[name]["payout"]+=acc["payout_yen"]
            counts[name]["marks"]+=acc["marks"]
            counts[name]["entered"]+=int(post["entered"])
            if name=="new":
                assert post["shadow_used_as_real_stake"] is False
                assert (post["skipped_shadow_net_yen"] is None)==bool(post["entered"])
    for name in ("old","new"):
        a=summary["comparison"][name]
        v=counts[name]
        assert a["actual_staked_yen"]==v["stake"]
        assert a["actual_payout_proxy_yen"]==v["payout"]
        assert a["actual_net_yen"]==v["payout"]-v["stake"]
        assert a["actual_marks"]==v["marks"]
        assert a["entered"]==v["entered"]
        assert a["races"]==len(before)
        checkpoint=json.loads((d/name/"checkpoint.json").read_text(encoding="utf-8"))
        weights=np.asarray(checkpoint["weights"],dtype=np.float64)
        assert weights.shape==(D,) and model_digest(weights)==checkpoint["weights_sha256"]
        assert checkpoint["policy_sha256"]==BehaviorPolicy.load(d/name/"behavior_policy.json").digest()
        assert checkpoint["dates"]==[start,end]
    assert summary["real_cash_excludes_shadow"]
    print("GEN008_WEEKLY_AUDIT_PASS",week,len(before),summary["comparison"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--week",type=int,default=1)
    check(parser.parse_args().week)

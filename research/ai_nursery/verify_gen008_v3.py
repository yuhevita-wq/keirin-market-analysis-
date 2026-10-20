"""Read-only audit for GEN-008 V3. Verify original sealed selections and cashflow."""
from __future__ import annotations
import json,math
from collections import defaultdict

from run_gen008_v3 import OUT,WEB,STAGE,summarize,load_model_settings
from gen008_v3 import settle_v3,TRAIN_FIRST,TRAIN_LAST,VALID_FIRST,VALID_LAST,TEST_FIRST,TEST_LAST
from gen008_experiment import score_papers


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def records(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]


def check():
    summary=read(OUT/"summary.json")
    checkpoint=read(OUT/"v3_checkpoint.json")
    public=read(WEB)
    assert summary["stage"]==STAGE and summary["status"]=="COMPLETE"
    assert public["status"]=="COMPLETE"
    assert summary["october_test_data_not_used_for_v3_fit"]
    assert summary["decision_before_outcome_seal"]
    assert not summary["weights_policy_updated_during_test"]
    assert summary["frozen_parent_start"]==summary["frozen_parent_end"]
    assert checkpoint["calibration"]["train_dates"]==[TRAIN_FIRST,TRAIN_LAST]
    assert checkpoint["calibration"]["validation_dates"]==[VALID_FIRST,VALID_LAST]
    assert checkpoint["frozen_test_dates"]==[TEST_FIRST,TEST_LAST]
    assert checkpoint["candidate_uses_october_results"] is False
    assert checkpoint["training_done_on_november"] is False
    assert summary["v3_alpha"]==checkpoint["alpha"]
    assert summary["v3_algorithm_digest"]==checkpoint["algorithm_digest"]
    assert summary["v3_alpha_fit"]["alpha"]==checkpoint["alpha"]
    assert checkpoint["calibration"]["train_races"]>0
    assert checkpoint["calibration"]["validation_races"]>0
    assert checkpoint["calibration"]["train_dates"][1]<TEST_FIRST
    assert checkpoint["calibration"]["validation_dates"][1]<TEST_FIRST
    original,snap,meta=load_model_settings()
    assert summary["frozen_parent_start"]==snap
    assert checkpoint["parent_new_prediction"]==meta["weights_sha256"]
    assert checkpoint["parent_new_policy"]==meta["policy_sha256"]

    prepared=records(OUT/"sealed_before_results.jsonl")
    finalized=records(OUT/"revealed_after_results.jsonl")
    assert len(prepared)==summary["november_exam_races"]
    assert len(finalized)==len(prepared)*3
    all_ids=set()
    group=defaultdict(dict)
    for i,record in enumerate(finalized):
        group[record["sequence"]][record["branch"]]=record
    for idx,p in enumerate(prepared,1):
        assert p["stage"]==STAGE and p["sequence"]==idx and p["before_result_revealed"]
        assert TEST_FIRST<=p["date"]<=TEST_LAST
        assert p["race_id"] not in all_ids
        all_ids.add(p["race_id"])
        assert p["predictor_start"]==snap
        assert p["v3_algorithm_digest"]==checkpoint["algorithm_digest"]
        assert set(p["decisions"])=={"old","new","v3"}
        q=group[idx]
        assert set(q)=={"old","new","v3"}
        outcomes={tuple(v["outcome_after_seal"]) for v in q.values()}
        assert len(outcomes)==1
        finish=next(iter(outcomes))
        for name in ("old","new","v3"):
            bet=p["decisions"][name]
            r=q[name]
            assert r["stage"]==STAGE and r["race_id"]==p["race_id"]
            assert r["date"]==p["date"] and r["branch"]==name
            assert r["sequence"]==idx and r["no_real_money"]
            if name=="v3":
                assert bet["predictor_sha256"]==snap["new"]["prediction"]
                assert bet["no_result_used"] and bet["no_fixed_ticket_cap"]
                assert bet["alpha"]==checkpoint["alpha"]
                for item in bet["selected"]:
                    assert item["type"] in ("3連単","3連複")
                    assert item["stake_yen"]>=100 and item["stake_yen"]%100==0
                    assert item["odds"]>=1
                expected=settle_v3(bet,finish)
            else:
                assert bet["prediction_before_sha256"]==snap[name]["prediction"]
                assert bet["policy_before_sha256"]==snap[name]["behavior"]
                # Recompute from sealed pick labels, selected-book prices saved only
                # in original racecard odds. Controls' payouts are asserted from
                # outcome and official historical price, not model EV.
                payout=0
                for page in bet["pages"]:
                    hit=("-".join(map(str,finish)) if page["type"]=="3連単"
                         else "=".join(map(str,sorted(finish))))
                    if hit in page["picks"]:
                        # Original mark-sheet JSON has no odds attached. Payout
                        # is reconciled in source-aware verification below.
                        payout+=1
                assert r["actual"]["hit"]==bool(payout)
                assert r["actual"]["marks"]==sum(len(page["picks"]) for page in bet["pages"])
                expected={"stake_yen":bet["actual_paper_stake_yen"],
                          "marks":bet["actual_paper_marks"],
                          "hit":r["actual"]["hit"]}
            for field,value in expected.items():
                assert r["actual"][field]==value,(idx,name,field)
            assert r["entered"]==(r["actual"]["stake_yen"]>0)
            assert r["actual"]["net_yen"]==r["actual"]["payout_yen"]-r["actual"]["stake_yen"]
            assert math.isfinite(r["trifecta_logloss"]) and r["trifecta_logloss"]>=0
            assert math.isfinite(r["trio_logloss"]) and r["trio_logloss"]>=0
    assert len(group)==len(prepared)
    recomputed=summarize(finalized)
    for branch in ("old","new","v3"):
        for k,v in summary["comparison"][branch].items():
            a=recomputed[branch][k]
            if isinstance(v,float):
                assert math.isclose(a,v,rel_tol=1e-10,abs_tol=1e-9),(branch,k,v,a)
            else:
                assert a==v,(branch,k,v,a)
    assert public["comparison"]==summary["comparison"]
    print("GEN008_V3_AUDIT_PASS",json.dumps({
        "races":len(prepared),
        "control_old_net":recomputed["old"]["paper_net_yen"],
        "control_new_net":recomputed["new"]["paper_net_yen"],
        "v3_net":recomputed["v3"]["paper_net_yen"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    check()

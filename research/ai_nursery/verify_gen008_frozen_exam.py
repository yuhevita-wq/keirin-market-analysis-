"""Read-only audit of frozen Oct 2023 GEN-008 old/new holdout."""
from __future__ import annotations
import hashlib,json,math
from collections import defaultdict
from gen008_frozen_exam import OUT,WEB,FIRST,LAST,STAGE,BRANCHES,load_frozen

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def records(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def check():
    summary=read(OUT/"summary.json")
    assert summary["status"]=="COMPLETE" and summary["id"]==STAGE
    assert summary["kind"]=="FROZEN_OUT_OF_SAMPLE"
    assert summary["exam_dates"]==[FIRST,LAST] and summary["days"]==14
    assert summary["no_training_on_exam"] and not summary["weights_or_policy_updated"]
    assert summary["same_races_for_both"] and summary["decisions_sealed_before_result"]
    assert summary["shadow_not_in_actual_cash"] and summary["historical_final_odds_only"]
    pre=records(OUT/"sealed_pre_result_choices.jsonl")
    post=records(OUT/"revealed_forward_results.jsonl")
    assert len(pre)==summary["races"]>0 and len(post)==2*len(pre)
    ids=[r["race_id"] for r in pre]
    assert len(ids)==len(set(ids))
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest()==summary["race_ids_sha256"]
    _,hashes=load_frozen()
    assert hashes==summary["frozen_start"]==summary["frozen_end"]
    totals={k:{"stake":0,"payout":0,"marks":0,"entered":0,"hits":0,
               "tri_loss":0.0,"trio_loss":0.0} for k in BRANCHES}
    shadow_skips=shadow_profitable=shadow_net=shadow_positive=0
    days=defaultdict(int)
    for sealed,old,new in zip(pre,post[::2],post[1::2]):
        assert sealed["stage"]==STAGE and FIRST<=sealed["date"]<=LAST
        assert sealed["before_outcome_read"]
        assert all(k not in sealed for k in ("outcome","finish_order","payout_yen","net_yen"))
        assert sealed["race_id"]==old["race_id"]==new["race_id"]
        days[sealed["date"]]+=1
        for name,r in zip(BRANCHES,(old,new)):
            decision=sealed[name]
            assert decision["id"]==sealed["race_id"]
            assert decision["branch"]==r["branch"]==name
            assert r["stage"]==STAGE and r["date"]==sealed["date"]
            assert decision["prediction_before_sha256"]==r["prediction_sha256"]==hashes[name]["prediction"]
            assert decision["policy_before_sha256"]==r["policy_sha256"]==hashes[name]["behavior"]
            assert math.isclose(decision["temperature_before"],hashes[name]["temperature"])
            actual=r["actual"]
            assert actual["net_yen"]==actual["payout_yen"]-actual["stake_yen"]
            assert actual["stake_yen"]==decision["actual_paper_stake_yen"]==100*decision["actual_paper_marks"]
            assert r["entered"]==bool(decision["pages"])
            assert not r["shadow_counted_as_actual_cash"]
            assert r["trifecta_logloss"]>=0 and math.isfinite(r["trifecta_logloss"])
            assert r["trio_logloss"]>=0 and math.isfinite(r["trio_logloss"])
            a=totals[name]
            a["entered"]+=int(r["entered"])
            a["stake"]+=actual["stake_yen"]
            a["payout"]+=actual["payout_yen"]
            a["marks"]+=actual["marks"]
            a["hits"]+=int(actual["hit"])
            a["tri_loss"]+=r["trifecta_logloss"]
            a["trio_loss"]+=r["trio_logloss"]
            if name=="new" and not r["entered"]:
                assert r["skipped_shadow_net_yen"] is not None
                shadow_skips+=1
                shadow_net+=r["skipped_shadow_net_yen"]
                shadow_profitable+=int(r["skipped_shadow_net_yen"]>0)
                shadow_positive+=max(r["skipped_shadow_net_yen"],0)
            else:
                assert r["skipped_shadow_net_yen"] is None
    for name,a in totals.items():
        score=summary["comparison"][name]
        expected={"races":len(pre),"entered":a["entered"],"skipped":len(pre)-a["entered"],
                  "actual_marks":a["marks"],"hit_races":a["hits"],
                  "actual_staked_yen":a["stake"],"actual_payout_proxy_yen":a["payout"],
                  "actual_net_yen":a["payout"]-a["stake"]}
        for k,v in expected.items():
            assert score[k]==v,(name,k,score[k],v)
        assert math.isclose(score["mean_trifecta_logloss"],a["tri_loss"]/len(pre))
        assert math.isclose(score["mean_trio_logloss"],a["trio_loss"]/len(pre))
    s=summary["comparison"]["new"]
    assert s["skipped_with_presealed_shadow"]==shadow_skips
    assert s["profitable_skipped_shadows"]==shadow_profitable
    assert s["missed_profitable_shadow_yen"]==shadow_positive
    assert s["shadow_net_on_skips_yen"]==shadow_net
    assert sum(days.values())==summary["races"]
    assert len(summary["daily"])==14
    assert sum(x["races"] for x in summary["daily"])==summary["races"]
    assert read(WEB)["comparison"]==summary["comparison"]
    print("GEN008_FROZEN_EXAM_AUDIT_PASS",
          json.dumps({"races":len(pre),"comparison":summary["comparison"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    check()

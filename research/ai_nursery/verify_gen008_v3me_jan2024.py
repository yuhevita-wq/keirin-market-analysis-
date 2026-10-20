"""Read-only recomputation of frozen V3-ME original coefficients on independent January 2024."""
from __future__ import annotations

import json,math
from collections import defaultdict

from run_gen008_v3me_jan2024 import OUT,WEB,STAGE,summarize,source_rows
from gen008_v3_market_error import (
 market_features,corrected_probability,best_expected_value_stats,portfolio,settle,
 FIRST_FIT,LAST_FIT,FIRST_DIAG,LAST_DIAG,FIRST_TEST,LAST_TEST,
)
from gen008_frozen_exam import load_frozen
from parallel_gen008 import pure_pre_result
from gen008_experiment import score_papers


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def lines(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]


def audit():
    report=read(OUT/"summary.json")
    model=read(OUT/"market_error_fit.json")
    frontend=read(WEB)
    assert report["stage"]==STAGE and report["status"]=="COMPLETE"
    assert report["former_october_holdout_now_dev"]
    assert report["december_outcomes_never_used_to_fit"]
    assert report["january_2024_outcomes_never_used_to_fit"]
    assert report["frozen_prior_market_error_parameters"]
    assert report["test_period"]==["2024-01-01","2024-01-14"]
    frozen=read(OUT.parent/"dec-2023-expanded-authentic-prices/market_error_fit.json")
    assert model["algorithm_sha256"]==frozen["algorithm_sha256"]
    assert model["coefficients"]==frozen["coefficients"]
    assert report["pre_result_decisions_sealed_before_outcomes"]
    assert report["old_new_v3_frozen_unchanged"] and report["same_races_for_all_four"]
    assert report["historical_final_odds_only"]
    assert not report["actual_real_bets"]
    assert model["train_dates"]==[FIRST_FIT,LAST_FIT]
    assert model["diagnostic_dates"]==[FIRST_DIAG,LAST_DIAG]
    assert model["january_2024_untouched_forward_period"]==["2024-01-01","2024-01-14"]
    assert model["no_october_holdout_claim"] and model["future_december_labels_used_in_fit"] is False
    assert frontend["status"]=="COMPLETE" and frontend["comparison"]==report["comparison"]
    original,checkpoints=load_frozen()
    assert report["frozen_parent_start"]==checkpoints==report["frozen_parent_end"]
    sealed=lines(OUT/"decisions_sealed_before_result.jsonl")
    scores=lines(OUT/"settlements_after_result.jsonl")
    train,validation,testing,providers=source_rows()
    assert model["train_races"]==len(train)==report["train_races"]
    assert len(validation)==report["diagnostic_races"]
    assert len(testing)==len(sealed)==report["test_races"]
    assert len(scores)==4*len(sealed)
    by_id={r["race_id"]:r for r in testing}
    by_step=defaultdict(dict)
    for r in scores:
        assert r["branch"] not in by_step[r["sequence"]]
        by_step[r["sequence"]][r["branch"]]=r
    ids=set()
    for index,p in enumerate(sealed,1):
        assert p["stage"]==STAGE and p["sequence"]==index and p["pre_result"]
        rid=p["race_id"]
        assert rid not in ids and rid in by_id
        ids.add(rid)
        row=by_id[rid]
        pre=pure_pre_result(row)
        finish=row["outcome"]
        assert p["date"]==pre["race"]["race_date"]
        assert "2024-01-01"<=p["date"]<="2024-01-14"
        assert p["market_model_sha256"]==model["algorithm_sha256"]
        assert p["input_model_sha256"]==checkpoints["new"]["prediction"]
        feat=market_features(pre,original["new"].weights,original["new"].temperature)
        probs=corrected_probability(feat,model["coefficients"])
        recomputed=portfolio(pre,feat,probs)
        assert recomputed==p["v3_market_error"],(index,"non-reproducible presettlement")
        expected_ev=best_expected_value_stats(pre,feat,probs)
        for key,v in expected_ev.items():
            a=p["estimated_ev_summary"][key]
            if isinstance(v,float): assert math.isclose(a,v,rel_tol=1e-12,abs_tol=1e-12)
            else:assert a==v
        group=by_step[index]
        assert set(group)=={"old","new","v3","v3_market_error"}
        for branch in group:
            record=group[branch]
            assert record["race_id"]==rid and record["date"]==p["date"]
            assert tuple(record["finish_only_after_seal"])==tuple(finish)
            assert not record["real_money_bet"]
            if branch in ("old","new"):
                assert p[branch]["prediction_before_sha256"]==checkpoints[branch]["prediction"]
                assert p[branch]["policy_before_sha256"]==checkpoints[branch]["behavior"]
                actual=score_papers(p[branch]["pages"],pre["odds"],finish)
            elif branch=="v3":
                from gen008_v3 import settle_v3
                actual=settle_v3(p["v3"],finish)
            else:
                actual=settle(p["v3_market_error"],finish)
            assert record["actual"]==actual,(index,branch,"settlement mismatch")
            assert record["entered"]==(actual["stake_yen"]>0)
            assert math.isfinite(record["trifecta_logloss"]) and record["trifecta_logloss"]>=0
            assert math.isfinite(record["trio_logloss"]) and record["trio_logloss"]>=0
    recomputed=summarize(scores)
    for branch in recomputed:
        for field,actual in recomputed[branch].items():
            reported=report["comparison"][branch][field]
            if isinstance(actual,float):
                assert math.isclose(actual,reported,rel_tol=1e-12,abs_tol=1e-12)
            else:assert actual==reported
    assert report["test_races"]==len(by_step)
    print("GEN008_V3ME_JAN24_READONLY_AUDIT_PASS",json.dumps({
        "races":len(sealed),"learned_coefs":model["coefficients"],
        "v3_market_error":recomputed["v3_market_error"],
        "old":recomputed["old"]},ensure_ascii=False),flush=True)


if __name__=="__main__":audit()

"""Independent deterministic audit of exhaustive GEN-008 paper-mark experiment.

Recompute every full 3連単+3連複 mark, every hypothetical settlement,
each of four branches' pre-result distributions and the training gradient.
No results from test split can affect any training delta.
"""
from __future__ import annotations
import gzip,hashlib,json,math
from collections import defaultdict

import numpy as np
from gen008_all_marks_experiment import (
    OUT,WEB,STAGE,FIRST_TRAIN,LAST_TRAIN,FIRST_TEST,LAST_TEST,BRANCHES,
    UNIT,LR,L2,MAX_DELTA_NORM,load_frozen,read_calibration,select_rows,
    aligned_distribution,revised_prior,all_marks,paper_settlement,metrics,
    gradient_full_sheet,positive_shadow,shadow_settlement,digest
)


def entries(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)


def summary_recompute():
    if not OUT.exists() or not WEB.exists():
        raise RuntimeError("Missing finished all-sheet immutable records")
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    assert summary["stage"]==STAGE and summary["status"]=="COMPLETE"
    assert summary["train_dates"]==[FIRST_TRAIN,LAST_TRAIN]
    assert summary["test_dates"]==[FIRST_TEST,LAST_TEST]
    assert summary["all_four_same_full_sheet"] and summary["100_yen_per_combination"]
    assert summary["all_3rentan_and_3renpuku_marked"] and summary["outcome_read_after_pre_result_seal"]
    assert summary["test_period_never_updates_weights_or_coefficients"]
    assert summary["original_models_untouched"] and summary["controls_frozen_start"]==summary["controls_frozen_end"]
    assert summary["full_sheet_stakes_not_real_money"]
    parents,snap=load_frozen()
    calib=read_calibration(parents,snap)
    assert snap==summary["controls_frozen_start"]
    train,test,sources=select_rows()
    assert len(train)==summary["train_races"] and len(test)==summary["test_races"]
    rows=[("train",r) for r in train]+[("test",r) for r in test]
    saved_choices=list(entries(OUT/"sealed_four_predictions.jsonl"))
    saved_settled=list(entries(OUT/"outcomes_and_learning.jsonl"))
    assert len(saved_choices)==len(saved_settled)==len(rows)
    coefficients={branch:np.zeros(5,dtype=float) for branch in BRANCHES}
    scores=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    with gzip.open(OUT/"all_combinations_mark_sheets.jsonl.gz","rt",encoding="utf-8") as f:
        for i,(split,row) in enumerate(rows):
            sheet=json.loads(f.readline())
            sealed=saved_choices[i]
            settled=saved_settled[i]
            pre={"race_id":row["race_id"],"entries":row["entries"],
                 "race":row["race"],"odds":row["odds"]}
            feat,bases=aligned_distribution(pre,parents,calib)
            exact=all_marks(pre,feat)
            assert sheet==exact,("altered full combination mark sheet",i)
            assert sealed["full_sheet_sha256"]==digest(sheet)
            assert sealed["step"]==settled["step"]==i+1
            assert sealed["race_id"]==settled["race_id"]==pre["race_id"]
            assert sealed["split"]==settled["split"]==split
            assert sealed["pre_result"] and settled["after_pre_result_seal"]
            assert sealed["full_sheet_mark_count"]==sheet["marks"]
            assert sealed["all_four_branches_virtual_full_sheet_stake_yen"]==sheet["virtual_stake_yen"]
            assert sheet["same_full_sheet_for_four_branches"] and sheet["no_real_purchase"]
            assert sheet["virtual_stake_yen"]==sheet["marks"]*UNIT
            assert all(x["yen"]==UNIT for book in sheet["markets"].values() for x in book)
            assert list(row["outcome"])==settled["result"]
            assert set(sealed["four_branches"])==set(settled["per_branch"])==set(BRANCHES)
            actual=paper_settlement(pre,feat,row["outcome"])
            assert settled["full_sheet_cashflow"]==actual
            assert actual["winning_marks"]==2 and actual["losing_marks"]==sheet["marks"]-2
            assert actual["stake_yen"]==actual["marks"]*UNIT
            assert actual["payout_yen"]==actual["3連単_paid_yen"]+actual["3連複_paid_yen"]
            for key in ("races","marks","stake_yen","payout_yen","net_yen"):
                scores[split]["full_sheet"][key]+=(1 if key=="races" else actual[key])
            winner=feat["keys"].index("-".join(map(str,row["outcome"])))
            for name in BRANCHES:
                choice=sealed["four_branches"][name]
                record=settled["per_branch"][name]
                before=coefficients[name].copy()
                assert np.allclose(before,choice["learned_delta_before"],atol=1e-14)
                q=revised_prior(bases[name],feat["x"],before)
                hashed=hashlib.sha256(q.astype("<f8").tobytes()).hexdigest()
                assert choice["probabilities_sha256"]==hashed
                assert choice["shadow_is_not_spent_in_full_sheet"]
                want_shadow=positive_shadow(pre,feat,q)
                assert choice["positive_ev_shadow"]==want_shadow
                want_cash=shadow_settlement(want_shadow,row["outcome"])
                assert record["shadow_cashflow"]==want_cash
                assert record["probability_metrics"]==metrics(feat,q,winner)
                assert record["untrained_source_probability_metrics"]==metrics(feat,bases[name],winner)
                assert record["full_sheet_same_for_all"]
                metrics_b=record["probability_metrics"]
                metrics_before=record["untrained_source_probability_metrics"]
                agg=scores[split][name]
                agg["races"]+=1
                agg["ordered_logloss"]+=metrics_b["3連単_logloss"]
                agg["unordered_logloss"]+=metrics_b["3連複_logloss"]
                agg["untrained_source_ordered_logloss"]+=metrics_before["3連単_logloss"]
                agg["untrained_source_unordered_logloss"]+=metrics_before["3連複_logloss"]
                agg["shadow_races_entered"]+=int(want_cash["marks"]>0)
                agg["shadow_marks"]+=want_cash["marks"]
                agg["shadow_stake_yen"]+=want_cash["stake_yen"]
                agg["shadow_payout_yen"]+=want_cash["payout_yen"]
                agg["shadow_net_yen"]+=want_cash["net_yen"]
                agg["shadow_hit_races"]+=int(want_cash["hit"])
                if split=="train":
                    grad=gradient_full_sheet(feat,q,winner)
                    coefficients[name]=coefficients[name]*(1-LR*L2)+LR*grad
                    norm=float(np.linalg.norm(coefficients[name]))
                    if norm>MAX_DELTA_NORM:coefficients[name]*=MAX_DELTA_NORM/norm
                else:
                    assert np.array_equal(coefficients[name],before)
                assert record["training_update"]==(split=="train")
                assert np.allclose(record["delta_before"],before,atol=1e-14)
                assert np.allclose(record["delta_after"],coefficients[name],atol=1e-14)
                assert np.all(np.isfinite(coefficients[name]))
        assert not f.readline(),"Additional sheets not corresponding to tested racecards"
    for v in summary["branch_specific_incremental_learning"]:
        assert np.allclose(coefficients[v["branch"]],
             v["incremental_market_error_coefficients"],atol=1e-14)
    for split in ("train","test"):
        expected=summary["comparison"][split]
        found=scores[split]["full_sheet"]
        for key in ("races","marks","stake_yen","payout_yen","net_yen"):
            assert expected["all_marks_virtual"][key]==found[key],(split,key)
        for branch in BRANCHES:
            expected_b=expected[branch]
            seen=scores[split][branch]
            for k,v in seen.items():
                if k in ("ordered_logloss","unordered_logloss",
                          "untrained_source_ordered_logloss",
                          "untrained_source_unordered_logloss"):continue
                assert expected_b[k]==v,(split,branch,k,v,expected_b[k])
            for actual_key,src_key in (
                ("mean_trifecta_logloss","ordered_logloss"),
                ("mean_trio_logloss","unordered_logloss"),
                ("untrained_source_trifecta_logloss","untrained_source_ordered_logloss"),
                ("untrained_source_trio_logloss","untrained_source_unordered_logloss")
            ):
                assert math.isclose(expected_b[actual_key],
                  seen[src_key]/seen["races"],rel_tol=1e-12,abs_tol=1e-12)
    published=json.loads(WEB.read_text(encoding="utf-8"))
    assert published["status"]=="COMPLETE" and published["comparison"]==summary["comparison"]
    assert published["full_sheet_stakes_not_real_money"]
    print("GEN008_ALL_MARKS_AUDIT_PASS",json.dumps({
       "training":len(train),"holdout":len(test),
       "test_full_sheet_net":summary["comparison"]["test"]["all_marks_virtual"]["net_yen"],
       "test_models":{b:{"shadow_net":summary["comparison"]["test"][b]["shadow_net_yen"],
          "new_logloss":summary["comparison"]["test"][b]["mean_trifecta_logloss"],
          "original_logloss":summary["comparison"]["test"][b]["untrained_source_trifecta_logloss"]}
         for b in BRANCHES},
       "immutable_model_snapshots":True},ensure_ascii=False),flush=True)


if __name__=="__main__":
    summary_recompute()

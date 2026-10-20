"""Post-hoc fixed-policy deletion controls; strictly NO model/policy change.

Compare the saved deletion decision to two controls at the EXACT number of
remaining tickets within each race and market:
 A) Uniform random deletion, using mathematical expected payout (no RNG)
 B) Cheapest-decimal-odds first with that same number of marks.
This tells whether the deletion *ranking* adds value beyond spending less.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import chapter1_full_sheet_deletion as engine

OUT=engine.OUT/"fixed_count_controls_2023.json"
SEED_STUDY=engine.REPORT
TAG="chapter1-full-sheet-deletion-2023-v1"


def sum_stats(rows,models,coverage):
    results={}
    for role in engine.ROLES:
        portfolio=defaultdict(lambda:{"races":0,"retained_marks":0,
                                       "retained_win_tickets":0,"stake":0.,
                                       "learned_paid":0.,"random_expected_paid":0.,
                                       "favorite_paid":0.})
        for row in (r for r in rows if r["role"]==role):
            mask=engine.delete_from_full(
                models[role][row["kind"]],row,coverage[role])
            n=len(mask)
            k=int(mask.sum())
            true=row["winner_index"]
            payout=engine.STAKE*float(row["prices"][true])
            favorite=np.argsort(row["prices"],kind="stable")[:k]
            fav_hit=bool(np.any(favorite==true))
            group=portfolio[row["kind"]]
            group["races"]+=1
            group["retained_marks"]+=k
            group["retained_win_tickets"]+=int(mask[true])
            group["stake"]+=k*engine.STAKE
            group["learned_paid"]+=payout*bool(mask[true])
            group["random_expected_paid"]+=payout*(k/n)
            group["favorite_paid"]+=payout*fav_hit
        merged={k:dict(v) for k,v in portfolio.items()}
        total={k:sum(v[k] for v in merged.values())
               for k in ("races","retained_marks","retained_win_tickets",
                         "stake","learned_paid","random_expected_paid",
                         "favorite_paid")}
        merged["combined"]=total
        for k,v in merged.items():
            v["learned_roi"]=100*v["learned_paid"]/v["stake"] if v["stake"] else None
            v["random_matched_count_expected_roi"]=100*v["random_expected_paid"]/v["stake"] if v["stake"] else None
            v["cheap_odds_same_count_roi"]=100*v["favorite_paid"]/v["stake"] if v["stake"] else None
            v["learned_minus_random_expected_yen"]=v["learned_paid"]-v["random_expected_paid"]
            v["learned_minus_cheapest_yen"]=v["learned_paid"]-v["favorite_paid"]
        results[role]=merged
    return results


def load_archives(hashes):
    models={role:{} for role in engine.ROLES}
    local=engine.OUT/"audit_immutable_weights"
    local.mkdir(exist_ok=True,parents=True)
    for role in engine.ROLES:
        for kind in engine.KINDS:
            slug="trifecta" if kind=="3連単" else "trio"
            fname="delete_"+role+"_"+slug+".joblib"
            path=local/fname
            if not path.exists():
                subprocess.run(["gh","release","download",TAG,
                                "--pattern",fname,"--dir",str(local)],
                               check=True,timeout=120)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=hashes[fname]:
                raise RuntimeError("Saved deletion model was silently modified")
            models[role][kind]=joblib.load(path)
    return models


def main():
    if OUT.exists():
        print("POSTHOC_CONTROL_REPORT_ALREADY_SEALED",flush=True)
        return
    if not SEED_STUDY.exists():
        raise RuntimeError("Cannot score controls without sealed deletion result")
    original=json.loads(SEED_STUDY.read_text("utf-8"))
    if original["status"]!="FINISHED_DELETE_FROM_FULL_SHEET_RESEARCH":
        raise RuntimeError("Not the committed original result")
    coverage=original["october_chosen_coverage"]
    models=load_archives(original["learner_artifact_sha256"])
    accuracy_asset,calibration,_=engine.duel.fetch_immutable_weights()
    all_books,_=engine.load_historical_books()
    eval_books=[r for r in all_books if "2023-11-01"<=r[1]["date"]<="2023-12-31"]
    data=engine.prepare(eval_books,accuracy_asset,calibration)
    scored=sum_stats(data["2023-11"]+data["2023-12"],models,coverage)
    for role in engine.ROLES:
        result=scored[role]["combined"]
        original_result=original["historical_frozen_checks"][role]
        if (abs(result["learned_roi"]-original_result["remaining_return_percent"])>1e-8
                or abs(result["learned_paid"]-original_result["remaining_payout_yen"])>1e-8
                or result["retained_marks"]!=original_result["marked_tickets_remaining"]):
            raise RuntimeError("Different dataset or frozen model: refuses counterfactual")
    report={
        "status":"POST_SEALED_EQUAL_MARK_COUNT_CONTROLS",
        "computed_at":datetime.now(timezone.utc).isoformat(),
        "parent_result_status":original["status"],
        "no_retraining_and_no_policy_reselection":True,
        "payouts_from_same_completed_2023_races":True,
        "random_is_expectation_not_a_randomized_lucky_sample":True,
        "uniform_random_same_number_of_marks_within_each_market_and_race":True,
        "cheapest_odds_same_number_of_marks_within_each_market_and_race":True,
        "retained_coverage_frozen_in_october":coverage,
        "results":scored,
        "limitations":[
            "This comparison is a post-hoc DIAGNOSTIC. It cannot be used to change model, weights or the previously scored November/December policy.",
            "Uniform random analytical expectancy does not require an RNG and is an exact per-race/per-market expectation, not a simulated realized ROI.",
            "The cheapest odds baseline follows the same number of tickets for every market/race, but is not an independently optimized betting system.",
            "These historical final odds are not pre-close executable."
        ]
    }
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DELETION_FIXED_COUNT_CONTROLS",json.dumps(
        {role:scored[role]["combined"] for role in engine.ROLES},
        ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

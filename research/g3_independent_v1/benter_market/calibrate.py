#!/usr/bin/env python3
"""Train/validate the *G3-adapted* Benter fundamental+public first-win bridge.

Input is HISTORICAL audited G3 race records, not current-race odds.
Requires matched historical G3 independent q (prior-year source), complete
2車単 market odds, and actual 1st result. No data available -> fails closed.
No implicit/default alpha beta. No F1 odds/outcomes allowed.
"""
from __future__ import annotations
import argparse,json,math,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from combination import fit_combination,score_combination,validate_historical_row,blended_first

def load(path):
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:raise ValueError("Cannot fit Benter market stage without actual historical G3 matched prices")
    seen=set()
    for r in rows:
        if r["race_id"] in seen:raise ValueError("Repeated historical race ID")
        seen.add(r["race_id"])
        validate_historical_row(r)
        if r["race_date"]>="2026-07-01":
            raise ValueError("Keep validation before 2026H2; no modern target labels in fit")
    return rows

def daily_ci(rows,co,phase,seed=20261018,n_rep=1600):
    grouped=defaultdict(list)
    for row in rows:
        f,p,w=validate_historical_row(row)
        c=blended_first(f,p,co["alpha"],co["beta"])
        grouped[row["race_date"]].append((
            math.log(c[w]/p[w]),
            math.log(c[w]/f[w])
        ))
    days=sorted(grouped)
    if len(days)<5:raise ValueError("Need >4 observed different G3 days")
    sum_mat=np.array([np.sum(grouped[d],axis=0) for d in days])
    counts=np.array([len(grouped[d]) for d in days])
    gen=np.random.default_rng(seed)
    indices=gen.integers(len(days),size=(n_rep,len(days)))
    samples=sum_mat[indices].sum(axis=1)/counts[indices].sum(axis=1)[:,:,None] if False else (
        sum_mat[indices].sum(axis=1)/counts[indices].sum(axis=1)[:,None])
    return {
      "days":len(days),
      "improvement_combined_over_public_mean":float(sum_mat[:,0].sum()/counts.sum()),
      "improvement_combined_over_fundamental_mean":float(sum_mat[:,1].sum()/counts.sum()),
      "public_95pct_day_cluster":[float(x) for x in np.quantile(samples[:,0],[.025,.975])],
      "fundamental_95pct_day_cluster":[float(x) for x in np.quantile(samples[:,1],[.025,.975])]
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--history",required=True,type=Path)
    ap.add_argument("--phase",required=True,choices=["final","T-15"])
    ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    rows=load(a.history)
    if any(r["odds_phase"]!=a.phase for r in rows):
        raise ValueError("Historical final and T-15 cannot be treated as the same price distribution")
    dev=[r for r in rows if "2023-01-01"<=r["race_date"]<"2025-01-01"]
    val=[r for r in rows if "2025-01-01"<=r["race_date"]<"2026-01-01"]
    test=[r for r in rows if "2026-01-01"<=r["race_date"]<"2026-07-01"]
    if min(map(len,(dev,val,test)))<300:
        raise ValueError(f"Insufficient matched independent q + exacta G3 market cases by period: {len(dev),len(val),len(test)}")
    base=fit_combination(dev,a.phase)
    val_score=score_combination(val,a.phase,base)
    # No model family search using 2026 result. Refit beta on 2023-25,
    # evaluate 2026-H1 exactly once; store all resulting diagnostics.
    co=fit_combination(dev+val,a.phase)
    test_score=score_combination(test,a.phase,co)
    bstrap=daily_ci(test,co,a.phase)
    # This gate is a statistical research criterion, not profitability.
    passes=(bstrap["public_95pct_day_cluster"][0]>0 and
            bstrap["fundamental_95pct_day_cluster"][0]>0)
    report={
        "schema":"G3_BENTER_SECOND_STAGE_MARKET_CALIBRATION_V1",
        "status":("G3_BENTER_PUBLIC_COMBINATION_CHRONOLOGICALLY_VALIDATED"
                  if passes else "G3_BENTER_PUBLIC_COMBINATION_NOT_YET_VALIDATED"),
        "alpha":co["alpha"],"beta":co["beta"],
        "scope":"G3 S級 only",
        "odds_phase":a.phase,
        "proxy_method":"NORMALIZED_COMPLETE_2CAR_TAN_INVERSE_ODDS_AS_MARKET_PROXY",
        "methodology":"Benter multinomial-logit combination f^alpha p^beta; Keirin proxy p from complete observed 2車単 quotes is researcher adaptation",
        "training_2023_2024":len(dev),
        "validation_2025":len(val),
        "refit_2023_2025":len(dev)+len(val),
        "evaluation_2026h1":len(test),
        "validation_before_refit":val_score,
        "evaluation_post_refit":test_score,
        "paired_day_interval":bstrap,
        "train_only_G3_and_prior_year_independent_first":True,
        "unobserved_new_race_odds_used_in_fit":False,
        "past_final_odds_are_not_T_15":True,
        "exotic_conditional_probabilities_after_first_marginal_reweighting_verified":False,
        "price_impact_and_multi_pool_kelly_verified":False,
        "is_betting_profitability_verified":False,
        "history_sha256":hashlib.sha256(a.history.read_bytes()).hexdigest(),
        "warning":"Passing this alpha-beta gate validates first win probability log-loss increment only. Does not validate 2/3 conditional fit under blended market or EV+/stakes. Future 7-way market comparison remains exploratory."
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("G3_BENTER_MARKET_STAGE",json.dumps({
      "status":report["status"],"alpha":co["alpha"],"beta":co["beta"],
      "training":report["refit_2023_2025"],"test":report["evaluation_2026h1"],
      "CI":bstrap
    },ensure_ascii=False),flush=True)
if __name__=="__main__":main()

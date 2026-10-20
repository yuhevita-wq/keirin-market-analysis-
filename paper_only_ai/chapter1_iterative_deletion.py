"""Continuous deletion learning from complete 3連単 and 3連複 mark sheets.

Four independently updated deletion scorers: accuracy/calibration x trifecta/trio.
Base 2023 deletion scorers + 2023 accuracy/calibration models are IMMUTABLE.
New learned incremental weight layers are trained in strictly chronological
2024 prequential order, one quarter PER GitHub workflow run.

Decisions are sealed before outcomes update weights:
   ALL marks -> sequentially delete worst expected-loss marks -> score -> learn.
No fixed mark-count cap or preselected formation. Keep original 2023 coverage
policies untouched. The 2024 history has been consulted by other studies,
so prequential ROI is exploratory, NOT a pristine untouched blind test.
After Q4: stop and preserve checkpoints. Do not use 2025 without a separate
precommitted, one-time frozen evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

import chapter1_accuracy_calibration_duel as duel
import chapter1_blind_exam as exam
import chapter1_full_sheet_deletion as original
import backtest_line_rf_2024_q1 as bt
import backtest_roi_2024_q1 as prices

ROOT=Path("docs/paper-only-ai-2023/chapter1_deletion_continual")
STATE=ROOT/"checkpoint.json"
ROLES=("accuracy","calibration")
KINDS=("3連単","3連複")
YEAR=2024
QUARTERS=(1,2,3,4)
ST=100
ETA=0.035
SHRINK=0.00003
CAP=1.5
N_FEATURES=20
COVERAGE={"accuracy":0.35,"calibration":0.50}
ORIGINAL_TAG="chapter1-full-sheet-deletion-2023-v1"
REF="docs/paper-only-ai-2023/chapter1_full_sheet_deletion/deletion_2023.json"


def base_checkpoint():
    return {
        "schema":"chapter1-autonomous-deletion-2024-v1",
        "training_started":"2024-01-01",
        "processed_quarters":[],
        "last_completed_race_date":None,
        "weight_corrections":{r:{k:[0.0]*N_FEATURES for k in KINDS} for r in ROLES},
        "parent_ids":{"accuracy":"acc-g08-c01","calibration":"g01-c01"},
        "frozen_2023_coverage":COVERAGE,
        "completed_reports":[],
        "status":"TRAINING",
        "never_tune_on_2024_development_roi":True,
    }


def load_state():
    obj=json.loads(STATE.read_text("utf-8")) if STATE.exists() else base_checkpoint()
    if obj.get("schema")!=base_checkpoint()["schema"]:
        raise RuntimeError("Wrong checkpoint schema")
    if obj.get("frozen_2023_coverage")!=COVERAGE:
        raise RuntimeError("Historical deletion policy changed after test")
    past=obj["processed_quarters"]
    if past != list(QUARTERS[:len(past)]) or len(past)>4:
        raise RuntimeError("Out-of-order or overwritten quarter")
    if not obj.get("never_tune_on_2024_development_roi"):
        raise RuntimeError("State is contaminated with outcome tuning")
    for role in ROLES:
        for kind in KINDS:
            v=np.asarray(obj["weight_corrections"][role][kind],dtype=float)
            if v.shape!=(N_FEATURES,) or not np.isfinite(v).all():
                raise RuntimeError("Invalid online deletion model weights")
    return obj


def frozen_deletion_parents():
    report=json.loads(Path(REF).read_text("utf-8"))
    if report["status"]!="FINISHED_DELETE_FROM_FULL_SHEET_RESEARCH":
        raise RuntimeError("2023 model is not an immutable research artifact")
    if report["october_chosen_coverage"]!=COVERAGE:
        raise RuntimeError("Not the exact 2023 chosen policies")
    models={role:{} for role in ROLES}
    dest=ROOT/"frozen_2023_deletion_base"
    dest.mkdir(parents=True,exist_ok=True)
    for role in ROLES:
        for kind in KINDS:
            name="delete_"+role+"_"+("trifecta" if kind=="3連単" else "trio")+".joblib"
            path=dest/name
            if not path.is_file():
                subprocess.run(["gh","release","download",ORIGINAL_TAG,
                                "--pattern",name,"--dir",str(dest)],
                               check=True,timeout=120)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=report["learner_artifact_sha256"][name]:
                raise RuntimeError("Tampered frozen deletion scorer: "+name)
            models[role][kind]=joblib.load(path)
    return models


def ticket_probabilities(row,parent,correction):
    """Normalize ticket protection weights over the entire original sheet.

    Learning affects which marks to DELETE and may never generate any ticket
    not in the complete real odds book.
    """
    x=row["x"]
    if x.shape[1]!=N_FEATURES:
        raise RuntimeError("Unexpected pre-result feature schema")
    init=np.clip(parent.predict_proba(x)[:,1],1e-9,1.0)
    z=np.log(init) + np.clip(x,-5,5) @ correction
    z-=np.max(z)
    q=np.exp(z)
    q/=q.sum()
    if not np.isfinite(q).all() or abs(float(q.sum())-1)>1e-8:
        raise RuntimeError("Not a full-ticket normalized distribution")
    return q


def remove_from_all(q,prices,coverage):
    """Sequential elimination by smallest expected retained payoff, no cap."""
    q=np.asarray(q,dtype=float)
    price=np.asarray(prices,dtype=float)
    if len(q)!=len(price) or len(q)<2 or not (0<coverage<=1):
        raise RuntimeError("Incomplete full market/invalid coverage")
    if np.ptp(q*price)<1e-10:
        return np.ones(len(q),dtype=bool)
    keep=np.ones(len(q),dtype=bool)
    mass=1.0
    # All marks are ON. Turn them OFF one at a time in this order.
    for ix in np.argsort(q*price,kind="stable"):
        if mass-q[ix]+1e-12<coverage:continue
        if keep.sum()<=1:break
        keep[ix]=False
        mass-=q[ix]
    if not keep.any():raise RuntimeError("Deleted all marks")
    return keep


def update_after_result(theta,x,q,winning_index,odds):
    """Payout-cost weighted conditional multinomial update AFTER decision.

    The winner's influence is bounded to stabilize longshot variance; the
    actual price is NEVER capped in the realized ROI accounting.
    """
    features=np.clip(np.asarray(x,dtype=float),-5,5)
    true=features[winning_index]
    expected=q @ features
    scaled=1.0 + min(math.log1p(float(odds)),8.0)/5.0
    grad=scaled*(true-expected)
    norm=float(np.linalg.norm(grad))
    if norm>10:grad=10*grad/norm
    new=(1.0-SHRINK)*np.asarray(theta,dtype=float)+ETA*grad
    new=np.clip(new,-CAP,CAP)
    if not np.isfinite(new).all():raise RuntimeError("Unsafe online update")
    return new


def source_entries(quarter):
    return (exam.quarter_sources(YEAR,quarter)+
            ["data/2024/s_class_yosen/entries.csv"])


def valid_quarter_rows(q):
    start=f"{YEAR}-{(q-1)*3+1:02d}-01"
    last_month=q*3
    end=f"{YEAR}-{last_month:02d}-{31 if q in (1,4) else 30:02d}"
    bt.LABELS="data/result_universe_v2/combined/results_2024.jsonl.gz"
    bt.START,bt.END=start,end
    bt.sources=lambda:source_entries(q)
    prices.ODDS="data/odds_only/odds_2024.jsonl.gz"
    cards,ca=bt.load_2024_line_cards()
    odds,oa=prices.collect_2024_books()
    failures=Counter()
    matched=[]
    for rid,race in cards.items():
        if rid not in odds:
            failures["missing_odds"]+=1
            continue
        try:
            bk=prices.decode_odds(odds[rid],rid,set(race["cars"]))
        except (TypeError,ValueError,KeyError) as e:
            failures[type(e).__name__+":"+str(e)[:55]]+=1
            continue
        matched.append((rid,race,bk))
    matched.sort(key=lambda a:(a[1]["date"],a[0]))
    if q in (1,2) and len(matched)<500:
        raise RuntimeError("Expected 2024 Q1/Q2 aligned races missing")
    if not matched:
        raise RuntimeError("No eligible completed races/odds for quarter")
    return matched,{"from":start,"to":end,"cards":len(cards),
                    "aligned_races":len(matched),"cards_audit":ca,
                    "odds_audit":oa,"dropped":dict(failures)}


def records_for_batch(rows,accuracy_asset,calibrated):
    chunk=[(rid,race) for rid,race,_ in rows]
    pred,_=duel.probabilities_both(chunk,accuracy_asset,calibrated)
    by_race=[]
    for rid,race,books in rows:
        objective=bt.actual_tickets(race)
        groups=[]
        for role in ROLES:
            for kind in KINDS:
                prices_for=books[kind]
                keys=list(prices_for)
                pp=np.array([float(prices_for[t]) for t in keys],dtype=float)
                qmarket=1.0/pp
                qmarket=qmarket/qmarket.sum()
                qoriginal=np.asarray([pred[role][rid][kind][t] for t in keys],dtype=float)
                x=original.feature_rows(race,keys,qmarket,qoriginal,kind)
                if x.shape[1]!=N_FEATURES or not np.all(np.isfinite(x)):
                    raise RuntimeError("Incompatible feature model")
                groups.append({
                    "role":role,"kind":kind,"prices":pp,
                    "x":x,"model_p":qoriginal,
                    # The outcome label is carried separately, never passed
                    # to the decision functions until both decisions are fixed.
                    "winner_index":keys.index(objective[kind]),
                })
        by_race.append((rid,race["date"],groups))
    return by_race


def empty_stats():
    return {"races":0,"full_marks":0,"kept_marks":0,"removed_marks":0,
            "winning_marks_kept":0,"winning_marks_deleted":0,
            "stake":0.0,"payout":0.0,"saved_stake":0.0,
            "lost_payout":0.0,"full_stake":0.0,"full_payout":0.0,
            "random_expected_payout_same_count":0.0,
            "cheapest_odds_payout_same_count":0.0,
            "parent_stake":0.0,"parent_payout":0.0}


def mark_score(stats,prices,win,mask,parent_mask):
    n=len(prices)
    k=int(mask.sum())
    hit=bool(mask[win])
    payoff=ST*float(prices[win])
    saved=ST*(n-k)
    forfeited=(not hit)*payoff
    stats["races"]+=1
    stats["full_marks"]+=n
    stats["kept_marks"]+=k
    stats["removed_marks"]+=n-k
    stats["winning_marks_kept"]+=int(hit)
    stats["winning_marks_deleted"]+=int(not hit)
    stats["stake"]+=ST*k
    stats["payout"]+=payoff*hit
    stats["saved_stake"]+=saved
    stats["lost_payout"]+=forfeited
    stats["full_stake"]+=ST*n
    stats["full_payout"]+=payoff
    stats["random_expected_payout_same_count"]+=payoff*k/n
    favorite_ix=np.argsort(prices,kind="stable")[:k]
    stats["cheapest_odds_payout_same_count"]+=payoff*int(win in favorite_ix)
    stats["parent_stake"]+=ST*int(parent_mask.sum())
    stats["parent_payout"]+=payoff*int(parent_mask[win])


def summarize(s):
    rate=lambda payment,capital:100*payment/capital if capital else None
    a=dict(s)
    a["roi_percent"]=rate(s["payout"],s["stake"])
    a["all_mark_roi_percent"]=rate(s["full_payout"],s["full_stake"])
    a["original_frozen_delete_roi_percent"]=rate(s["parent_payout"],s["parent_stake"])
    a["uniform_random_same_count_roi_percent"]=rate(
        s["random_expected_payout_same_count"],s["stake"])
    a["cheapest_odds_same_count_roi_percent"]=rate(
        s["cheapest_odds_payout_same_count"],s["stake"])
    a["profit_yen"]=s["payout"]-s["stake"]
    a["value_of_deletion_vs_all_yen"]=s["saved_stake"]-s["lost_payout"]
    if not np.isclose(a["profit_yen"]-(s["full_payout"]-s["full_stake"]),
                      a["value_of_deletion_vs_all_yen"],atol=1e-5):
        raise RuntimeError("Deleted and missed cash do not balance")
    return a


def model_digest(theta):
    buf=b"".join(np.asarray(theta[r][k],dtype="<f8").tobytes()
                  for r in ROLES for k in KINDS)
    return hashlib.sha256(buf).hexdigest()


def execute_one_quarter(state,parent_models,original_predictions):
    quarter=len(state["processed_quarters"])+1
    if quarter not in QUARTERS:
        raise RuntimeError("All 2024 quarters have been trained")
    complete,audit=valid_quarter_rows(quarter)
    theta={r:{k:np.asarray(state["weight_corrections"][r][k],dtype=float)
              for k in KINDS} for r in ROLES}
    old_sha=model_digest(theta)
    totals={r:{k:empty_stats() for k in KINDS} for r in ROLES}
    monthly={}
    last_date=state["last_completed_race_date"] or "2023-12-31"
    progress=0
    for offset in range(0,len(complete),16):
        inputs=complete[offset:offset+16]
        for rid,day,groups in records_for_batch(inputs,*original_predictions):
            if day<last_date:
                raise RuntimeError("Non-chronological training order")
            last_date=day
            # Both branches and markets choose masks BEFORE any outcome is read.
            decisions=[]
            for row in groups:
                role,kind=row["role"],row["kind"]
                frozen_model=parent_models[role][kind]
                q=ticket_probabilities(row,frozen_model,theta[role][kind])
                keep=remove_from_all(q,row["prices"],COVERAGE[role])
                baseline=original.delete_from_full(
                    frozen_model,row,COVERAGE[role])
                decisions.append((row,q,keep,baseline))
            # Reveal result only AFTER every model's deletion action is sealed.
            # A result from one market cannot affect decision in another.
            for row,q,keep,baseline in decisions:
                role,kind=row["role"],row["kind"]
                winner=row["winner_index"]
                mark_score(totals[role][kind],row["prices"],winner,keep,baseline)
            for row,q,keep,baseline in decisions:
                role,kind=row["role"],row["kind"]
                theta[role][kind]=update_after_result(
                    theta[role][kind],row["x"],q,row["winner_index"],
                    row["prices"][row["winner_index"]])
            progress+=1
        if offset%256==0 or offset+16>=len(complete):
            print("DELETE_PREQUENTIAL_TRAINING",quarter,progress,len(complete),
                  flush=True)
    report={"status":"PREQUENTIAL_TRAINING_QUARTER_COMPLETE",
            "year":YEAR,"quarter":quarter,
            "date_boundary":[audit["from"],audit["to"]],
            "races":progress,"coverage_2023_precommitted":COVERAGE,
            "old_weight_sha256":old_sha,"new_weight_sha256":model_digest(theta),
            "stats":{r:{k:summarize(totals[r][k]) for k in KINDS}
                     for r in ROLES},
            "audit":audit,
            "first_decide_then_reveal_outcome_then_update":True,
            "no_2024_roi_based_model_selection":True,
            "final_odds_not_realtime":True,
            "2024_prequential_training_not_untouched_evaluation":True,
            "comparison_is_aligned_same_races_all_four_models":True,
            "created":datetime.now(timezone.utc).isoformat()}
    ROOT.mkdir(exist_ok=True,parents=True)
    output=ROOT/f"quarter_{YEAR}_q{quarter}.json"
    if output.exists():raise RuntimeError("Quarter already graded, refusing rewrite")
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    state["processed_quarters"].append(quarter)
    state["completed_reports"].append(str(output))
    state["last_completed_race_date"]=last_date
    state["weight_corrections"]={
        r:{k:theta[r][k].tolist() for k in KINDS} for r in ROLES}
    state["status"]="TRAINING" if quarter<4 else "ALL_2024_QUARTERS_TRAINED"
    STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",
                     encoding="utf-8")
    print("DELETE_ITERATION_COMPLETE",json.dumps({
        "quarter":quarter,"races":progress,
        "model_sha256":report["new_weight_sha256"],
        "by_role":{r:{k:{
            "roi":report["stats"][r][k]["roi_percent"],
            "baseline":report["stats"][r][k]["original_frozen_delete_roi_percent"],
            "random":report["stats"][r][k]["uniform_random_same_count_roi_percent"]}
            for k in KINDS} for r in ROLES},
        "checkpoint_status":state["status"]
    },ensure_ascii=False),flush=True)


def main():
    state=load_state()
    if state["status"]=="ALL_2024_QUARTERS_TRAINED":
        print("DELETE_TRAINING_COMPLETE__NO_REPEAT_BACKTEST",flush=True)
        return
    original_model=duel.fetch_immutable_weights()
    frozen=frozen_deletion_parents()
    execute_one_quarter(state,frozen,original_model[:2])


if __name__=="__main__":
    main()

"""Frozen, out-of-sample same-race growth examination for every infant generation.

Audit-only: never updates weights, policies or bankroll checkpoints.
Fixed holdout 2023-06-01..07 MUST remain excluded from ALL later training.
The exact same joined races and deterministic per-race RNG are used for everyone.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import date, timedelta

import numpy as np

from train_week import ROOT, WEB, D, load_cards, load_results, load_odds, eligible, feature_matrix, softmax
from train_week2 import make_sheets
from train_week3 import spend_marks
from train_week4 import award_excess
from behavior_policy import BehaviorPolicy, make_learned_sheets

OUT=ROOT/"research/ai_nursery/growth_audit"
FIRST=date(2023,6,1)
LAST=FIRST+timedelta(days=6)
SEED=20261025
BOOTSTRAPS=400
COST=100


def stable_seed(race_id):
    return int.from_bytes(hashlib.sha256(("same-races-"+str(race_id)).encode()).digest()[:8],"little")


def complete_holdout():
    cards,_=load_cards()
    week={rid:obj for rid,obj in cards.items()
          if FIRST.isoformat()<=obj["race"].get("race_date","")<=LAST.isoformat()}
    results=load_results()
    prices=load_odds(set(week).intersection(results))
    absent=Counter()
    available=[]
    for rid,obj in week.items():
        if rid not in results:
            absent["missing_result"]+=1;continue
        if rid not in prices:
            absent["missing_complete_odds"]+=1;continue
        row={**obj,"odds":prices[rid],"outcome":results[rid],"race_id":rid}
        if not eligible(row):
            absent["ineligible_racecard_or_market"]+=1;continue
        available.append(row)
    available.sort(key=lambda r:(
        r["race"]["race_date"],r["race"].get("start_time",""),r["race"].get("track",""),
        int(r["race"].get("race_no","0") or 0),r["race_id"]))
    if len(available)<100:
        raise RuntimeError("Common untouched 7-day holdout too small: "+str(len(available)))
    return available,dict(absent),len(week)


def read_generations():
    registry=json.loads((WEB/"generations.json").read_text(encoding="utf-8"))
    out=[]
    for x in registry["generations"]:
        cp=ROOT/x["checkpoint"]
        if not cp.exists():
            continue
        saved=json.loads(cp.read_text(encoding="utf-8"))
        w=np.asarray(saved["weights"],dtype=np.float64)
        if w.shape!=(D,) or hashlib.sha256(w.tobytes()).hexdigest()!=saved["weights_sha256"]:
            raise RuntimeError("Bad frozen checkpoint "+x["id"])
        if saved["training_span"][1]>=FIRST.isoformat():
            raise RuntimeError("Holdout was used in training by "+x["id"])
        learned=None
        if int(x["id"].split("-")[1])>=6:
            path=ROOT/x["training_records"]/"behavior_policy.json"
            if not path.exists():
                raise RuntimeError("Missing frozen learned behavior policy for "+x["id"])
            learned=BehaviorPolicy.load(path)
            if learned.update_count==0:raise RuntimeError("Behavior never trained "+x["id"])
        out.append((x["id"],saved,w,learned))
    if len(out)<5:
        raise RuntimeError("Expected original five generations, found "+str(len(out)))
    return out


def choice_history(arms,p,row,gen,rng,bank_yen,learned):
    if gen=="GEN-001":
        k=int(rng.choice(len(arms),p=p))
        a=arms[k]
        return [{"type":a["market"],"picks":[a["selection"]],"sheet_number":1}]
    if gen in ("GEN-002","GEN-003","GEN-004","GEN-005"):
        return make_sheets(arms,p,row["odds"],row["entries"],rng)
    if learned is not None:
        pages,_=make_learned_sheets(arms,p,row["odds"],row["entries"],bank_yen,rng,learned)
        return pages
    raise RuntimeError("No trained/verified behavior policy for "+gen)


def conditional_probs(arms,all_probs,market):
    ids=[i for i,a in enumerate(arms) if a["market"]==market]
    p=all_probs[ids].astype(float)
    p=p/p.sum()
    odds=np.asarray([arms[i]["odds"] for i in ids],dtype=float)
    market_base=1.0/odds
    market_base=market_base/market_base.sum()
    return ids,p,market_base


def brier(probs,correct):
    return float(np.dot(probs,probs)+1-2*probs[correct])


def score_one(row,gen,weights,bank,learned):
    arms=feature_matrix(row["entries"],row["odds"])
    X=np.stack([a["x"] for a in arms])
    prior=np.array([a["prior"] for a in arms],dtype=float)
    prob=softmax(X@weights+prior)
    rng=np.random.default_rng(stable_seed(row["race_id"]))
    # GEN-007 optimizes pure NET profit; wallet is ledger-only, never a policy input.
    action_wallet=10000 if int(gen.split("-")[1])>=7 else bank
    marks=choice_history(arms,prob,row,gen,rng,action_wallet,learned)
    truth={
        ("3連単","-".join(map(str,row["outcome"]))),
        ("3連複","=".join(map(str,sorted(row["outcome"]))))
    }
    exposures=Counter((page["type"],ticket) for page in marks for ticket in page["picks"])
    hit=sum(exposures[t] for t in truth)>0
    stake=COST*sum(exposures.values())
    returns=int(round(sum(COST*exposures[k]*row["odds"][k[0]][k[1]]
                          for k in truth if k in exposures)))
    wallet_after_marks,_,refills=spend_marks(marks,bank)
    wallet_next,reward=award_excess(wallet_after_marks+returns)
    metrics={}
    for m,suffix in (("3連単","tan"),("3連複","fuku")):
        ids,q,base=conditional_probs(arms,prob,m)
        idx=next(j for j,i in enumerate(ids)
                 if (arms[i]["market"],arms[i]["selection"]) in truth)
        top=int(np.argmax(q))
        metrics["logloss_"+suffix]=-math.log(max(float(q[idx]),1e-30))
        metrics["market_logloss_"+suffix]=-math.log(max(float(base[idx]),1e-30))
        metrics["brier_"+suffix]=brier(q,idx)
        metrics["market_brier_"+suffix]=brier(base,idx)
        metrics["top1_hit_"+suffix]=int(top==idx)
        metrics["top1_confidence_"+suffix]=float(q[top])
    return {
        "race_id":row["race_id"],"date":row["race"]["race_date"],
        "race_type":row["race"].get("race_type",""),
        "marks":sum(exposures.values()),"sheets":len(marks),"skipped":not marks,
        "hit":hit,"stake_yen":stake,"payout_proxy_yen":returns,
        "net_proxy_yen":returns-stake,
        "wallet_before_yen":bank,"wallet_after_yen":wallet_next,
        "refill_events":len(refills),"reward_yen":reward,
        **metrics
    }


def paired_ci(new,old,metric):
    diffs=np.asarray([float(a[metric])-float(b[metric]) for a,b in zip(new,old)],dtype=float)
    seed=42
    rng=np.random.default_rng(seed)
    sampled=np.array([np.mean(rng.choice(diffs,len(diffs),replace=True))
                      for _ in range(BOOTSTRAPS)])
    lo,hi=np.quantile(sampled,[0.025,0.975])
    return {
        "mean_difference":float(diffs.mean()),
        "paired_bootstrap_95_percentile_interval":[float(lo),float(hi)]
    }


def aggregate(rows):
    def avg(f):return float(np.mean([r[f] for r in rows]))
    stake=sum(x["stake_yen"] for x in rows)
    paid=sum(x["payout_proxy_yen"] for x in rows)
    return {
        "race_count":len(rows),
        "skipped_races":sum(x["skipped"] for x in rows),
        "skip_rate":avg("skipped"),
        "mark_count":sum(x["marks"] for x in rows),
        "sheets":sum(x["sheets"] for x in rows),
        "marks_per_race":avg("marks"),
        "sheets_per_race":avg("sheets"),
        "hit_races":sum(x["hit"] for x in rows),
        "hit_rate":avg("hit"),
        "stakes_yen":stake,
        "payout_proxy_yen":paid,
        "net_proxy_yen":paid-stake,
        "roi_proxy_percent":100*paid/stake if stake else None,
        "net_per_race_yen":avg("net_proxy_yen"),
        **{key:avg(key) for key in (
            "logloss_tan","logloss_fuku","market_logloss_tan","market_logloss_fuku",
            "brier_tan","brier_fuku","market_brier_tan","market_brier_fuku",
            "top1_hit_tan","top1_hit_fuku","top1_confidence_tan","top1_confidence_fuku")}
    }


def main():
    data,missing,raw_count=complete_holdout()
    generation_data=read_generations()
    per_generation={}
    summary=[]
    for gen,checkpoint,weights,learned in generation_data:
        bank=10000
        rows=[]
        for r in data:
            item=score_one(r,gen,weights,bank,learned)
            rows.append(item)
            bank=item["wallet_after_yen"]
        per_generation[gen]=rows
        metrics=aggregate(rows)
        metrics.update({
            "generation":gen,"last_training_date":checkpoint["training_span"][1],
            "frozen_checkpoint_sha256":checkpoint["weights_sha256"],
            "learned_stop_or_skip_policy":learned is not None,
            "refill_events":sum(x["refill_events"] for x in rows),
            "reward_yen":sum(x["reward_yen"] for x in rows),
            "ending_wallet_yen":bank,
            "behavior_policy":("learned enter/skip and page stop/continue, FROZEN weights" if learned is not None
                              else "GEN-001 one-mark stochastic" if gen=="GEN-001"
                              else "GEN-002 fixed neutral 50-percent page stopping; never learns skipping"),
        })
        summary.append(metrics)
        print("GROWTH_AUDIT",gen,len(rows),"logloss_tan",round(metrics["logloss_tan"],5),
              "roi",round(metrics["roi_proxy_percent"],3),flush=True)
    baseline=per_generation["GEN-001"]
    comparisons={}
    for gen,rows in per_generation.items():
        if gen=="GEN-001":continue
        comparisons[gen]={
            "vs_GEN-001_tan_logloss":paired_ci(rows,baseline,"logloss_tan"),
            "vs_GEN-001_fuku_logloss":paired_ci(rows,baseline,"logloss_fuku"),
            "vs_GEN-001_net_per_race_yen":paired_ci(rows,baseline,"net_proxy_yen"),
            "vs_GEN-001_marks_per_race":paired_ci(rows,baseline,"marks"),
        }
    parent_comparisons={}
    learning_verdict={}
    for i,(gen,_,_,learned) in enumerate(generation_data):
        if i==0:
            learning_verdict[gen]={"classification":"baseline","description":"Original infant starting generation"}
            continue
        parent=generation_data[i-1][0]
        rows=per_generation[gen]
        old=per_generation[parent]
        deltas={
            "tan_logloss":paired_ci(rows,old,"logloss_tan"),
            "fuku_logloss":paired_ci(rows,old,"logloss_fuku"),
            "net_per_race_yen":paired_ci(rows,old,"net_proxy_yen"),
            "marks_per_race":paired_ci(rows,old,"marks"),
            "sheets_per_race":paired_ci(rows,old,"sheets"),
            "skip_rate":paired_ci(rows,old,"skipped")
        }
        parent_comparisons[gen]={"parent":parent,**deltas}
        def flag(metric):
            low,high=deltas[metric]["paired_bootstrap_95_percentile_interval"]
            return "improvement" if high<0 else "deterioration" if low>0 else "undetermined"
        def utility():
            low,high=deltas["net_per_race_yen"]["paired_bootstrap_95_percentile_interval"]
            return "improvement" if low>0 else "deterioration" if high<0 else "undetermined"
        movement=any(
            0>v["paired_bootstrap_95_percentile_interval"][1] or
            0<v["paired_bootstrap_95_percentile_interval"][0]
            for v in (deltas["marks_per_race"],deltas["sheets_per_race"],deltas["skip_rate"]))
        learning_verdict[gen]={
            "prediction_tan":flag("tan_logloss"),
            "prediction_fuku":flag("fuku_logloss"),
            "paper_utility":utility(),
            "observed_behavior_distribution_changed":bool(movement),
            "stop_skip_policy_actually_trained":learned is not None,
            "causal_behavior_learning_proven":False,
            "note":"Paired bootstrap on frozen shared history; this cannot alone establish a causal effect of training. A changed paper count is not a trained stop/skip policy if the control still uses a fixed 50-percent coin."
        }
    report={
        "title":"全世代共通の完全未学習試験（固定2023年6月）",
        "created_by":"frozen and deterministic audit; strictly no checkpoint updates",
        "status":"COMPLETE","holdout_start":FIRST.isoformat(),"holdout_end":LAST.isoformat(),
        "no_training_or_policy_tuning_on_holdout":True,
        "holdout_is_reusable_only_for_fixed_evaluation_not_model_selection":True,
        "race_count":len(data),"raw_racecards":raw_count,"excluded":missing,
        "race_id_set_sha256":hashlib.sha256("\n".join(r["race_id"] for r in data).encode()).hexdigest(),
        "generation_count":len(summary),
        "shared_test_starting_paper_wallet_yen":10000,
        "metrics":summary,
        "paired_comparisons":comparisons,
        "parent_comparisons":parent_comparisons,
        "learning_verdict":learning_verdict,
        "interpretation":{
            "prediction_scores":"All checkpoints predict identical future races; proper conditional-market log loss and Brier; lower is better.",
            "behavior_scores":"Frozen papers under same per-race seed. Older generations lack learned stop/skip policies: changes in counts do not prove learning of those actions.",
            "utility_scores":"Paper odds and stakes; mark count and exposure differ by generation, making naive total wins incomparable.",
            "market_baseline":"within each market, normalized inverse final odds, as a comparator not as the baby's training signal.",
            "statistical_uncertainty":"paired race-level bootstrap percentiles with 400 resamples; not causal attribution.",
            "no_proof":"Results are final-odds backtest only. No confirmed pre-close price or profitable live betting.",
            "holdout_policy":"Do not train, adjust thresholds, select hyperparameters or otherwise tune on June 1-7 data. If violated, create a new untouched holdout."
        }
    }
    OUT.mkdir(parents=True,exist_ok=True)
    WEB.mkdir(parents=True,exist_ok=True)
    (OUT/"holdout_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (WEB/"nursery-growth-audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (OUT/"per_race.jsonl").open("w",encoding="utf-8") as f:
        for gen,rows in per_generation.items():
            for x in rows:f.write(json.dumps({"generation":gen,**x},ensure_ascii=False,separators=(",",":"))+"\n")
    print("GROWTH_AUDIT_COMPLETE",json.dumps({
        "holdout":report["holdout_start"]+".."+report["holdout_end"],
        "races":len(data),"generations":len(summary),
        "race_id_sha256":report["race_id_set_sha256"][:12]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

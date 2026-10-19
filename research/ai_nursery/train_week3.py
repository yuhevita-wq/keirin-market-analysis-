"""GEN-003, 2023-01-15..21: inherits GEN-002 and learns virtual bankroll bookkeeping.

Only new instruction: give the baby 10,000 virtual yen initially; whenever
funds are exhausted (less than the next 100-yen mark), supply another 10,000.
Previously learned 256 weights, formations, sheet sampling, and learning rule
are NOT changed. There is no real-money transaction.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from train_week2 import (
    ROOT, WEB, D, LR, COST_PER_MARK, load_cards, load_results, load_odds,
    eligible, feature_matrix, softmax, make_sheets,
)

GEN="GEN-003"
PARENT="GEN-002"
PARENT_DIR=ROOT/"research/ai_nursery/outputs/GEN-002"
OUT=ROOT/"research/ai_nursery/outputs/GEN-003"
FIRST=date(2023,1,15)
LAST=FIRST+timedelta(days=6)
SEED=20261021
INITIAL_YEN=10000
REPLENISH_YEN=10000


def spend_marks(pages, bank):
    """Pre-result money flow. Unlimited replenishments as needed per 100-yen mark."""
    before=bank
    deposits=[]
    spent=0
    for page in pages:
        for mark_no,_ in enumerate(page["picks"],1):
            if bank<COST_PER_MARK:
                event={
                    "page_no":page["sheet_number"],"mark_no":mark_no,
                    "balance_before_yen":bank,
                    "amount_yen":REPLENISH_YEN
                }
                deposits.append(event)
                bank+=REPLENISH_YEN
            assert bank>=COST_PER_MARK
            bank-=COST_PER_MARK
            spent+=COST_PER_MARK
    assert bank==before+len(deposits)*REPLENISH_YEN-spent
    return bank,spent,deposits


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("GEN-003 already saved: refusing to overwrite learned weights or history")
    prior=json.loads((PARENT_DIR/"checkpoint.json").read_text(encoding="utf-8"))
    if prior.get("id")!=PARENT or prior.get("after_step")!=1029 or prior.get("training_span",[])[-1]!="2023-01-14":
        raise RuntimeError("Incorrect GEN-002 parent checkpoint")
    weights=np.asarray(prior["weights"],dtype=np.float64)
    parent_hash=hashlib.sha256(weights.tobytes()).hexdigest()
    if weights.shape!=(D,) or parent_hash!=prior["weights_sha256"]:
        raise RuntimeError("GEN-002 checkpoint tampered or incompatible")
    rng=np.random.default_rng(SEED)

    cards,_=load_cards()
    week={
        rid:obj for rid,obj in cards.items()
        if FIRST.isoformat()<=obj["race"].get("race_date","")<=LAST.isoformat()
    }
    actual=load_results()
    odds=load_odds(set(week).intersection(actual))
    missing=Counter()
    usable=[]
    for rid,obj in week.items():
        if rid not in actual:
            missing["missing_result"]+=1
            continue
        if rid not in odds:
            missing["missing_complete_odds"]+=1
            continue
        item={**obj,"odds":odds[rid],"outcome":actual[rid],"race_id":rid}
        if not eligible(item):
            missing["incomplete_or_inconsistent"]+=1
            continue
        usable.append(item)
    usable.sort(key=lambda r:(
        r["race"]["race_date"],
        r["race"].get("start_time",""),
        r["race"].get("track",""),
        int(r["race"].get("race_no","0") or 0),
        r["race_id"]
    ))
    if not usable:
        raise RuntimeError("No valid full-card, full-odds, results joins for the GEN-003 week")

    OUT.mkdir(parents=True,exist_ok=True)
    WEB.mkdir(parents=True,exist_ok=True)
    bank=INITIAL_YEN
    total_deposited=INITIAL_YEN
    total_replenishments=0
    journal=[]
    with (OUT/"formation_mark_sheets.jsonl").open("x",encoding="utf-8") as marks:
        for n,item in enumerate(usable,1):
            before_hash=hashlib.sha256(weights.tobytes()).hexdigest()
            before_bank=bank
            arms=feature_matrix(item["entries"],item["odds"])
            X=np.stack([a["x"] for a in arms])
            probs=softmax(X@weights+np.array([a["prior"] for a in arms]))
            sheets=make_sheets(arms,probs,item["odds"],item["entries"],rng)
            if not sheets:
                raise RuntimeError("A full paper bundle should not be empty")
            counted=Counter()
            for sheet in sheets:
                for combo in sheet["picks"]:
                    counted[(sheet["type"],combo)]+=1

            bank_after_paper,spent,refills=spend_marks(sheets,bank)
            assert spent==100*sum(counted.values())
            total_deposited+=len(refills)*REPLENISH_YEN
            total_replenishments+=len(refills)
            bank=bank_after_paper

            # Phase one: the child has ONLY racecards + odds, no outcome.
            sealed={
                "step":prior["after_step"]+n,"generation_step":n,
                "generation":GEN,"parent":PARENT,
                "race_id":item["race_id"],"date":item["race"]["race_date"],
                "track":item["race"].get("track"),
                "race_no":item["race"].get("race_no"),
                "race_type":item["race"].get("race_type"),
                "checkpoint_before_sha256":before_hash,
                "balance_before_yen":before_bank,
                "replenishments_pre_result":refills,
                "replenished_pre_result_yen":len(refills)*REPLENISH_YEN,
                "sheet_count":len(sheets),
                "sheets":sheets,
                "total_marks":sum(counted.values()),
                "unique_marks":len(counted),
                "repeated_marks":sum(counted.values())-len(counted),
                "total_staked_yen":spent,
                "balance_after_marks_yen":bank
            }
            marks.write(json.dumps(sealed,ensure_ascii=False,separators=(",",":"))+"\n")
            marks.flush()
            os.fsync(marks.fileno())

            # Phase two: reveal the finishing order only after all sheets are sealed.
            finish=item["outcome"]
            truth={
                ("3連単","-".join(map(str,finish))),
                ("3連複","=".join(map(str,sorted(finish))))
            }
            true_indices=[j for j,a in enumerate(arms)
                          if (a["market"],a["selection"]) in truth]
            if len(true_indices)!=2:
                raise RuntimeError("Complete markets must contain two outcome tickets")
            loss=-sum(math.log(max(float(probs[j]),1e-30)) for j in true_indices)/2
            weights-=LR*(probs@X-X[true_indices].mean(axis=0))
            wins=sum(counted[t] for t in truth)
            paid=int(round(sum(COST_PER_MARK*counted[t]*item["odds"][t[0]][t[1]]
                               for t in truth if t in counted)))
            bank+=paid
            journal.append({
                "step":sealed["step"],"generation_step":n,
                "race_id":sealed["race_id"],"date":sealed["date"],"track":sealed["track"],
                "race_no":sealed["race_no"],"race_type":sealed["race_type"],
                "sheet_count":sealed["sheet_count"],"marks":sealed["total_marks"],
                "spent_yen":spent,"replenishments":len(refills),
                "replenished_yen":len(refills)*REPLENISH_YEN,
                "balance_before_yen":before_bank,"balance_after_marks_yen":bank_after_paper,
                "finish_order":"-".join(map(str,finish)),
                "hit":wins>0,"winning_ticket_exposures":wins,
                "payout_proxy_yen":paid,"net_proxy_yen":paid-spent,
                "balance_after_result_yen":bank,"loss_before_update":float(loss),
                "checkpoint_before_sha256":before_hash,
                "checkpoint_after_sha256":hashlib.sha256(weights.tobytes()).hexdigest()
            })
            if n%50==0:
                print("GEN003_PROGRESS",n,len(usable),"bank",bank,
                      "replenishments",total_replenishments,flush=True)

    capital_spent=sum(x["spent_yen"] for x in journal)
    total_paid=sum(x["payout_proxy_yen"] for x in journal)
    assert bank==total_deposited+total_paid-capital_spent
    daily=[]
    for d in (FIRST+timedelta(days=i) for i in range(7)):
        rows=[x for x in journal if x["date"]==d.isoformat()]
        daily.append({
            "date":d.isoformat(),"races":len(rows),
            "sheets":sum(x["sheet_count"] for x in rows),
            "marks":sum(x["marks"] for x in rows),
            "hit_races":sum(x["hit"] for x in rows),
            "spent_yen":sum(x["spent_yen"] for x in rows),
            "payout_proxy_yen":sum(x["payout_proxy_yen"] for x in rows),
            "replenishments":sum(x["replenishments"] for x in rows),
            "closing_balance_yen":rows[-1]["balance_after_result_yen"] if rows else None,
            "mean_loss":float(np.mean([x["loss_before_update"] for x in rows])) if rows else None
        })
    summary={
        "id":GEN,"parent_id":PARENT,
        "status":"COMPLETE_BABY_BANKROLL_EXPERIMENT",
        "dates":[FIRST.isoformat(),LAST.isoformat()],
        "source":"actual historical racecard + full final odds + sealed paper formation then confirmed finish",
        "instruction_added":"最初に仮想資金10000円。次の100円のマークに足りなくなれば10000円補充する。ほかの予想ルールは追加しない。",
        "prior_checkpoint_sha256":parent_hash,
        "parent_step":prior["after_step"],
        "total_step":prior["after_step"]+len(journal),
        "races":len(journal),
        "raw_racecards_in_week":len(week),
        "missing_races":dict(missing),
        "race_types":dict(Counter(x["race_type"] for x in journal)),
        "sheets":sum(x["sheet_count"] for x in journal),
        "marks":sum(x["marks"] for x in journal),
        "multi_sheet_races":sum(x["sheet_count"]>1 for x in journal),
        "max_sheets_in_one_race":max(x["sheet_count"] for x in journal),
        "max_marks_in_one_race":max(x["marks"] for x in journal),
        "hit_races":sum(x["hit"] for x in journal),
        "hit_rate_percent":100*sum(x["hit"] for x in journal)/len(journal),
        "initial_capital_yen":INITIAL_YEN,
        "replenishment_unit_yen":REPLENISH_YEN,
        "replenishment_events":total_replenishments,
        "replenishment_amount_yen":total_replenishments*REPLENISH_YEN,
        "total_contributed_capital_yen":total_deposited,
        "total_spent_yen":capital_spent,
        "total_payout_proxy_yen":total_paid,
        "net_betting_proxy_yen":total_paid-capital_spent,
        "final_balance_yen":bank,
        "balance_minus_total_contributions_yen":bank-total_deposited,
        "roi_proxy_percent":100*total_paid/capital_spent if capital_spent else None,
        "mean_loss":float(np.mean([x["loss_before_update"] for x in journal])),
        "first_30_loss":float(np.mean([x["loss_before_update"] for x in journal[:30]])),
        "last_30_loss":float(np.mean([x["loss_before_update"] for x in journal[-30:])),
        "daily":daily,
        "limitations":[
            "No monetary transfers: all capital, replenishment, stakes and payouts are hypothetical.",
            "The public joined odds are FINAL odds, not contemporaneously available odds at sale time.",
            "Public joined odds contain only 3連単 and 3連複, not all ticket types.",
            "Replenishment is triggered only if next 100-yen mark cannot be funded; no balance goes negative.",
            "Bankroll deposits are not income: net profit = payouts minus stakes, not final balance alone.",
            "Generation-two stochastic sheet-stop policy is inherited unchanged; not a learned budgeting policy.",
            "Model still trains after each sealed historical result. These are in-sample learning outcomes.",
            "Payouts are final-odds proxy; not audited exact official payout slips."
        ]
    }
    checkpoint={
        "id":GEN,"parent_id":PARENT,"dimension":D,"after_step":summary["total_step"],
        "weights":[float(x) for x in weights],
        "weights_sha256":hashlib.sha256(weights.tobytes()).hexdigest(),
        "parent_weights_sha256":parent_hash,"random_seed":SEED,
        "training_span":[FIRST.isoformat(),LAST.isoformat()],
        "bankroll_balance_yen":bank,"replenishment_events":total_replenishments
    }
    outputs=[
        (OUT/"summary.json",summary),
        (OUT/"checkpoint.json",checkpoint),
        (WEB/"nursery-gen003-summary.json",summary),
        (WEB/"nursery-gen003-checkpoint.json",checkpoint),
        (WEB/"nursery-gen003-history.json",journal)
    ]
    for path,obj in outputs:
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2 if path.name=="summary.json" else None)+"\n",encoding="utf-8")
    with (OUT/"learning_history.jsonl").open("x",encoding="utf-8") as f:
        for x in journal:
            f.write(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n")
    print("GEN003_COMPLETE",json.dumps({
        "races":len(journal),"marks":summary["marks"],
        "replenishments":total_replenishments,"final_balance":bank,
        "payout_proxy":total_paid,"total_staked":capital_spent,
        "end_weights_sha256":checkpoint["weights_sha256"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

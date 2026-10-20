"""GEN-008 four-child ALL-COMBINATIONS paper-mark training laboratory.

No real wagers. Each eligible race buys every published 3連単 permutation
and every 3連複 combination for exactly JPY 100 in a *virtual* ledger.
This can never be a profitable buying rule in expectation under a takeout.

Each of OLD / NEW / V3 / V3-ME inherits its frozen source predictor and
independently learns a five-parameter *incremental* market-error correction.
On each TRAIN race the entire page is sealed before reading the result, then
all losing tickets and exactly one winning ticket in each pool yield a
multinomial gradient (both ordered and unordered). Each branch learns from
the complete opportunity set, not only from the marks it would have picked.

Train 2024-02-01..07. OOS 2024-02-08..14: absolutely no updates.
Report full-page accounting AND deterministic, pre-result, EV-positive
shadow tickets; shadow scores NEVER count toward full-page settlement.
"""
from __future__ import annotations

import gzip,hashlib,itertools,json,math,os
from collections import defaultdict
from pathlib import Path

import numpy as np

from train_week import ROOT,feature_matrix,softmax
from source_bridge import joined_races
from gen008_frozen_exam import load_frozen
from gen008_v3 import blend
from gen008_v3_market_error import market_features,corrected_probability

FIRST_TRAIN,LAST_TRAIN="2024-02-01","2024-02-07"
FIRST_TEST,LAST_TEST="2024-02-08","2024-02-14"
BRANCHES=("old","new","v3","v3_market_error")
OUT=ROOT/"research/ai_nursery/outputs/GEN-008-ALL-MARKS/2024-02-01-to-14"
WEB=ROOT/"docs/ai-lab/gen008-all-marks-feb2024.json"
ME_FIT=ROOT/"research/ai_nursery/outputs/GEN-008-V3-MARKET-ERROR/dec-2023-expanded-authentic-prices/market_error_fit.json"
V3_CKPT=ROOT/"research/ai_nursery/outputs/GEN-008-V3/first-independent-november-2023/v3_checkpoint.json"
UNIT=100
LR=0.018
L2=0.002
MAX_DELTA_NORM=1.6
STAGE="GEN-008-FOUR-BRANCH-ALL-MARKS-2024-FEB"


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,
                separators=(",",":"),allow_nan=False).encode()).hexdigest()


def seal(handle,value):
    handle.write(json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    handle.flush()


def aligned_distribution(pre,parents,calibration):
    """Shared 3連単 state space; independent four frozen pre-result priors."""
    new=parents["new"]
    feat=market_features(pre,new.weights,new.temperature)
    keys=feat["keys"]
    market=feat["baseline"]
    new_q=feat["new"]
    arm=feature_matrix(pre["entries"],pre["odds"])
    old_arms=[a for a in arm if a["market"]=="3連単"]
    old_logits=np.asarray([float(a["x"]@parents["old"].weights+a["prior"])
                           for a in old_arms])
    old_vector=softmax(old_logits)
    old_map={a["selection"]:float(p) for a,p in zip(old_arms,old_vector)}
    old_q=np.asarray([old_map[k] for k in keys])
    v3_q=blend(new_q,market,calibration["v3_alpha"])
    me_q=corrected_probability(feat,calibration["me_coefficients"])
    result={"old":old_q,"new":new_q,"v3":v3_q,"v3_market_error":me_q}
    for name,p in result.items():
        if abs(float(np.sum(p))-1)>1e-8 or np.any(p<=0):
            raise AssertionError("Invalid coherent ordered market: "+name)
    return feat,result


def revised_prior(base,features,delta):
    logits=np.log(np.maximum(base,1e-100))+features@delta
    logits-=logits.max()
    ex=np.exp(logits)
    q=ex/ex.sum()
    assert abs(float(q.sum())-1)<1e-8
    return q


def metrics(f,q,winner):
    group=f["trio_group"]
    p_trio=np.bincount(group,weights=q,minlength=len(f["trio_names"]))
    return {"3連単_logloss":-math.log(max(float(q[winner]),1e-50)),
            "3連複_logloss":-math.log(max(float(p_trio[group[winner]]),1e-50))}


def all_marks(pre,f):
    """Explicit entire paper ticket list, numeric odds and one JPY100 mark each."""
    k=f["keys"]
    three=[{"ticket":t,"odds":float(pre["odds"]["3連単"][t]),"yen":UNIT}
           for t in k]
    trio=[{"ticket":t,"odds":float(pre["odds"]["3連複"][t]),"yen":UNIT}
          for t in f["trio_names"]]
    cars=sorted(pre["entries"])
    assert len(three)==len(cars)*(len(cars)-1)*(len(cars)-2)
    assert len(trio)==math.comb(len(cars),3)
    assert set(x["ticket"] for x in three)==set(pre["odds"]["3連単"])
    assert set(x["ticket"] for x in trio)==set(pre["odds"]["3連複"])
    return {"race_id":pre["race_id"],"date":pre["race"]["race_date"],
            "markets":{"3連単":three,"3連複":trio},
            "mark_yen":UNIT,"marks":len(three)+len(trio),
            "virtual_stake_yen":UNIT*(len(three)+len(trio)),
            "same_full_sheet_for_four_branches":True,
            "no_real_purchase":True}


def paper_settlement(pre,f,outcome):
    """Exactly ONE paid ticket in each market, not four independent bets."""
    t="-".join(map(str,outcome))
    c="=".join(map(str,sorted(outcome)))
    number=len(f["keys"])+len(f["trio_names"])
    stake=UNIT*number
    odds3=float(pre["odds"]["3連単"][t])
    odds2=float(pre["odds"]["3連複"][c])
    paid=int(round(UNIT*(odds3+odds2)))
    return {"marks":number,"stake_yen":stake,"payout_yen":paid,
            "net_yen":paid-stake,"3連単_winner":t,"3連複_winner":c,
            "3連単_paid_yen":int(round(UNIT*odds3)),
            "3連複_paid_yen":int(round(UNIT*odds2)),
            "winning_marks":2,"losing_marks":number-2}


def gradient_full_sheet(f,q,winner):
    """All possible loss marks + winner feedback, for BOTH ticket markets.

    d log p(ordered winner) / d theta = X_win - E_q[X].
    d log p(unordered winner) / d theta =
        E_q[X|winning trio] - E_q[X].
    Every nonwinning ticket appears in the expectation term.
    """
    x=f["x"]
    avg=q@x
    mask=f["trio_group"]==f["trio_group"][winner]
    conditional=(q[mask]@x[mask])/q[mask].sum()
    g=.5*((x[winner]-avg)+(conditional-avg))
    assert g.shape==(x.shape[1],) and np.all(np.isfinite(g))
    return g


def positive_shadow(pre,f,q):
    """All positive estimated EV tickets, no fixed count/cap/formation."""
    tri_prob=np.bincount(f["trio_group"],weights=q,minlength=len(f["trio_names"]))
    three=[]
    trio=[]
    for key,p in zip(f["keys"],q):
        odds=float(pre["odds"]["3連単"][key])
        ev=float(p)*odds-1.
        if ev>0:
            three.append({"ticket":key,"ev":ev,"p":float(p),"odds":odds,"yen":UNIT})
    for key,p in zip(f["trio_names"],tri_prob):
        odds=float(pre["odds"]["3連複"][key])
        ev=float(p)*odds-1.
        if ev>0:
            trio.append({"ticket":key,"ev":ev,"p":float(p),"odds":odds,"yen":UNIT})
    return {"3連単":three,"3連複":trio}


def shadow_settlement(chosen,outcome):
    tri="-".join(map(str,outcome))
    trio="=".join(map(str,sorted(outcome)))
    marks=sum(len(v) for v in chosen.values())
    stake=marks*UNIT
    paid=sum(int(round(x["yen"]*x["odds"])) for market,win in
        (("3連単",tri),("3連複",trio))
        for x in chosen[market] if x["ticket"]==win)
    return {"marks":marks,"stake_yen":stake,"payout_yen":paid,
            "net_yen":paid-stake,"hit":paid>0,
            "not_full_sheet_cashflow":True}


def control_state(parents,calibration):
    return {"models":{
            b:{"source":"frozen September/October 2023",
               "new_parent_prediction_sha256":calibration["new_parent_hash"],
               "delta":[0.]*len(calibration["feature_names"])}
            for b in BRANCHES},
            "source_snapshots":calibration["parents"]}


def read_calibration(parents,snap):
    v3=json.loads(V3_CKPT.read_text(encoding="utf-8"))
    me=json.loads(ME_FIT.read_text(encoding="utf-8"))
    assert me["status"]=="FIT_COMPLETE"
    assert me["algorithm_sha256"]=="48c37ededf914ddffe3f20a5e85bbe18343d8206d6f116db0fdf147221c28cfe"
    assert v3["parent_new_prediction"]==snap["new"]["prediction"]
    assert me["input_NEW_weights_sha256"]==snap["new"]["prediction"]
    assert v3["alpha"]==me["frozen_v3_alpha"]
    assert me["train_dates"][1]<FIRST_TRAIN
    return {"v3_alpha":float(v3["alpha"]),
            "me_coefficients":me["coefficients"],
            "me_sha256":me["algorithm_sha256"],
            "new_parent_hash":snap["new"]["prediction"],
            "feature_names":me["feature_names"],"parents":snap,
            "source_fitted_before_first_train":True}


def select_rows():
    train,r1=joined_races(2024,FIRST_TRAIN,LAST_TRAIN)
    test,r2=joined_races(2024,FIRST_TEST,LAST_TEST)
    assert len(train)>=20 and len(test)>=20
    ids1={r["race_id"] for r in train}
    ids2={r["race_id"] for r in test}
    assert ids1.isdisjoint(ids2)
    assert all(FIRST_TRAIN<=r["race"]["race_date"]<=LAST_TRAIN for r in train)
    assert all(FIRST_TEST<=r["race"]["race_date"]<=LAST_TEST for r in test)
    return train,test,{"train":r1,"test":r2}


def run():
    if OUT.exists() or WEB.exists():
        raise RuntimeError("All-mark experiment is immutable: refuse overwrite")
    parents,snap=load_frozen()
    calib=read_calibration(parents,snap)
    train,test,source=select_rows()
    all_rows=[("train",r) for r in train]+[("test",r) for r in test]
    assert len({r["race_id"] for _,r in all_rows})==len(all_rows)
    OUT.mkdir(parents=True,exist_ok=False)
    delta={b:np.zeros(len(calib["feature_names"]),dtype=float) for b in BRANCHES}
    scored=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    perday=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    changes=[]
    stage_data=[]
    with gzip.open(OUT/"all_combinations_mark_sheets.jsonl.gz","wt",encoding="utf-8",compresslevel=6) as papers, \
         (OUT/"sealed_four_predictions.jsonl").open("x",encoding="utf-8") as decisions, \
         (OUT/"outcomes_and_learning.jsonl").open("x",encoding="utf-8") as settled:
        for index,(stage,row) in enumerate(all_rows,1):
            pre={"race_id":row["race_id"],"entries":row["entries"],
                 "race":row["race"],"odds":row["odds"]}
            f,bases=aligned_distribution(pre,parents,calib)
            full=all_marks(pre,f)
            seal(papers,full)
            sheet_id=digest(full)
            snapshots={}
            q_by_branch={}
            shadow_by_branch={}
            for b in BRANCHES:
                q=revised_prior(bases[b],f["x"],delta[b])
                q_by_branch[b]=q
                shadow=positive_shadow(pre,f,q)
                shadow_by_branch[b]=shadow
                snapshots[b]={
                  "model":b,
                  "learned_delta_before":[float(v) for v in delta[b]],
                  "probabilities_sha256":hashlib.sha256(q.astype("<f8").tobytes()).hexdigest(),
                  "logloss_is_not_available_before_result":True,
                  "positive_ev_shadow":shadow,
                  "shadow_is_not_spent_in_full_sheet":True,
                }
            entry={"stage":STAGE,"step":index,"split":stage,"date":pre["race"]["race_date"],
                   "race_id":pre["race_id"],"full_sheet_sha256":sheet_id,
                   "full_sheet_mark_count":full["marks"],
                   "all_four_branches_virtual_full_sheet_stake_yen":full["virtual_stake_yen"],
                   "pre_result":True,"four_branches":snapshots}
            seal(decisions,entry)
            # No result available to any prediction, shadow, or mark sheet above.
            outcome=row["outcome"]
            win="-".join(map(str,outcome))
            j=f["keys"].index(win)
            full_cash=paper_settlement(pre,f,outcome)
            total_record={
                "step":index,"split":stage,"race_id":pre["race_id"],
                "date":pre["race"]["race_date"],"result":list(outcome),
                "after_pre_result_seal":True,"full_sheet_cashflow":full_cash,
                "per_branch":{},
            }
            scored[stage]["full_sheet"]["races"]+=1
            for metric in ("marks","stake_yen","payout_yen","net_yen"):
                scored[stage]["full_sheet"][metric]+=full_cash[metric]
            perday[stage][entry["date"]]["full_net_yen"]+=full_cash["net_yen"]
            for b in BRANCHES:
                p=q_by_branch[b]
                ll=metrics(f,p,j)
                baseline_ll=metrics(f,bases[b],j)
                sh=shadow_settlement(shadow_by_branch[b],outcome)
                bstats=scored[stage][b]
                bstats["races"]+=1
                bstats["ordered_logloss"]+=ll["3連単_logloss"]
                bstats["unordered_logloss"]+=ll["3連複_logloss"]
                bstats["untrained_source_ordered_logloss"]+=baseline_ll["3連単_logloss"]
                bstats["untrained_source_unordered_logloss"]+=baseline_ll["3連複_logloss"]
                bstats["shadow_races_entered"]+=int(sh["marks"]>0)
                bstats["shadow_marks"]+=sh["marks"]
                bstats["shadow_stake_yen"]+=sh["stake_yen"]
                bstats["shadow_payout_yen"]+=sh["payout_yen"]
                bstats["shadow_net_yen"]+=sh["net_yen"]
                bstats["shadow_hit_races"]+=int(sh["hit"])
                perday[stage][entry["date"]][b+"_shadow_net"]+=sh["net_yen"]
                before=delta[b].copy()
                if stage=="train":
                    gradient=gradient_full_sheet(f,p,j)
                    delta[b]=delta[b]*(1.-LR*L2)+LR*gradient
                    norm=float(np.linalg.norm(delta[b]))
                    if norm>MAX_DELTA_NORM:
                        delta[b]*=MAX_DELTA_NORM/norm
                    assert np.all(np.isfinite(delta[b]))
                else:
                    assert np.array_equal(delta[b],before)
                total_record["per_branch"][b]={
                  "probability_metrics":ll,"untrained_source_probability_metrics":baseline_ll,
                  "shadow_cashflow":sh,
                  "delta_before":[float(x) for x in before],
                  "delta_after":[float(x) for x in delta[b]],
                  "training_update":stage=="train",
                  "full_sheet_same_for_all":True}
            seal(settled,total_record)
            if index%100==0:
                print("GEN008_ALL_MARKS_PROGRESS",index,len(all_rows),stage,flush=True)
    ending={name:{"prediction":__import__("gen008_experiment").model_digest(model.weights),
                  "behavior":model.policy.digest(),"temperature":model.temperature}
            for name,model in parents.items()}
    assert ending==snap, "Source OLD/NEW predictors were modified"
    for b in BRANCHES:
        changes.append({"branch":b,"incremental_market_error_coefficients":[float(v) for v in delta[b]],
                        "trained_on_full_result_likelihood_only":True})
    comparison={}
    for stage in ("train","test"):
        full=dict(scored[stage]["full_sheet"])
        full["roi_percent"]=100*full["payout_yen"]/full["stake_yen"]
        total={"all_marks_virtual":full}
        for b in BRANCHES:
            metrics_agg=dict(scored[stage][b])
            nr=metrics_agg["races"]
            metrics_agg["mean_trifecta_logloss"]=metrics_agg.pop("ordered_logloss")/nr
            metrics_agg["mean_trio_logloss"]=metrics_agg.pop("unordered_logloss")/nr
            metrics_agg["untrained_source_trifecta_logloss"]=metrics_agg.pop("untrained_source_ordered_logloss")/nr
            metrics_agg["untrained_source_trio_logloss"]=metrics_agg.pop("untrained_source_unordered_logloss")/nr
            metrics_agg["shadow_roi_percent"]=(100*metrics_agg["shadow_payout_yen"]/metrics_agg["shadow_stake_yen"]
                                            if metrics_agg["shadow_stake_yen"] else None)
            total[b]=metrics_agg
        comparison[stage]=total
    state={"stage":STAGE,"status":"COMPLETE","kind":"FULL_FIELD_MARK_SHEET_SUPERVISION",
           "train_dates":[FIRST_TRAIN,LAST_TRAIN],"test_dates":[FIRST_TEST,LAST_TEST],
           "train_races":len(train),"test_races":len(test),
           "controls_frozen_start":snap,"controls_frozen_end":ending,
           "me_original_digest":calib["me_sha256"],
           "original_models_untouched":True,
           "branch_specific_incremental_learning":changes,
           "all_3rentan_and_3renpuku_marked":True,
           "100_yen_per_combination":True,"all_four_same_full_sheet":True,
           "full_sheet_stakes_not_real_money":True,
           "outcome_read_after_pre_result_seal":True,
           "test_period_never_updates_weights_or_coefficients":True,
           "counterfactual_shadow_not_in_full_cash":True,
           "frozen_final_odds_only":True,
           "source_reports":source,"comparison":comparison,
           "daily":{split:[{"date":d,**dict(v)} for d,v in sorted(days.items())]
                    for split,days in perday.items()},
           "note":"Buying every outcome trivially hits both pools, without implying a return advantage. Market-error models are judged by frozen out-of-sample scoring and separated EV-positive shadow."}
    (OUT/"summary.json").write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    WEB.parent.mkdir(parents=True,exist_ok=True)
    WEB.write_text(json.dumps({"stage":STAGE,"status":"COMPLETE",
        "train_races":len(train),"test_races":len(test),
        "comparison":comparison,"source":str((OUT/"summary.json").relative_to(ROOT)),
        "full_sheet_stakes_not_real_money":True},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("GEN008_ALL_MARKS_DONE",json.dumps({
        "train_races":len(train),"test_races":len(test),
        "full_sheet":comparison["test"]["all_marks_virtual"],
        "test_models":{b:comparison["test"][b] for b in BRANCHES},
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()

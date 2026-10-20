"""Deletion-first 3連単/3連複 learner for Chapter 1.

State transition is FULL_SHEET -> remove individual tickets, not generation
of a formation from an empty card. An action model learns how costly it is
to delete the actual winning mark, using the completed race's payout as the
label. No race result appears in pre-race features or the deletion decisions.

Frozen parent branches:
  accuracy = acc-g08-c01, calibration = g01-c01.
New training (both branches independently): September 2023.
Coverage target selection: October 2023 only.
Historical scoring without re-selection: November and December 2023.
All these dates have been consulted in prior studies, so this is an
exploratory historical experiment, NOT an independent science holdout.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import chapter1_accuracy_calibration_duel as duel
import chapter1_market_error as features
import backtest_line_rf_2024_q1 as bt
import backtest_roi_2024_q1 as pricing

OUT = Path("docs/paper-only-ai-2023/chapter1_full_sheet_deletion")
REPORT = OUT / "deletion_2023.json"
KINDS = ("3連単","3連複")
ROLES = ("accuracy","calibration")
COVERAGE = (0.20,0.35,0.50,0.65,0.80,0.90,0.98)
STAKE = 100
MAX_WINNER_LABEL_WEIGHT = 250
SEED = 230306021


def feature_rows(race, keys, market_p, model_p, kind):
    """Only card/line/odds/model-prior values, never the true finish."""
    odds_only = features._feature_matrix(race, keys, market_p, kind)
    model_p=np.asarray(model_p,dtype=float)
    if len(model_p)!=len(keys) or np.any(model_p<=0):
        raise RuntimeError("Model probability and ticket keys misaligned")
    # Conditioned on a FULL ticket universe, the learning signal is a
    # marginal mark deletion, not the chosen 1st/2nd/3rd combinational mask.
    columns=np.column_stack((
        odds_only,
        np.clip(np.log(model_p/market_p),-5,5),
        np.clip(np.log(model_p*len(keys)),-6,6),
        np.clip(np.log(market_p*len(keys)),-6,6),
    ))
    if not np.isfinite(columns).all():raise RuntimeError("Invalid deletion features")
    return columns.astype(np.float32)


def load_historical_books():
    bt.LABELS="data/result_universe_v2/combined/results_2023.jsonl.gz"
    bt.START,bt.END="2023-09-01","2023-12-31"
    bt.sources=lambda:[
        "data/a_class_12/2023_09_10/entries.csv",
        "data/a_class_12/2023_11_12/entries.csv",
        "data/a3_challenge/2023_09_10/entries.csv",
        "data/a3_challenge/2023_11_12/entries.csv",
        "data/2023/s_class_f1_all_parts/2023_q3/entries.csv",
        "data/2023/s_class_f1_all_parts/2023_q4/entries.csv",
        "data/2023/s_class_yosen/entries.csv",
    ]
    pricing.ODDS="data/odds_only/odds_2023.jsonl.gz"
    cards,card_audit=bt.load_2024_line_cards()
    odds,odds_audit=pricing.collect_2024_books()
    joined=[]
    failures=Counter()
    for rid,race in cards.items():
        if rid not in odds:
            failures["no_matching_odds"]+=1
            continue
        try:
            book=pricing.decode_odds(odds[rid],rid,set(race["cars"]))
        except (ValueError,KeyError,TypeError) as exc:
            failures[type(exc).__name__+":"+str(exc)[:70]]+=1
            continue
        joined.append((rid,race,book))
    joined.sort(key=lambda t:(t[1]["date"],t[0]))
    months=Counter(row[1]["date"][:7] for row in joined)
    for month in ("2023-09","2023-10","2023-11","2023-12"):
        if months[month]<150:
            raise RuntimeError("Insufficient paired complete full-sheet data: "+month)
    return joined,{"eligible_cards":len(cards),
                   "complete_book_races":len(joined),
                   "months":dict(months),
                   "reasons_for_exclusion":dict(failures),
                   "cards":card_audit,"odds":odds_audit}


def prepare(rows,accuracy_asset,calibration):
    """Precompute predictions for all marks; never consult the outcome in
    constructing the sheet or its covariates."""
    by_month=defaultdict(list)
    for offset in range(0,len(rows),18):
        segment=rows[offset:offset+18]
        chunk=[(rid,race) for rid,race,_ in segment]
        predictions,_=duel.probabilities_both(chunk,accuracy_asset,calibration)
        for rid,race,book in segment:
            truth=bt.actual_tickets(race)
            for kind in KINDS:
                keys=list(book[kind])
                prices=np.asarray([book[kind][k] for k in keys],dtype=np.float64)
                model_market=1/prices
                model_market/=model_market.sum()
                idx=keys.index(truth[kind])
                for role in ROLES:
                    model_p=np.asarray([predictions[role][rid][kind][k] for k in keys],
                                       dtype=np.float64)
                    if not np.isclose(np.sum(model_p),1,rtol=0,atol=1e-7):
                        raise RuntimeError("Incomplete original all-mark probability")
                    x=feature_rows(race,keys,model_market,model_p,kind)
                    by_month[race["date"][:7]].append({
                        "race_id":rid,"date":race["date"],"kind":kind,
                        "role":role,"x":x,"model_p":model_p,
                        "prices":prices,"winner_index":idx,
                        "tickets":tuple(keys),
                    })
        if offset%270==0 or offset+18>=len(rows):
            print("DELETE_ALL_MARKS_PREDICTED",offset+len(segment),len(rows),flush=True)
    return by_month


def cost_aware_train(rows,kind,role):
    """A deleted losing mark saves 100 yen; a deleted winner forfeits
    its actual payout. Weighted supervised mark-level deletion-cost labels.
    Scores are rankings of protected marks, not calibrated probabilities.
    """
    subset=[r for r in rows if r["kind"]==kind and r["role"]==role]
    if len(subset)<150:raise RuntimeError("No action-learning cohort")
    X=np.concatenate([r["x"] for r in subset])
    y=np.zeros(len(X),dtype=np.int8)
    weights=np.ones(len(X),dtype=np.float64)
    position=0
    for row in subset:
        i=position+row["winner_index"]
        y[i]=1
        weights[i]=min(float(row["prices"][row["winner_index"]]),
                       MAX_WINNER_LABEL_WEIGHT)
        position+=len(row["x"])
    if int(y.sum())!=len(subset) or position!=len(y):
        raise RuntimeError("Winner label problem in all-mark learner")
    model=HistGradientBoostingClassifier(
        learning_rate=0.055,max_iter=105,max_leaf_nodes=15,
        max_depth=5,min_samples_leaf=110,l2_regularization=15,
        early_stopping=False,random_state=SEED+(0 if kind=="3連単" else 10)
        +(0 if role=="accuracy" else 1))
    model.fit(X,y,sample_weight=weights)
    return model,{"races":len(subset),"full_marks":len(y),
                  "actual_winning_marks":int(y.sum()),
                  "winner_label":("paid odds capped at "
                                  +str(MAX_WINNER_LABEL_WEIGHT)
                                  +" in TRAINING loss only"),
                  "total_label_weight":float(weights.sum()),
                  "model_params":model.get_params()}


def delete_from_full(model,row,coverage):
    """Begin with all marks. Delete one worst mark at a time until the
    retained probability mass would drop below the selected coverage.
    No fixed formation, rank box, mark count or preferred winners.
    """
    if not 0<coverage<=1:raise ValueError("Coverage outside 0..1")
    n=len(row["prices"])
    scores=model.predict_proba(row["x"])[:,1]
    if n<3 or len(scores)!=n:raise RuntimeError("Invalid full marked sheet")
    if not np.isfinite(scores).all():raise RuntimeError("Invalid deletion score")
    # A tie means the model has not learned a reliable removal ordering:
    # retaining the entire sheet is the conservative experimental action.
    if float(np.ptp(scores))<1e-9:
        return np.ones(n,dtype=bool)
    order=np.argsort(scores,kind="stable")
    marked=np.ones(n,dtype=bool)
    mass=1.0
    for i in order:
        after=mass-float(row["model_p"][i])
        if after+1e-12 < coverage:
            continue
        if np.count_nonzero(marked)<=1:
            break
        marked[i]=False
        mass=after
    if not marked.any():
        raise RuntimeError("No retained mark: broken coverage constraint")
    return marked


def score_rows(rows,trained,coverage_by_role,include_months=True):
    out={}
    for role in ROLES:
        records=[r for r in rows if r["role"]==role]
        full_stake=full_return=stake=paid=forfeited=0.0
        full_marks=retained=deleted_winners=winning_kept=0
        full_races=set()
        monthly=defaultdict(lambda:{"full_stake":0.,"full_payout":0.,
                                     "stake":0.,"payout":0.,"marks_kept":0,
                                     "marks_deleted":0,"winners_kept":0,
                                     "winners_deleted":0})
        by_market=defaultdict(lambda:{"stake":0.,"payout":0.,
                                      "full_stake":0.,"full_payout":0.,
                                      "kept":0,"all":0,"winners_kept":0})
        for row in records:
            kind=row["kind"]
            model=trained[role][kind]
            kept=delete_from_full(model,row,coverage_by_role[role])
            n=len(kept)
            keep_count=int(kept.sum())
            idx=row["winner_index"]
            hit=bool(kept[idx])
            payoff=STAKE*float(row["prices"][idx])
            total=STAKE*n
            cost=STAKE*keep_count
            winnings=payoff if hit else 0.
            full_races.add(row["race_id"])
            full_marks+=n
            retained+=keep_count
            full_stake+=total
            full_return+=payoff
            stake+=cost
            paid+=winnings
            forfeited+=payoff-winnings
            winning_kept+=int(hit)
            deleted_winners+=int(not hit)
            key=row["date"][:7]
            for detail in (monthly[key],by_market[kind]):
                detail["stake"]+=cost
                detail["payout"]+=winnings
            m=monthly[key]
            m["full_stake"]+=total
            m["full_payout"]+=payoff
            m["marks_kept"]+=keep_count
            m["marks_deleted"]+=n-keep_count
            m["winners_kept"]+=int(hit)
            m["winners_deleted"]+=int(not hit)
            m=by_market[kind]
            m["full_stake"]+=total
            m["full_payout"]+=payoff
            m["kept"]+=keep_count
            m["all"]+=n
            m["winners_kept"]+=int(hit)
        saved=full_stake-stake
        if not np.isclose((paid-stake)-(full_return-full_stake),
                          saved-forfeited,atol=1e-5):
            raise AssertionError("Deletion accounting must reconcile exactly")
        def roi(st,ret):
            return 100*ret/st if st else None
        out[role]={
            "races":len(full_races),
            "eligible_full_marks":full_marks,
            "marked_tickets_remaining":retained,
            "deleted_marks":full_marks-retained,
            "deleted_fraction":(full_marks-retained)/full_marks if full_marks else None,
            "winning_tickets_retained":winning_kept,
            "winning_tickets_deleted":deleted_winners,
            "full_sheet_investment_yen":round(full_stake,2),
            "full_sheet_payout_yen":round(full_return,2),
            "full_sheet_return_percent":roi(full_stake,full_return),
            "saved_investment_yen":round(saved,2),
            "forfeited_winning_payout_yen":round(forfeited,2),
            "net_change_from_full_sheet_yen":round(saved-forfeited,2),
            "remaining_investment_yen":round(stake,2),
            "remaining_payout_yen":round(paid,2),
            "remaining_net_profit_yen":round(paid-stake,2),
            "remaining_return_percent":roi(stake,paid),
            "per_month":({k:{**v,
                             "return_percent":roi(v["stake"],v["payout"]),
                             "full_return_percent":roi(v["full_stake"],v["full_payout"])}
                          for k,v in sorted(monthly.items())}
                         if include_months else {}),
            "by_market":{k:{**v,
                            "return_percent":roi(v["stake"],v["payout"]),
                            "full_return_percent":roi(v["full_stake"],v["full_payout"])}
                         for k,v in sorted(by_market.items())},
        }
    if out["accuracy"]["races"]!=out["calibration"]["races"]:
        raise RuntimeError("Duel branches not compared on the same races")
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if REPORT.exists():
        print("DELETE_FIRST_HISTORICAL_REPORT_SEALED_NO_RETUNING",flush=True)
        return
    accuracy_model,calibration_model,original_checksums=duel.fetch_immutable_weights()
    joined,audit=load_historical_books()
    data=prepare(joined,accuracy_model,calibration_model)
    train=data["2023-09"]
    develop=data["2023-10"]
    historical=data["2023-11"]+data["2023-12"]
    trained={role:{} for role in ROLES}
    traininfo={role:{} for role in ROLES}
    archives={}
    for role in ROLES:
        for kind in KINDS:
            model,info=cost_aware_train(train,kind,role)
            trained[role][kind]=model
            traininfo[role][kind]=info
            name="delete_"+role+"_"+("trifecta" if kind=="3連単" else "trio")+".joblib"
            path=OUT/name
            joblib.dump(model,path,compress=3)
            archives[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            print("DELETE_SCORE_LEARNED",role,kind,info["full_marks"],flush=True)
    table=[]
    picked={}
    for coverage in COVERAGE:
        check=score_rows(develop,trained,{role:coverage for role in ROLES},
                         include_months=False)
        for role in ROLES:
            m=check[role]
            # Oct training is the ONLY place where retention policy is
            # chosen. A meaningful sample prevents picking on 1 lucky hit.
            eligible=(m["marked_tickets_remaining"]>=500 and
                      m["winning_tickets_retained"]>=25)
            table.append({
                "role":role,"coverage":coverage,"october_qualified":eligible,
                "october_return_percent":m["remaining_return_percent"],
                "october_return_yen":m["remaining_payout_yen"],
                "october_net_profit_yen":m["remaining_net_profit_yen"],
                "october_deleted_marks":m["deleted_marks"],
                "october_winners_kept":m["winning_tickets_retained"],
            })
    for role in ROLES:
        eligible=[r for r in table if r["role"]==role and r["october_qualified"]]
        if not eligible:
            raise RuntimeError("No sufficiently sampled October deletion policy")
        # Robust objective is ROI; net profit and coverage only deterministic
        # tie-breakers. NEVER inspect November/December to choose.
        best=max(eligible,key=lambda r:(r["october_return_percent"],
                                        r["october_net_profit_yen"],
                                        r["coverage"]))
        picked[role]=best["coverage"]
    checks=score_rows(historical,trained,picked)
    pass_month={}
    for role in ROLES:
        mon=checks[role]["per_month"]
        pass_month[role]=all(mon[month]["return_percent"] is not None and
                             mon[month]["return_percent"]>100
                             for month in ("2023-11","2023-12"))
    report={
        "status":"FINISHED_DELETE_FROM_FULL_SHEET_RESEARCH",
        "created_at":datetime.now(timezone.utc).isoformat(),
        "frozen_parent_sha256":original_checksums,
        "learner_artifact_sha256":archives,
        "training_period":"2023-09",
        "deletion_policy_selection_period":"2023-10",
        "frozen_historical_scoring_period":["2023-11","2023-12"],
        "data_audit":audit,
        "both_original_sheets_fully_marked_first":True,
        "result_not_used_to_decide_pre_race_deletions":True,
        "full_sheet_before_every_removal":True,
        "ranking_trained_to_protect_high_payout_winning_marks":True,
        "coverage_candidates_not_ticket_count_limits":COVERAGE,
        "october_candidate_table":table,
        "october_chosen_coverage":picked,
        "historical_frozen_checks":checks,
        "both_months_profitable":pass_month,
        "chapter1_goal_reached":False,
        "limitations":[
            "The two original AIs are already 2023-trained/selected frozen models. The accuracy branch was selected on October, so October tuning is not an untouched nested holdout.",
            "g01-c01 used September labels for its earlier temperature fitting, so September action training does not provide an independent statistical test.",
            "November/December were consulted in other past studies; this is an exploratory historical check only.",
            "All odds are FINAL historical odds, not necessarily bookable 15 minutes before close.",
            "Weighted training labels cap odds at 250 for stability; ALL realized payouts are scored at genuine recorded full odds.",
            "A positive net change versus buying every ticket does NOT imply a profitable portfolio: require remaining_return_percent > 100 separately.",
            "Both ticket markets are evaluated in full. Not a winning formation or preselected ticket-count template.",
            "This tests payoff-sensitive ordered DELETION, distinct from the old p*odds>1 buy-from-empty policy.",
        ],
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                      encoding="utf-8")
    print("FULL_SHEET_DELETION_FINISHED",json.dumps({
        "matched_races":audit["complete_book_races"],"coverage":picked,
        "accuracy":{k:checks["accuracy"][k] for k in
                    ("remaining_return_percent","deleted_marks",
                     "remaining_net_profit_yen","winning_tickets_retained")},
        "calibration":{k:checks["calibration"][k] for k in
                       ("remaining_return_percent","deleted_marks",
                        "remaining_net_profit_yen","winning_tickets_retained")},
        "both_months_profitable":pass_month,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

"""Second generation: nonlinear price-error learning from finalized 2023 races.

Train boosted rider/line/market predictors on 2023 Jan-Aug outcomes.
Calibrate with 2023 September, choose a *development* purchase policy in
October, then use Nov/Dec only for a frozen chronological verification.
No model or ticket rule is refitted on Nov/Dec or later results.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import chapter1_market_error as base

ROOT=Path("docs/paper-only-ai-2023/chapter1_market_boost")
REPORT=ROOT/"report_2023.json"
KINDS=("3連単","3連複")
ALPHAS=(0.15,0.3,0.5,0.8,1.0,1.35,1.7)
BUFFERS=(0.,0.05,0.1,0.2,0.35)
TICKET_MIN=300
HIT_MIN=20
SEED=230306021


def extend_features(row):
    _,_,features,logmarket,_,_=row
    # Odds-derived quantities are available BEFORE the race.
    augmented=np.column_stack((
        features, logmarket/9,
        np.full(len(logmarket),np.log(len(logmarket))/6)
    ))
    return np.asarray(augmented,dtype=np.float32)


def training_arrays(rows):
    X=np.concatenate([extend_features(r) for r in rows],axis=0)
    y=np.zeros(len(X),dtype=np.int8)
    pos=0
    for r in rows:
        y[pos+r[-1]]=1
        pos+=len(r[3])
    if pos != len(X) or y.sum()!=len(rows):
        raise RuntimeError("Each training race must have exactly one winning ticket")
    return X,y


def fit_boost(rows,kind):
    X,y=training_arrays(rows)
    model=HistGradientBoostingClassifier(
        loss="log_loss",learning_rate=0.055,max_iter=100,
        max_leaf_nodes=15,max_depth=6,min_samples_leaf=100,
        l2_regularization=20.0,max_bins=128,
        early_stopping=False,random_state=SEED+(0 if kind=="3連単" else 1))
    # Fit all tickets, not just winners or the most popular picks.
    model.fit(X,y)
    return model,{"samples":int(len(X)),"winners":int(y.sum()),
                  "feature_count":int(X.shape[1]),
                  "model_parameters":model.get_params()}


def attach_boost_scores(rows,model,batch=50):
    out=[]
    for i in range(0,len(rows),batch):
        chunk=rows[i:i+batch]
        lengths=[len(r[3]) for r in chunk]
        X=np.concatenate([extend_features(r) for r in chunk],axis=0)
        p=model.predict_proba(X)[:,1]
        offset=0
        for r,n in zip(chunk,lengths):
            log_score=np.log(np.clip(p[offset:offset+n],1e-12,1.0))
            out.append((*r,log_score))
            offset+=n
    return out


def blend(row, alpha):
    logmarket=row[3]
    loglearned=row[6]
    z=(1-alpha)*logmarket+alpha*loglearned
    z=z-float(np.max(z))
    p=np.exp(z)
    p/=p.sum()
    if abs(p.sum()-1)>1e-8 or not np.all(np.isfinite(p)):
        raise RuntimeError("Bad boosted price probability")
    return p


def nll(rows,alpha):
    return float(np.mean([-np.log(max(float(blend(r,alpha)[r[5]]),1e-14))
                          for r in rows]))


def book_return(data,alphas,margin):
    investment=payout=predicted=0.
    tickets=hits=0
    ledger=defaultdict(lambda:{"investment_yen":0.,"payout_yen":0.,
                               "tickets":0,"hits":0})
    for kind in KINDS:
        for row in data[kind]:
            prob=blend(row,alphas[kind])
            allowed=prob*row[4]>(1.0+margin)
            cnt=int(allowed.sum())
            winner=bool(allowed[row[5]])
            cost=100.0*cnt
            returns=100.0*float(row[4][row[5]]) if winner else 0.0
            investment+=cost
            payout+=returns
            predicted+=100.*float(np.sum((prob*row[4])[allowed]))
            tickets+=cnt
            hits+=int(winner)
            mon=ledger[row[0][:7]]
            mon["investment_yen"]+=cost
            mon["payout_yen"]+=returns
            mon["tickets"]+=cnt
            mon["hits"]+=int(winner)
    return {"purchased_tickets":tickets,"winning_tickets":hits,
            "investment_yen":round(investment,2),
            "payout_yen":round(payout,2),
            "net_profit_yen":round(payout-investment,2),
            "return_percent":100*payout/investment if investment else None,
            "expected_return_percent":100*predicted/investment if investment else None,
            "months":{k:{**v,"return_percent":100*v["payout_yen"]/v["investment_yen"]
                        if v["investment_yen"] else None}
                      for k,v in sorted(ledger.items())}}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if REPORT.exists():
        print("NONLINEAR_HISTORICAL_CHECK_ALREADY_DONE__REFUSE_RETUNING",flush=True)
        return
    splits,audit=base._load()
    models={}
    train_audit={}
    validations={}
    predictions={kind:{} for kind in KINDS}
    for kind in KINDS:
        model,fit_audit=fit_boost(splits[kind]["train"],kind)
        models[kind]=model
        train_audit[kind]=fit_audit
        for stage in ("validation","develop","test"):
            predictions[kind][stage]=attach_boost_scores(splits[kind][stage],model)
        values=[{"alpha":a,"september_nll":nll(predictions[kind]["validation"],a)}
                for a in ALPHAS]
        best=min(values,key=lambda v:v["september_nll"])
        validations[kind]={"candidate_alphas":values,"best_alpha":best["alpha"],
                            "market_nll":nll(predictions[kind]["validation"],0)}
        blob=ROOT/("model_"+("trifecta" if kind=="3連単" else "trio")+".joblib")
        joblib.dump(model,blob,compress=3)
        validations[kind]["artifact_sha256"]=hashlib.sha256(blob.read_bytes()).hexdigest()
        print("BOOST_STAGE_TRAINED",kind,fit_audit,validations[kind],flush=True)
    # Never let November/December influence which October development policy wins.
    development={kind:predictions[kind]["develop"] for kind in KINDS}
    test={kind:predictions[kind]["test"] for kind in KINDS}
    model_choices={
        "market":{kind:0.0 for kind in KINDS},
        "september_best_nll":{kind:validations[kind]["best_alpha"] for kind in KINDS},
    }
    for alpha in ALPHAS:
        model_choices["common_alpha_"+str(alpha)]={kind:alpha for kind in KINDS}
    candidates=[]
    for label,alphas in model_choices.items():
        for margin in BUFFERS:
            oct_score=book_return(development,alphas,margin)
            allowed=(oct_score["purchased_tickets"]>=TICKET_MIN
                     and oct_score["winning_tickets"]>=HIT_MIN)
            candidates.append({"name":label,"alphas":alphas,
                               "entry_margin":margin,
                               "october":oct_score,
                               "eligible":allowed})
    eligible=[c for c in candidates if c["eligible"]]
    choice=max(eligible,key=lambda c:(c["october"]["return_percent"],
                                     c["october"]["net_profit_yen"])) if eligible else None
    check=None
    if choice:
        historical=book_return(test,choice["alphas"],choice["entry_margin"])
        nov=historical["months"].get("2023-11",{})
        dec=historical["months"].get("2023-12",{})
        passed=(historical["return_percent"] is not None and
                historical["return_percent"]>100 and
                historical["purchased_tickets"]>=TICKET_MIN and
                all(v.get("return_percent") is not None and
                    v["return_percent"]>100 for v in (nov,dec)))
        check={"selected_on_october":{"name":choice["name"],
                "entry_margin":choice["entry_margin"],
                "alphas":choice["alphas"],
                "october_roi":choice["october"]["return_percent"]},
               "nov_dec_historical_verification":historical,
               "both_months_profitable":bool(passed)}
    report={
        "schema":"nonlinear-market-error-learner-v1",
        "date":datetime.now(timezone.utc).isoformat(),
        "data_provenance":audit,"fit":train_audit,
        "september_calibration":validations,
        "october_choices":[{"name":x["name"],"entry_margin":x["entry_margin"],
                             "eligible":x["eligible"],
                             "return_percent":x["october"]["return_percent"],
                             "tickets":x["october"]["purchased_tickets"],
                             "hits":x["october"]["winning_tickets"],
                             "net_yen":x["october"]["net_profit_yen"]}
                            for x in candidates],
        "nov_dec_frozen_check":check,
        "learning_date_end":"2023-08-31",
        "september_tuning_did_not_use_nov_dec":True,
        "october_policy_did_not_use_nov_dec":True,
        "future_2024plus_used":False,
        "fully_untouched_test":False,
        "betting_at_actual_15min_advance_odds":False,
        "final_odds_only":True,
        "model_artifacts":{kind:validations[kind]["artifact_sha256"] for kind in KINDS},
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                      encoding="utf-8")
    print("MARKET_BOOST_COMPLETE",json.dumps({
        "choice":check["selected_on_october"] if check else None,
        "nov_dec":check["nov_dec_historical_verification"]["return_percent"] if check else None,
        "profitable_both_months":check["both_months_profitable"] if check else False
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

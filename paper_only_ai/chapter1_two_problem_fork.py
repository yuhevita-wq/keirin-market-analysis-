"""Fork both frozen 2024 Chapter-1 paper AIs into two separate learning objectives.

Four INDEPENDENT lines:
  accuracy/probability, accuracy/price, calibration/probability, calibration/price.
Each line is maintained for BOTH complete 3連単 and 3連複 ticket markets.

PROBABILITY: score all tickets as a normalized categorical distribution and
train with unweighted proper log loss (plus implicit Brier/NLL diagnostics).
PRICE: learn a risk-bounded cash-vs-all-tickets allocation policy by directly
maximizing realized log wealth, not label accuracy. An always-cash policy is
allowed; it may be the best choice when no useful edges are found.

All decisions are computed BEFORE the realized order is accessed. Train 2025
chronologically with one quarter per run and immutable 2024 base model.
2025 was ALREADY examined in prior work. Development metrics are NOT a blind
profitable-strategy discovery. A frozen 2026 H1 retrospective comparison is
separately scored; 2026 has been examined in other project work as well and
available entries are A-class only, not an S-class-equivalent universe.
ODDS ARE FINAL and simulated allocation is NOT executable live performance.

This experiment uses predeclared regularized learning rates, cash allocation
and risk budget. Do not choose a winner based on observed test performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import chapter1_iterative_deletion as previous
import chapter1_frozen_2025 as frozen
import chapter1_accuracy_calibration_duel as duel
import chapter1_blind_exam as exam

ROOT = Path("docs/paper-only-ai-2023/chapter1_deletion_continual/two_problem_fork")
STATE = ROOT / "state.json"
ROLES = previous.ROLES
KINDS = previous.KINDS
NF = previous.N_FEATURES
PROB_ETA = 0.012
PRICE_ETA = 0.06
REGULARIZER = 0.00012
FRACTIONAL_BUDGET = 0.02
STAKE = 100
EXPECTED_BASE_DIGEST = "d99e43fc8b0165f670fe5be811c5a5bdfa5e0d5bcc802f41f339fda160f49f28"


def initial_state():
    _, sealed = frozen.frozen_state()
    if sealed["weight_model_sha256"] != EXPECTED_BASE_DIGEST:
        raise RuntimeError("Immutable source AI differs from announced 2024 version")
    return {
        "schema": "paper-only-two-problem-fork-2025-v1",
        "status": "TRAIN_2025_CHRONOLOGICAL",
        "source_frozen_2024": sealed,
        "2025_is_previously_explored_development_data": True,
        "training_quarters": [],
        "next_quarter": 1,
        "weights": {
            role: {kind: {
                "probability": {"beta": 0.25, "theta": [0.0]*NF},
                "price": {"beta": 0.25, "theta": [0.0]*NF, "cash": 0.0},
            } for kind in KINDS} for role in ROLES
        },
    }


def load_state():
    if not STATE.exists():
        return initial_state()
    s = json.loads(STATE.read_text("utf-8"))
    if s["schema"] != "paper-only-two-problem-fork-2025-v1":
        raise RuntimeError("Unknown fork format")
    _, sealed = frozen.frozen_state()
    if sealed != s["source_frozen_2024"] or sealed["weight_model_sha256"] != EXPECTED_BASE_DIGEST:
        raise RuntimeError("Fork 2024 ancestry checkpoint mutated")
    quarters = s["training_quarters"]
    if quarters != list(range(1, len(quarters)+1)):
        raise RuntimeError("Nonchronological or tampered learning ledger")
    if s["next_quarter"] != len(quarters)+1:
        raise RuntimeError("Unsealed checkpoint ledger")
    return s


def hash_weights(w):
    return hashlib.sha256(json.dumps(w, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def market_prob(odds):
    p = np.asarray(odds, dtype=float)
    if len(p)<2 or not np.isfinite(p).all() or np.any(p<=1):
        raise RuntimeError("Invalid full ticket odds")
    q=1/p
    return q/q.sum()


def transform(row, old_prior, m):
    """A frozen fixed feature transform. No outcomes or current-period fit."""
    x = np.asarray(row["x"], dtype=np.float64)
    if x.shape != (len(m), NF):
        raise RuntimeError("Malformed 20-feature input")
    if np.any(~np.isfinite(x)) or len(old_prior)!=len(m) or np.any(old_prior<=0):
        raise RuntimeError("Malformed original prediction")
    features = np.tanh(np.clip(x, -5, 5) / 3.0)
    delta = np.clip(np.log(np.asarray(old_prior, dtype=float)/m), -4, 4)
    phi = np.column_stack((delta,features))
    return phi


def stable_softmax(z):
    z=np.asarray(z,dtype=float)
    t=np.exp(z-np.max(z))
    return t/t.sum()


def probability_forward(w, m, phi):
    params=np.r_[float(w["beta"]), np.asarray(w["theta"],dtype=float)]
    p=stable_softmax(np.log(m)+phi@params)
    if np.any(p<=0) or not np.isfinite(p).all():
        raise RuntimeError("Bad probability prediction")
    return p


def price_forward(w, m, phi):
    params=np.r_[float(w["beta"]), np.asarray(w["theta"],dtype=float)]
    ticket_logits=np.log(m)+phi@params
    cash_logit=float(w["cash"])
    logits=np.r_[cash_logit,ticket_logits]
    distribution=stable_softmax(logits)
    cash=float(distribution[0])
    tickets=distribution[1:]
    if cash<=0 or np.any(tickets<=0) or abs(cash+tickets.sum()-1)>1e-8:
        raise RuntimeError("Bad cash-vs-ticket policy")
    return cash,tickets


def probability_update(w, p, phi, win):
    """Gradient descent on proper categorical log loss; outcome read only now."""
    g=phi[win]-p@phi
    g=np.clip(g,-4,4)
    params=np.r_[float(w["beta"]), np.asarray(w["theta"],dtype=float)]
    params=(1-REGULARIZER)*params+PROB_ETA*g
    w["beta"]=float(np.clip(params[0],0,1.25))
    w["theta"]=np.clip(params[1:],-1,1).tolist()


def price_gradient(cash,ticket_policy,phi,odds,win):
    """Exact gradient of log(1-B+B*cash+B*pi_win*O_win).

    The outside CASH option has its own trainable intercept and no payoff.
    Every candidate ticket participates, including longshots and unwon marks.
    This is an economic/log-wealth objective, not a log-loss proxy.
    """
    extended=np.column_stack((phi,np.zeros(len(phi),dtype=float)))
    cash_feature=np.zeros(phi.shape[1]+1,dtype=float)
    cash_feature[-1]=1.0
    expected=extended.T@ticket_policy+cash*cash_feature
    price=float(odds[win])
    wealth=1-FRACTIONAL_BUDGET+FRACTIONAL_BUDGET*(cash+price*ticket_policy[win])
    if not wealth>0 or not np.isfinite(wealth):
        raise RuntimeError("Nonpositive portfolio wealth")
    grad=(FRACTIONAL_BUDGET/wealth)*(
        cash*(cash_feature-expected)+
        price*ticket_policy[win]*(extended[win]-expected))
    return grad,float(wealth)


def price_update(w, g):
    params=np.r_[float(w["beta"]), np.asarray(w["theta"],dtype=float),float(w["cash"])]
    params=(1-REGULARIZER)*params+PRICE_ETA*np.clip(g,-4,4)
    w["beta"]=float(np.clip(params[0],-1,1.25))
    w["theta"]=np.clip(params[1:-1],-1,1).tolist()
    w["cash"]=float(np.clip(params[-1],-5,8))


def empty_metrics():
    return {"races":0,"probability_nll_sum":0.0,"frozen2024_nll_sum":0.0,
            "market_nll_sum":0.0,
            "probability_brier_sum":0.0,"market_brier_sum":0.0,
            "probability_top1_hits":0,"market_top1_hits":0,
            "price_log_wealth_sum":0.0,"price_expected_staked_bankroll_fraction":0.0,
            "price_realized_payout_bankroll_fraction":0.0,
            "price_cash_allocation_sum":0.0,
            "price_positive_signal_bets":0,
            "price_positive_signal_payout_yen":0.0}


def record_metrics(s,p,old,m,cash,policy,odds,win,wealth):
    """Only after prediction, when all eight decisions are already fixed."""
    s["races"]+=1
    s["probability_nll_sum"]+=-math.log(float(p[win]))
    s["frozen2024_nll_sum"]+=-math.log(float(old[win]))
    s["market_nll_sum"]+=-math.log(float(m[win]))
    s["probability_brier_sum"]+=float(np.dot(p,p)+1-2*p[win])
    s["market_brier_sum"]+=float(np.dot(m,m)+1-2*m[win])
    s["probability_top1_hits"]+=int(int(np.argmax(p))==win)
    s["market_top1_hits"]+=int(int(np.argmax(m))==win)
    s["price_log_wealth_sum"]+=math.log(wealth)
    s["price_expected_staked_bankroll_fraction"]+=FRACTIONAL_BUDGET*(1-cash)
    s["price_realized_payout_bankroll_fraction"]+=FRACTIONAL_BUDGET*float(policy[win]*odds[win])
    s["price_cash_allocation_sum"]+=cash
    # An optional prespecified 100-yen economic signal diagnostic. NOT the
    # actual portfolio and NOT a claim of calibrated event probability.
    signal=(policy/(1-cash)*odds)>1
    s["price_positive_signal_bets"]+=int(np.count_nonzero(signal))
    s["price_positive_signal_payout_yen"]+=STAKE*float(odds[win])*bool(signal[win])


def summarize(s):
    n=s["races"]
    invested=s["price_expected_staked_bankroll_fraction"]
    payoff=s["price_realized_payout_bankroll_fraction"]
    t=s["price_positive_signal_bets"]
    return {
        **s,
        "probability_mean_nll":s["probability_nll_sum"]/n if n else None,
        "frozen2024_mean_nll":s["frozen2024_nll_sum"]/n if n else None,
        "market_mean_nll":s["market_nll_sum"]/n if n else None,
        "probability_mean_brier":s["probability_brier_sum"]/n if n else None,
        "market_mean_brier":s["market_brier_sum"]/n if n else None,
        "probability_top1_accuracy_percent":100*s["probability_top1_hits"]/n if n else None,
        "market_top1_accuracy_percent":100*s["market_top1_hits"]/n if n else None,
        "price_total_fractional_bankroll_invested":invested,
        "price_total_fractional_bankroll_returned":payoff,
        "price_portfolio_roi_percent":100*payoff/invested if invested else None,
        "price_mean_log_growth_per_race":s["price_log_wealth_sum"]/n if n else None,
        "price_compound_wealth_multiplier":math.exp(s["price_log_wealth_sum"]) if s["price_log_wealth_sum"]<700 else None,
        "price_mean_cash_fraction":s["price_cash_allocation_sum"]/n if n else None,
        "price_positive_signal_flat100_roi_percent":100*s["price_positive_signal_payout_yen"]/(STAKE*t) if t else None,
    }


def load_aligned(year,q):
    if year==2025:
        frozen.configure_year()
        return frozen.load_2025_aligned(q)
    if year!=2026 or q not in (1,2):
        raise RuntimeError("Only 2025 development and 2026 H1 A-class validation are approved")
    # No 2026 S-category rider/line entries have been verified in the repository.
    # Do NOT silently include missing S files or falsely claim a nationwide test.
    intervals={1:("01_02","03_04"),2:("03_04","05_06")}[q]
    paths=([f"data/a_class_12/{year}_{s}/entries.csv" for s in intervals]+
           [f"data/a3_challenge/{year}_{s}/entries.csv" for s in intervals])
    start=f"{year}-{(q-1)*3+1:02d}-01"
    end=f"{year}-{q*3:02d}-{31 if q==1 else 30:02d}"
    previous.bt.LABELS=f"data/result_universe_v2/combined/results_{year}.jsonl.gz"
    previous.bt.START,previous.bt.END=start,end
    previous.bt.sources=lambda:paths
    previous.prices.ODDS=f"data/odds_only/odds_{year}.jsonl.gz"
    cards,card_audit=previous.bt.load_2024_line_cards()
    odds,odds_audit=previous.prices.collect_2024_books()
    matched=[]
    failures=Counter()
    for rid,race in cards.items():
        if rid not in odds:
            failures["missing_odds"]+=1
            continue
        try:
            book=previous.prices.decode_odds(odds[rid],rid,set(race["cars"]))
        except (ValueError,TypeError,KeyError) as exc:
            failures[type(exc).__name__+":"+str(exc)[:55]]+=1
            continue
        matched.append((rid,race,book))
    matched.sort(key=lambda row:(row[1]["date"],row[0]))
    if len(matched)<100:
        raise RuntimeError("Too few 2026 complete A-class races")
    return matched,{"from":start,"to":end,"eligible_cards":len(cards),
                    "qualified_races":len(matched),"classes":"A1/A2/A3 only",
                    "complete_odds":True,"card_audit":card_audit,
                    "odds_audit":odds_audit,"excluded":dict(failures)}


def quarter(q,train):
    target=ROOT/(f"{'training_2025' if train else 'frozen_2026'}_q{q}.json")
    if target.exists():
        raise RuntimeError("Quarter already evaluated: preserving first result")
    state=load_state()
    if train:
        if q!=state["next_quarter"] or state["status"]!="TRAIN_2025_CHRONOLOGICAL":
            raise RuntimeError("Training quarters must run exactly once in chronological order")
    else:
        if state["training_quarters"]!=[1,2,3,4] or state["status"]!="2025_TRAINING_COMPLETE":
            raise RuntimeError("Must freeze all 2025 learning before 2026 test")
    theta,anchor=frozen.frozen_state()
    if state["source_frozen_2024"]!=anchor:
        raise RuntimeError("Model ancestry changed")
    year=2025 if train else 2026
    records,audit=load_aligned(year,q)
    parents=previous.frozen_deletion_parents()
    accuracy_asset,calibrated,_=duel.fetch_immutable_weights()
    metrics={role:{kind:empty_metrics() for kind in KINDS} for role in ROLES}
    weights=state["weights"]
    weights_before=hash_weights(weights)
    seen=0
    for start in range(0,len(records),16):
        batch=records[start:start+16]
        aligned=previous.records_for_batch(batch,accuracy_asset,calibrated)
        for rid,day,groups in aligned:
            planned=[]
            for row in groups:
                role,kind=row["role"],row["kind"]
                w=weights[role][kind]
                odds=np.asarray(row["prices"],dtype=float)
                m=market_prob(odds)
                old=previous.ticket_probabilities(row,parents[role][kind],theta[role][kind])
                phi=transform(row,old,m)
                p=probability_forward(w["probability"],m,phi)
                cash,portfolio=price_forward(w["price"],m,phi)
                planned.append((row,odds,m,old,phi,p,cash,portfolio))
            # The true finish is accessed only after all four groups have
            # produced BOTH (probability and price) decisions.
            for row,odds,m,old,phi,p,cash,portfolio in planned:
                win=int(row["winner_index"])
                role,kind=row["role"],row["kind"]
                g,wealth=price_gradient(cash,portfolio,phi,odds,win)
                record_metrics(metrics[role][kind],p,old,m,cash,portfolio,odds,win,wealth)
                if train:
                    probability_update(weights[role][kind]["probability"],p,phi,win)
                    price_update(weights[role][kind]["price"],g)
            seen+=1
        if start%256==0 or start+16>=len(records):
            print("TWO_PROBLEM_PROGRESS",year,q,seen,len(records),"train",train,flush=True)
    if len(records)!=seen:
        raise RuntimeError("Incomplete quarter")
    _,after=frozen.frozen_state()
    if anchor!=after:
        raise RuntimeError("Original 2024 checkpoint changed while learning")
    if not train and hash_weights(weights)!=weights_before:
        raise RuntimeError("Evaluation mutated trained 2025 parameters")
    report={
        "status":"FORK_2025_PREQUENTIAL_LEARNING_QUARTER" if train else "FORK_2026_FROZEN_A_CLASS_VALIDATION",
        "year":year,"quarter":q,"races":seen,
        "base_2024":anchor,"weights_before":weights_before,
        "training_during_scoring":train,
        "2025_has_already_been_examined":True,
        "2026_has_already_been_examined_in_other_project_work":True,
        "2026_only_A_class":not train,
        "final_odds_only":True,
        "fixed_fractional_bankroll_budget_per_market":FRACTIONAL_BUDGET,
        "probability_objective":"online full-book categorical log loss",
        "price_objective":"online fixed-risk cash-option log wealth (not NLL)",
        "no_fixed_number_of_tickets":True,
        "data_audit":audit,
        "metrics":{role:{kind:summarize(metrics[role][kind]) for kind in KINDS} for role in ROLES},
    }
    ROOT.mkdir(parents=True,exist_ok=True)
    if train:
        state["training_quarters"].append(q)
        state["next_quarter"]=q+1
        state["status"]="2025_TRAINING_COMPLETE" if q==4 else "TRAIN_2025_CHRONOLOGICAL"
        state["2025_last_trained_day"]=f"2025-{q*3:02d}-{31 if q in (1,4) else 30:02d}"
        report["weights_after"]=hash_weights(weights)
        report["updated_after_each_race"]=True
        state.setdefault("training_reports",[]).append(str(target))
        state["weights"]=weights
    else:
        report["weights_after"]=weights_before
    report["completed_at"]=datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if train:
        STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("TWO_PROBLEM_QUARTER_COMPLETE",year,q,seen,"train",train,
          "prob_nll",[(role,kind,report["metrics"][role][kind]["probability_mean_nll"])
                      for role in ROLES for kind in KINDS],flush=True)


def audit():
    s=load_state()
    if s["status"]!="2025_TRAINING_COMPLETE":
        raise RuntimeError("Four 2025 learning quarters required")
    filenames=[ROOT/f"frozen_2026_q{q}.json" for q in (1,2)]
    docs=[json.loads(p.read_text("utf-8")) for p in filenames]
    digest=hash_weights(s["weights"])
    if any(x["training_during_scoring"] or x["weights_before"]!=digest or
           x["weights_after"]!=digest for x in docs):
        raise RuntimeError("2026 test was refitted or model versions changed")
    summary={}
    for role in ROLES:
        summary[role]={}
        for kind in KINDS:
            metrics=empty_metrics()
            for doc in docs:
                source=doc["metrics"][role][kind]
                for name in metrics:
                    metrics[name]+=source[name]
            summary[role][kind]=summarize(metrics)
    result={
        "status":"FOUR_FORKS_TRAINED_2025_AND_FROZEN_2026_H1_A_CLASS_EVALUATED",
        "training":"2025_Q1_Q4_chronological_prequential",
        "training_2025_previously_examined":True,
        "out_of_time_year":"2026_H1",
        "2026_period_used_by_other_project_experiments":True,
        "eligible_classes":"A1/A2/A3 ONLY; S excluded because entries were unverified",
        "races":sum(d["races"] for d in docs),
        "races_by_quarter":[d["races"] for d in docs],
        "2024_source_fingerprint":s["source_frozen_2024"],
        "frozen_2025_weights_sha256":digest,
        "evaluation_fit_calls":0,
        "markets":list(KINDS),
        "cash_allowed":True,
        "price_budget_per_market_race":FRACTIONAL_BUDGET,
        "actual_prices":"final_odds_only",
        "model_results":summary,
        "limitations":[
            "The probability branch minimizes proper multiclass log loss.",
            "The price branch maximizes realized cash-inclusive log wealth with strictly pre-result actions.",
            "Price ROI measures continuous allocation across ALL tickets; it is not the ROI of purchasing every positive-EV 100-yen mark.",
            "There is no clean unexamined 2026 project-wide holdout, and the 2026 data has only A-class verified entries.",
            "A 2025 prequential model is not a 2025 out-of-time test. Negative results must remain visible.",
            "Historical final odds do not establish 15-minute preclose execution feasibility.",
        ]
    }
    out=ROOT/"summary_2026_h1.json"
    if out.exists():raise RuntimeError("Already completed audit; do not rewrite")
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("TWO_PROBLEM_FINAL_AUDIT",json.dumps({
        role:{kind:{"prob_nll":summary[role][kind]["probability_mean_nll"],
                    "market_nll":summary[role][kind]["market_mean_nll"],
                    "price_roi":summary[role][kind]["price_portfolio_roi_percent"],
                    "price_log_growth":summary[role][kind]["price_mean_log_growth_per_race"]}
              for kind in KINDS} for role in ROLES},ensure_ascii=False),flush=True)


def checks():
    s=initial_state()
    for role in ROLES:
        for kind in KINDS:
            w=s["weights"][role][kind]
            m=np.array([.2,.3,.5])
            phi=np.zeros((3,NF+1),dtype=float)
            p=probability_forward(w["probability"],m,phi)
            c,t=price_forward(w["price"],m,phi)
            if not np.allclose(p,m,atol=1e-10) or abs(c-.5)>1e-10:
                raise RuntimeError("Wrong neutral initialization")
            odds=np.array([2.3,3.4,9.5])
            g,W=price_gradient(c,t,phi,odds,2)
            if not W>0 or len(g)!=NF+2:
                raise RuntimeError("Price objective missing grad")
            eps=1e-5
            base=w["price"]["cash"]
            w["price"]["cash"]=base+eps
            cp,tp=price_forward(w["price"],m,phi)
            wp=1-FRACTIONAL_BUDGET+FRACTIONAL_BUDGET*(cp+tp[2]*odds[2])
            w["price"]["cash"]=base-eps
            cm,tm=price_forward(w["price"],m,phi)
            wm=1-FRACTIONAL_BUDGET+FRACTIONAL_BUDGET*(cm+tm[2]*odds[2])
            w["price"]["cash"]=base
            estimated=(math.log(wp)-math.log(wm))/(2*eps)
            if abs(estimated-g[-1])>1e-8:
                raise RuntimeError("Price economic gradient failed numerical check")
    print("TWO_PROBLEM_CHECK_OK",s["source_frozen_2024"]["weight_model_sha256"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--check",action="store_true")
    group.add_argument("--train-quarter",type=int)
    group.add_argument("--eval-quarter",type=int)
    group.add_argument("--audit",action="store_true")
    args=p.parse_args()
    if args.check:checks()
    elif args.train_quarter is not None:quarter(args.train_quarter,True)
    elif args.eval_quarter is not None:quarter(args.eval_quarter,False)
    else:audit()

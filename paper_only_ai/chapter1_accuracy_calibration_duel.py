"""Precommitted identical-ticket accuracy vs calibration AI experiment.

ACC: accuracy-selected acc-g08-c01, trained 2023 Jan-Aug, selected in Oct.
CAL: classwise/conditional-ECE-selected g01-c01, trained Jan-Aug with Sep
     stage temperature calibration. Both derive 2nd and 3rd conditionals
     from the SAME immutable g01-c01 models.

Test: Completed 2024 Q1, Q2 races with identical complete trifecta/trio
books, actual outcomes, published LINE, same candidate ticket universe.
No retraining, outcome-dependent choice, stake threshold tuning, or entry
based on rank. Payouts are at FINAL odds, not 15-min pre-close odds.

Kelly is explicitly an ILLUSTRATIVE cash-capped portfolio adaptation of
Walsh/Joshi's 1/8 Kelly: correlated bets across markets matter.
2024 historical results have been used in other project experiments.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

import chapter1_accuracy_ai as acc
import chapter1_blind_exam as previous
import backtest_line_rf_2024_q1 as bet
import backtest_roi_2024_q1 as prices
from compare_lines_2023 import feature

REPORT = Path("docs/paper-only-ai-2023/chapter1_accuracy_calibration_duel/duel_2024q1_q2.json")
YEAR = 2024
KINDS = ("3連単", "3連複")
ROLES = ("accuracy", "calibration")
QUARTERS = (1, 2)
STAKE_YEN = 100
MIN_PAIRED_RACES = 900


def declared_candidates():
    accuracy_report = json.loads(Path(
        "docs/paper-only-ai-2023/chapter1_accuracy_ai/latest.json"
    ).read_text("utf-8"))
    calibration_state = json.loads(Path(
        "docs/paper-only-ai-2023/autonomous_chapter1/state.json"
    ).read_text("utf-8"))
    ac = accuracy_report["current_best"]
    ca = calibration_state["champions"]["calibration"]
    if accuracy_report["status"] != "CONFIRMED_HISTORICAL_GOAL" or ac["id"] != "acc-g08-c01":
        raise RuntimeError("Accuracy selected AI must remain frozen as preregistered")
    if ca["id"] != "g01-c01":
        raise RuntimeError("Calibration model must remain frozen as preregistered")
    if "october" not in ac or "development" not in ca:
        raise RuntimeError("Missing model selection audit")
    return ac, ca


def fetch_immutable_weights():
    ac, ca = declared_candidates()
    local = Path(os.environ.get("DUEL_ACCURACY_MODEL_DIR", "frozen_accuracy_ai"))
    filename = "accuracy_champion_acc-g08-c01.joblib"
    path = local / filename
    if not path.is_file():
        local.mkdir(parents=True,exist_ok=True)
        subprocess.run(["gh","release","download",
                        "chapter1-accuracy_champion_acc-g08-c01",
                        "--pattern",filename,"--dir",str(local)],
                       check=True,timeout=120)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ac["model_asset"]["sha256"]:
        raise RuntimeError("Accuracy model release does not match archived SHA256")
    asset = joblib.load(path)
    if asset["config"] != ac["config"]:
        raise RuntimeError("Accuracy model's config altered")
    if asset["selection"] != ac["october"]:
        raise RuntimeError("Accuracy model's selecting accuracy altered")
    columns = acc.selected_columns(ac["config"])
    if not np.array_equal(asset["columns"], columns):
        raise RuntimeError("Accuracy trained feature columns inconsistent")
    if float(asset["temperature"]) != float(ac["temperature"]):
        raise RuntimeError("Accuracy model's fitted temperature changed")
    with tempfile.TemporaryDirectory(prefix="duel_calibration_") as td:
        calibrated,cal_digest = previous.load_frozen_champion(ca, td)
    return (asset, calibrated,
            {"accuracy_joblib": digest, "calibration_generation_01_release":cal_digest})


def accuracy_first_distribution(chunk, accuracy_asset):
    model=accuracy_asset["model"]
    cols=accuracy_asset["columns"]
    temp=float(accuracy_asset["temperature"])
    races={rid:race for rid,race in chunk}
    base_rows=np.asarray([feature(race,car,use_line=True)
                          for rid,race in chunk
                          for car in sorted(race["cars"],key=int)],
                         dtype=np.float32)
    relative=acc.race_relative_features(races)
    transformed=acc.candidate_inputs(base_rows,relative,
                                     accuracy_asset["config"])
    scores=np.clip(model.predict_proba(transformed[:,cols])[:,1],1e-12,1.0)
    out={}
    start=0
    for rid,race in chunk:
        cars=sorted(race["cars"],key=int)
        raw=np.power(scores[start:start+len(cars)],1.0/temp)
        if not len(raw) or not np.all(np.isfinite(raw)):
            raise RuntimeError("Invalid accuracy winner probability")
        normalized=raw/raw.sum()
        out[rid]=dict(zip(cars,normalized))
        start+=len(cars)
    if start!=len(scores):
        raise RuntimeError("Accuracy prediction/race alignment mismatch")
    return out


def probabilities_both(chunk, accuracy_asset, calibrated_models):
    """No outcome/odds appears in either model's feature calculations.

    Only the first-place factor differs; both get exactly the same stage-2
    and stage-3 conditional model distribution and race lineup.
    """
    base=bet.infer_new_chunk(chunk,calibrated_models)
    new_first=accuracy_first_distribution(chunk,accuracy_asset)
    cal_first={}
    accur={}
    for rid,race in chunk:
        cars=sorted(race["cars"],key=int)
        x=np.asarray([feature(race,car,use_line=True) for car in cars],
                     dtype=np.float32)
        raw=calibrated_models[1].predict_proba(x)[:,1]
        denom=float(raw.sum())
        if denom<=0:
            raise RuntimeError("Empty calibration 1st-place distribution")
        cal_first[rid]=dict(zip(cars,raw/denom))
        trifecta={}
        trio=defaultdict(float)
        for ticket,p in base[rid]["3連単"].items():
            a=ticket[0]
            adjusted=float(p)*(new_first[rid][a]/cal_first[rid][a])
            trifecta[ticket]=adjusted
            trio[tuple(sorted(ticket,key=int))]+=adjusted
        total=math.fsum(trifecta.values())
        if abs(total-1)>1e-8:
            raise RuntimeError("Probability factorization not normalized")
        if abs(math.fsum(trio.values())-1)>1e-8:
            raise RuntimeError("Trio probability factorization not normalized")
        accur[rid]={"3連単":trifecta,"3連複":dict(trio)}
    return {"accuracy":accur,"calibration":base}, {"accuracy":new_first,"calibration":cal_first}


class CalibEce:
    def __init__(self):
        self.p=[]
        self.y=[]
        self.one_win=0
        self.exact_top1=0
        self.trio_top1=0
        self.n=0
    def update(self,first, race, pred):
        cars=sorted(race["cars"],key=int)
        values=np.asarray([first[k] for k in cars])
        labels=np.asarray([int(k==race["order"][0]) for k in cars])
        if abs(float(values.sum())-1)>1e-8 or labels.sum()!=1:
            raise RuntimeError("Broken 1st-place categorical distribution")
        self.p.extend(values.tolist())
        self.y.extend(labels.tolist())
        self.one_win+=int(cars[int(values.argmax())]==race["order"][0])
        self.exact_top1+=int(max(pred["3連単"],key=pred["3連単"].get)==race["order"])
        correct_trio=tuple(sorted(race["order"],key=int))
        self.trio_top1+=int(max(pred["3連複"],key=pred["3連複"].get)==correct_trio)
        self.n+=1
    def summary(self):
        p=np.asarray(self.p)
        y=np.asarray(self.y)
        idx=np.minimum((20*p).astype(int),19)
        ece=sum(float(np.mean(idx==i)*abs(y[idx==i].mean()-p[idx==i].mean()))
                for i in range(20) if np.any(idx==i))
        return {
            "races":self.n, "first_place_correct":self.one_win,
            "first_place_accuracy_percent":100*self.one_win/self.n,
            "single_exact_trifecta_correct":self.exact_top1,
            "single_exact_trifecta_percent":100*self.exact_top1/self.n,
            "single_unordered_trio_correct":self.trio_top1,
            "single_unordered_trio_percent":100*self.trio_top1/self.n,
            "winner_positive_class_binned_ece20":ece,
            "winner_probability_occupied_bins20":int(np.unique(idx).size),
            "bin_80pct_nonempty_proxy_passed":int(np.unique(idx).size)>=16,
            "note":"20bin pooled winner ECE; NOT the original paper's full binary NBA classwise-ECE",
        }


def bootstrap_paired_annual(ledger, iterations=1000):
    """Race bootstrap, pair-preserving, for fixed-payout ROI difference."""
    a=np.asarray(ledger["accuracy"],dtype=float)
    c=np.asarray(ledger["calibration"],dtype=float)
    if a.shape!=c.shape or len(a)<100 or a.shape[1]!=2:
        raise RuntimeError("Paired bootstrap requires same race set")
    rng=np.random.default_rng(230306021)
    diff=[]
    for _ in range(iterations):
        ix=rng.integers(0,len(a),len(a))
        ainv,aret=np.sum(a[ix],axis=0)
        cinv,cret=np.sum(c[ix],axis=0)
        if ainv and cinv:
            diff.append(100*((aret/ainv)-(cret/cinv)))
    return {
        "method":"paired bootstrap resampling aligned races, 1000 replicates",
        "point_difference_accuracy_minus_calibration_pp":
            100*(a[:,1].sum()/a[:,0].sum()-c[:,1].sum()/c[:,0].sum()),
        "p025":float(np.quantile(diff,0.025)),
        "p975":float(np.quantile(diff,0.975)),
        "caution":"Race bootstrap ignores time correlation and bet execution uncertainty",
    }


def main():
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    if REPORT.exists():
        print("DUEL_ALREADY_SCORED__NO_POSTHOC_TUNING",flush=True)
        return
    accuracy_asset,calibration,checksums=fetch_immutable_weights()
    card_audit={}
    market_audit={}
    exclusions=Counter()
    ledger={role:{kind:bet.Ledger() for kind in KINDS}
            for role in ROLES}
    kelly={role:{kind:bet.KellyLedger() for kind in KINDS+("combined",)}
           for role in ROLES}
    metrics={role:CalibEce() for role in ROLES}
    by_period={}
    by_category={}
    per_race={role:[] for role in ROLES}
    total_market={role:defaultdict(lambda:{"investment_yen":0.,"payout_yen":0.,
                                          "ticket_count":0,"hit_count":0})
                  for role in ROLES}
    total_races=0
    for quarter in QUARTERS:
        start=f"{YEAR}-{(quarter-1)*3+1:02d}-01"
        end=f"{YEAR}-{quarter*3:02d}-{31 if quarter==1 else 30:02d}"
        bet.LABELS=f"data/result_universe_v2/combined/results_{YEAR}.jsonl.gz"
        bet.START,bet.END=start,end
        bet.sources=lambda q=quarter: (previous.quarter_sources(YEAR,q)+
                                      ["data/2024/s_class_yosen/entries.csv"])
        prices.ODDS=f"data/odds_only/odds_{YEAR}.jsonl.gz"
        cards,audit=bet.load_2024_line_cards()
        odds,oa=prices.collect_2024_books()
        card_audit[f"2024q{quarter}"]=audit
        market_audit[f"2024q{quarter}"]=oa
        intersection=sorted(set(cards)&set(odds),
                            key=lambda rid:(cards[rid]["date"],rid))
        books={}
        matching=[]
        for rid in intersection:
            try:
                books[rid]=prices.decode_odds(
                    odds[rid],rid,set(cards[rid]["cars"]))
            except (KeyError,ValueError,TypeError) as e:
                exclusions[type(e).__name__+":"+str(e)[:80]]+=1
                continue
            matching.append((rid,cards[rid]))
        if len(matching)<300:
            raise RuntimeError(f"Too few comparable race/odds pairs for Q{quarter}")
        by_period[f"2024q{quarter}"]={
            "eligible_cards":len(cards),
            "joined_before_price_verification":len(intersection),
            "qualified_races":len(matching),
            "matched_fraction_of_cards":len(matching)/len(cards),
        }
        for offset in range(0,len(matching),20):
            chunk=matching[offset:offset+20]
            distributions,first=probabilities_both(
                chunk,accuracy_asset,calibration)
            for rid,race in chunk:
                total_races+=1
                results=bet.actual_tickets(race)
                book=books[rid]
                for role in ROLES:
                    pred=distributions[role][rid]
                    bet.safe_records(pred,book)
                    metrics[role].update(first[role][rid],race,pred)
                    item_invest=item_payout=0.
                    raw_both=[]
                    for kind in KINDS:
                        prices_kind=book[kind]
                        actual=results[kind]
                        all_prob=pred[kind]
                        selected={}
                        kraw=[]
                        for ticket,p in all_prob.items():
                            odd=prices_kind[ticket]
                            if float(p)*float(odd)>1.0:
                                selected[ticket]=(kind,p,odd)
                                kraw.append(((kind,ticket),odd,p,ticket==actual))
                        stake,payout=ledger[role][kind].update(
                            selected,actual,prices_kind,all_prob)
                        item_invest+=stake
                        item_payout+=payout
                        kelly[role][kind].step(kraw)
                        raw_both.extend(kraw)
                        group=total_market[role][race["source"],kind]
                        group["investment_yen"]+=stake
                        group["payout_yen"]+=payout
                        group["ticket_count"]+=len(selected)
                        group["hit_count"]+=int(actual in selected)
                    kelly[role]["combined"].step(raw_both)
                    per_race[role].append((item_invest,item_payout))
                    group=total_market[role][race["date"][:7],"combined"]
                    group["investment_yen"]+=item_invest
                    group["payout_yen"]+=item_payout
                    group["ticket_count"]+=len(raw_both)
                    group["hit_count"]+=int(item_payout>0)
            if offset % 300==0 or offset+20>=len(matching):
                print("DUEL_PROGRESS",quarter,offset+len(chunk),len(matching),
                      flush=True)
    if total_races<MIN_PAIRED_RACES:
        raise RuntimeError("Insufficient full-year matched race sample")
    results={}
    for role in ROLES:
        kind_metrics={kind:ledger[role][kind].summary() for kind in KINDS}
        invest=sum(ledger[role][kind].invest for kind in KINDS)
        returned=sum(ledger[role][kind].paid for kind in KINDS)
        count=sum(ledger[role][kind].tickets for kind in KINDS)
        kind_metrics["combined"]={
            "eligible_races":total_races,
            "purchased_tickets":count,
            "investment_yen":round(invest,2),
            "payout_yen":round(returned,2),
            "net_profit_yen":round(returned-invest,2),
            "return_rate_percent":100*returned/invest if invest else None,
            "roi_percent":100*(returned-invest)/invest if invest else None,
            "model_expected_roi":100*(sum(ledger[role][kind].expected for kind in KINDS)-invest)/invest if invest else None,
        }
        results[role]={
            "fixed_stake_100_yen":kind_metrics,
            "first_place_and_one_pick_diagnostics":metrics[role].summary(),
            "one_eighth_kelly":{kind:kelly[role][kind].summary()
                                for kind in KINDS+("combined",)},
            "by_category_and_month":{
                "+".join(k):{
                    **v,
                    "return_rate_percent":100*v["payout_yen"]/v["investment_yen"]
                         if v["investment_yen"] else None}
                for k,v in sorted(total_market[role].items())},
        }
    a=results["accuracy"]["fixed_stake_100_yen"]["combined"]
    c=results["calibration"]["fixed_stake_100_yen"]["combined"]
    report={
        "status":"COMPLETED_SAME_RACES_FROZEN_ACCURACY_VS_CALIBRATION_DUEL",
        "computed_at":datetime.now(timezone.utc).isoformat(),
        "paper_hypothesis":"Accuracy model selection versus calibration model selection: which generates greater realized betting returns?",
        "accuracy_branch_id":"acc-g08-c01",
        "calibration_branch_id":"g01-c01",
        "frozen_model_releases_sha256":checksums,
        "compared_periods":by_period,
        "same_races_both_branches":True,
        "all_2024_labels_used_only_for_metrics":True,
        "refitted_models_in_2024":False,
        "no_2024_selection_of_model_or_betting_rule":True,
        "book_odds_are_final":True,
        "final_odds_not_necessarily_available_15_minutes_before_deadline":True,
        "selection_rule":"Purchase EVERY ticket with model_probability times decimal_final_odds > 1.0",
        "fixed_100yen_unlimited_capital":True,
        "fractional_kelly_is_cash_capped_correlation_unoptimized_extension":True,
        "allowed_markets":["3連単","3連複"],
        "eligible_races":total_races,
        "results":results,
        "paired_roi_difference":{
            "accuracy_minus_calibration_pp":
                a["return_rate_percent"]-c["return_rate_percent"],
            "net_yen_accuracy_minus_calibration":
                a["net_profit_yen"]-c["net_profit_yen"],
            "bootstrap":bootstrap_paired_annual(per_race),
        },
        "book_coverage":card_audit,
        "market_coverage":market_audit,
        "excluded_prices":dict(exclusions),
        "limitations":[
            "2024 Q1-Q2 results and odds have already been consulted elsewhere in this project; not a completely pristine untouched project-wide holdout.",
            "The models are independently selected on accuracy versus conditional calibration, but the accuracy AI reuses the same frozen stage-2 and 3 models as the calibration AI.",
            "Neither branch is optimized for ROI on the evaluation period. The rules are identical and frozen.",
            "The paper's strict original NBA classwise ECE with 20 bins and at least 80% occupied bins is NOT identically implemented for the entire Keirin multi-outcome market.",
            "This replicates the original experiment's intent, not a literal NBA two-outcome clinical replication.",
            "Kelly bets within/across trifecta-trio outcomes are correlated; single-ticket 1/8 Kelly with race-level cash cap is only an illustrative adaptation.",
            "Missing published LINE, feature data, complete finish order and all odds for any ticket lead to exclusion, never invented prices.",
            "Realized ROI from final odds cannot be interpreted as practical purchasability before the betting cutoff.",
        ],
    }
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("DUEL_COMPLETED",json.dumps({
        "eligible_races":total_races,
        "accuracy_percent":a["return_rate_percent"],
        "calibration_percent":c["return_rate_percent"],
        "accuracy_trifecta_hit":results["accuracy"]["first_place_and_one_pick_diagnostics"]["single_exact_trifecta_percent"],
        "calibration_trifecta_hit":results["calibration"]["first_place_and_one_pick_diagnostics"]["single_exact_trifecta_percent"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()

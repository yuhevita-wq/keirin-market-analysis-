"""Unrestricted odds-surface observation of Liu/Liu/Teo (2023) two-solution diversification.
Every valid trifecta permutation is eligible. No hand-selected formation shapes.
No model or purchase-rule edits. Aggregates only, and FINAL-odds proxy only.
"""
from __future__ import annotations
import itertools,json,math,sys,os
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.snowberg_observation_corrected_20261027 import observe as corrected
OUT=ROOT/"research/picking_winners_unrestricted_20261028"
AUDIT=Counter()
STATS=defaultdict(Counter)
CACHE={}

def make_cache(cars):
    if cars not in CACHE:
        orders=list(itertools.permutations(cars,3))
        a=np.asarray(orders,dtype=np.int8)
        score=sum((a[:,None,j]==a[None,:,j]).astype(np.float32) for j in range(3))
        CACHE[cars]=(orders,score,{o:i for i,o in enumerate(orders)})
    return CACHE[cars]

def record(method,day,n,rtype,win,ids,orders,score,q,book):
    outcome_idx=orders.index(tuple(win))
    k=len(ids)
    actual_best=int(score[np.asarray(ids),outcome_idx].max())
    hit=int(outcome_idx in ids)
    payout=100.0*float(book[tuple(win)]) if hit else 0.0
    bought_q=float(q[np.asarray(ids)].sum())
    overlapping_positions=int(score[ids[0],ids[1]])
    for group in (("all",),("year",day[:4]),("cars",str(n)),("type",str(rtype))):
        c=STATS[(method,*group)]
        c["races"]+=1
        c["bets"]+=k
        c["stake_yen"]+=100*k
        c["payout_final_odds_yen"]+=payout
        c["exact_hits"]+=hit
        c["partial_best_score_sum"]+=actual_best
        c["partial_best_score_0"]+=int(actual_best==0)
        c["partial_best_score_1"]+=int(actual_best==1)
        c["partial_best_score_2"]+=int(actual_best==2)
        c["partial_best_score_3"]+=int(actual_best==3)
        c["expected_market_hits"]+=bought_q
        c["same_position_overlap_sum"]+=overlapping_positions

def capture(rid,day,books,finish,rtype,source):
    AUDIT["joined_callback_races"]+=1
    if source!="sealed_four_markets_orientation_verified":
        raise RuntimeError("Unverified odds orientation")
    book=books["3rentan"]
    cars=tuple(sorted({c for o in book for c in o}))
    if len(cars)<3 or len(finish)!=3 or tuple(finish) not in book:
        AUDIT["invalid_podium"]+=1;return
    orders,score,ix=make_cache(cars)
    if set(book)!=set(orders) or any(not math.isfinite(float(v)) or float(v)<1.0 for v in book.values()):
        AUDIT["noncomplete_market"]+=1;return
    inv=np.asarray([1.0/float(book[o]) for o in orders],dtype=np.float64)
    q=inv/inv.sum()
    # Market-only candidate solution 1: maximum expected number of correct positions.
    first=int(np.argmax(score@q))
    # Full unrestricted search for solution 2 maximizing market-expected best of two scores.
    best=np.maximum(score,score[first][None,:])@q
    best[first]=-np.inf
    second=int(np.argmax(best))
    # Comparison: two highest normalized market-support tickets (equal stakes).
    top=np.argsort(-q,kind="stable")[:2].tolist()
    diversified=[first,second]
    record("market_top_two",day,len(cars),rtype,finish,top,orders,score,q,book)
    record("partial_score_diversified_two",day,len(cars),rtype,finish,diversified,orders,score,q,book)
    AUDIT["usable"]+=1
    AUDIT["same_ticket_sets"]+=int(set(top)==set(diversified))
    AUDIT["different_ticket_sets"]+=int(set(top)!=set(diversified))
    AUDIT["same_first_selection"]+=int(top[0]==first)
    AUDIT["cars_"+str(len(cars))]+=1
    AUDIT["year_"+day[:4]]+=1

def main():
    if not os.environ.get(corrected.old.bridge.SECRET_ENV):
        raise RuntimeError("Refuse partial source: existing sealed four-market key unavailable")
    # Preserve the existing public-price orientation gate and sealed stream.
    corrected.base_observe=capture
    corrected.old.save=lambda state: print("CHECKPOINT",state,dict(AUDIT),flush=True)
    corrected.old.sealed()
    assert AUDIT["usable"]>0
    rows=[]
    for key,c in sorted(STATS.items()):
        d={"method":key[0],"group":list(key[1:]),**dict(c)}
        d["roi_final_odds_proxy"]=c["payout_final_odds_yen"]/c["stake_yen"]
        d["hit_rate"]=c["exact_hits"]/c["races"]
        d["mean_best_partial_score"]=c["partial_best_score_sum"]/c["races"]
        d["mean_same_position_overlap"]=c["same_position_overlap_sum"]/c["races"]
        d["empirical_hits_over_market_q"]=c["exact_hits"]/c["expected_market_hits"] if c["expected_market_hits"] else None
        rows.append(d)
    report={
        "status":"UNRESTRICTED_TWO_SOLUTION_OBSERVATION_COMPLETE",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source_commit":os.environ.get("GITHUB_SHA","unknown"),
        "paper":"Picking Winners: Diversification through Portfolio Optimization (Liu et al. 2023)",
        "scope":"Existing full sealed 2021-2026H1 four-market archive, no class/race/line limitation",
        "question":"Does a market-probability-weighted best-of-two partial-placement-score design improve realized partial matching or ROI relative to top-two market tickets?",
        "methods":{
            "candidate_space":"All distinct ordered trifecta permutations of all entrant numbers, no formation template",
            "candidate_one":"argmax of market-expected number of exact-position matches",
            "candidate_two":"argmax of market-expected max position matches over two tickets, excluding ticket one",
            "baseline":"Two trifecta tickets with highest normalized inverse final odds",
            "same_stake":"100 yen per ticket; exactly 2 tickets per strategy FOR PAPER OBSERVATION ONLY (not a final formation rule)",
            "market_proxy":"normalized 1/final_odds, NOT calibrated outcome probability",
            "profit_measure":"final-odds return proxy, not T-15 executable or official settlement"
        },
        "audit":dict(AUDIT),"orientation_examples":list(corrected.VALIDATED.values()),
        "groups":rows,
        "limitations":[
            "This is an explicit experimental translation of a multi-game partial-score paper; NOT a theorem transferring to exact-order single-race wagers.",
            "Choice of exact-position partial score is a testable surrogate, NOT the paper's original multi-game scoring rule.",
            "Both candidate selectors optimize market-implied score and may simply reproduce market consensus; no independent predictive edge claimed.",
            "Exactly two tickets tests the paper's two-solution theorem, not any imposed production formation size.",
            "Historical final odds are not T-15 available odds; ROI is descriptive only.",
            "In-sample descriptive observations across all years, not untouched test period.",
            "No rider attributes, predicted line, or scores used, and no conditions searched after seeing outcomes.",
            "Only aggregate metrics published, no plaintext odds or private secret published.",
            "No AI, model weights, purchase rules, or website changed."
        ]
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("FINAL",json.dumps({"status":report["status"],"audit":report["audit"],"overall":[d for d in rows if d["group"]==["all"]]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Three PRE-SELECTED random historical races, tested with genuine formation +
nonuniform ticket stakes + skip. No race selection based on hits or payouts.

2021-2024 is calibration; 2025-2026H1 is the never-trained test universe.
Random sample uses market validity only and independent verification that a
result label exists. Seed and selected IDs saved before settlement in report.
Candidate geometries are generated anew for each race, rather than choosing
from a catalog of 20 old equal-stake formations.
"""
from __future__ import annotations
import gzip,itertools,json,math,random,heapq
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2]
ODDS=ROOT/"data/odds_only"
RESULTS=ROOT/"data/result_universe_v2/combined"
OUT=ROOT/"research/three_random_joint_formation_allocation_20261029"
SEED=20261029
TRAIN=(2021,2022,2023,2024)
TEST=(2025,2026)
RISK_BANKROLL_YEN=100000
FRACTION_RACE_CAP=.02
SMOOTH_EXPECTED=25.
MIN_GROSS_EV=1.025
MIN_CANDIDATE_POINTS=2
MAX_CANDIDATE_POINTS=12
CANDIDATE_TOP=14

def rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def load_labels():
    labels={};bad=set()
    for p in sorted(RESULTS.glob("results_20??.jsonl.gz")):
        for r in rows(p):
            rid=str(r.get("race_id",""))
            w=tuple(r.get("finish_order") or ())
            if len(w)!=3 or len(set(w))!=3 or any(type(c)!=int or c<1 or c>9 for c in w):continue
            if rid in labels and labels[rid]!=w:bad.add(rid)
            else:labels[rid]=w
    for rid in bad:labels.pop(rid,None)
    return labels

def market(row,actual):
    raw=(row.get("odds") or {}).get("3連単") or {}
    try:p={tuple(map(int,k.split("-"))):float(v) for k,v in raw.items()}
    except (TypeError,ValueError):return None
    cars=sorted({c for t in p for c in t})
    n=len(cars)
    if n not in (5,6,7,8,9) or len(p)!=n*(n-1)*(n-2) or actual not in p:return None
    if any(not math.isfinite(o) or o<=0 for o in p.values()):return None
    invsum=sum(1/o for o in p.values())
    support=Counter()
    for t,o in p.items():
        for car in t:support[car]+=1/o
    ordered=sorted(cars,key=lambda c:(-support[c],c))
    rank={car:i for i,car in enumerate(ordered)}
    return {"prices":p,"cars":cars,"n":n,"invsum":invsum,
        "ordered":ordered,"rank":rank,
        "favorite":min(p,key=lambda t:(p[t],t))}

def train_and_select(labels):
    seen=set();actual=defaultdict(float);expected=defaultdict(float)
    reservoir=[];rng=random.Random(SEED);test_total=train_total=0
    for path in sorted(ODDS.glob("odds_20??.jsonl.gz")):
        year=int(path.name[5:9])
        for r in rows(path):
            rid=str(r.get("race_id",""))
            if rid in seen:raise RuntimeError("duplicate race id")
            seen.add(rid)
            finish=labels.get(rid)
            if finish is None:continue
            m=market(r,finish)
            if m is None:continue
            if year in TRAIN:
                train_total+=1
                for t,o in m["prices"].items():
                    keys=tuple(m["rank"][c] for c in t)
                    expected[keys]+=(1/o)/m["invsum"]
                actual[tuple(m["rank"][c] for c in finish)]+=1
            elif year in TEST:
                test_total+=1
                # True reservoir sample of verified usable races, not payout conditioned.
                if len(reservoir)<3:reservoir.append((year,rid,finish,m))
                else:
                    j=rng.randrange(test_total)
                    if j<3:reservoir[j]=(year,rid,finish,m)
    # posteriors learn unconditional ordered-rank price error. No trainer sees
    # results in 2025 or 2026.
    factors={}
    all_keys=set(expected)|set(actual)
    for k in all_keys:
        factors[k]=(actual[k]+SMOOTH_EXPECTED)/(expected[k]+SMOOTH_EXPECTED)
    return actual,expected,factors,reservoir,train_total,test_total

def calibrated_probs(m,factors):
    base={t:(1/o)/m["invsum"] for t,o in m["prices"].items()}
    weights={t:base[t]*factors.get(tuple(m["rank"][c] for c in t),1.) for t in base}
    z=sum(weights.values())
    return {t:v/z for t,v in weights.items()}

def possible_sets(cars):
    choices={k:list(itertools.combinations(cars,k)) for k in range(1,5)}
    # Arbitrary independent sets per finish slot, up to 12 valid unique tickets;
    # deliberately NOT a catalog of previous support-rank formation patterns.
    size_triples=set()
    for ka in range(1,5):
        for kb in range(1,5):
            for kc in range(1,5):
                if ka*kb*kc<=18:size_triples.add((ka,kb,kc))
    for sizes in sorted(size_triples):
        for slots in itertools.product(*(choices[k] for k in sizes)):
            t=tuple(sorted(x for x in itertools.product(*slots) if len(set(x))==3))
            if MIN_CANDIDATE_POINTS<=len(t)<=MAX_CANDIDATE_POINTS:
                yield slots,t

def find_candidates(m,probs):
    p=m["prices"];cars=m["cars"]
    edge={t:probs[t]*p[t] for t in p}
    potential={t:max(0.,edge[t]-MIN_GROSS_EV)*probs[t] for t in p}
    # Stream millions of possible Cartesian boxes without keeping the entire
    # candidate universe in RAM. Deduplicate only the top-scoring distinct
    # concrete ticket sets; zero-edge boxes never enter the heap.
    ranked=[];tested=0;positive_boxes=0
    serial=0
    TOP_HEAP=2048
    for slots,tickets in possible_sets(cars):
        tested+=1
        positive=[t for t in tickets if edge[t]>MIN_GROSS_EV]
        if not positive:continue
        positive_boxes+=1
        score=sum(potential[t] for t in positive) - .000002*len(tickets)
        score*=1 if len(positive)>=2 else .5
        serial+=1
        item=(score,serial,slots,tickets,positive)
        if len(ranked)<TOP_HEAP:heapq.heappush(ranked,item)
        elif score>ranked[0][0]:heapq.heapreplace(ranked,item)
    ordered=sorted(ranked,key=lambda r:(-r[0],len(r[3]),r[2]))
    unique=[];seen_top=set()
    for score,_,slots,tickets,positive in ordered:
        if tickets in seen_top:continue
        seen_top.add(tickets)
        unique.append((score,slots,tickets,positive))
        if len(unique)>=CANDIDATE_TOP:break
    return unique,tested,positive_boxes,edge

def kelly(tickets,m,probs):
    # exact portfolio objective for disjoint mutually exclusive ordered outcomes.
    # Stake fraction_i = fraction of a reference experimental bankroll.
    p=[probs[t] for t in tickets]
    o=[m["prices"][t] for t in tickets]
    P=sum(p)
    if not tickets or max(x*y for x,y in zip(p,o))<=1:
        return [0.]*len(tickets),0.
    def loggrow(f):
        S=sum(f);cash=1-S
        if cash<=0:return -1e5
        return sum(q*math.log(cash+f_i*odd) for f_i,q,odd in zip(f,p,o))+(1-P)*math.log(cash)
    def jac(f):
        S=sum(f);cash=1-S
        den=[cash+fi*oi for fi,oi in zip(f,o)]
        global_loss=sum(q/d for q,d in zip(p,den))+(1-P)/cash
        return [-(q*oi/d-global_loss) for q,oi,d in zip(p,o,den)]
    start=[0.]*len(tickets)
    opt=minimize(lambda f:-loggrow(f),start,
      jac=jac,bounds=[(0,FRACTION_RACE_CAP)]*len(tickets),
      constraints=[{"type":"ineq","fun":lambda f:FRACTION_RACE_CAP-sum(f),
                    "jac":lambda f:[-1.]*len(f)}],
      method="SLSQP",options={"maxiter":160,"ftol":1e-12})
    if not opt.success:return start,0.
    return list(opt.x),loggrow(opt.x)

def cash_round(f):
    # Nearest 100yen, 0 allowed. Does not force betting on each of the
    # initial rectangle's tickets.
    return [max(0,100*int(math.floor(x*RISK_BANKROLL_YEN/100+0.5))) for x in f]

def shape_cover(tickets):
    # Greedy exact cover of positive-stake tickets with full Cartesian boxes.
    # Strictly no ticket outside purchased set, no duplicated purchased ticket.
    paid=set(tickets);left=set(paid);parts=[]
    while left:
        a=sorted({t[0] for t in left});b=sorted({t[1] for t in left});c=sorted({t[2] for t in left})
        ra=[tuple(x for i,x in enumerate(a) if mask>>i&1) for mask in range(1,1<<len(a))]
        rb=[tuple(x for i,x in enumerate(b) if mask>>i&1) for mask in range(1,1<<len(b))]
        rc=[tuple(x for i,x in enumerate(c) if mask>>i&1) for mask in range(1,1<<len(c))]
        best=None;best_t=None
        for A,B,C in itertools.product(ra,rb,rc):
            tt={t for t in itertools.product(A,B,C) if len(set(t))==3}
            if not tt or not tt.issubset(left):continue
            if best_t is None or len(tt)>len(best_t):
                best=(A,B,C);best_t=tt
        assert best and best_t
        parts.append({"1着":list(best[0]),"2着":list(best[1]),"3着":list(best[2]),
          "formation":"-".join("".join(map(str,side)) for side in best),
          "points":len(best_t)})
        left-=best_t
    assert sum(x["points"] for x in parts)==len(paid)
    return parts

def evaluate(yr,rid,finish,m,factors):
    prob=calibrated_probs(m,factors)
    candidates,tested,positive_boxes,edge=find_candidates(m,prob)
    # Pick candidate by ex-ante *cash-rounded* objective (not actual outcome).
    top=[]
    for score,slots,full,positive in candidates:
        # solve fractional Kelly within positive-estimated-edge entries; the
        # resulting bought set can be the subset of a larger initial geometry
        eligible=[t for t in full if edge[t]>MIN_GROSS_EV]
        fractions,gain=kelly(eligible,m,prob)
        money=cash_round(fractions)
        stake_by_t={t:s for t,s in zip(eligible,money) if s>=100}
        spending=sum(stake_by_t.values())
        if spending>int(RISK_BANKROLL_YEN*FRACTION_RACE_CAP):
            raise AssertionError("stake exceeds per-race risk cap")
        expected_profit=sum(s*(edge[t]-1) for t,s in stake_by_t.items())
        top.append((expected_profit,gain,score,slots,full,stake_by_t))
    top.sort(key=lambda x:(-x[0],-x[1],-x[2],x[3]))
    if top:
        profit,gain,score,slots,original,stakes=top[0]
        # no forced bet; no expected gain => all zero
        selected=bool(stakes) and profit>0
        if not selected:stakes={}
        diagnostic=[{"rank":i+1,
            "geometry":"-".join("".join(map(str,a)) for a in item[3]),
            "original_valid_points":len(item[4]),
            "ex_ante_expected_profit_yen":round(item[0],3),
            "risk_fractional_log_growth":round(item[1],9)} for i,item in enumerate(top[:5])]
        initial=slots;initial_set=original
    else:
        selected=False;stakes={};diagnostic=[];initial=None;initial_set=[]
    purchased=sorted(stakes)
    payment=sum(stakes.values())
    win_stake=stakes.get(finish,0)
    payout=round(win_stake*m["prices"][finish]) if win_stake else 0
    return {
      "race_id":rid,"date_year":yr,"random_selection_seed":SEED,
      "car_count":m["n"],
      "rank_support_order":m["ordered"],"favorite":"-".join(map(str,m["favorite"])),
      "favorite_odds":m["prices"][m["favorite"]],
      "market_overround":round(m["invsum"],5),
      "all_market_formations_tested":tested,
      "estimated_positive_edge_geometries":positive_boxes,
      "decision":"ADOPT_EXPERIMENTAL" if selected else "SKIP",
      "initial_full_rectangle":{
        "formation":"-".join("".join(map(str,a)) for a in initial),
        "valid_points":len(initial_set)} if initial else None,
      "purchased_formations_union":shape_cover(purchased) if purchased else [],
      "ticket_stakes":[{"ticket":"-".join(map(str,t)),
        "odds":m["prices"][t],"market_probability":round(1/m["prices"][t]/m["invsum"],7),
        "calibrated_probability":round(prob[t],7),
        "estimated_gross_roi_pct":round(edge[t]*100,3),
        "yen":stakes[t]} for t in purchased],
      "total_stake_yen":payment,
      "ex_ante_expected_profit_yen":round(sum(s*(edge[t]-1) for t,s in stakes.items()),2),
      "settlement_actual_finish":"-".join(map(str,finish)),
      "settlement_hit":bool(win_stake),
      "settlement_payout_yen":payout,
      "settlement_profit_yen":payout-payment,
      "top_5_candidate_rectangles_before_settlement":diagnostic
    }

def main():
    labels=load_labels()
    actual,expected,factors,picked,train_n,test_n=train_and_select(labels)
    if train_n<27000 or test_n<17000:raise RuntimeError("Unexpectedly small whole-history universe")
    result=[evaluate(*row,factors) for row in picked]
    spent=sum(r["total_stake_yen"] for r in result)
    pay=sum(r["settlement_payout_yen"] for r in result)
    d={"status":"THREE_RANDOM_RACES_APPLIED_AND_SETTLED",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "pre_registered_seed":SEED,
        "selection_rule":"Reservoir random 3 distinct from every complete, result-verifiable 5-9car 2025-26H1 market; actual finishes NEVER used for selection or stake estimation.",
        "training_races":train_n,"random_pool_races":test_n,
        "research_protocol":{
           "probabilities":"Historical 2021-2024 realized ranks vs market-normalized implied rank triplet frequency, shrink expected count 25; probabilities renormalized per race",
           "formation":"Search all 1..4-car slot subsets with product slot sizes<=18, keep 2..12 actual ordered tickets, no predetermined line/template; rank first7? NO, use actual car sets.",
           "candidate":"Highest estimated net cash expected value after optimizing selected geometry, before any result is observed",
           "staking":"Mutually-exclusive multi-ticket Kelly log-growth optimization with 2% TOTAL bankroll exposure cap; 100yen rounding; max risk bankroll example 100,000yen, 0 stake allowed.",
           "market_safety":"Only tickets with model p*odds >1.025 allowed inside a bet; if no positive rounded-size set, skip.",
           "model_is_unvalidated":"In-sample calibrator with final odds, prior studies have inspected later periods; no guarantee of profit or real-world availability.",
           "not_modified":"No live buy rules, predictor, or websites modified."},
        "races":result,
        "totals":{"races_random":len(result),"races_bought":sum(r["decision"]!="SKIP" for r in result),
                  "stake_yen":spent,"payout_yen":pay,"profit_yen":pay-spent,
                  "roi_pct":round(100*pay/spent,3) if spent else None}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"result.json").write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RANDOM_3_RESULTS",json.dumps(d,ensure_ascii=False),flush=True)

if __name__=="__main__":main()

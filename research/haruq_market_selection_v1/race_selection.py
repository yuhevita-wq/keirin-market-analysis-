"""HaruQ market race-selection study. FINAL-odds archival diagnostics ONLY.
Never modifies flagship HaruQ. 2024 tunes the race gate; 2025/2026 test
the frozen gate. Winning results are only used AFTER choosing the ticket.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np

GATES = (
    "max_ev_105", "max_ev_110", "max_ev_120", "robust_ev_105",
    "robust_ev_110", "max_ev_110_and_cross", "max_ev_110_and_sparse",
    "max_ev_110_and_robust_100", "robust_ev_105_and_cross")
STAKE = 100

def probs(logq, x, theta):
    z = np.asarray(logq, dtype=np.float64) + np.asarray(x, dtype=np.float64) @ np.asarray(theta, dtype=np.float64)
    z -= np.max(z)
    p = np.exp(z)
    return p/p.sum()

def research_row(r, theta):
    p = probs(r.logq,r.x,theta)
    market = probs(r.logq,r.x,np.zeros_like(theta))
    ev = p*r.odds
    ev_shrunken = np.minimum.reduce([
        probs(r.logq,r.x,theta*v)*r.odds for v in (0.50,0.75,1.00)])
    if not np.isclose(p.sum(),1.0,atol=1e-6) or not np.isfinite(ev).all():
        raise ValueError("Invalid probability mass or market book")
    j,k = int(np.argmax(ev)),int(np.argmax(ev_shrunken))
    # r.winner is not accessed before j,k selection.
    w = int(r.winner)
    return {
        "id":str(r.rid),"date":str(r.date),"n":int(len(ev)),
        "market_q_winner":float(market[w]),"model_p_winner":float(p[w]),
        "odds_j":float(r.odds[j]),"odds_k":float(r.odds[k]),
        "hit_j":int(j==w),"hit_k":int(k==w),
        "max_ev":float(ev[j]),"max_robust_ev":float(ev_shrunken[k]),
        "robust_ev_j":float(ev_shrunken[j]),
        "positive_count":int((ev>=1.10).sum()),
        "cross":float(abs(r.x[j,1])) if r.cross_market else None,
    }

def limits(rows):
    cross=[r["cross"] for r in rows if r["cross"] is not None]
    counts=[r["positive_count"] for r in rows]
    return {
        "cross_q75":float(np.quantile(cross,0.75)) if cross else float("inf"),
        "count_median":float(np.median(counts)) if counts else float("inf")}

def decision(r, name, lm):
    if name=="max_ev_105":return r["max_ev"]>=1.05,"j"
    if name=="max_ev_110":return r["max_ev"]>=1.10,"j"
    if name=="max_ev_120":return r["max_ev"]>=1.20,"j"
    if name=="robust_ev_105":return r["max_robust_ev"]>=1.05,"k"
    if name=="robust_ev_110":return r["max_robust_ev"]>=1.10,"k"
    if name=="max_ev_110_and_cross":
        return r["max_ev"]>=1.10 and r["cross"] is not None and r["cross"]>=lm["cross_q75"],"j"
    if name=="max_ev_110_and_sparse":
        return r["max_ev"]>=1.10 and r["positive_count"]<=lm["count_median"],"j"
    if name=="max_ev_110_and_robust_100":
        return r["max_ev"]>=1.10 and r["robust_ev_j"]>=1.0,"j"
    if name=="robust_ev_105_and_cross":
        return r["max_robust_ev"]>=1.05 and r["cross"] is not None and r["cross"]>=lm["cross_q75"],"k"
    raise ValueError(name)

def ledger(rows,name,lm):
    chosen=[(r,k) for r in rows for take,k in [decision(r,name,lm)] if take]
    hits=sum(r["hit_"+k] for r,k in chosen)
    paid=sum(r["odds_"+k]*STAKE for r,k in chosen if r["hit_"+k])
    n=len(chosen)
    return {
       "races_seen":len(rows),"races_selected":n,
       "nonparticipating_races":len(rows)-n,
       "hits":hits,"stake_yen":n*STAKE,
       "final_odds_payout_proxy_yen":round(paid,2),
       "final_odds_roi_percent":round(100*paid/(n*STAKE),3) if n else None,
       "official_payouts_verified":False,"T15_price_verified":False}

def halves(rows,name,lm):
    return {str(h):ledger([r for r in rows if (int(r["date"][5:7])<=6)==(h==1)],name,lm) for h in (1,2)}

def select_gate(rows,lm):
    report={}
    for name in GATES:
        a,b=ledger(rows,name,lm),halves(rows,name,lm)
        qualified=(a["races_selected"]>=150 and a["hits"]>=6 and all(
            d["races_selected"]>=40 and d["hits"]>=2 and
            d["final_odds_roi_percent"] is not None and d["final_odds_roi_percent"]>100
            for d in b.values()))
        report[name]={"eligible":qualified,"whole":a,"halves":b}
    options=[name for name in GATES if report[name]["eligible"]]
    return (max(options,key=lambda n:(report[n]["whole"]["final_odds_roi_percent"],
                    report[n]["whole"]["races_selected"],-GATES.index(n))) if options else None),report

def calibration(rows):
    if not rows:return None
    diffs=np.array([math.log(max(r["model_p_winner"],1e-30))-
                    math.log(max(r["market_q_winner"],1e-30)) for r in rows])
    mean=float(np.mean(diffs))
    se=float(np.std(diffs,ddof=1)/math.sqrt(len(diffs)))
    return {"races":len(rows),"logloss_gain_vs_market":mean,
            "standard_error":se,"normal_95pct_interval":[mean-1.96*se,mean+1.96*se]}

def test():
    from types import SimpleNamespace
    import itertools
    combos=list(itertools.permutations(range(1,8),3))
    oo=np.array([100+0.1*i for i in range(len(combos))],dtype=float)
    q=1/oo;q=q/q.sum()
    xx=np.zeros((len(oo),3));xx[:,0]=np.log(q)/5;xx[:,1]=np.linspace(-1,1,len(oo))
    rr=SimpleNamespace(logq=np.log(q),x=xx,odds=oo,winner=0,
                       rid="demo",date="2024-04-01",cross_market=True)
    theta=np.array([0.1,-0.1,0.])
    before=research_row(rr,theta)
    rr.winner=12
    after=research_row(rr,theta)
    for k in ("max_ev","max_robust_ev","positive_count","cross"):
        assert before[k]==after[k],k
    assert before["n"]==210
    assert abs(probs(rr.logq,rr.x,theta).sum()-1)<1e-9
    lm={"cross_q75":0.1,"count_median":100}
    assert all("hits" in ledger([before,after],n,lm) for n in GATES)
    print("MARKET_RACE_SELECTION_SELF_TEST_PASS")

def run():
    from research.market_probability_v1 import train as old
    years,counts,folders=old.get_records(old.ROOT)
    for y in (2023,2024,2025,2026):
        if not years[y]:raise RuntimeError("Missing training/evaluation year "+str(y))
    data={y:old.pack(years[y]) for y in (2023,2024,2025,2026)}
    # 0.02 comes from previous probability-only 2024 validation,
    # and is NOT optimized on 2025 or 2026 ROI.
    lam=0.02
    p24=old.optimize(data[2023],lam)
    v24=[research_row(r,p24) for r in years[2024]]
    lm=limits(v24)
    chosen,val=select_gate(v24,lm)
    print("VALIDATION_SELECTED_GATE",chosen,flush=True)
    p25=old.optimize(old.join((data[2023],data[2024])),lam)
    p26=old.optimize(old.join((data[2023],data[2024],data[2025])),lam)
    out={
      "status":"RESEARCH_FINAL_ODDS_NOT_EXECUTABLE",
      "source":"Existing market_probability_v1 retrospective S-class 7-or-fewer-car model",
      "scope":"complete archived S-class 3連単 markets and race cards only; not all grades",
      "train_2023":"model for 2024 selection",
      "validation_2024":{"quantile_limits":lm,"predeclared_candidates":val,
                         "gate_fixed_before_tests":chosen},
      "test_2025":"model fitted on 2023-24, gate frozen from 2024",
      "test_2026_h1":"model fitted on 2023-25, same gate (period seen in prior research)",
      "data_counts":{str(y):len(years[y]) for y in years},
      "discovery_counts":counts,"tests":{},
      "method":"P(c)=q(c)*exp(X(c)*theta)/sum_all_orders(q*exp(X*theta)); sum P=1; EV=P*O",
      "stake_per_selected_race_yen":STAKE,
      "limits":[
        "NO T-15 odds: uses final prices only",
        "Displayed final-odds payouts are proxies, NOT verified official settled dividends",
        "2026H1 previously examined, not untouched",
        "2024 gate selection using observed returns risks overfit",
        "Retrospective race-card field timestamps are not independently certified",
        "One-ticket-per-race is controlled experimental comparison, NOT a HaruQ purchase-rule change",
        "No user funds, betting APIs, frozen flagship model, web or production files modified"
      ]
    }
    for name,year,theta in (("2025",2025,p25),("2026H1",2026,p26)):
        rows=[research_row(r,theta) for r in years[year]]
        out["tests"][name]={
            "calibration":calibration(rows),
            "all_predeclared_gates":{n:ledger(rows,n,lm) for n in GATES},
            "selected_gate":ledger(rows,chosen,lm) if chosen else None,
            "selected_gate_halves":halves(rows,chosen,lm) if chosen else None,
            "market_error_score_median":{
                "max_EV":float(np.median([r["max_ev"] for r in rows])),
                "robust_EV":float(np.median([r["max_robust_ev"] for r in rows]))}}
    path=Path("data/audits/haruq_market_race_selection_v1.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("MARKET_RACE_SELECTION_COMPLETE",json.dumps({
      n:x["selected_gate"] for n,x in out["tests"].items()},ensure_ascii=False),flush=True)

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--self-test",action="store_true")
    a=pa.parse_args()
    test() if a.self_test else run()

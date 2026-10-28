"""Full-archive test of previous-result / next-favorite mispricing hypotheses.
Retrospective: every year already observed. No strategy or AI modification.
"""
import sys,os,json,math
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.casino_sequence_observation_20261028 import observe as base
OUT=ROOT/"research/casino_sequence_hypothesis_20261028"
BANDS=["5","10","20","50","100","300","1000","above1000"]
def capture(rid,day,books,finish,rtype,source):
    base.capture(rid,day,books,finish,rtype,source)
    if rid not in base.RECORDS:return
    for kind,m in base.RECORDS[rid]["markets"].items():
        b=books[kind];lo=min(b.values());tied=[t for t,o in b.items() if o==lo]
        target=(tuple(finish[:2]) if kind=="2shatan" else tuple(sorted(finish[:2]))
          if kind=="2shafuku" else tuple(finish) if kind=="3rentan" else tuple(sorted(finish)))
        # One unit split evenly across all tied minimum-odds tickets.
        hit=float(target in tied)/len(tied)
        m.update(favorite_odds=lo,favorite_hit=hit,favorite_return=hit*lo,
          favorite_ties=len(tied))
def pairs():
    groups=defaultdict(dict)
    for r in base.RECORDS.values():
        key=(r["day"],r["venue"])
        if r["number"] in groups[key]:raise RuntimeError("Duplicate race number")
        groups[key][r["number"]]=r
    result=[]
    for races in groups.values():
        for n,r in races.items():
            if n-1 in races:result.append((races[n-1],r))
    return result
def raw_report(ps):
    cells=defaultdict(Counter)
    for p,c in ps:
        for kind,m in c["markets"].items():
            b=base.band(p["markets"][kind]["winning_odds"])
            for y in ("ALL",c["day"][:4]):
                s=cells[y,kind,b];s["n"]+=1
                for k,v in {"support":m["top"],"hit":m["favorite_hit"],
                  "return":m["favorite_return"],"odds":m["favorite_odds"],
                  "calibration":m["favorite_hit"]-m["top"],
                  "tie_races":int(m["favorite_ties"]>1)}.items():s[k]+=v
    return [{"year":y,"market":k,"previous_band":b,"n":s["n"],
      **{v:s[v]/s["n"] for v in ["support","hit","return","odds","calibration"]},
      "tie_races":int(s["tie_races"])} for (y,k,b),s in sorted(cells.items())]
def fixed_effects(ps,kind):
    # The continuous predictor uses ALL observed previous-price bands.
    # A log-odds slope is a prespecified summary, not assumed causal.
    groups=defaultdict(list)
    for p,c in ps:
        m=c["markets"][kind]
        oddsbin=math.floor(math.log(m["favorite_odds"])/0.1)
        key=(c["day"][:4],c["venue"],c["number"],c["type"],c["cars"],oddsbin)
        x=math.log(p["markets"][kind]["winning_odds"])
        groups[key].append((c["day"],x,m["favorite_hit"]-m["top"],m["favorite_return"]))
    # Within-stratum demean both predictor AND outcome.
    sx=0.;sxy=np.zeros(2);clusters=defaultdict(lambda:np.zeros(3))
    used=0;varying=0;rows=[]
    for data in groups.values():
        arr=np.array([r[1:] for r in data],dtype=float)
        d=arr-arr.mean(axis=0)
        if float(d[:,0]@d[:,0])<1e-14:continue
        used+=len(data);varying+=1
        sx+=float(d[:,0]@d[:,0]);sxy+=d[:,0]@d[:,1:]
        rows.append((data,d))
    if sx<=0:return {"status":"NO_WITHIN_STRATUM_VARIATION"}
    beta=sxy/sx
    for data,d in rows:
        score=d[:,0,None]*(d[:,1:]-d[:,0,None]*beta)
        for original,sc in zip(data,score):clusters[original[0]][:2]+=sc
    G=len(clusters)
    scores=np.array([v[:2] for v in clusters.values()])
    se=np.sqrt(np.sum(scores*scores,axis=0)*G/max(1,G-1))/sx
    outcomes=[]
    for i,name in enumerate(["hit_minus_market_support","return_units"]):
        b=float(beta[i]);s=float(se[i])
        p=math.erfc(abs(b/s)/math.sqrt(2)) if s>0 else None
        outcomes.append({"outcome":name,"slope_per_log_previous_odds":b,
          "cluster_daily_se":s,"ci95_normal":[b-1.96*s,b+1.96*s],
          "effect_per_doubling_previous_odds":b*math.log(2),"two_sided_p":p})
    return {"status":"ESTIMATED","market":kind,"all_pairs":len(ps),
      "used_pairs_within_varying_strata":used,"excluded_no_variation":len(ps)-used,
      "varying_strata":varying,"date_clusters":G,"outcomes":outcomes}
def save():
    ps=pairs()
    tests=[fixed_effects(ps,k) for k in ("2shatan","2shafuku","3rentan","3renpuku")]
    # Holm adjustment across 4 markets x 2 outcomes, not across raw descriptive cells.
    items=[o for t in tests for o in t.get("outcomes",[]) if o["two_sided_p"] is not None]
    ordered=sorted(items,key=lambda o:o["two_sided_p"]);running=0.
    for i,o in enumerate(ordered):
        running=max(running,min(1.,(len(ordered)-i)*o["two_sided_p"]))
        o["holm_p_eight_tests"]=running
    report={"status":"COMPLETE","generated_utc":datetime.now(timezone.utc).isoformat(),
      "source_commit":os.environ.get("GITHUB_SHA"),"source_audit":dict(base.AUDIT),
      "archive_audit":dict(base.old.audit),"adjacent_pairs":len(ps),
      "hypotheses":["Lower previous realized odds predict worse next-favorite calibration and return.",
        "Higher previous realized odds predict better next-favorite calibration and return.",
        "Alternative: association disappears after conditioning on current market and race context."],
      "fixed_before_run":{"scope":"All available validated non-girls four-market 2021-01-01..2026-06-30",
        "favorite":"Minimum-odds ticket; total unit split uniformly across minimum-odds ties.",
        "raw_bands":BANDS,"primary_predictor":"log(previous winning odds), all bands",
        "expected_sign":"positive for both calibration and return slopes",
        "fixed_effects":["year","venue","current_race_number","current_race_type","current_car_count",
          "floor(log(current favorite odds)/0.1)"],
        "inference":"Within-stratum OLS, calendar-date clustered sandwich SE; normal 95% CI; Holm over eight tests."},
      "raw":raw_report(ps),"adjusted":tests,
      "orientation_checks":list(base.corrected.VALIDATED.values()),
      "limits":["Retrospective hypothesis test on already observed years; no untouched holdout claim.",
        "Final odds return proxy, not official payouts, and not executable pre-close betting results.",
        "No before/after odds snapshots or confirmed preceding-result publication times.",
        "No individual betting histories; cannot identify psychological mechanism or hot hand.",
        "Rider score/line composition not controlled; context strata and market price do not remove all confounding.",
        "Log-linear slope is a summary; raw price-band tables must also be examined for nonlinearity.",
        "Odds-bin conditioning is approximate and sparse strata reduce adjusted sample, explicitly counted.",
        "Daily clustering addresses within-date dependence, not all cross-date dependence.",
        "Inverse odds normalized within market are support proxies; hit-minus-support is calibration, not profit.",
        "Same observations create correlated market tests; Holm remains valid without independence.",
        "No plaintext source quotes or keys published; no archive, AI or purchase-rule changes."]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    compact={k:v for k,v in report.items() if k not in ("raw","orientation_checks")}
    compact["raw"]=[r for r in report["raw"] if r["year"]=="ALL"]
    (OUT/"compact.json").write_text(json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(compact,ensure_ascii=False),flush=True)
def progress(status):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"progress.json").write_text(json.dumps({"status":status,"audit":dict(base.AUDIT),
      "archive_audit":dict(base.old.audit)},ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":
    if not os.environ.get(base.old.bridge.SECRET_ENV):raise RuntimeError("Full archive key unavailable")
    base.corrected.base_observe=capture
    base.old.save=progress
    try:
        base.old.sealed();save();progress("COMPLETE")
    except Exception:
        progress("FAILED_PARTIAL_NOT_CONCLUSION");raise

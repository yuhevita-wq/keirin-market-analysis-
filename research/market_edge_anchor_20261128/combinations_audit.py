"""Exact four-condition patterns, race-level residuals, chronological selection.
No new predictive features or monetary decisions."""
from train import load,build,baseline_matrix,LogisticRegression,OUT
import json, math
import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import product
NAMES=["U1","U2","O1","O2"]
PERIODS={"discovery":("2025-07-01","2026-01-01"),"confirmation":("2026-01-01","2026-04-01"),"test":("2026-04-01","2026-07-01")}
def aggregate(data,clean):
    d1,d2=data[1],data[2]
    masks={
      "U1":(d1,(d1.history_top2_min>0.036474499851465225)&(d1.car_max>3.5)&(d1.market_anchor_max>0.07931389287114143)),
      "O1":(d1,(d1.history_top2_min<=0.036474499851465225)&((d1.history_top2_min>=0)&(d1.history_n_min>0) if clean else True)),
      "U2":(d2,(d2.car_max>3.5)&(d2.line_position_max>2.5)&(d2.market_entropy>0.7395333647727966)),
      "O2":(d2,(d2.car_max<=3.5)&(d2.history_n_min>45.5))}
    out=d1[["rid","date"]].drop_duplicates().set_index("rid")
    for k,(d,m) in masks.items():
        z=d.loc[m,["rid","y","q","cal"]].copy()
        z["gap"]=(z.y-z.cal)*(1 if k.startswith("U") else -1)
        z["raw_gap"]=(z.y-z.q)*(1 if k.startswith("U") else -1)
        z=z.groupby("rid").agg(gap=("gap","mean"),raw_gap=("raw_gap","mean"),n=("gap","size"),actual=("y","mean"),market=("cal","mean"))
        out=out.join(z.add_prefix(k+"_"))
    out["mask"]=["".join("1" if v else "0" for v in row) for row in out[[k+"_n" for k in NAMES]].notna().to_numpy()]
    out["u"]=out[["U1_gap","U2_gap"]].mean(axis=1)
    out["o"]=out[["O1_gap","O2_gap"]].mean(axis=1)
    return out
def summary(z,vals):
    a=pd.DataFrame({"date":z.date,"x":vals}).dropna()
    if a.empty:return None
    m=float(a.x.mean());n=len(a)
    g=a.groupby("date").x.agg(["sum","size"]);D=len(g)
    se=float(np.sqrt(D/(D-1)*np.sum((g["sum"]-m*g["size"])**2))/n) if D>1 else None
    return dict(mean=m,ci95=[m-1.96*se,m+1.96*se] if se is not None else None,lower_adjusted=m-norm.ppf(1-.05/18)*se if se is not None else None,days=D)
def evaluate(d,mask):
    z=d[d["mask"]==mask];ref=d[(d["mask"]!=mask)&d.u.notna()&d.o.notna()]
    result=dict(races=len(z),coverage=len(z)/len(d),conditions={},u=summary(z,z.u),o=summary(z,z.o))
    if not len(z):return result
    lifts={"U":[],"O":[]}
    for k in NAMES:
        if mask[NAMES.index(k)]=="0":continue
        g=k+"_gap";v=ref[g].dropna();m=float(z[g].mean())
        lift=m-float(v.mean()) if len(v) else None
        if lift is not None:lifts[k[0]].append(lift)
        result["conditions"][k]=dict(gap=m,raw_gap=float(z[k+"_raw_gap"].mean()),actual=float(z[k+"_actual"].mean()),market=float(z[k+"_market"].mean()),reference_races=len(v),reference_gap=float(v.mean()) if len(v) else None,lift=lift)
    for k,v in lifts.items():
        result[k.lower()+"_lift"]=float(np.mean(v)) if v else None
    return result
def qualify(x):
    return (x["races"]>=100 and x["u"] is not None and x["o"] is not None
        and min(x["u"]["days"],x["o"]["days"])>=30
        and x["u"]["lower_adjusted"] is not None and x["o"]["lower_adjusted"] is not None
        and min(x["u"]["lower_adjusted"],x["o"]["lower_adjusted"])>0
        and all(v is not None and v>0 for v in (x.get("u_lift"),x.get("o_lift"))))
def main():
    data=build(load())
    for k in (1,2):
        d=data[k];a=d.date<"2025-07-01"
        m=LogisticRegression(max_iter=500,C=1).fit(baseline_matrix(d[a]),d.loc[a,"y"])
        d["cal"]=m.predict_proba(baseline_matrix(d))[:,1]
    del data[3]
    d=data[1];o=d.history_top2_min<=0.036474499851465225
    audit=dict(o1_candidates=int(o.sum()),missing_or_no_history=int((o&((d.history_top2_min<0)|(d.history_n_min<=0))).sum()))
    report={"audit":audit,"modes":{},"notes":["Exact presence/absence masks ordered U1,U2,O1,O2. All nine under+over masks tested.",
    "Race-equal signed candidate-average probability residuals; compare each condition with same condition in other overlap races in the same period. No payout optimization.",
    "Fixed discovery eligibility: >=100 races, >=30 days, both positive Bonferroni one-sided lower bounds, both descriptive lifts positive. Rank by minimum lift. Confirmation applies same criteria. Test remains diagnostic, previous periods reused.",
    "Clean sensitivity excludes missing/no-history O1 candidates only; original thresholds otherwise frozen. Final odds, exacta proxy, historical provenance limitations remain."]}
    csvs=[]
    for mode in ("original","observed_history_only"):
        table=aggregate(data,mode!="original")
        periods={name:table[table.date.between(lo,hi,inclusive="left")] for name,(lo,hi) in PERIODS.items()}
        masks=["".join(map(str,t)) for t in product((0,1),repeat=4) if any(t[:2]) and any(t[2:])]
        res={mask:{name:evaluate(z,mask) for name,z in periods.items()} for mask in masks}
        ranked=sorted([k for k in masks if qualify(res[k]["discovery"])],key=lambda k:min(res[k]["discovery"]["u_lift"],res[k]["discovery"]["o_lift"]),reverse=True)
        confirmed=[k for k in ranked if qualify(res[k]["confirmation"])]
        # Freeze choice before evaluating final period, do not choose on test.
        choice=confirmed[0] if confirmed else None
        for k,z in periods.items():
            z=z.copy();z["period"]=k;z["mode"]=mode;z["selected"]=z["mask"].eq(choice) if choice else False;csvs.append(z.reset_index())
        report["modes"][mode]=dict(all_races={k:len(v) for k,v in periods.items()},patterns=res,discovery_ranked=ranked,confirmed=confirmed,selected=choice,test_pass=bool(choice and qualify(res[choice]["test"])))
    pd.concat(csvs,ignore_index=True).to_csv(OUT/"race_patterns.csv.gz",index=False,compression="gzip")
    (OUT/"combination_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print("COMBINATION_RESULT",json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=="__main__":main()

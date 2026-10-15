"""Odds-only 3連複 probability correction using 3連単 pooled probabilities.

The alpha value is selected only on 2024 and frozen for 2025/2026 tests.
Features: trio and trifecta final odds. Labels: actual 3-rider set only.
"""
from __future__ import annotations
import json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from train import ROOT,gather

OUT=ROOT/"data/audits/odds_only_trio_alpha_sweep_2023_2026.json"

def loss(r,alpha):
    # This is a geometric interpolation/extrapolation, normalised per market.
    l=np.log(r.trio_q.astype(np.float64))*(1-alpha)+np.log(r.trio_p.astype(np.float64))*alpha
    m=l.max()
    z=np.exp(l-m)
    z/=z.sum()
    return -math.log(max(float(z[r.trio_y]),1e-30))

def measure(rows,alpha,relative_to=1.0):
    a=np.array([loss(r,alpha) for r in rows],dtype=np.float64)
    b=np.array([loss(r,relative_to) for r in rows],dtype=np.float64)
    delta=b-a
    m=float(delta.mean())
    day=defaultdict(list)
    for r,v in zip(rows,delta):
        day[r.date].append(float(v-m))
    g=len(day);n=len(delta)
    se=math.sqrt((g/(g-1))*sum(sum(v)**2 for v in day.values()))/n if g>1 else None
    return {"races":n,"alpha":float(alpha),"logloss":float(a.mean()),
            "improvement_vs_trifecta_aggregate":m,
            "date_cluster_se":se,
            "approx_95pct_ci_gain":[m-1.96*se,m+1.96*se] if se else None}

def main():
    years,skipped=gather()
    testing=years[2024]
    f=lambda alpha:sum(loss(r,alpha) for r in testing)/len(testing)
    res=minimize_scalar(f,bounds=(-1,4),method="bounded",options={"xatol":0.005})
    alpha=float(res.x)
    sweep={}
    for y in (2023,2024,2025,2026):
        data=years[y]
        sweep[str(y)]={str(v):measure(data,v) for v in (0.0,0.5,1.0,1.25,1.5,2.0,2.5,3.0,4.0)}
        sweep[str(y)]["selected"]=measure(data,alpha)
    report={
      "status":"EXPLORATORY_ODDS_ONLY_TRIO_GEOMETRIC_BLEND",
      "market_inputs_only":["trio_final_odds.csv","trifecta_final_odds.csv"],
      "labels_only":["results.csv finish_position and car_no"],
      "alpha_selection_year":2024,
      "alpha_selection_range":[-1,4],
      "selected_alpha":alpha,
      "definition":"trio q_alpha(s) proportional to q_trio(s)^(1-alpha) * q_trifecta_aggregate(s)^alpha",
      "test_years":[2025,2026],
      "races_by_year":{str(y):len(years[y]) for y in (2023,2024,2025,2026)},
      "skipped":skipped,
      "yearly":sweep,
      "warning":"Entire experiment uses archived final odds; 2024 is a tuning set. No realized profit or prospective T-15 evidence."
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ALPHA_SELECTED",alpha,"2024_LOGLOSS",res.fun,"TEST",json.dumps({y:sweep[str(y)]["selected"] for y in (2025,2026)}),flush=True)

if __name__=="__main__":
    main()

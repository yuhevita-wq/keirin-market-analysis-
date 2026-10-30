#!/usr/bin/env python3
"""Report each of 12 months of 2023 complete reverse-time V6 holdout."""
import json
from pathlib import Path
p=Path("data/audits/v6_2023_full_calendar_heldout_from_2024_fitting.json")
d=json.loads(p.read_text())
assert d["state"]=="COMPLETE_2023_HISTORICAL_HELDOUT_FROM_2024_REFIT_V6_VALIDATION"
tot={"AAA":[0.,0],"AAB":[0.,0],"ABA":[0.,0],"ABB":[0.,0],"ABC":[0.,0]}
for month,row in sorted(d["monthly_pattern_validation"].items()):
    pats=row["predicted_vs_observed_patterns"]
    n=row["evaluated"]
    z={"month":month,"source":row["source"],"eval":n,
       "patterns":{k:{"expected":round(v["expected"],3),"actual":v["actual"],
                       "delta":round(v["observed_minus_expected"],3)}
                   for k,v in sorted(pats.items())}}
    for k,v in pats.items():
        tot[k][0]+=v["expected"];tot[k][1]+=v["actual"]
    print("V6_2023_MONTHLY_FULL_VALIDATION",json.dumps(z,ensure_ascii=False),flush=True)
for k,(pred,actual) in sorted(tot.items()):
    actual_year=d["cells"]["PAT"][k]["actual"]
    model_year=d["cells"]["PAT"][k]["expected"]
    assert actual==actual_year
    assert abs(pred-model_year)<.02
print("V6_2023_ALL_12_MONTHS_RECONCILED",json.dumps({
    "races":d["coverage"]["evaluated"],
    "sources":sum(x["source"] for x in d["monthly_pattern_validation"].values()),
    "months":len(d["monthly_pattern_validation"]),
    "AAA_months_underpredicted":sum(1 for x in d["monthly_pattern_validation"].values()
            if x["predicted_vs_observed_patterns"]["AAA"]["observed_minus_expected"]>0),
    "AAA_months_overpredicted":sum(1 for x in d["monthly_pattern_validation"].values()
            if x["predicted_vs_observed_patterns"]["AAA"]["observed_minus_expected"]<0)
},ensure_ascii=False),flush=True)

#!/usr/bin/env python3
"""Read only completed original full V6 structural census; summarize role pairs.

This is POSTPROCESSING of immutable GitHub Actions artifact. No refit.
Prevent unsupported causal transfers by checking all partition sums.
"""
import json,sys
from pathlib import Path
src=Path(sys.argv[1]); dest=Path(sys.argv[2])
d=json.loads(src.read_text(encoding="utf-8"))
assert d["state"]=="FROZEN_V6_ALL_TICKET_DONOR_RECIPIENT_STRUCTURAL_AUDIT_COMPLETE"
r=d["results"]["2024_to_2026"]
assert r["races"]==60776 and r["tickets"]==13994280
def convert(row):
    return {"key":row[0],"predicted":row[1]["expected"],
            "observed":row[1]["actual"],
            "observed_minus_predicted":row[1]["observed_minus_expected"],
            "z_diagnostic":row[1]["race_binomial_z_diagnostic"],
            "mass_error_pct_points_per_race":row[1]["mass_error_pp_per_race"]}
out={"state":"V6_DONOR_RECEIVER_ROLE_PAIR_EXTRACTION_CONFIRMED",
     "run_source":"https://github.com/yuhevita-wq/keirin-market-analysis-/actions/runs/40972234043",
     "races":r["races"],"tickets":r["tickets"],"axes":{},
     "caveat":"No odds teacher, no new V6 model, no correction; years have been inspected before."}
for axis in ("PAT_R1","PAT_RK1","P_PAT_R1_R2","P_PAT_R1_RK1","P_PAT"):
    members=r["cells"][axis].items()
    assert sum(z["actual"] for _,z in members)==60776
    members=r["cells"][axis].items()
    assert abs(sum(z["expected"] for _,z in members)-60776)<.1
    ss=[]
    for key,z in r["cells"][axis].items():
        if z["expected"]<80 or abs(z["observed_minus_expected"])<25:continue
        yearly=[d["results"][yr]["cells"][axis].get(key) for yr in ("2024","2025","2026")]
        if not all(v and v["expected"]>=10 for v in yearly):continue
        diff=z["observed_minus_expected"]
        if not all((v["observed_minus_expected"]>0)==(diff>0) for v in yearly):continue
        item=convert((key,z))
        item["by_year"]={yr:{"predicted":v["expected"],"observed":v["actual"],"difference":v["observed_minus_expected"]}
                         for yr,v in zip(("2024","2025","2026"),yearly)}
        ss.append(item)
    out["axes"][axis]={
        "stable_donors":sorted((z for z in ss if z["observed_minus_predicted"]<0),
                               key=lambda z:z["observed_minus_predicted"])[:25],
        "stable_receivers":sorted((z for z in ss if z["observed_minus_predicted"]>0),
                                  key=lambda z:-z["observed_minus_predicted"])[:25]}
out["AAA_AAB_conditional"]={
    "predicted_total_1_2_same_line":r["cells"]["PAT"]["AAA"]["expected"]+r["cells"]["PAT"]["AAB"]["expected"],
    "actual_total_1_2_same_line":r["cells"]["PAT"]["AAA"]["actual"]+r["cells"]["PAT"]["AAB"]["actual"],
    "predicted_third_remains_same_line_given_1_2_same_line_pct":100*r["cells"]["PAT"]["AAA"]["expected"]/(r["cells"]["PAT"]["AAA"]["expected"]+r["cells"]["PAT"]["AAB"]["expected"]),
    "observed_third_remains_same_line_given_1_2_same_line_pct":100*r["cells"]["PAT"]["AAA"]["actual"]/(r["cells"]["PAT"]["AAA"]["actual"]+r["cells"]["PAT"]["AAB"]["actual"])
}
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
for axis,x in out["axes"].items():
    print("V6_ROLEPAIR_" +axis,json.dumps({
        "donors":x["stable_donors"][:12],"receivers":x["stable_receivers"][:12]},ensure_ascii=False),flush=True)
print("V6_AAA_AAB_THIRD_PLACE_CONDITIONAL",json.dumps(out["AAA_AAB_conditional"]),flush=True)
print("V6_ROLEPAIR_POSTPROCESS_COMPLETE")

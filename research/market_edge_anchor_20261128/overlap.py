"""Audit overlap of already frozen market residual leaves; no refitting."""
from train import load,build,study,OUT
import json
import numpy as np
import pandas as pd
def main():
    data=build(load())
    rules={
      "U1":(1,lambda d:(d.history_top2_min>0.036474499851465225)&(d.car_max>3.5)&(d.market_anchor_max>0.07931389287114143)),
      "O1":(1,lambda d:d.history_top2_min<=0.036474499851465225),
      "U2":(2,lambda d:(d.car_max>3.5)&(d.line_position_max>2.5)&(d.market_entropy>0.7395333647727966)),
      "O2":(2,lambda d:(d.car_max<=3.5)&(d.history_n_min>45.5))}
    # No missing-history candidates may silently satisfy O1.
    rows=[]
    for k,d in data.items():
        if k not in (1,2):continue
        # Fit only the frozen market baseline, exactly as in the preceding study.
        from train import baseline_matrix,LogisticRegression
        a=d.date<"2025-07-01"
        model=LogisticRegression(max_iter=500,C=1).fit(baseline_matrix(d[a]),d.loc[a,"y"])
        d["cal"]=model.predict_proba(baseline_matrix(d))[:,1]
    periods={"discovery":("2025-07-01","2026-01-01"),"confirmation":("2026-01-01","2026-04-01"),"test":("2026-04-01","2026-07-01")}
    report={}
    for label,(start,end) in periods.items():
        raceids=set(data[1].loc[data[1].date.between(start,end,inclusive="left"),"rid"])
        groups={}
        for name,(kind,fn) in rules.items():
            d=data[kind]; z=d[d.date.between(start,end,inclusive="left")&fn(d)]
            groups[name]=z
        sets={name:set(z.rid) for name,z in groups.items()}
        pairs={}
        for u in ("U1","U2"):
            for o in ("O1","O2"):
                ids=sets[u]&sets[o]
                result={"races":len(ids),"share_all":len(ids)/len(raceids),"share_under":len(ids)/len(sets[u]) if sets[u] else None}
                for name in (u,o):
                    z=groups[name][groups[name].rid.isin(ids)]
                    result[name]={"candidates":len(z),"raw_market":float(z.q.mean()) if len(z) else None,"calibrated_market":float(z.cal.mean()) if len(z) else None,"actual":float(z.y.mean()) if len(z) else None,"gap":float((z.y-z.cal).mean()) if len(z) else None}
                pairs[u+"+"+o]=result
                for rid in sorted(ids):
                    rows.append(dict(period=label,pair=u+"+"+o,race_id=rid))
        any_u=sets["U1"]|sets["U2"];any_o=sets["O1"]|sets["O2"]
        report[label]={"all_races":len(raceids),"condition_counts":{k:len(v) for k,v in sets.items()},"any_overlap":len(any_u&any_o),"overlap_excluding_O1":len(any_u&sets["O2"]),"all_four":len(set.intersection(*sets.values())),"pairs":pairs}
    pd.DataFrame(rows).to_csv(OUT/"overlap_races.csv",index=False)
    (OUT/"overlap_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print("OVERLAP_RESULT",json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Empirical survivor-line opportunity census, without model fitting.

Uses frozen outcome order only to GROUP completed races, NOT as a pre-race
feature. Exact 1st+2nd order is conditioned on when computing 3rd groups.
Counts per-group remaining eligible riders and model-agnostic uniform odds.
"""
import csv,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from model_survivor_v6 import get_h1,get_races,H2,H2RACES,H2RESULT,SEP,SEPRACES,SEPRESULT,origin,pair_case

def line_shape(r):
    lid=r["line_idx"];sz=r["size"]
    return "-".join(str(n) for n in sorted([int(sz[np.flatnonzero(lid==v)[0]]) for v in np.unique(lid)],reverse=True))
def analyze(races):
    by=defaultdict(lambda:{"n":0,"origins":Counter(),"remaining_counts":defaultdict(Counter),
                            "roles":Counter(),"opportunities":Counter(),"role_opps":Counter()})
    all_race=defaultdict(lambda:{"n":0,"origins":Counter(),"opportunities":Counter()})
    for r in races:
        sh=line_shape(r);i,j,k=r["win_idx"];pair=pair_case(r);case=(sh,pair)
        group=by[case]
        globalg=all_race[pair]
        group["n"]+=1;globalg["n"]+=1
        actual=origin(r,i,j,k)
        group["origins"][actual]+=1;globalg["origins"][actual]+=1
        # conditional outcome distribution for one 3rd among 5+ remaining
        remaining=defaultdict(list)
        for c in range(len(r["cars"])):
            if c in (i,j):continue
            category=origin(r,i,j,c)
            remaining[category].append(c)
            group["opportunities"][category]+=1
            globalg["opportunities"][category]+=1
            role={0:"leader",1:"bante"}.get(int(r["pos_idx"][c]),"third_plus")
            group["role_opps"][role]+=1
        for category,cands in remaining.items():
            group["remaining_counts"][category][str(len(cands))]+=1
        group["roles"][{0:"leader",1:"bante"}.get(int(r["pos_idx"][k]),"third_plus")]+=1
    def packing(d):
        return {k:{
            "races":v["n"],"actual_counts":dict(v["origins"]),
            "actual_shares":{x:round(y/v["n"],6) for x,y in v["origins"].items()},
            "remaining_eligible_rider_opportunities":dict(v["opportunities"]),
            "uniform_remaining_share":{x:round(y/(sum(v["opportunities"].values())),6) for x,y in v["opportunities"].items()},
            **({"third_roles":dict(v["roles"]),
                "third_role_opportunities":dict(v["role_opps"]),
                "counts_of_remaining_eligible_by_origin":{origin:dict(counts)
                      for origin,counts in v["remaining_counts"].items()}}
                 if "roles" in v else {})
        } for k,v in d.items()}
    return packing({f"{shape}|{pair}":data for (shape,pair),data in by.items()}),packing(all_race)

def main():
    sources=[
        ("fit_JulAug_2026",get_races(H2,H2RACES,H2RESULT,"h2")),
        ("test_Sep_2026",get_races(SEP,SEPRACES,SEPRESULT,"sep")),
        ("reverse_H1_2026",get_h1())
    ]
    report={}
    for name,(races,audit) in sources:
        patterns,by_pair=analyze(races)
        ordered=sorted(patterns.items(),key=lambda x:-x[1]["races"])
        report[name]={
            "races":len(races),"audit":audit,
            "case_totals":by_pair,
            "largest_patterns":dict(ordered[:20]),
            "all_patterns":patterns,
            "shape_totals":dict(Counter(line_shape(r) for r in races))
        }
    out=HERE/"remaining_line_census.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    compact={}
    for name,z in report.items():
        compact[name]={"n":z["races"],
            "shapes":dict(sorted(z["shape_totals"].items(),key=lambda k:-k[1])[:10]),
            "case_totals":z["case_totals"],
            "322":{k:v for k,v in z["all_patterns"].items() if k.startswith("3-2-2|")}}
    print("CENSUS",json.dumps(compact,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()

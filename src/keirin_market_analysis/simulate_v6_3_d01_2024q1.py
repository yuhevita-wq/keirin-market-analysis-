from __future__ import annotations

import csv
import json
from pathlib import Path

SCHEME_VERSION = "v6.3-D01"
BASE_SCHEME_VERSION = "v6.1"
DEVELOPMENT_DATASET = "2024Q1"
ANALYSIS_SOURCE = "v6.1-HM01"
STATUS = "DEVELOPMENT"
ENTRY_FILTER_NAME = "M_PRE_FORMATION_MASS"
ENTRY_FILTER_THRESHOLD = 0.35640013538348414
ENTRY_FILTER_DERIVATION = "midpoint of v6.1-HM01 hit median 0.3805313011341475 and miss median 0.3322689696328208; fixed before D01 simulation"
STAKE = 100

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "artifacts" / "v6_1_hm01_2024q1" / "v6_1_hm01_races.csv"
OUT = ROOT / "artifacts" / "v6_3_d01_2024q1"


def to_int(x):
    try: return int(float(x))
    except (TypeError, ValueError): return 0

def to_float(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def max_losing_streak(rows):
    ordered=sorted(rows,key=lambda r:(r["race_date"],r["race_id"]));cur=best=0
    for r in ordered:
        if to_int(r["hit"]):cur=0
        else:cur+=1;best=max(best,cur)
    return best

def summarize(rows):
    b=len(rows);h=sum(to_int(r["hit"]) for r in rows);t=sum(to_int(r["final_bet_count"]) for r in rows);p=sum(to_int(r["payout_yen"]) for r in rows);s=t*STAKE
    return {"bet_races":b,"hit_races":h,"miss_races":b-h,"hit_rate_pct":100*h/b if b else None,"tickets":t,"avg_tickets_per_race":t/b if b else None,"stake_yen":s,"payout_yen":p,"profit_yen":p-s,"roi_pct":100*p/s if s else None,"max_losing_streak":max_losing_streak(rows)}
def main():
    with IN_CSV.open("r",encoding="utf-8-sig",newline="") as f:base=list(csv.DictReader(f))
    selected=[];removed=[]
    for r in base:
        m=to_float(r.get("M_pre"));(selected if m is not None and m>=ENTRY_FILTER_THRESHOLD else removed).append(r)
    result={"scheme_version":SCHEME_VERSION,"base_scheme_version":BASE_SCHEME_VERSION,"development_dataset":DEVELOPMENT_DATASET,"analysis_source":ANALYSIS_SOURCE,"status":STATUS,"entry_filter":{"name":ENTRY_FILTER_NAME,"rule":f"M_pre >= {ENTRY_FILTER_THRESHOLD}","threshold":ENTRY_FILTER_THRESHOLD,"derivation":ENTRY_FILTER_DERIVATION},"base_v6_1":summarize(base),"v6_3_d01":summarize(selected),"removed_by_filter":summarize(removed),"development_warning":"2024Q1 was used to derive D01. This output remains development-only."}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"v6_3_d01_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    if selected:
        with (OUT/"v6_3_d01_selected_races.csv").open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=list(selected[0].keys()));w.writeheader();w.writerows(selected)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
    from develop_v6_4_d02_q2q3 import main as x1;x1()
    from simulate_v6_4_d02_2024q1 import main as x2;x2()
    from inspect_v6_4_d02_q1_random3 import main as x3;x3()
    from simulate_v7_0_f01_2024q1 import main as x4;x4()
    from simulate_v7_1_f02_2024q1 import main as x5;x5()
    from simulate_v7_2_f03_2024q1 import main as x6;x6()
    from simulate_v7_3_f04_2024q1 import main as x7;x7()

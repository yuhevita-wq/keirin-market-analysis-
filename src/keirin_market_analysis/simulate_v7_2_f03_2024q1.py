from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from v7_2_f03_cross_market_value import build_v7_2_f03

SCHEME_VERSION = "v7.2-F03"
DATASET = "2024Q1"
STAKE = 100
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/2024/s_class_f1_all_parts/2024_q1"
OUT = ROOT / "artifacts/v7_2_f03_2024q1"


def read_rows(name):
    with (DATA / name).open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def pint(x):
    try: return int(float(x))
    except (TypeError, ValueError): return None


def pfloat(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def parse_trio(s):
    xs = tuple(sorted(int(x) for x in (s or "").strip().replace("=", "-").replace(",", "-").split("-") if x.strip().isdigit()))
    return xs if len(xs) == 3 and len(set(xs)) == 3 else None


def parse_tf(s):
    xs = tuple(int(x) for x in (s or "").strip().replace("=", "-").replace(",", "-").split("-") if x.strip().isdigit())
    return xs if len(xs) == 3 and len(set(xs)) == 3 else None


def parse_lines(s):
    out=[]
    for raw in (s or "").strip().split("/"):
        if not raw.strip(): continue
        ps=raw.strip().split("-")
        if not all(x.strip().isdigit() for x in ps): return None
        out.append(tuple(int(x.strip()) for x in ps))
    flat=[x for line in out for x in line]
    return tuple(out) if out and len(flat)==len(set(flat)) else None


def load():
    races={r["race_id"]:r for r in read_rows("races.csv")}
    trio,tf,payouts=defaultdict(dict),defaultdict(dict),defaultdict(dict)
    for r in read_rows("trio_final_odds.csv"):
        c,o=parse_trio(r.get("combination")),pfloat(r.get("odds"))
        if c and o and o>0 and r.get("odds_status")=="available": trio[r["race_id"]][c]=o
    for r in read_rows("trifecta_final_odds.csv"):
        c,o=parse_tf(r.get("combination")),pfloat(r.get("odds"))
        if c and o and o>0 and r.get("odds_status")=="available": tf[r["race_id"]][c]=o
    for r in read_rows("payouts.csv"):
        if r.get("bet_code")!="trifecta" and r.get("ticket_type")!="3連単": continue
        if r.get("status")!="paid": continue
        c,y=parse_tf(r.get("combination")),pint(r.get("payout_yen"))
        if c and y is not None: payouts[r["race_id"]][c]=y
    return races,trio,tf,payouts


def max_streak(rows):
    cur=best=0
    for r in sorted(rows,key=lambda x:(x["race_date"],x["race_id"])):
        if r["hit"]: cur=0
        else:
            cur+=1; best=max(best,cur)
    return best


def main():
    races,trio,tf,payouts=load()
    rows=[]; fail=Counter(); sizes=Counter(); attr=defaultdict(int); attr["races_csv"]=len(races)
    for rid,r in sorted(races.items(),key=lambda kv:(kv[1].get("race_date",""),kv[0])):
        if r.get("meeting_grade")!="F1" or not (r.get("race_type") or "").startswith("Ｓ級"): continue
        attr["f1_s"]+=1
        if pint(r.get("entry_count"))!=7: continue
        attr["seven_car"]+=1
        if len(trio.get(rid,{}))!=35: continue
        attr["complete_trio35"]+=1
        if len(tf.get(rid,{}))!=210: continue
        attr["complete_tf210"]+=1
        cars=sorted({x for c in trio[rid] for x in c})
        lines=parse_lines(r.get("predicted_line_formation"))
        if len(cars)!=7 or lines is None or set(x for line in lines for x in line)!=set(cars): continue
        attr["complete_line"]+=1
        if rid not in payouts or not payouts[rid]: continue
        attr["has_tf_payout"]+=1; attr["population"]+=1
        d=build_v7_2_f03(trio[rid],tf[rid],r.get("predicted_line_formation") or "")
        if not d.get("buy"):
            fail[str(d.get("reason"))]+=1; continue
        attr["entry_pass"]+=1
        tickets=tuple(d["tickets"]); paid=payouts[rid]; win=[t for t in tickets if t in paid]
        payout=sum(paid[t] for t in win); size=f"{len(d['first'])}-{len(d['second'])}-{len(d['third'])}"; sizes[size]+=1
        rows.append({"race_id":rid,"race_date":r.get("race_date"),"track":r.get("track"),"race_no":pint(r.get("race_no")),"formation":d["formation"],"size_pattern":size,"ticket_count":d["ticket_count"],"modeled_profit_units":d["modeled_profit_units"],"modeled_roi":d["modeled_roi"],"hit":int(bool(win)),"payout_yen":payout,"winning_ticket":";".join("-".join(map(str,t)) for t in win)})
    attr["bet_races"]=len(rows)
    hits=sum(r["hit"] for r in rows); tickets=sum(r["ticket_count"] for r in rows); payout=sum(r["payout_yen"] for r in rows); stake=tickets*STAKE
    result={"scheme_version":SCHEME_VERSION,"dataset":DATASET,"entry_rule":"unchanged v6.1 pre-formation gate","formation_rule":"maximize modeled cross-market total profit over market-ranked nested formations","fixed_place_counts":False,"price_cut":False,"attrition":dict(attr),"entry_fail_reasons":dict(fail),"summary":{"bet_races":len(rows),"hit_races":hits,"miss_races":len(rows)-hits,"hit_rate_pct":100*hits/len(rows) if rows else None,"tickets":tickets,"avg_tickets_per_race":tickets/len(rows) if rows else None,"min_tickets_per_race":min((r["ticket_count"] for r in rows),default=None),"max_tickets_per_race":max((r["ticket_count"] for r in rows),default=None),"stake_yen":stake,"payout_yen":payout,"profit_yen":payout-stake,"roi_pct":100*payout/stake if stake else None,"max_losing_streak":max_streak(rows)},"size_pattern_distribution":dict(sorted(sizes.items(),key=lambda kv:(-kv[1],kv[0])))}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"v7_2_f03_2024q1_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    if rows:
        with (OUT/"v7_2_f03_2024q1_races.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("V7_2_F03_Q1_RESULT_BEGIN"); print(json.dumps(result,ensure_ascii=False,indent=2)); print("V7_2_F03_Q1_RESULT_END")


if __name__=="__main__": main()
